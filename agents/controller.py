from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from agents.common import SUPPORTED_COLLISION_SYSTEMS
from pomapf_env.target_projection import get_square_target

ARPE_BRANCH = 0
AO_BRANCH = 1
NUM_BRANCHES = 2
NUM_PRIMITIVE_ACTIONS = 5
SWITCHER_CROP_SIZE = 11
SWITCHER_SPATIAL_SHAPE = (3, SWITCHER_CROP_SIZE, SWITCHER_CROP_SIZE)
SWITCHER_COORD_DIM = 2
SWITCHER_VECTOR_DIM = 2 * SWITCHER_COORD_DIM + 2 * NUM_PRIMITIVE_ACTIONS
SWITCHER_FIELD_SHAPES = {
    "obs": SWITCHER_SPATIAL_SHAPE,
    "xy": (SWITCHER_COORD_DIM,),
    "target_xy": (SWITCHER_COORD_DIM,),
    "caar_action": (NUM_PRIMITIVE_ACTIONS,),  # 权重沿用此字段名，实际表示 ARPE 候选动作。
    "aoreplan_action": (NUM_PRIMITIVE_ACTIONS,),
}


def switcher_observation_space():
    # 输入：无。输出：Switcher 的观测空间，包含三通道局部图、两组坐标和两组动作编码。
    from gymnasium import spaces

    return spaces.Dict(
        {
            key: spaces.Box(
                -1024.0 if key in {"xy", "target_xy"} else 0.0,
                1024.0 if key in {"xy", "target_xy"} else 1.0,
                shape=shape,
                dtype=np.float32,
            )
            for key, shape in SWITCHER_FIELD_SHAPES.items()
        }
    )


def _one_hot(actions: np.ndarray) -> np.ndarray:
    # 输入：N 个动作编号 [N]。输出：对应的五维独热编码 [N,5]。
    result = np.zeros((len(actions), NUM_PRIMITIVE_ACTIONS), dtype=np.float32)
    result[np.arange(len(actions)), actions] = 1.0
    return result


def build_switcher_state(
    observations: Sequence[Mapping],
    arpe_actions: Sequence[int],
    aoreplan_actions: Sequence[int],
) -> dict[str, np.ndarray]:
    # 输入：N 个局部观测，以及 ARPE、AORePlan 各自给出的 N 个候选动作。
    # 输出：批量网络输入字典；obs 为障碍、其他智能体、目标投影三张 11×11 矩阵。
    count = len(observations)
    arpe = np.asarray(arpe_actions, dtype=np.int64).reshape(-1)
    aoreplan = np.asarray(aoreplan_actions, dtype=np.int64).reshape(-1)
    if arpe.shape != (count,) or aoreplan.shape != (count,):
        raise RuntimeError("Switcher action arrays have inconsistent lengths.")
    if np.any((arpe < 0) | (arpe >= NUM_PRIMITIVE_ACTIONS)):
        raise RuntimeError("ARPE produced an invalid primitive action.")
    if np.any((aoreplan < 0) | (aoreplan >= NUM_PRIMITIVE_ACTIONS)):
        raise RuntimeError("AORePlan produced an invalid primitive action.")

    spatial_rows = []
    xy_rows = []
    target_rows = []
    expected_crop = (SWITCHER_CROP_SIZE, SWITCHER_CROP_SIZE)
    for observation in observations:
        obstacles = np.asarray(observation["obstacles"], dtype=np.float32)
        agents = np.asarray(observation["agents"], dtype=np.float32)
        xy = np.asarray(observation["xy"], dtype=np.float32).reshape(2)
        target = np.asarray(observation["target_xy"], dtype=np.float32).reshape(2)
        if obstacles.shape != expected_crop or agents.shape != expected_crop:
            raise RuntimeError(
                "Switcher requires aligned 11x11 obstacle and agent crops."
            )
        spatial_rows.append(
            np.stack(
                [
                    np.clip(obstacles, 0.0, 1.0),
                    np.clip(agents, 0.0, 1.0),
                    get_square_target(*xy, *target, SWITCHER_CROP_SIZE // 2),
                ],
                axis=0,
            )
        )
        xy_rows.append(xy)
        target_rows.append(target)

    return {
        "obs": np.asarray(spatial_rows, dtype=np.float32),
        "xy": np.asarray(xy_rows, dtype=np.float32),
        "target_xy": np.asarray(target_rows, dtype=np.float32),
        "caar_action": _one_hot(arpe),
        "aoreplan_action": _one_hot(aoreplan),
    }


@dataclass(frozen=True)
class PreparedSwitcherStep:
    # 保存本步两套候选及网络输入；等待 resolve_actions 决定实际执行哪一个。
    arpe_actions: tuple[int, ...]
    aoreplan_actions: tuple[int, ...]
    switch_allowed_mask: tuple[bool, ...]
    switcher_state: Mapping[str, np.ndarray]


class SwitcherController:
    def __init__(self, arpe, aoreplan):
        # 输入：ARPE 策略和 AORePlan 分支对象。
        # 输出：无返回值；初始化控制器，并清空两分支的历史状态。
        self.arpe = arpe
        self.aoreplan = aoreplan
        self._pending: PreparedSwitcherStep | None = None
        self.after_reset()

    def set_grid_config(self, grid_config) -> None:
        # 输入：地图配置。输出：无返回值；检查冲突模式并配置 ARPE 的共享轨迹。
        collision_system = getattr(grid_config, "collision_system", None)
        if collision_system not in SUPPORTED_COLLISION_SYSTEMS:
            raise ValueError(
                "HRSS requires a supported collision system; received "
                f"collision_system={collision_system!r}, supported "
                f"{SUPPORTED_COLLISION_SYSTEMS}."
            )
        self.arpe.set_grid_config(grid_config)

    def set_env(self, env) -> None:
        # 输入：实际环境。输出：无返回值；供 ARPE 获取冲突处理后的真实位置。
        self.arpe.set_env(env)

    def after_reset(self) -> None:
        # 输入：无。输出：无返回值；清空两分支记忆、未提交的候选和上一回合人数。
        self.arpe.after_reset()
        self.aoreplan.reset()
        self._pending = None
        self._agent_count: int | None = None

    @staticmethod
    def _primitive_actions(actions, count: int, source: str) -> tuple[int, ...]:
        # 输入：候选动作、预期人数 count 和来源名称 source。
        # 输出：长度为 count 的整数动作元组；拒绝缺失、越界或非整数动作。
        values = np.asarray(actions, dtype=object).reshape(-1)
        if values.shape != (count,):
            raise RuntimeError(f"{source} returned the wrong number of actions.")
        converted = []
        for action in values:
            try:
                integer = int(action)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"{source} returned invalid action {action!r}.") from exc
            if integer != action or not 0 <= integer < NUM_PRIMITIVE_ACTIONS:
                raise RuntimeError(f"{source} returned invalid action {action!r}.")
            converted.append(integer)
        return tuple(converted)

    def prepare_actions(self, observations, rewards=None, dones=None, infos=None):
        # 输入：全部智能体的局部观测及上一步环境反馈。
        # 输出：PreparedSwitcherStep；生成两套候选、Switcher 输入以及非等待掩码。
        # 此时尚未执行候选，必须随后调用 resolve_actions，不能直接进入下一步。
        if self._pending is not None:
            raise RuntimeError("The previous switch decision was not applied.")
        raw_observations = tuple(observations)
        count = len(raw_observations)
        if count < 1:
            raise RuntimeError("Switcher received an empty agent batch.")
        arpe_actions = self._primitive_actions(
            self.arpe.act(raw_observations, rewards, dones, infos), count, "ARPE"
        )
        step = self.aoreplan.propose(raw_observations)
        try:
            if len(step.planned_mask) != count or not all(step.planned_mask):
                raise RuntimeError("AORePlan must provide one primitive action per agent.")
            aoreplan_actions = self._primitive_actions(step.actions, count, "AORePlan")
            switcher_state = build_switcher_state(
                raw_observations,
                arpe_actions,
                aoreplan_actions,
            )
        except Exception:
            # 构建输入失败时取消这批规划提议，避免下一步误以为它们已经执行。
            self.aoreplan.commit([False] * count)
            raise
        self._pending = PreparedSwitcherStep(
            arpe_actions=arpe_actions,
            aoreplan_actions=aoreplan_actions,
            # 仅按 wait 判断；候选动作相同也沿用现有 Switcher 调用，不改变采样序列。
            switch_allowed_mask=tuple(action != 0 for action in aoreplan_actions),
            switcher_state=switcher_state,
        )
        return self._pending

    def resolve_actions(self, branches: Sequence[int]) -> tuple[int, ...]:
        # 输入：仅非等待候选对应的分支编号，0 选 ARPE、1 选 AORePlan。
        # 输出：全部智能体的最终动作元组，并向规划器提交动作是否匹配的反馈。
        pending = self._pending
        if pending is None:
            raise RuntimeError(
                "prepare_actions() must be called before resolve_actions()."
            )
        count = len(pending.arpe_actions)
        switch_allowed = np.asarray(pending.switch_allowed_mask, dtype=bool)
        eligible_count = int(switch_allowed.sum())
        requested = np.asarray(branches, dtype=np.int64).reshape(-1)
        if requested.shape != (eligible_count,) or np.any(
            (requested < 0) | (requested >= NUM_BRANCHES)
        ):
            raise ValueError(
                "Switcher choices must match the number of non-wait AORePlan actions."
            )

        executed_ao = np.zeros(count, dtype=bool)
        executed_ao[switch_allowed] = requested == AO_BRANCH
        final_actions = tuple(
            ao if use_ao else arpe
            for arpe, ao, use_ao in zip(
                pending.arpe_actions, pending.aoreplan_actions, executed_ao
            )
        )
        # 比较实际提交的动作，而不是分支编号：选 ARPE 但动作相同，也要保留规划器反馈。
        # 这里只确认提交；是否真正移动成功，要等下一步的实际位置才能知道。
        commit = [
            actual == ao for actual, ao in zip(final_actions, pending.aoreplan_actions)
        ]
        try:
            self.aoreplan.commit(commit)
        except Exception:
            self._pending = None
            raise

        if self._agent_count is None:
            self._agent_count = count
        elif self._agent_count != count:
            raise RuntimeError("Agent count changed without an environment reset.")
        self._pending = None
        return final_actions

    def after_step(self, dones: Sequence[bool]) -> None:
        # 输入：各智能体的结束标记。输出：无返回值；转交 ARPE 并核对人数是否一致。
        flags = tuple(bool(value) for value in dones)
        self.arpe.after_step(flags)
        if self._agent_count is not None and len(flags) != self._agent_count:
            raise RuntimeError("Done mask and agent count differ.")


__all__ = [
    "AO_BRANCH",
    "ARPE_BRANCH",
    "NUM_BRANCHES",
    "NUM_PRIMITIVE_ACTIONS",
    "SWITCHER_COORD_DIM",
    "SWITCHER_CROP_SIZE",
    "SWITCHER_FIELD_SHAPES",
    "SWITCHER_SPATIAL_SHAPE",
    "SWITCHER_VECTOR_DIM",
    "PreparedSwitcherStep",
    "SwitcherController",
    "build_switcher_state",
    "switcher_observation_space",
]
