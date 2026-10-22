from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from pydantic import Extra, Field

from agents.switcher import Switcher, SwitcherConfig
from agents.arpe import ARPE, ARPEWeights
from agents.controller import SwitcherController
from agents.common import AlgoBase
from planning.branch import AORePlanBranch


class HRSSConfig(AlgoBase, extra=Extra.forbid):
    # HRSS 的设备、种子、Switcher 权重和单次搜索扩展上限。
    name: Literal["HRSS"] = "HRSS"
    switcher: SwitcherConfig = SwitcherConfig()
    max_planning_steps: int = Field(10_000, gt=0)


class HRSS:
    def __init__(
        self,
        cfg: HRSSConfig,
        *,
        candidate: ARPEWeights | None = None,
        candidate_factory: Callable = ARPE,
        planner_factory: Callable = AORePlanBranch,
        switcher_factory: Callable = Switcher,
        controller_factory: Callable = SwitcherController,
    ):
        # 输入：cfg 为运行配置，candidate 可指定 ARPE 权重；各 factory 用于构造对应组件。
        # 输出：无返回值；载入 ARPE、AORePlan 和 Switcher，并建立共用的决策控制器。
        switcher_cfg = cfg.switcher.copy(deep=True, update={"seed": cfg.seed})
        self.switcher = switcher_factory(switcher_cfg)
        if candidate is None:
            candidate = self.switcher.candidate_artifact
        if candidate is None:
            raise RuntimeError(
                "Switcher checkpoint does not contain ARPE candidate paths."
            )
        self.candidate = candidate_factory(
            candidate,
            seed=int(cfg.seed or 0),
            device=str(cfg.device),
        )
        planner = planner_factory(max_steps=cfg.max_planning_steps, seed=cfg.seed)
        self.controller = controller_factory(self.candidate, planner)
        self.device = getattr(self.candidate, "device", cfg.device)

    def set_grid_config(self, grid_config):
        # 输入：地图及冲突处理配置。输出：无返回值；初始化当前地图的共享轨迹。
        self.controller.set_grid_config(grid_config)

    def set_env(self, env):
        # 输入：已经 reset 的环境。输出：无返回值；绑定实际地图和智能体位置。
        self.controller.set_env(env)

    def after_step(self, dones):
        # 输入：各智能体的结束标记。输出：无返回值；将实际结束情况传给 ARPE。
        self.controller.after_step(dones)

    def after_reset(self):
        # 输入：无。输出：无返回值；清空两分支的回合记忆，并重置 Switcher 的采样种子。
        self.controller.after_reset()
        self.switcher.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        # 输入：全部智能体的局部观测及可选的上一步奖励、结束标记、环境信息。
        # 输出：每个智能体最终提交给模拟器的动作列表，编号为 0 等待、1 上、2 下、3 左、4 右。
        prepared = self.controller.prepare_actions(observations, rewards, dones, infos)
        switch_allowed = np.asarray(prepared.switch_allowed_mask, dtype=bool)
        # AORePlan 提议等待时直接用 ARPE，其余智能体交给 Switcher 按概率选分支。
        branches = np.empty(0, dtype=np.int64)
        if switch_allowed.any():
            branches = self.switcher.choose(
                {
                    key: np.asarray(value)[switch_allowed]
                    for key, value in prepared.switcher_state.items()
                }
            )
        return list(self.controller.resolve_actions(branches))


__all__ = [
    "HRSS",
    "HRSSConfig",
]
