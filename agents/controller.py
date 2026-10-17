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
    "caar_action": (NUM_PRIMITIVE_ACTIONS,),
    "aoreplan_action": (NUM_PRIMITIVE_ACTIONS,),
}


def switcher_observation_space():
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
    result = np.zeros((len(actions), NUM_PRIMITIVE_ACTIONS), dtype=np.float32)
    result[np.arange(len(actions)), actions] = 1.0
    return result


def build_switcher_state(
    observations: Sequence[Mapping],
    arpe_actions: Sequence[int],
    aoreplan_actions: Sequence[int],
) -> dict[str, np.ndarray]:

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

    state = {
        "obs": np.asarray(spatial_rows, dtype=np.float32),
        "xy": np.asarray(xy_rows, dtype=np.float32),
        "target_xy": np.asarray(target_rows, dtype=np.float32),
        "caar_action": _one_hot(arpe),
        "aoreplan_action": _one_hot(aoreplan),
    }
    return state


@dataclass(frozen=True)
class PreparedSwitcherStep:
    arpe_actions: tuple[int, ...]
    aoreplan_actions: tuple[int, ...]
    switch_allowed_mask: tuple[bool, ...]
    switcher_state: Mapping[str, np.ndarray]


class SwitcherController:
    def __init__(self, arpe, aoreplan):
        self.arpe = arpe
        self.aoreplan = aoreplan
        self._pending: PreparedSwitcherStep | None = None
        self.after_reset()

    def set_grid_config(self, grid_config) -> None:
        collision_system = getattr(grid_config, "collision_system", None)
        if collision_system not in SUPPORTED_COLLISION_SYSTEMS:
            raise ValueError(
                "HRSS requires a supported collision system; received "
                f"collision_system={collision_system!r}, supported "
                f"{SUPPORTED_COLLISION_SYSTEMS}."
            )
        self.arpe.set_grid_config(grid_config)

    def set_env(self, env) -> None:
        self.arpe.set_env(env)

    def after_reset(self) -> None:
        self.arpe.after_reset()
        self.aoreplan.reset()
        self._pending = None
        self._agent_count: int | None = None

    @staticmethod
    def _coerce_arpe_actions(actions, count: int) -> tuple[int, ...]:
        values = np.asarray(actions, dtype=object).reshape(-1)
        if values.shape != (count,):
            raise RuntimeError("ARPE returned the wrong number of actions.")
        converted = []
        for action in values:
            try:
                integer = int(action)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError(f"ARPE returned invalid action {action!r}.") from exc
            if integer != action or not 0 <= integer < NUM_PRIMITIVE_ACTIONS:
                raise RuntimeError(f"ARPE returned invalid action {action!r}.")
            converted.append(integer)
        return tuple(converted)

    @staticmethod
    def _validate_aoreplan_step(step, count: int):
        fields = (
            step.actions,
            step.planned_mask,
        )
        if any(len(values) != count for values in fields):
            raise RuntimeError("AORePlan returned the wrong number of actions.")
        actions = []
        for action, planned in zip(step.actions, step.planned_mask):
            try:
                integer = int(action)
            except (TypeError, ValueError, OverflowError) as exc:
                raise RuntimeError("AORePlan must return a primitive action.") from exc
            if (
                not planned
                or integer != action
                or not 0 <= integer < NUM_PRIMITIVE_ACTIONS
            ):
                raise RuntimeError("AORePlan must return a valid primitive action.")
            actions.append(integer)
        return tuple(actions)

    def prepare_actions(self, observations, rewards=None, dones=None, infos=None):
        if self._pending is not None:
            raise RuntimeError("The previous switch decision was not applied.")
        raw_observations = tuple(observations)
        count = len(raw_observations)
        if count < 1:
            raise RuntimeError("Switcher received an empty agent batch.")
        arpe_actions = self._coerce_arpe_actions(
            self.arpe.act(raw_observations, rewards, dones, infos),
            count,
        )
        step = self.aoreplan.propose(raw_observations)
        try:
            aoreplan_actions = self._validate_aoreplan_step(step, count)
            switcher_state = build_switcher_state(
                raw_observations,
                arpe_actions,
                aoreplan_actions,
            )
        except Exception:
            self.aoreplan.commit([False] * count)
            raise
        self._pending = PreparedSwitcherStep(
            arpe_actions=arpe_actions,
            aoreplan_actions=aoreplan_actions,
            switch_allowed_mask=tuple(action != 0 for action in aoreplan_actions),
            switcher_state=switcher_state,
        )
        return self._pending

    def resolve_actions(self, branches: Sequence[int]) -> tuple[int, ...]:
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

        selected = np.full(count, ARPE_BRANCH, dtype=np.int64)
        selected[switch_allowed] = requested
        executed_ao = selected == AO_BRANCH
        final_actions = [
            pending.aoreplan_actions[index]
            if executed_ao[index]
            else pending.arpe_actions[index]
            for index in range(count)
        ]

        commit = [
            bool(pending.aoreplan_actions[index] == final_actions[index])
            for index in range(count)
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
        return tuple(int(value) for value in final_actions)

    def after_step(self, dones: Sequence[bool]) -> None:
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
