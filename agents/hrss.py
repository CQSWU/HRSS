from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from pydantic import Extra, Field

from agents.switcher import Switcher, SwitcherConfig
from agents.arpe import (
    ARPE,
    ARPEWeights,
)
from agents.controller import SwitcherController
from agents.common import AlgoBase
from planning.branch import AORePlanBranch


class HRSSConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["HRSS"] = "HRSS"
    switcher: SwitcherConfig = SwitcherConfig()
    max_planning_steps: int = Field(10_000, gt=0)


class HRSS:
    def set_grid_config(self, grid_config):
        self.controller.set_grid_config(grid_config)

    def set_env(self, env):
        self.controller.set_env(env)

    def after_step(self, dones):
        self.controller.after_step(dones)

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
        switcher_cfg = cfg.switcher.copy(
            deep=True,
            update={"seed": cfg.seed},
        )
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
        planner = planner_factory(
            max_steps=cfg.max_planning_steps,
            seed=cfg.seed,
        )
        self.controller = controller_factory(self.candidate, planner)
        self.device = getattr(self.candidate, "device", cfg.device)

    def after_reset(self):
        self.controller.after_reset()
        self.switcher.after_reset()

    def act(self, observations, rewards=None, dones=None, infos=None):
        prepared = self.controller.prepare_actions(
            observations,
            rewards,
            dones,
            infos,
        )
        switch_allowed = np.asarray(
            prepared.switch_allowed_mask,
            dtype=bool,
        )
        if np.any(switch_allowed):
            active_state = {
                key: np.asarray(value)[switch_allowed]
                for key, value in prepared.switcher_state.items()
            }
            branches = self.switcher.choose(active_state)
        else:
            branches = np.empty(0, dtype=np.int64)
        return list(self.controller.resolve_actions(branches))


__all__ = [
    "HRSS",
    "HRSSConfig",
]
