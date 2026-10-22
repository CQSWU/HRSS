from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.envs.create_env import create_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from agents.common import resolve_device
from learning.config import Environment, checkpoint_experiment_config
from learning.arpe_actor_critic import InferenceCorrection
from learning.grid_memory import MultipleGridMemory
from pomapf_env.trace import SharedTrace
from pomapf_env.wrappers import MatrixObservationWrapper
from train_arpe import register_custom_components, validate_config

TRACE_RADIUS = 5


@dataclass(frozen=True)
class ARPEWeights:
    project_root: Path
    weights_path: Path
    checkpoint_path: Path
    base_weights_path: Path
    base_checkpoint_path: Path

    @classmethod
    def from_mapping(
        cls,
        mapping: Mapping[str, object],
        project_root: Path,
    ) -> "ARPEWeights":
        root = Path(project_root).resolve()
        return cls(
            project_root=root,
            **{
                field: (root / str(mapping[field])).resolve()
                for field in cls._path_fields()
            },
        )

    def as_dict(self) -> dict[str, object]:
        result = {}
        for field in self._path_fields():
            path = getattr(self, field)
            result[field] = (
                path.relative_to(self.project_root).as_posix()
                if path.is_relative_to(self.project_root)
                else str(path)
            )
        return result

    @classmethod
    def _path_fields(cls):
        return [field.name for field in fields(cls) if field.name != "project_root"]


class ARPE:
    def __init__(
        self,
        artifact: ARPEWeights,
        *,
        seed: int,
        device: str,
    ):
        self.artifact = artifact
        self.seed = int(seed)
        register_custom_components()
        config = json.loads(
            (artifact.weights_path / "config.json").read_text(encoding="utf-8")
        )
        runtime_config = deepcopy(config["full_config"])
        runtime_config["experiment_settings"]["epom_base_weights_path"] = str(
            artifact.base_weights_path
        )
        _, self.cfg = validate_config(checkpoint_experiment_config(runtime_config))
        self.cfg.base_checkpoint_path = str(artifact.base_checkpoint_path)
        env = create_env(self.cfg.env, cfg=self.cfg, env_config={})
        try:
            if "tau" not in env.observation_space.spaces:
                raise RuntimeError(
                    "ARPE requires a checkpoint trained with the separate tau observation."
                )
            self.ppo = create_actor_critic(
                self.cfg, env.observation_space, env.action_space
            )
        finally:
            env.close()
        self.device = resolve_device(str(device))
        if self.device.type == "mps":
            self.ppo.float()
        self.ppo.model_to_device(self.device)
        self.ppo.load_state_dict(self._load_checkpoint()["model"], strict=True)
        self.ppo.eval()
        self.ppo.requires_grad_(False)
        self.ppo.inference_correction = InferenceCorrection()

        env_cfg = Environment(**self.cfg.full_config["environment"])
        if int(env_cfg.tau_radius) != TRACE_RADIUS:
            raise RuntimeError(
                "Runtime tau_radius disagrees with checkpoint config: "
                f"{env_cfg.tau_radius} != {TRACE_RADIUS}"
            )
        self.grid_memory_radius = int(env_cfg.grid_memory_obs_radius)
        self.grid_memory = MultipleGridMemory()
        self.rnn_states = None
        self.aco = SharedTrace(rho=env_cfg.tau_rho)
        self.env = None
        self._numpy_rng = np.random.default_rng(self.seed)

    def _load_checkpoint(self):
        path = Path(self.artifact.checkpoint_path).expanduser()
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        self.checkpoint_path = path.resolve()
        log.info("Loading checkpoint: %s", self.checkpoint_path)
        return torch.load(
            self.checkpoint_path,
            map_location="cpu" if self.device.type == "mps" else self.device,
            weights_only=False,
        )

    def set_grid_config(self, grid_config) -> None:
        self.aco.configure_from_grid_config(grid_config, clear=True)

    def set_env(self, env) -> None:
        self.env = env
        grid_obstacles = getattr(getattr(env, "grid", None), "obstacles", None)
        if grid_obstacles is not None:
            self.aco.configure_from_obstacle_mask(
                np.asarray(grid_obstacles, dtype=bool), clear=True
            )

    def after_reset(self) -> None:
        torch.manual_seed(self.seed)
        self._clear_memory()
        self._numpy_rng = np.random.default_rng(self.seed)

    def _clear_memory(self) -> None:
        self.rnn_states = None
        self.aco.clear()
        self.grid_memory.clear()

    def _global_positions(self):
        grid = getattr(self.env, "grid", None)
        positions = getattr(grid, "positions_xy", None)
        if positions is None and grid is not None and hasattr(grid, "get_agents_xy"):
            positions = grid.get_agents_xy()
        if positions is None:
            raise RuntimeError(
                "ARPE requires global grid positions. Call set_env(env) after env.reset()."
            )
        return np.asarray(positions, dtype=np.int64)

    def act(self, observations, rewards=None, dones=None, infos=None):
        observations = deepcopy(observations)
        num_agents = len(observations)
        self.grid_memory.update(observations)
        self.grid_memory.modify_observation(observations, self.grid_memory_radius)
        observations = MatrixObservationWrapper.to_matrix(observations)
        if self.rnn_states is None or len(self.rnn_states) != num_agents:
            self.rnn_states = torch.zeros(
                [num_agents, get_rnn_size(self.cfg)],
                dtype=torch.float32,
                device=self.device,
            )
        if self.aco.tau is None:
            raise RuntimeError(
                "Trace state is not initialised. Call set_env() after env.reset() and before act()."
            )
        positions = self._global_positions()
        self.aco.observe_for_inference(
            observations, positions=positions, radius=TRACE_RADIUS
        )
        with torch.no_grad():
            obs_torch = TensorDict(
                {
                    key: torch.from_numpy(np.stack([obs[key] for obs in observations]))
                    .to(self.device)
                    .float()
                    for key in observations[0]
                }
            )
            obs_torch = prepare_and_normalize_obs(self.ppo, obs_torch)
            policy_outputs = self.ppo(obs_torch, self.rnn_states)
            self.rnn_states = policy_outputs["new_rnn_states"]
        logits = policy_outputs["action_logits"].float().cpu().numpy()
        probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return np.asarray([self._numpy_rng.choice(5, p=row) for row in probabilities])

    def after_step(self, dones) -> None:
        if all(dones):
            self._clear_memory()


__all__ = [
    "ARPEWeights",
    "ARPE",
]
