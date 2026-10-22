from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Literal, Mapping

import gymnasium as gym
import numpy as np
import torch
from pydantic import Extra
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.utils import log

from agents.controller import (
    NUM_BRANCHES,
    SWITCHER_FIELD_SHAPES,
    switcher_observation_space,
)
from agents.arpe import ARPEWeights
from agents.common import AlgoBase, resolve_device
from learning.config import checkpoint_experiment_config
from train_arpe import register_custom_components, validate_config


class SwitcherConfig(AlgoBase, extra=Extra.forbid):
    name: Literal["Switcher"] = "Switcher"
    path_to_weights: str = "weights/HRSS-Switcher-Final-1B"


class Switcher:
    def __init__(self, cfg: SwitcherConfig):
        self.cfg = cfg
        path = Path(cfg.path_to_weights)
        register_custom_components()
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        full_config = deepcopy(config["full_config"])
        declaration = full_config.pop("candidate_policy", None)
        if declaration is not None and not isinstance(declaration, dict):
            raise RuntimeError("Switcher candidate_policy declaration is malformed.")
        self.candidate_artifact = (
            ARPEWeights.from_mapping(declaration, Path(__file__).resolve().parents[1])
            if declaration is not None else None
        )

        _, flat_config = validate_config(checkpoint_experiment_config(full_config))
        if flat_config.encoder_custom != "switcher":
            raise RuntimeError("Checkpoint is not a Switcher policy.")
        if bool(flat_config.use_rnn):
            raise RuntimeError("Switcher checkpoint must be feed-forward.")

        observation_space = gym.spaces.Dict(
            {
                **switcher_observation_space().spaces,
                "switch_allowed": gym.spaces.Box(
                    0.0, 1.0, shape=(1,), dtype=np.float32
                ),
            }
        )
        action_space = gym.spaces.Discrete(NUM_BRANCHES)
        actor = create_actor_critic(flat_config, observation_space, action_space)
        self.device = resolve_device(str(cfg.device).lower())
        actor.model_to_device(self.device)

        self.checkpoint_path = self._resolve_checkpoint(
            path / f"checkpoint_p{flat_config.policy_index}"
        )
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        actor.load_state_dict(checkpoint["model"])
        actor.eval()
        actor.requires_grad_(False)

        self.ppo = actor
        self.rnn_state_size = get_rnn_size(flat_config)
        self.after_reset()

    @staticmethod
    def _resolve_checkpoint(checkpoint_dir) -> Path:
        directory = Path(checkpoint_dir)
        candidates = sorted(directory.glob("checkpoint_*")) or sorted(
            directory.glob("best_*.pth")
        )
        if not candidates:
            raise FileNotFoundError(f"No Switcher checkpoint in {directory}.")
        path = Path(candidates[-1]).resolve()
        log.info("Loading Switcher checkpoint: %s", path)
        return path

    def after_reset(self) -> None:
        torch.manual_seed(int(self.cfg.seed or 0))

    def choose(self, state: Mapping[str, np.ndarray]) -> np.ndarray:
        arrays = {}
        count = None
        for key, trailing_shape in SWITCHER_FIELD_SHAPES.items():
            if key not in state:
                raise ValueError(f"Switcher state is missing {key!r}.")
            array = np.asarray(state[key], dtype=np.float32)
            if (
                array.ndim != len(trailing_shape) + 1
                or tuple(array.shape[1:]) != trailing_shape
            ):
                raise ValueError(
                    f"Switcher field {key!r} expected [N,{trailing_shape}], "
                    f"got {array.shape}."
                )
            if count is None:
                count = len(array)
            elif len(array) != count:
                raise ValueError("Switcher state fields have different batches.")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Switcher field {key!r} is non-finite.")
            arrays[key] = array
        arrays["switch_allowed"] = np.asarray(
            state.get("switch_allowed", arrays["aoreplan_action"][:, :1] < 0.5),
            dtype=np.float32,
        )
        if arrays["switch_allowed"].shape != (count, 1) or not np.all(
            np.isin(arrays["switch_allowed"], (0.0, 1.0))
        ):
            raise ValueError("Switcher switch_allowed must be a binary [N,1] mask.")
        if not count:
            raise ValueError("Switcher received an empty batch.")
        if not np.all(arrays["aoreplan_action"][:, 0] == 0.0):
            raise ValueError("Only non-wait AORePlan states may enter Switcher.")
        rnn_states = torch.zeros(
            (count, self.rnn_state_size),
            dtype=torch.float32,
            device=self.device,
        )
        with torch.no_grad():
            observations = TensorDict(
                {
                    key: torch.as_tensor(value, dtype=torch.float32, device=self.device)
                    for key, value in arrays.items()
                }
            )
            observations = prepare_and_normalize_obs(self.ppo, observations)
            outputs = self.ppo(observations, rnn_states)
            return outputs["actions"].cpu().numpy().astype(np.int64)


__all__ = [
    "Switcher",
    "SwitcherConfig",
]
