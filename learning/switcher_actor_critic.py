from __future__ import annotations

import math

import torch
from sample_factory.algo.utils.action_distributions import (
    get_action_distribution,
    sample_actions_log_probs,
)
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.algo.utils.torch_utils import calc_num_elements
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.encoder import Encoder
from sample_factory.model.model_utils import nonlinearity
from torch import nn

from agents.controller import (
    NUM_BRANCHES,
    SWITCHER_FIELD_SHAPES,
    SWITCHER_SPATIAL_SHAPE,
    SWITCHER_VECTOR_DIM,
)


class SwitcherEncoder(Encoder):
    def __init__(self, cfg, obs_space):
        super().__init__(cfg)
        if tuple(obs_space["obs"].shape) != SWITCHER_SPATIAL_SHAPE:
            raise ValueError("Switcher received the wrong spatial shape.")

        self.spatial = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nonlinearity(cfg),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nonlinearity(cfg),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nonlinearity(cfg),
        )
        spatial_size = calc_num_elements(self.spatial, SWITCHER_SPATIAL_SHAPE)
        self.vector = nn.Sequential(
            nn.Linear(SWITCHER_VECTOR_DIM, 32),
            nonlinearity(cfg),
        )
        self.fusion = nn.Sequential(
            nn.Linear(spatial_size + 32, int(cfg.hidden_size)),
            nonlinearity(cfg),
        )
        self.encoder_out_size = int(cfg.hidden_size)

    def forward(self, x):
        spatial = self.spatial(x["obs"].float()).flatten(1)

        coords = torch.cat([x["xy"], x["target_xy"]], dim=-1).float()
        coords = coords / coords.abs().clamp_min(64.0)
        branch_actions = torch.cat(
            [x["caar_action"], x["aoreplan_action"]], dim=-1
        ).float()
        vector = self.vector(torch.cat([coords, branch_actions], dim=-1))
        return self.fusion(torch.cat([spatial, vector], dim=-1))

    def get_out_size(self) -> int:
        return self.encoder_out_size


class SwitcherActorCritic(ActorCriticSharedWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg):
        if not cfg.actor_critic_share_weights:
            raise ValueError("Switcher requires shared actor-critic weights.")
        if getattr(action_space, "n", None) != NUM_BRANCHES:
            raise ValueError("Switcher requires exactly two branch actions.")
        for key, shape in SWITCHER_FIELD_SHAPES.items():
            if key not in obs_space.spaces or tuple(obs_space[key].shape) != shape:
                raise ValueError(f"Switcher expected field {key!r} with shape {shape}.")
        if cfg.use_rnn:
            raise ValueError("Switcher is intentionally feed-forward.")
        super().__init__(model_factory, obs_space, action_space, cfg)

        probability = float(cfg.switcher_initial_ao_probability)
        if not 0.0 < probability < 1.0:
            raise ValueError(
                "switcher_initial_ao_probability must be between zero and one."
            )
        output = self.action_parameterization.distribution_linear
        nn.init.normal_(output.weight, mean=0.0, std=1e-3)
        with torch.no_grad():
            output.bias.copy_(
                torch.tensor(
                    [math.log(1.0 - probability), math.log(probability)],
                    dtype=output.bias.dtype,
                )
            )

        self.encoder_out_size = self.encoder.get_out_size()
        self.core_out_size = self.core.get_out_size()

    def forward_head(self, normalized_obs_dict):
        encoded = self.encoder(normalized_obs_dict)
        switch_allowed = normalized_obs_dict.get("switch_allowed")
        if switch_allowed is None:
            switch_allowed = normalized_obs_dict["aoreplan_action"][:, :1] < 0.5
        return torch.cat([encoded, switch_allowed.float()], dim=-1)

    def forward_core(self, head_output, rnn_states):
        encoded = head_output[:, : self.encoder_out_size]
        switch_allowed = head_output[:, self.encoder_out_size :]
        core_output, new_rnn_states = self.core(encoded, rnn_states)
        return torch.cat([core_output, switch_allowed], dim=-1), new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        policy_state = core_output[:, : self.core_out_size]
        switch_allowed = core_output[:, self.core_out_size :].squeeze(-1) > 0.5
        decoder_output = self.decoder(policy_state)
        values = self.critic_linear(decoder_output).squeeze()
        result = TensorDict(values=values)
        if values_only:
            return result

        count = decoder_output.shape[0]
        logits = decoder_output.new_full((count, NUM_BRANCHES), -1.0e9)
        logits[:, 0] = 0.0
        active_distribution = None
        if torch.any(switch_allowed):
            active_logits, active_distribution = self.action_parameterization(
                decoder_output[switch_allowed]
            )
            logits[switch_allowed] = active_logits

        self.last_action_distribution = get_action_distribution(
            self.action_space,
            logits,
        )
        result["action_logits"] = logits
        if sample_actions:
            actions = torch.zeros(count, dtype=torch.int64, device=logits.device)
            log_probs = torch.zeros(count, dtype=logits.dtype, device=logits.device)
            if active_distribution is not None:
                active_actions, active_log_probs = sample_actions_log_probs(
                    active_distribution
                )
                actions[switch_allowed] = active_actions.squeeze(-1)
                log_probs[switch_allowed] = active_log_probs
            result["actions"] = actions
            result["log_prob_actions"] = log_probs
        return result
