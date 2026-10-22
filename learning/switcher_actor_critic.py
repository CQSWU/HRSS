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
        # 输入：网络配置和观测空间。
        # 输出：无返回值；建立三层卷积、坐标/候选动作编码，以及共享特征融合层。
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
        # 输入：obs [N,3,11,11]、xy/target_xy 各 [N,2]，两套候选动作各 [N,5]。
        # 输出：共享特征 [N,hidden_size]，当前权重使用 128 维，供 Actor 和 Critic 共用。
        spatial = self.spatial(x["obs"].float()).flatten(1)

        coords = torch.cat([x["xy"], x["target_xy"]], dim=-1).float()
        coords = coords / coords.abs().clamp_min(64.0)
        branch_actions = torch.cat(
            [x["caar_action"], x["aoreplan_action"]], dim=-1
        ).float()
        vector = self.vector(torch.cat([coords, branch_actions], dim=-1))
        return self.fusion(torch.cat([spatial, vector], dim=-1))

    def get_out_size(self) -> int:
        # 输入：无。输出：共享编码向量的维数，供后续网络自动确定层大小。
        return self.encoder_out_size


class SwitcherActorCritic(ActorCriticSharedWeights):
    def __init__(self, model_factory, obs_space, action_space, cfg):
        # 输入：模型工厂、观测空间、二分支动作空间和配置。
        # 输出：无返回值；建立共享编码器、二输出 Actor 和单输出 Critic，不使用 GRU。
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
        # 这里只控制新建模型的初始分支概率，加载已有权重后会被检查点参数覆盖。
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
        # 输入：预处理后的观测字典，可附带 switch_allowed [N,1]。
        # 输出：共享编码加一列允许切换标记；没有显式标记时，由 AORePlan 是否等待判断。
        encoded = self.encoder(normalized_obs_dict)
        switch_allowed = normalized_obs_dict.get("switch_allowed")
        if switch_allowed is None:
            switch_allowed = normalized_obs_dict["aoreplan_action"][:, :1] < 0.5
        return torch.cat([encoded, switch_allowed.float()], dim=-1)

    def forward_core(self, head_output, rnn_states):
        # 输入：编码与切换标记的拼接张量，以及框架要求的空循环状态。
        # 输出：（core 特征与原标记的拼接，空状态）；标记不参与特征学习。
        encoded = head_output[:, : self.encoder_out_size]
        switch_allowed = head_output[:, self.encoder_out_size :]
        core_output, new_rnn_states = self.core(encoded, rnn_states)
        return torch.cat([core_output, switch_allowed], dim=-1), new_rnn_states

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        # 输入：core 输出；values_only 表示只求价值，sample_actions 表示需要采样分支。
        # 输出：状态价值、可选的两个分支分数、采样分支及其对数概率。
        # Actor/Critic 共享前面的特征，但各有输出头；价值学习仍包含规则接管的状态。
        policy_state = core_output[:, : self.core_out_size]
        switch_allowed = core_output[:, self.core_out_size :].squeeze(-1) > 0.5
        decoder_output = self.decoder(policy_state)
        values = self.critic_linear(decoder_output).squeeze()
        result = TensorDict(values=values)
        if values_only:
            return result

        count = decoder_output.shape[0]
        # AORePlan 等待时固定选 ARPE，且这些行不参与 Actor 计算或随机采样。
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
            # 只对允许切换的行采样，避免规则接管行改变其他智能体的随机采样序列。
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
