from __future__ import annotations

from pathlib import Path
import torch
from sample_factory.algo.utils.action_distributions import get_action_distribution
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.model.actor_critic import ActorCriticSharedWeights
from sample_factory.model.encoder import ResBlock
from torch import nn
from torch.nn.utils.rnn import PackedSequence

ENTROPY_EPS = 1e-10


class InferenceCorrection:
    @staticmethod
    def gate(entropy):
        # 输入：每个智能体的基础策略熵 [B]。
        # 输出：是否启用学习修正的布尔列 [B,1]；熵严格大于 0.01 才启用。
        return (entropy > 0.01).unsqueeze(-1)

    def apply(self, base_logits, residual, entropy):
        # 输入：基础分数、学习修正均为 [B,5]，基础策略熵为 [B]。
        # 输出：修正后的五个动作分数；低熵时保持基础分数不变。
        return base_logits + self.gate(entropy).to(base_logits) * residual


class TraceEncoder(nn.Module):
    OUTPUT_SIZE = 32
    TRACE_SIZE = 11

    def __init__(self, cfg):
        # 输入：Sample Factory 的网络配置 cfg。
        # 输出：无返回值；建立卷积、两个残差块和 32 维全连接输出。
        super().__init__()
        channels = 32
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            ResBlock(cfg, channels, channels),
            ResBlock(cfg, channels, channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(channels * self.TRACE_SIZE * self.TRACE_SIZE, self.OUTPUT_SIZE),
            nn.ReLU(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # 输入：局部 Shared Trace [B,1,11,11]，已由环境完成均值中心化。
        # 输出：轨迹特征 [B,32]；B 是本次处理的智能体或训练样本数量。
        if tuple(inputs.shape[1:]) != (1, self.TRACE_SIZE, self.TRACE_SIZE):
            raise ValueError(
                f"Paper Trace Encoder requires [B,1,11,11], got {tuple(inputs.shape)}."
            )
        return self.network(inputs)


class _FrozenEPOMActorCritic(ActorCriticSharedWeights):
    NUM_ACTIONS = 5
    TRAINABLE_PREFIXES = ()

    def _load_and_freeze_base(self, settings):
        # 输入：包含 EPOM-L 权重目录的 settings；cfg 可指定具体检查点。
        # 输出：无返回值；加载并冻结基础网络，新增 Actor/Critic 分支仍可训练。
        directory = Path(settings["epom_base_weights_path"]).expanduser()
        if not directory.is_absolute():
            directory = Path(__file__).resolve().parents[1] / directory
        checkpoint_path = getattr(self.cfg, "base_checkpoint_path", None)
        if checkpoint_path is None:
            checkpoints = sorted(
                path
                for path in (directory / "checkpoint_p0").glob("*.pth")
                if not path.name.startswith("best_")
            )
            if not checkpoints:
                raise FileNotFoundError(f"No checkpoint under {directory}.")
            checkpoint_path = checkpoints[-1]
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        incompatible = self.load_state_dict(checkpoint["model"], strict=False)
        missing = [
            key
            for key in incompatible.missing_keys
            if not key.startswith(self.TRAINABLE_PREFIXES)
        ]
        if missing or incompatible.unexpected_keys:
            raise RuntimeError(
                f"EPOM checkpoint mismatch: missing={missing}, "
                f"unexpected={incompatible.unexpected_keys}"
            )

        self._frozen_base_modules = tuple(
            module
            for module in (
                self.encoder,
                self.core,
                self.decoder,
                self.action_parameterization,
                self.critic_linear,
                getattr(self, "obs_normalizer", None),
                getattr(self, "returns_normalizer", None),
            )
            if module is not None
        )
        for module in self._frozen_base_modules:
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad_(False)

    def train(self, mode=True):
        # 输入：mode=True 开启训练，False 开启推理。
        # 输出：模型自身；无论哪种模式，冻结的基础网络始终保持 eval 状态。
        super().train(mode)
        for module in getattr(self, "_frozen_base_modules", ()):
            module.eval()
        return self

    @staticmethod
    def _base_entropy(logits):
        # 输入：基础策略的五个动作分数 [B,5]。
        # 输出：Shannon 熵 [B]，使用自然对数；动作概率越均匀，熵越大。
        probabilities = torch.softmax(logits, dim=-1)
        return -(probabilities * torch.log(probabilities + ENTROPY_EPS)).sum(-1)

    @staticmethod
    def _packed_like(reference, data):
        # 输入：参考变长序列 reference 和替换后的有效样本 data。
        # 输出：保留原序列长度、排序信息的新 PackedSequence，不改变时间对应关系。
        return PackedSequence(
            data,
            reference.batch_sizes,
            reference.sorted_indices,
            reference.unsorted_indices,
        )


class ARPEActorCritic(_FrozenEPOMActorCritic):
    TRAINABLE_PREFIXES = (
        "actor_trace_encoder.",
        "trace_fusion_head.",
        "trace_multiplier_head.",
        "critic_trace_encoder.",
        "critic_fusion_head.",
        "trace_value_head.",
    )

    @staticmethod
    def _validate_spatial_contract(
        obs_shape: tuple[int, ...],
        tau_shape: tuple[int, ...],
    ) -> None:
        # 输入：基础观测和局部轨迹的形状，不含批次维。
        # 输出：无返回值；尺寸不符合网络输入要求时抛出异常。
        if tau_shape != (1, 11, 11):
            raise ValueError(
                f"Paper ARPE requires the [1,11,11] trace crop, got {tau_shape}."
            )
        if (
            len(obs_shape) != 3
            or obs_shape[-2] != obs_shape[-1]
            or obs_shape[-1] % 2 != 1
            or obs_shape[-1] < 11
        ):
            raise ValueError(
                "Paper ARPE requires an odd square EPOM observation of at "
                f"least 11x11, got {obs_shape}."
            )

    def __init__(self, model_factory, obs_space, action_space, cfg):
        # 输入：模型工厂、观测空间、五动作空间和训练配置。
        # 输出：无返回值；建立冻结 EPOM-L，以及参数独立的轨迹 Actor 和 Critic。
        if not cfg.actor_critic_share_weights:
            raise ValueError("Paper ARPE requires shared base weights.")
        if "tau" not in obs_space.spaces:
            raise ValueError("Paper ARPE requires a tau observation.")
        if getattr(action_space, "n", None) != self.NUM_ACTIONS:
            raise ValueError(f"Expected five discrete actions, got {action_space}.")

        ActorCriticSharedWeights.__init__(
            self, model_factory, obs_space, action_space, cfg
        )
        settings = cfg.full_config["experiment_settings"]
        self._validate_spatial_contract(
            tuple(obs_space["obs"].shape), tuple(obs_space["tau"].shape)
        )
        self.core_out_size = int(self.core.get_out_size())

        self.actor_trace_encoder = TraceEncoder(cfg)
        if self.core_out_size != 512:
            raise ValueError("ARPE requires the 512D EPOM hidden state.")
        self.critic_trace_encoder = TraceEncoder(cfg)
        # 每条分支融合：轨迹特征 32 + 基础 GRU 隐藏状态 512 + 基础分数 5。
        self.critic_fusion_head = nn.Sequential(nn.Linear(549, 256), nn.ReLU())
        self.trace_value_head = nn.Linear(256, 1)
        self.trace_fusion_head = nn.Sequential(nn.Linear(549, 256), nn.ReLU())
        # 保留检查点中的层名；这里输出的是五个加性修正量，不是乘法系数。
        self.trace_multiplier_head = nn.Sequential(nn.Linear(256, self.NUM_ACTIONS))
        for module in (
            self.actor_trace_encoder,
            self.trace_multiplier_head,
            self.trace_value_head,
            self.trace_fusion_head,
            self.critic_trace_encoder,
            self.critic_fusion_head,
        ):
            module.apply(self.initialize_weights)
        nn.init.zeros_(self.trace_multiplier_head[-1].weight)
        nn.init.zeros_(self.trace_multiplier_head[-1].bias)

        self._load_and_freeze_base(settings)
        # 这些字段已保存在现有权重中，仅保留用于加载；当前推理门槛见 gate()。
        self.register_buffer(
            "fixed_entropy_threshold",
            torch.tensor(0.46371241, dtype=torch.float64),
        )
        self.register_buffer(
            "paper_entropy_gate_version",
            torch.tensor(0, dtype=torch.int64),
        )
        self.register_buffer(
            "independent_critic_version", torch.tensor(1, dtype=torch.int64)
        )
        self.register_buffer(
            "allaction_residual_version", torch.tensor(2, dtype=torch.int64)
        )
        self.actor_trace_embedding_size = self.actor_trace_encoder.OUTPUT_SIZE
        self.critic_trace_embedding_size = self.critic_trace_encoder.OUTPUT_SIZE
        self.head_extra_size = 2 * TraceEncoder.OUTPUT_SIZE

    def forward_head(self, normalized_obs_dict):
        # 输入：预处理后的观测字典，基础网络用 obs/xy/target_xy，轨迹分支用 tau。
        # 输出：拼接特征 [B,576]，依次为基础编码 512、Actor 轨迹 32、Critic 轨迹 32。
        with torch.no_grad():
            base_context = self.encoder(normalized_obs_dict)
        tau = normalized_obs_dict["tau"].float()
        actor_trace = self.actor_trace_encoder(tau)
        critic_trace = self.critic_trace_encoder(tau)
        return torch.cat(
            [base_context.detach(), actor_trace, critic_trace], dim=-1
        )

    def forward_core(self, head_output, rnn_states):
        # 输入：head 的普通批次或变长序列，以及每个智能体的基础 GRU 状态。
        # 输出：（新隐藏状态与两条轨迹特征的拼接，新 GRU 状态）；轨迹特征不经过 GRU。
        packed = isinstance(head_output, PackedSequence)
        data = head_output.data if packed else head_output
        context, extras = data[:, :-self.head_extra_size], data[:, -self.head_extra_size:]
        if packed:
            context = self._packed_like(head_output, context)
        with torch.no_grad():
            core_output, new_states = self.core(context, rnn_states)
        hidden = core_output.data if packed else core_output
        combined = torch.cat([hidden.detach(), extras], dim=-1)
        if packed:
            combined = self._packed_like(core_output, combined)
        return combined, new_states

    def _split_core(self, core_output):
        # 输入：core 的拼接输出 [B,576]。
        # 输出：隐藏状态 [B,512]、Actor 轨迹 [B,32]、Critic 轨迹 [B,32]。
        return core_output.split(
            (self.core_out_size, self.actor_trace_embedding_size, self.critic_trace_embedding_size),
            dim=-1,
        )

    def _critic_values(
        self,
        hidden: torch.Tensor,
        critic_trace: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        # 输入：基础隐藏状态 [B,512]、Critic 轨迹特征 [B,32]、基础分数 [B,5]。
        # 输出：独立 Critic 的状态价值 [B]，不与基础策略的价值相加。
        features = self.compose_paper_entropy_fusion_input(
            critic_trace, hidden, base_logits
        )
        return self.trace_value_head(self.critic_fusion_head(features)).squeeze(-1)

    @staticmethod
    def compose_paper_entropy_fusion_input(
        trace_feature: torch.Tensor,
        frozen_hidden: torch.Tensor,
        base_logits: torch.Tensor,
    ) -> torch.Tensor:
        # 输入：轨迹特征 [B,32]、冻结基础网络的隐藏状态 [B,512] 和分数 [B,5]。
        # 输出：融合输入 [B,549]；梯度只流向轨迹特征，不反传到基础网络。
        batch = trace_feature.shape[0]
        expected = ((batch, 32), (batch, 512), (batch, 5))
        actual = (
            tuple(trace_feature.shape),
            tuple(frozen_hidden.shape),
            tuple(base_logits.shape),
        )
        if actual != expected:
            raise ValueError(
                f"Paper entropy fusion expects trace32+h512+z5, got {actual}."
            )
        return torch.cat(
            [trace_feature, frozen_hidden.detach(), base_logits.detach()], dim=-1
        )

    def _apply_configured_correction_rule(
        self,
        base_logits,
        raw_correction,
    ):
        # 输入：基础分数与学习分支输出的原始修正，均为 [B,5]。
        # 输出：最终分数 [B,5]；当前训练直接相加，部署时由 InferenceCorrection 控制。
        base_logits = base_logits.detach()
        inference = getattr(self, "inference_correction", None)
        if inference is not None:
            if self.training:
                raise RuntimeError(
                    "Inference correction must not be used for training."
                )
            return inference.apply(
                base_logits,
                raw_correction,
                self._base_entropy(base_logits),
            )
        return base_logits + raw_correction

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
        # 输入：core 特征 [B,576]；values_only 表示只计算价值，sample_actions 表示需要采样。
        # 输出：包含 values、可选 action_logits/采样动作的字典；动作顺序为等待、上、下、左、右。
        if isinstance(core_output, PackedSequence):
            raise TypeError("Sample Factory must unpack PackedSequence before tail.")
        hidden, actor_trace, critic_trace = self._split_core(core_output)
        with torch.no_grad():
            decoder_output = self.decoder(hidden)
            base_logits, _ = self.action_parameterization(decoder_output)
        base_logits = base_logits.detach()
        values = self._critic_values(hidden, critic_trace, base_logits)
        result = TensorDict(values=values)
        if values_only:
            return result

        fusion_input = self.compose_paper_entropy_fusion_input(
            actor_trace, hidden, base_logits
        )
        actor_input = self.trace_fusion_head(fusion_input)
        raw_correction = self.trace_multiplier_head(actor_input)
        final_logits = self._apply_configured_correction_rule(
            base_logits,
            raw_correction,
        )
        self.last_action_distribution = get_action_distribution(
            self.action_space, final_logits
        )
        result["action_logits"] = final_logits
        self._maybe_sample_actions(sample_actions, result)
        return result
