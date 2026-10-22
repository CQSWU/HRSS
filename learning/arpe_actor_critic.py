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
        # Leave very confident base-policy decisions unchanged at inference.
        return (entropy > 0.01).unsqueeze(-1)

    def apply(self, base_logits, residual, entropy):
        return base_logits + self.gate(entropy).to(base_logits) * residual


class TraceEncoder(nn.Module):
    OUTPUT_SIZE = 32
    TRACE_SIZE = 11

    def __init__(self, cfg):
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
        # Encode each 1x11x11 trace crop as a 32-dimensional feature vector.
        if tuple(inputs.shape[1:]) != (1, self.TRACE_SIZE, self.TRACE_SIZE):
            raise ValueError(
                f"Paper Trace Encoder requires [B,1,11,11], got {tuple(inputs.shape)}."
            )
        return self.network(inputs)


class _FrozenEPOMActorCritic(ActorCriticSharedWeights):
    NUM_ACTIONS = 5
    TRAINABLE_PREFIXES = ()

    def _load_and_freeze_base(self, settings):
        # Load EPOM weights and train only the added trace modules.
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
        # Keep frozen base modules in evaluation mode during trace training.
        super().train(mode)
        for module in getattr(self, "_frozen_base_modules", ()):
            module.eval()
        return self

    @staticmethod
    def _base_entropy(logits):
        probabilities = torch.softmax(logits, dim=-1)
        return -(probabilities * torch.log(probabilities + ENTROPY_EPS)).sum(-1)

    @staticmethod
    def _packed_like(reference, data):
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
        # Actor and critic use separate trace encoders and fusion heads.
        self.critic_trace_encoder = TraceEncoder(cfg)
        self.critic_fusion_head = nn.Sequential(nn.Linear(549, 256), nn.ReLU())
        self.trace_value_head = nn.Linear(256, 1)
        self.trace_fusion_head = nn.Sequential(nn.Linear(549, 256), nn.ReLU())
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
        # Start with zero correction so initial actions follow the base policy.
        nn.init.zeros_(self.trace_multiplier_head[-1].weight)
        nn.init.zeros_(self.trace_multiplier_head[-1].bias)

        self._load_and_freeze_base(settings)
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
        # Encode the base observation without gradients and train both trace encoders.
        with torch.no_grad():
            base_context = self.encoder(normalized_obs_dict)
        tau = normalized_obs_dict["tau"].float()
        actor_trace = self.actor_trace_encoder(tau)
        critic_trace = self.critic_trace_encoder(tau)
        return torch.cat(
            [base_context.detach(), actor_trace, critic_trace], dim=-1
        )

    def forward_core(self, head_output, rnn_states):
        # Update the frozen base policy's memory while preserving trace gradients.
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
        # Predict a standalone value; do not add the frozen base critic's value.
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
        # Fuse trace features (32), frozen hidden state (512) and base logits (5).
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
        # Training adds the raw correction; inference uses the gate above.
        return base_logits + raw_correction

    def forward_tail(self, core_output, values_only: bool, sample_actions: bool):
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
        # Predict one additive logit correction for each of the five actions.
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
