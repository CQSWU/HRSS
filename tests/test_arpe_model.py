from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F
import yaml
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_dict import TensorDict
from sample_factory.envs.create_env import create_env
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from torch import nn
from torch.nn.utils.rnn import PackedSequence, pack_padded_sequence

from learning.config import Experiment
from learning.arpe_actor_critic import ARPEActorCritic
from train_arpe import register_custom_components, validate_config

ROOT = Path(__file__).resolve().parents[1]
FORMAL = ROOT / "configs" / "train_arpe.yaml"


def _load(path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@pytest.fixture
def correction_model():
    model = object.__new__(ARPEActorCritic)
    nn.Module.__init__(model)
    return model


@pytest.fixture(scope="module")
def full_model():
    register_custom_components()
    raw = _load(FORMAL)
    base = ROOT / raw["experiment_settings"]["epom_base_weights_path"]
    if not ((base / "config.json").is_file() or (base / "cfg.json").is_file()):
        pytest.skip("EPOM-L weights are distributed separately from Git")
    raw["global_settings"]["device"] = "cpu"
    _, cfg = validate_config(raw)
    env = create_env(cfg.env, cfg=cfg, env_config={})
    try:
        model = create_actor_critic(cfg, env.observation_space, env.action_space)
        observations, _ = env.reset()
        batch = TensorDict(
            {
                key: torch.from_numpy(
                    np.stack([observation[key] for observation in observations])
                ).float()
                for key in observations[0]
            }
        )
        batch = prepare_and_normalize_obs(model, batch)
    finally:
        env.close()
    return model, batch, cfg


def test_config_locks_the_selected_arpe_contract():
    experiment = Experiment(**_load(FORMAL))
    settings, environment = experiment.experiment_settings, experiment.environment
    assert settings.encoder_custom == "epom_trace_context"
    assert settings.hidden_size == 512
    assert settings.train_for_env_steps == 1_000_000_000
    assert settings.epom_base_weights_path == "weights/EPOM-L"
    assert environment.tau_radius == 5
    assert environment.grid_memory_obs_radius == 7
    assert environment.grid_config.map_name == "maps/train.yaml"


def test_network_restores_independent_trace_critic_and_605638_parameters(full_model):
    model, _, _ = full_model
    for encoder in (model.actor_trace_encoder, model.critic_trace_encoder):
        assert encoder.network[0].in_channels == 1
        assert encoder.network[0].out_channels == 32
        assert encoder.network[0].kernel_size == (3, 3)
        assert encoder.network[-2].in_features == 32 * 11 * 11
        assert encoder.network[-2].out_features == 32
        assert sum(p.numel() for p in encoder.parameters()) == 161_248
    for fusion in (model.trace_fusion_head, model.critic_fusion_head):
        assert fusion[0].in_features == 549
        assert fusion[0].out_features == 256
        assert sum(p.numel() for p in fusion.parameters()) == 140_800
    assert model.trace_multiplier_head[0].in_features == 256
    assert model.trace_multiplier_head[0].out_features == 5
    assert isinstance(model.trace_value_head, nn.Linear)
    assert model.trace_value_head.in_features == 256
    assert model.trace_value_head.out_features == 1
    assert sum(p.numel() for p in model.trace_multiplier_head.parameters()) == 1_285
    assert sum(p.numel() for p in model.trace_value_head.parameters()) == 257
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 605_638
    assert model.head_extra_size == 89
    for key, expected in (
        ("paper_entropy_gate_version", 0),
        ("independent_critic_version", 1),
        ("allaction_residual_version", 2),
    ):
        assert model.state_dict()[key].item() == expected


def test_fusion_input_is_trace32_h512_z5_and_detaches_frozen_epom():
    trace = torch.randn(3, 32, requires_grad=True)
    hidden = torch.randn(3, 512, requires_grad=True)
    logits = torch.randn(3, 5, requires_grad=True)
    fused = ARPEActorCritic.compose_paper_entropy_fusion_input(
        trace, hidden, logits
    )
    assert fused.shape == (3, 549)
    torch.testing.assert_close(fused[:, :32], trace)
    torch.testing.assert_close(fused[:, 32:544], hidden)
    torch.testing.assert_close(fused[:, 544:], logits)
    fused.sum().backward()
    assert trace.grad is not None
    assert hidden.grad is None
    assert logits.grad is None


def test_residual_keeps_raw_magnitude_and_mean(correction_model):
    raw = torch.tensor([[20.62, -30.0, 0.0, 2.0, -1.0], [4.0] * 5], requires_grad=True)
    base = torch.zeros_like(raw)
    final = correction_model._apply_configured_correction_rule(
        base, raw
    )
    assert torch.equal(final, raw)
    final.sum().backward()
    assert torch.equal(raw.grad, torch.ones_like(raw))


def test_training_keeps_actor_gradient_for_all_rows_and_actions(correction_model):
    base = torch.tensor(
        [[0.0, 0.4, 0.3, 0.2, 0.1], [20.0, 0.4, 0.3, 0.2, 0.1]],
        requires_grad=True,
    )
    raw = torch.tensor(
        [[0.4, -0.2, 0.1, 0.3, -0.1], [0.7, -0.4, 0.2, 0.1, -0.3]], requires_grad=True
    )
    final = correction_model._apply_configured_correction_rule(
        base, raw
    )
    reference_raw = raw.detach().clone().requires_grad_()
    expected = base.detach() + reference_raw
    torch.testing.assert_close(final, expected, rtol=0, atol=0)
    targets = torch.tensor([1, 2])
    F.cross_entropy(expected, targets, reduction="sum").backward()
    F.cross_entropy(final, targets, reduction="sum").backward()
    torch.testing.assert_close(raw.grad, reference_raw.grad, rtol=0, atol=0)
    assert torch.count_nonzero(raw.grad).item() == 10
    assert base.grad is None


def test_zero_output_preserves_base_logits_at_all_entropies(correction_model):
    base = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1], [20.0, 0.4, 0.3, 0.2, 0.1]])
    final = correction_model._apply_configured_correction_rule(
        base, torch.zeros_like(base)
    )
    torch.testing.assert_close(final, base, rtol=0, atol=0)


def test_training_formula_is_base_plus_raw_correction(correction_model):
    base = torch.tensor([[0.0, 0.4, 0.3, 0.2, 0.1], [20.0, 0.4, 0.3, 0.2, 0.1]])
    raw = torch.tensor([[2.0, -1.0, 0.0, 4.0, -3.0], [1.0, 2.0, 3.0, 4.0, 5.0]])
    final = correction_model._apply_configured_correction_rule(
        base, raw
    )
    expected = base + raw
    torch.testing.assert_close(final, expected, rtol=0, atol=0)


def test_actor_and_critic_have_independent_trace_gradients_and_frozen_base(full_model):
    model, _, _ = full_model
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.normal_(0.0, 0.01)
    actor_tau = torch.randn(4, 1, 11, 11, requires_grad=True)
    critic_tau = torch.randn(4, 1, 11, 11, requires_grad=True)
    hidden = torch.randn(4, 512, requires_grad=True)
    logits = torch.zeros(4, 5, requires_grad=True)
    actor_feature = model.actor_trace_encoder(actor_tau)
    fusion = model.compose_paper_entropy_fusion_input(actor_feature, hidden, logits)
    raw = model.trace_multiplier_head(model.trace_fusion_head(fusion))
    adjusted = model._apply_configured_correction_rule(
        logits.detach(), raw
    )
    values = model._critic_values(
        hidden, model.critic_trace_encoder(critic_tau), logits
    )
    F.cross_entropy(adjusted, torch.tensor([0, 1, 2, 3])).backward()
    assert actor_tau.grad is not None
    assert critic_tau.grad is None
    assert model.critic_trace_encoder.network[0].weight.grad is None
    model.zero_grad(set_to_none=True)
    actor_tau.grad = None
    values.square().mean().backward()
    assert actor_tau.grad is None
    assert critic_tau.grad is not None
    assert model.actor_trace_encoder.network[0].weight.grad is None
    assert model.critic_trace_encoder.network[0].weight.grad.norm().item() > 0
    assert model.critic_fusion_head[0].weight.grad.norm().item() > 0
    assert model.trace_value_head.weight.grad.norm().item() > 0
    assert hidden.grad is None
    assert logits.grad is None
    assert all(
        parameter.grad is None
        for module in model._frozen_base_modules
        for parameter in module.parameters()
    )


def test_full_forward_zero_initial_residual_preserves_base_logits(full_model):
    model, batch, cfg = full_model
    nn.init.zeros_(model.trace_multiplier_head[-1].weight)
    nn.init.zeros_(model.trace_multiplier_head[-1].bias)
    frozen_states = [
        {key: value.detach().clone() for key, value in module.state_dict().items()}
        for module in model._frozen_base_modules
    ]
    captured = {}
    hook = model.action_parameterization.register_forward_hook(
        lambda _module, _inputs, output: captured.update(base=output[0].detach())
    )
    try:
        outputs = model(batch, torch.zeros(len(batch["obs"]), cfg.hidden_size))
    finally:
        hook.remove()
    assert outputs["action_logits"].shape == (len(batch["obs"]), 5)
    assert outputs["values"].shape == (len(batch["obs"]),)
    for module, expected in zip(model._frozen_base_modules, frozen_states):
        assert all(not parameter.requires_grad for parameter in module.parameters())
        for key, value in module.state_dict().items():
            torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
    torch.testing.assert_close(outputs["action_logits"], captured["base"], rtol=0, atol=0)


def test_actor_and_critic_have_independent_parameters(full_model):
    model, _, _ = full_model
    actor = (
        model.actor_trace_encoder,
        model.trace_fusion_head,
        model.trace_multiplier_head,
    )
    critic = (
        model.critic_trace_encoder,
        model.critic_fusion_head,
        model.trace_value_head,
    )
    actor_parameters = [p for module in actor for p in module.parameters()]
    critic_parameters = [p for module in critic for p in module.parameters()]
    assert sum(p.numel() for p in actor_parameters) == 303_333
    assert sum(p.numel() for p in critic_parameters) == 302_305
    assert {id(p) for p in actor_parameters}.isdisjoint(
        id(p) for p in critic_parameters
    )
    assert all(p.requires_grad for p in actor_parameters + critic_parameters)
    assert int(model.allaction_residual_version) == 2


def test_packed_rollout_keeps_critic_and_tie_metadata_and_backpropagates(full_model):

    model, batch, cfg = full_model
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.normal_(0.0, 0.01)
    lengths = torch.tensor([3, 1, 2])
    sequence_count, padded_length = 3, 3
    observations = TensorDict({key: value[:9] for key, value in batch.items()})
    head = model.forward_head(observations).reshape(padded_length, sequence_count, -1)
    packed_head = pack_padded_sequence(head, lengths, enforce_sorted=False)
    states = torch.zeros(sequence_count, get_rnn_size(cfg))
    packed_core, new_states = model.forward_core(packed_head, states)

    assert isinstance(packed_core, PackedSequence)
    assert packed_core.data.shape == (6, 512 + 89)
    assert new_states.shape == states.shape
    for name in ("batch_sizes", "sorted_indices", "unsorted_indices"):
        torch.testing.assert_close(
            getattr(packed_core, name), getattr(packed_head, name)
        )
    torch.testing.assert_close(packed_core.data[:, -89:], packed_head.data[:, -89:])
    with torch.no_grad():
        reference_input = model._packed_like(
            packed_head, packed_head.data[:, :-89].detach()
        )
        reference_core, reference_states = model.core(reference_input, states)
    torch.testing.assert_close(packed_core.data[:, :512], reference_core.data)
    torch.testing.assert_close(new_states, reference_states)

    result = model.forward_tail(
        packed_core.data, values_only=False, sample_actions=False
    )
    assert result["action_logits"].shape == (6, 5)
    assert result["values"].shape == (6,)
    assert torch.isfinite(result["action_logits"]).all()
    assert torch.isfinite(result["values"]).all()
    loss = F.cross_entropy(result["action_logits"], torch.arange(6) % 5)
    loss = loss + result["values"].square().mean()
    loss.backward()
    for module in (
        model.actor_trace_encoder,
        model.critic_trace_encoder,
        model.trace_fusion_head,
        model.critic_fusion_head,
        model.trace_multiplier_head,
        model.trace_value_head,
    ):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(
            gradient is not None and torch.isfinite(gradient).all()
            for gradient in gradients
        )
    assert all(
        parameter.grad is None
        for module in model._frozen_base_modules
        for parameter in module.parameters()
    )
    with torch.no_grad():
        model.trace_multiplier_head[-1].weight.zero_()
        model.trace_multiplier_head[-1].bias.zero_()


def test_core_preserves_hidden_row_stride(full_model):
    model, batch, cfg = full_model
    observations = TensorDict({key: value[:2] for key, value in batch.items()})
    with torch.no_grad():
        head = model.forward_head(observations)
        core, _ = model.forward_core(head, torch.zeros(2, get_rnn_size(cfg)))
    hidden, *_ = model._split_core(core[:, :-10])
    assert hidden.shape == (2, 512)
    assert hidden.stride() == (601, 1)
    torch.testing.assert_close(core[:, 512:517], core[:, 522:527], rtol=0, atol=0)
