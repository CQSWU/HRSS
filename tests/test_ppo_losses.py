import subprocess
import sys
import inspect
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from learning.ppo_losses import _build_switcher_calculate_losses
from sample_factory.algo.learning.learner import Learner
from sample_factory.utils.attr_dict import AttrDict


class _FakeLearner:
    def __init__(self):
        self.cfg = SimpleNamespace(
            encoder_custom="switcher",
            use_rnn=False,
            with_vtrace=False,
            normalize_input=False,
            ppo_clip_value=0.2,
        )
        self.actor_weight = torch.nn.Parameter(torch.tensor(0.25))
        self.critic_weight = torch.nn.Parameter(torch.tensor(0.0))

    def _value_loss(
        self,
        new_values,
        old_values,
        target,
        clip_value,
        valids,
        num_invalids,
    ):
        del old_values, clip_value, num_invalids
        return ((new_values - target).square())[valids].mean()


def _upstream_like_calculate_losses(self, mb, num_invalids):
    del num_invalids
    valids = mb.valids
    selected_advantages = mb.advantages[valids]
    adv_std, adv_mean = torch.std_mean(selected_advantages)
    advantages = (mb.advantages - adv_mean) / torch.clamp_min(
        adv_std,
        1e-7,
    )
    policy_loss = -(self.actor_weight * advantages * mb.actor_signal)[valids].mean()
    exploration_loss = self.actor_weight * 0.0
    kl_loss = self.actor_weight * 0.0
    kl_old = torch.zeros(int(valids.sum().item()))

    values = self.critic_weight * mb.critic_features
    value_loss = self._value_loss(
        values,
        mb["values"],
        mb.returns,
        self.cfg.ppo_clip_value,
        valids,
        0,
    )
    summaries = {
        "ratio": torch.ones_like(mb.advantages),
        "values": values.unsqueeze(-1),
        "adv": advantages,
        "adv_std": adv_std,
        "adv_mean": adv_mean,
    }
    return (
        object(),
        policy_loss,
        exploration_loss,
        kl_old,
        kl_loss,
        value_loss,
        summaries,
    )


_PATCHED_CALCULATE_LOSSES = _build_switcher_calculate_losses(
    _upstream_like_calculate_losses
)


def _one_hot_actions(actions):
    return torch.nn.functional.one_hot(
        torch.as_tensor(actions),
        num_classes=5,
    ).float()


def _batch(actions, advantages, actor_signal, returns, valids=None, caar_actions=None):
    count = len(actions)
    return AttrDict(
        normalized_obs={
            "aoreplan_action": _one_hot_actions(actions),
            "caar_action": _one_hot_actions(
                [0] * count if caar_actions is None else caar_actions
            ),
        },
        valids=(
            torch.ones(count, dtype=torch.bool)
            if valids is None
            else torch.as_tensor(valids, dtype=torch.bool)
        ),
        advantages=torch.as_tensor(advantages, dtype=torch.float32),
        actor_signal=torch.as_tensor(actor_signal, dtype=torch.float32),
        critic_features=torch.ones(count, dtype=torch.float32),
        values=torch.zeros(count, dtype=torch.float32),
        returns=torch.as_tensor(returns, dtype=torch.float32),
    )


def _calculate(learner, batch):
    return _PATCHED_CALCULATE_LOSSES(learner, batch, 0)


def test_mixed_batch_normalizes_and_trains_actor_only_on_non_wait_rows():
    learner = _FakeLearner()
    batch = _batch(
        actions=[0, 1, 0, 4],
        advantages=[1000.0, 1.0, -1000.0, 3.0],
        actor_signal=[9.0, 1.0, -7.0, -1.0],
        returns=[1.0, 2.0, 3.0, 4.0],
    )

    losses = _calculate(learner, batch)
    policy_loss, value_loss, summaries = losses[1], losses[5], losses[6]
    actor_gradient = torch.autograd.grad(
        policy_loss,
        learner.actor_weight,
        retain_graph=True,
    )[0]

    reference = _FakeLearner()
    active_only = _batch(
        actions=[1, 4],
        advantages=[1.0, 3.0],
        actor_signal=[1.0, -1.0],
        returns=[2.0, 4.0],
    )
    reference_losses = _calculate(reference, active_only)
    reference_gradient = torch.autograd.grad(
        reference_losses[1],
        reference.actor_weight,
    )[0]

    assert torch.allclose(policy_loss, reference_losses[1])
    assert torch.allclose(actor_gradient, reference_gradient)
    assert torch.allclose(summaries["adv_mean"], torch.tensor(2.0))
    assert torch.allclose(
        summaries["adv_std"],
        torch.tensor(2.0).sqrt(),
    )

    assert torch.allclose(value_loss, torch.tensor(7.5))


def test_wait_advantages_cannot_change_actor_gradient():
    learner_a = _FakeLearner()
    learner_b = _FakeLearner()
    common = dict(
        actions=[0, 1, 0, 4],
        actor_signal=[10.0, 1.0, -10.0, -1.0],
        returns=[1.0, 2.0, 3.0, 4.0],
    )
    batch_a = _batch(advantages=[5.0, 1.0, 6.0, 3.0], **common)
    batch_b = _batch(advantages=[50000.0, 1.0, -90000.0, 3.0], **common)

    loss_a = _calculate(learner_a, batch_a)[1]
    loss_b = _calculate(learner_b, batch_b)[1]
    grad_a = torch.autograd.grad(loss_a, learner_a.actor_weight)[0]
    grad_b = torch.autograd.grad(loss_b, learner_b.actor_weight)[0]

    assert torch.allclose(loss_a, loss_b)
    assert torch.allclose(grad_a, grad_b)


def test_same_actions_are_excluded_from_actor_losses_and_normalization():
    learner = _FakeLearner()
    batch = _batch(
        actions=[0, 1, 2, 4, 3],
        caar_actions=[2, 4, 2, 1, 3],
        advantages=[1000.0, 1.0, -1000.0, 3.0, 50000.0],
        actor_signal=[9.0, 1.0, -7.0, -1.0, 20.0],
        returns=[1.0, 2.0, 3.0, 4.0, 5.0],
    )
    original_valids = batch.valids
    losses = _calculate(learner, batch)
    gradient = torch.autograd.grad(losses[1], learner.actor_weight)[0]

    reference = _FakeLearner()
    reference_losses = _calculate(
        reference,
        _batch([1, 4], [1.0, 3.0], [1.0, -1.0], [2.0, 4.0]),
    )
    reference_gradient = torch.autograd.grad(
        reference_losses[1], reference.actor_weight
    )[0]
    assert torch.equal(losses[1], reference_losses[1])
    assert torch.equal(gradient, reference_gradient)
    assert losses[6]["adv_mean"].item() == 2.0
    assert torch.allclose(losses[6]["adv_std"], torch.tensor(2.0).sqrt())
    assert losses[5].item() == 11.0
    assert batch.valids is original_valids
    assert batch.valids.all()


@pytest.mark.parametrize(
    "actions,caar_actions",
    [([1], [1]), ([1, 2, 3], [1, 2, 3]), ([0, 1, 2], [4, 1, 3])],
)
def test_fewer_than_two_real_choices_keep_critic_learning(actions, caar_actions):
    learner = _FakeLearner()
    count = len(actions)
    batch = _batch(
        actions=actions,
        caar_actions=caar_actions,
        advantages=[123.0] * count,
        actor_signal=[99.0] * count,
        returns=[2.0] * count,
    )
    losses = _calculate(learner, batch)
    total = sum(losses[index] for index in (1, 2, 4, 5))
    actor_gradient, critic_gradient = torch.autograd.grad(
        total, (learner.actor_weight, learner.critic_weight), allow_unused=True
    )
    assert torch.isfinite(total)
    assert all(losses[index].item() == 0.0 for index in (1, 2, 4))
    assert actor_gradient is None or actor_gradient.item() == 0.0
    assert critic_gradient.item() == -4.0
    assert losses[6]["adv_mean"].item() == 0.0
    assert losses[6]["adv_std"].item() == 0.0


def test_all_wait_batch_has_zero_actor_gradient_and_full_critic_gradient():
    learner = _FakeLearner()
    batch = _batch(
        actions=[0],
        advantages=[123.0],
        actor_signal=[99.0],
        returns=[2.0],
    )

    losses = _calculate(learner, batch)
    policy_loss, exploration_loss, kl_old, kl_loss = losses[1:5]
    value_loss, summaries = losses[5], losses[6]
    total_loss = policy_loss + exploration_loss + kl_loss + value_loss
    actor_gradient, critic_gradient = torch.autograd.grad(
        total_loss,
        (learner.actor_weight, learner.critic_weight),
        allow_unused=True,
    )

    assert torch.isfinite(total_loss)
    assert policy_loss.item() == 0.0
    assert exploration_loss.item() == 0.0
    assert kl_loss.item() == 0.0
    assert torch.equal(kl_old, torch.zeros(1))
    assert actor_gradient is None or actor_gradient.item() == 0.0
    assert critic_gradient is not None
    assert torch.isfinite(critic_gradient)
    assert critic_gradient.item() == -4.0
    assert summaries["adv_mean"].item() == 0.0
    assert summaries["adv_std"].item() == 0.0


def test_non_switcher_two_valid_rows_remain_bitwise_upstream():
    patched_learner = _FakeLearner()
    patched_learner.cfg.encoder_custom = "epom_trace_context"
    reference_learner = _FakeLearner()
    reference_learner.cfg.encoder_custom = "epom_trace_context"
    batch = _batch(
        actions=[1, 2, 3],
        advantages=[1.25, 999.0, -0.75],
        actor_signal=[0.5, 7.0, -2.0],
        returns=[2.0, 100.0, -3.0],
        valids=[True, False, True],
    )

    patched = _calculate(patched_learner, batch)
    reference = _upstream_like_calculate_losses(reference_learner, batch, 1)
    for index in (1, 2, 3, 4, 5):
        assert torch.equal(patched[index], reference[index])
    for key in ("ratio", "values", "adv", "adv_std", "adv_mean"):
        assert torch.equal(patched[6][key], reference[6][key])


def test_non_switcher_single_valid_row_uses_zero_population_std_without_nan():
    learner = _FakeLearner()
    learner.cfg.encoder_custom = "epom_trace_context"
    batch = _batch(
        actions=[1, 2, 3],
        advantages=[1000.0, 7.5, -1000.0],
        actor_signal=[9.0, 3.0, -7.0],
        returns=[100.0, 2.0, -100.0],
        valids=[False, True, False],
    )

    losses = _calculate(learner, batch)
    policy_loss, value_loss, summaries = losses[1], losses[5], losses[6]
    total = sum(losses[index] for index in (1, 2, 4, 5))
    actor_gradient, critic_gradient = torch.autograd.grad(
        total,
        (learner.actor_weight, learner.critic_weight),
        allow_unused=True,
    )

    assert torch.isfinite(total)
    assert summaries["adv_mean"].item() == 7.5
    assert summaries["adv_std"].item() == 0.0
    assert summaries["adv"][batch.valids].item() == 0.0
    assert policy_loss.item() == 0.0
    assert value_loss.item() == 4.0
    assert actor_gradient is not None and actor_gradient.item() == 0.0
    assert critic_gradient is not None and critic_gradient.item() == -4.0


class _TinyDistribution:
    def __init__(self, logits):
        self.logits = logits

    def log_prob(self, actions):
        log_probabilities = torch.log_softmax(self.logits, dim=-1)
        return log_probabilities.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)


class _TinyActorCritic(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor_weight = torch.nn.Parameter(torch.tensor(0.2))
        self.critic_weight = torch.nn.Parameter(torch.tensor(0.1))
        self.action_space = object()
        self._distribution = None

    def forward_head(self, normalized_obs):
        return normalized_obs["features"]

    def forward_core(self, head_outputs, rnn_states):
        return head_outputs, rnn_states

    def forward_tail(self, core_outputs, values_only, sample_actions):
        del values_only, sample_actions
        signal = core_outputs.squeeze(-1)
        logits = torch.stack(
            [self.actor_weight * signal, -self.actor_weight * signal], dim=-1
        )
        self._distribution = _TinyDistribution(logits)
        return {"values": self.critic_weight * signal}

    def action_distribution(self):
        return self._distribution


class _ActualUpstreamLossLearner:
    _policy_loss = staticmethod(Learner._policy_loss)
    _value_loss = Learner._value_loss

    def __init__(self):
        self.cfg = SimpleNamespace(
            encoder_custom="epom_trace_context",
            normalize_input=False,
            recurrence=1,
            ppo_clip_ratio=0.1,
            ppo_clip_value=0.2,
            use_rnn=False,
            with_vtrace=False,
            value_loss_coeff=0.5,
        )
        self.timing = SimpleNamespace(add_time=lambda _name: nullcontext())
        self.actor_critic = _TinyActorCritic()

    @staticmethod
    def exploration_loss_func(distribution, valids, num_invalids):
        del num_invalids
        probabilities = torch.softmax(distribution.logits, dim=-1)
        entropy = -(probabilities * torch.log_softmax(distribution.logits, dim=-1)).sum(
            dim=-1
        )
        return -0.01 * entropy[valids].mean()

    @staticmethod
    def kl_loss_func(action_space, old_logits, distribution, valids, num_invalids):
        del action_space, old_logits, num_invalids
        connected_zero = distribution.logits.sum(dim=-1) * 0.0
        selected = connected_zero[valids]
        return selected.detach(), selected.mean()


def _unwrap_sample_factory_loss():
    function = Learner._calculate_losses
    while hasattr(function, "_srslm_original_calculate_losses"):
        function = function._srslm_original_calculate_losses
    return function


def test_guard_executes_the_actual_sample_factory_advantage_path_for_singleton():
    upstream = _unwrap_sample_factory_loss()
    compact_source = " ".join(inspect.getsource(upstream).split())
    assert (
        "adv_std, adv_mean = torch.std_mean(masked_select(adv, valids, num_invalids))"
        in compact_source
    )
    guarded = _build_switcher_calculate_losses(upstream)
    learner = _ActualUpstreamLossLearner()
    features = torch.tensor([[1.0], [2.0], [3.0]])
    with torch.no_grad():
        initial_logits = torch.stack(
            [
                learner.actor_critic.actor_weight * features.squeeze(-1),
                -learner.actor_critic.actor_weight * features.squeeze(-1),
            ],
            dim=-1,
        )
        old_log_probabilities = torch.log_softmax(initial_logits, dim=-1)[:, 0]
    batch = AttrDict(
        normalized_obs={"features": features},
        valids=torch.tensor([False, True, False]),
        rnn_states=torch.zeros(3, 1),
        actions=torch.zeros(3, dtype=torch.long),
        log_prob_actions=old_log_probabilities,
        advantages=torch.tensor([1000.0, 7.5, -1000.0]),
        returns=torch.tensor([100.0, 2.0, -100.0]),
        values=torch.zeros(3),
        action_logits=torch.zeros(3, 2),
    )

    losses = guarded(learner, batch, 2)
    summaries = losses[6]
    total = sum(losses[index] for index in (1, 2, 4, 5))
    assert torch.isfinite(total)
    assert summaries["adv_std"].item() == 0.0
    assert summaries["adv_mean"].item() == 7.5
    assert summaries["adv"][batch.valids].item() == 0.0
    assert torch.isfinite(losses[1])
    assert torch.isfinite(losses[2])
    assert torch.isfinite(losses[4])
    assert torch.isfinite(losses[5])


@pytest.mark.parametrize("same_only", [False, True])
def test_actual_sample_factory_masks_same_actions_but_keeps_all_value_targets(same_only):
    guarded = _build_switcher_calculate_losses(_unwrap_sample_factory_loss())
    learner = _ActualUpstreamLossLearner()
    learner.cfg.encoder_custom = "switcher"
    features = torch.arange(1.0, 7.0).unsqueeze(-1)
    actions = [0, 1, 2, 4, 3, 1]
    candidate_actions = actions if same_only else [4, 4, 2, 1, 3, 2]
    logits = torch.stack(
        [0.2 * features.squeeze(-1), -0.2 * features.squeeze(-1)], dim=-1
    )
    batch = AttrDict(
        normalized_obs={
            "features": features,
            "aoreplan_action": _one_hot_actions(actions),
            "caar_action": _one_hot_actions(candidate_actions),
        },
        valids=torch.tensor([True, True, True, True, True, False]),
        rnn_states=torch.zeros(6, 1),
        actions=torch.zeros(6, dtype=torch.long),
        log_prob_actions=torch.log_softmax(logits, dim=-1)[:, 0],
        advantages=torch.tensor([1000.0, 1.0, -1000.0, 3.0, 50000.0, -90000.0]),
        returns=torch.arange(1.0, 7.0),
        values=torch.zeros(6),
        action_logits=logits,
    )
    original_valids = batch.valids
    losses = guarded(learner, batch, 1)
    expected_value_loss = learner._value_loss(
        0.1 * features.squeeze(-1), batch["values"], batch.returns,
        learner.cfg.ppo_clip_value, original_valids, 1,
    )
    assert torch.allclose(losses[5], expected_value_loss)
    assert batch.valids is original_valids
    if same_only:
        assert all(losses[index].item() == 0.0 for index in (1, 2, 4))
    else:
        assert losses[6]["adv_mean"].item() == 2.0
        assert torch.allclose(losses[6]["adv_std"], torch.tensor(2.0).sqrt())
        keep = torch.tensor([1, 3])
        active = AttrDict({
            key: ({k: v[keep] for k, v in value.items()}
                  if key == "normalized_obs" else value[keep])
            for key, value in batch.items()
        })
        reference = guarded(learner, active, 0)
        for index in (1, 2, 3, 4):
            assert torch.allclose(losses[index], reference[index])

    policy_gradient = torch.autograd.grad(
        sum(losses[index] for index in (1, 2, 4)),
        learner.actor_critic.actor_weight,
        retain_graph=True,
        allow_unused=True,
    )[0]
    critic_gradient = torch.autograd.grad(
        losses[5], learner.actor_critic.critic_weight
    )[0]
    assert torch.isfinite(critic_gradient) and critic_gradient.item() != 0.0
    if same_only:
        assert policy_gradient is None or policy_gradient.item() == 0.0
    else:
        assert torch.isfinite(policy_gradient) and policy_gradient.item() != 0.0


@pytest.mark.parametrize("encoder_custom", ["switcher", "epom_trace_context"])
def test_zero_valid_rows_are_rejected_before_loss_evaluation(encoder_custom):
    learner = _FakeLearner()
    learner.cfg.encoder_custom = encoder_custom
    batch = _batch(
        actions=[1, 2, 3],
        advantages=[1.0, 2.0, 3.0],
        actor_signal=[4.0, 5.0, 6.0],
        returns=[7.0, 8.0, 9.0],
        valids=[False, False, False],
    )

    try:
        _calculate(learner, batch)
    except RuntimeError as error:
        assert "zero valid samples" in str(error)
        assert "optimizer-state drift" in str(error)
    else:
        raise AssertionError("An empty valid mask must be rejected explicitly")


def test_train_import_installs_patch_in_a_fresh_process():
    code = """
import train_arpe
from sample_factory.algo.learning.learner import Learner

assert getattr(
    Learner._calculate_losses,
    '_srslm_switcher_actor_mask_patch',
    False,
)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr


def test_component_registration_repairs_a_missing_patch(monkeypatch):
    import train_arpe
    from sample_factory.algo.learning.learner import Learner

    patched = Learner._calculate_losses
    original = patched._srslm_original_calculate_losses
    monkeypatch.setattr(Learner, "_calculate_losses", original)

    monkeypatch.setattr(train_arpe, "_CUSTOM_COMPONENTS_REGISTERED", True)

    train_arpe.register_custom_components()

    assert getattr(
        Learner._calculate_losses,
        "_srslm_switcher_actor_mask_patch",
        False,
    )
