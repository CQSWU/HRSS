from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

import torch

_PATCH_ATTRIBUTE = "_hrss_switcher_actor_mask_patch"
_STD_MEAN_PATCH_LOCK = RLock()


class _DegenerateStdMeanTorchProxy:
    def __init__(self, base: Any):
        self._base = base

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)

    def std_mean(self, inputs, *args, **kwargs):
        if not args and not kwargs and inputs.numel() <= 1:
            if inputs.numel() == 1:
                return self._base.std_mean(inputs, correction=0)
            zero = inputs.new_zeros(())
            return zero, zero
        return self._base.std_mean(inputs, *args, **kwargs)


def _invalid_count(valids: torch.Tensor) -> int:
    return int(valids.numel() - valids.sum().item())


def _switch_allowed_mask(mb: Any) -> torch.Tensor:
    # Train branch selection only when candidates differ and the planner does not wait.
    actions = mb.normalized_obs["aoreplan_action"]
    if actions.ndim != 2 or actions.shape[-1] != 5:
        raise ValueError(
            "Switcher learner expects flattened five-way AORePlan one-hot "
            f"actions, got shape {tuple(actions.shape)}."
        )
    candidate_actions = mb.normalized_obs["caar_action"]
    if candidate_actions.shape != actions.shape:
        raise ValueError("Switcher candidate action tensors must have the same shape.")
    planner_actions = torch.argmax(actions, dim=-1)
    return planner_actions.ne(0) & planner_actions.ne(
        torch.argmax(candidate_actions, dim=-1)
    )


def _call_with_degenerate_std_guard(
    original: Callable,
    learner: Any,
    mb: Any,
    num_invalids: int,
):

    function_globals = original.__globals__
    with _STD_MEAN_PATCH_LOCK:
        base_torch = function_globals.get("torch")
        if base_torch is None or not hasattr(base_torch, "std_mean"):
            raise RuntimeError(
                "Cannot guard Sample Factory advantage normalization: the "
                "upstream loss function has no torch.std_mean binding."
            )
        function_globals["torch"] = _DegenerateStdMeanTorchProxy(base_torch)
        try:
            return original(learner, mb, num_invalids)
        finally:
            function_globals["torch"] = base_torch


def _calculate_with_valid_count_guard(
    original: Callable,
    learner: Any,
    mb: Any,
    num_invalids: int,
):

    valid_count = int(mb.valids.bool().sum().item())
    if valid_count >= 2:
        return original(learner, mb, num_invalids)
    if valid_count == 0:
        raise RuntimeError(
            "Sample Factory produced a minibatch with zero valid samples; "
            "the update is rejected before loss evaluation to avoid NaN and "
            "optimizer-state drift."
        )

    degenerate_num_invalids = _invalid_count(mb.valids.bool())
    return _call_with_degenerate_std_guard(
        original, learner, mb, degenerate_num_invalids
    )


def _build_switcher_calculate_losses(original: Callable) -> Callable:
    def patched(self, mb, num_invalids):
        if getattr(self.cfg, "encoder_custom", None) != "switcher":
            return _calculate_with_valid_count_guard(original, self, mb, num_invalids)

        if self.cfg.use_rnn:
            raise RuntimeError("Switcher actor masking requires use_rnn=false.")
        if self.cfg.with_vtrace:
            raise RuntimeError("Switcher actor masking requires with_vtrace=false.")
        if self.cfg.normalize_input:
            raise RuntimeError("Switcher actor masking requires normalize_input=false.")

        original_valids = mb.valids
        full_valids = original_valids.bool()
        switch_allowed = _switch_allowed_mask(mb)
        if switch_allowed.shape != full_valids.shape:
            raise ValueError(
                "AORePlan action mask and Sample Factory validity mask have "
                f"different shapes: {tuple(switch_allowed.shape)} versus "
                f"{tuple(full_valids.shape)}."
            )

        actor_valids = full_valids & switch_allowed
        full_num_invalids = _invalid_count(full_valids)
        actor_num_invalids = _invalid_count(actor_valids)
        actor_sample_count = int(actor_valids.sum().item())

        if actor_sample_count >= 2:
            mb.valids = actor_valids
            try:
                losses = _calculate_with_valid_count_guard(
                    original, self, mb, actor_num_invalids
                )
            finally:
                mb.valids = original_valids
        else:
            losses = _calculate_with_valid_count_guard(
                original, self, mb, full_num_invalids
            )

        (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            _masked_value_loss,
            loss_summaries,
        ) = losses

        values = loss_summaries["values"].squeeze(-1)
        # Keep all valid timesteps for value learning, including rule-controlled ones.
        value_loss = self._value_loss(
            values,
            mb["values"],
            mb.returns,
            self.cfg.ppo_clip_value,
            full_valids,
            full_num_invalids,
        )

        if actor_sample_count < 2:
            # Skip actor losses when too few real choices remain in the minibatch.
            zero = loss_summaries["values"].sum() * 0.0
            policy_loss = zero
            exploration_loss = zero
            kl_loss = zero
            kl_old = zero.detach().reshape(1)
            loss_summaries = dict(loss_summaries)
            loss_summaries["adv"] = torch.zeros_like(loss_summaries["adv"])
            loss_summaries["adv_mean"] = zero.detach()
            loss_summaries["adv_std"] = zero.detach()

        return (
            action_distribution,
            policy_loss,
            exploration_loss,
            kl_old,
            kl_loss,
            value_loss,
            loss_summaries,
        )

    setattr(patched, _PATCH_ATTRIBUTE, True)
    patched._hrss_original_calculate_losses = original
    return patched


def patch_switcher_learner_losses() -> None:

    from sample_factory.algo.learning.learner import Learner

    current = Learner._calculate_losses
    if getattr(current, _PATCH_ATTRIBUTE, False):
        return
    Learner._calculate_losses = _build_switcher_calculate_losses(current)
