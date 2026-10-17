from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from agents.arpe import ARPEWeights
from agents.controller import SwitcherController
from learning.config import Experiment
from learning.arpe_actor_critic import (
    ARPEActorCritic,
    InferenceCorrection,
)
from planning.branch import AORePlanStep

ROOT = Path(__file__).resolve().parents[1]


def test_inference_gate_is_strict_and_uses_raw_correction():
    profile = InferenceCorrection()
    entropy = torch.tensor([0.0, 0.01, 0.010001, 1.0], dtype=torch.float64)
    logits = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    raw = (logits / 3).requires_grad_()
    final = profile.apply(logits, raw, entropy)
    assert profile.gate(entropy).flatten().tolist() == [False, False, True, True]
    assert torch.equal(final[:2], logits[:2])
    assert torch.equal(final[2:], logits[2:] + raw[2:])
    final.sum().backward()
    assert torch.equal(raw.grad[:2], torch.zeros_like(raw[:2]))
    assert torch.equal(raw.grad[2:], torch.ones_like(raw[2:]))


def test_native_training_uses_only_base_and_all_action_correction():
    logits = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0]])
    raw = torch.tensor([[1.0, -1.0, 2.0, -2.0, 0.0]]).repeat(2, 1)
    model = SimpleNamespace(
        training=False,
        _base_entropy=ARPEActorCritic._base_entropy,
    )
    apply = ARPEActorCritic._apply_configured_correction_rule
    native = apply(model, logits, raw)
    torch.testing.assert_close(
        native, logits + raw, rtol=0, atol=0
    )
    model.inference_correction = InferenceCorrection()
    final = apply(model, logits, raw)
    assert torch.equal(final[0], logits[0])
    assert torch.equal(final[1], logits[1] + raw[1])
    model.training = True
    with pytest.raises(RuntimeError, match="training"):
        apply(model, logits, raw)


def _observations(position=(0, 0), target=(0, 5)):
    return [
        {
            "obstacles": np.zeros((11, 11), np.float32),
            "agents": np.zeros((11, 11), np.float32),
            "xy": position,
            "target_xy": target,
        }
    ]


class _Candidate:
    def act(self, *_args):
        return [1]

    def after_reset(self):
        pass


class _Planner:
    action = 4

    def reset(self):
        pass

    def propose(self, _observations):
        return AORePlanStep(
            actions=(self.action,),
            planned_mask=(True,),
        )

    def commit(self, selected):
        self.committed = tuple(selected)


@pytest.mark.parametrize("branch,expected_action", [(0, 1), (1, 3)])
def test_wait_only_controller_leaves_reverse_actions_to_switcher(
    branch, expected_action
):
    planner = _Planner()
    controller = SwitcherController(_Candidate(), planner)
    first = controller.prepare_actions(_observations())
    assert first.switch_allowed_mask == (True,)
    controller.resolve_actions([1])
    planner.action = 3
    second = controller.prepare_actions(_observations((0, 1)))

    assert second.switch_allowed_mask == (True,)
    assert not second.switcher_state["aoreplan_action"][:, 0].any()
    assert controller.resolve_actions([branch]) == (expected_action,)
    assert planner.committed == (branch == 1,)
    planner.action = 0
    third = controller.prepare_actions(_observations((0, 1)))
    assert third.switch_allowed_mask == (False,)
    assert third.switcher_state["aoreplan_action"][:, 0].all()
    assert controller.resolve_actions([]) == (1,)
    assert planner.committed == (False,)
    controller.after_reset()
    planner.action = 3
    assert controller.prepare_actions(_observations((0, 1))).switch_allowed_mask == (
        True,
    )


def test_training_and_deployment_share_wait_only_controller():
    import inspect
    from agents.srslm import SRSLM, SRSLMConfig
    from pomapf_env.switcher import SwitcherEnv

    assert SwitcherEnv.controller_class is SwitcherController
    assert (
        inspect.signature(SRSLM).parameters["controller_factory"].default
        is SwitcherController
    )
    assert "final_reverse_guard_enabled" not in SRSLMConfig.__fields__


def test_final_manifest_only_selects_weights():
    mapping = json.loads((ROOT / "configs/arpe.json").read_text())
    artifact = ARPEWeights.from_mapping(mapping, ROOT)
    assert ARPEWeights.from_mapping(artifact.as_dict(), ROOT) == artifact
    assert set(mapping) == {
        "weights_path", "checkpoint_path", "base_weights_path", "base_checkpoint_path"
    }
    assert not any("sha" in key.lower() for key in mapping)
    cfg = Experiment(
        **yaml.safe_load((ROOT / "configs/train_arpe.yaml").read_text())
    )
    assert cfg.experiment_settings.train_for_env_steps == 1_000_000_000


def test_final_weight_loading_and_small_cpu_forward_when_available():
    mapping = json.loads((ROOT / "configs/arpe.json").read_text())
    artifact = ARPEWeights.from_mapping(mapping, ROOT)
    switcher_dir = ROOT / "weights/SRSLM-Switcher-Final-1B"
    if (
        not artifact.checkpoint_path.is_file()
        or not (switcher_dir / "config.json").is_file()
    ):
        pytest.skip("Final model weights are distributed separately")
    from agents.srslm import SRSLM, SRSLMConfig
    from agents.switcher import SwitcherConfig
    from agents.controller import build_switcher_state
    from pomapf_env.tie_breaking import TIE_KEY
    from sample_factory.algo.utils.tensor_dict import TensorDict
    from sample_factory.model.model_utils import get_rnn_size

    source_config = artifact.weights_path / "config.json"
    source_bytes = source_config.read_bytes()
    policy = SRSLM(
        SRSLMConfig(
            device="cpu",
            switcher=SwitcherConfig(path_to_weights=str(switcher_dir), device="cpu"),
        ),
        candidate=artifact,
    )
    model = policy.candidate.ppo
    assert int(model.paper_entropy_gate_version) == 0
    assert isinstance(model.inference_correction, InferenceCorrection)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    config_before = deepcopy(model.cfg.full_config)
    batch = TensorDict(
        obs=torch.zeros(2, 3, 15, 15),
        xy=torch.zeros(2, 2),
        target_xy=torch.ones(2, 2),
        tau=torch.zeros(2, 1, 11, 11),
        tau_free_mask=torch.ones(2, 1, 11, 11),
    )
    batch[TIE_KEY] = torch.arange(5).float().repeat(2, 2, 1)
    captured = {}
    hooks = [
        model.action_parameterization.register_forward_hook(
            lambda _module, _inputs, output: captured.update(base=output[0].detach())
        ),
        model.trace_multiplier_head.register_forward_hook(
            lambda _module, _inputs, output: captured.update(raw=output.detach())
        ),
    ]
    try:
        with torch.no_grad():
            result = model(batch, torch.zeros(2, get_rnn_size(policy.candidate.cfg)))
    finally:
        for hook in hooks:
            hook.remove()
    gate = (model._base_entropy(captured["base"]) > 0.01).unsqueeze(-1)
    torch.testing.assert_close(
        result["action_logits"], captured["base"] + gate * captured["raw"], rtol=0, atol=0
    )
    assert result["action_logits"].shape == (2, 5)
    assert torch.isfinite(result["action_logits"]).all()
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key]), key
    assert model.cfg.full_config == config_before
    assert source_config.read_bytes() == source_bytes
    state = build_switcher_state(_observations() * 2, [1, 2], [4, 3])
    state["switch_allowed"] = np.ones((2, 1), dtype=np.float32)
    assert policy.switcher.choose(state).shape == (2,)
