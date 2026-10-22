import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

import agents.arpe as arpe_module
from agents.arpe import ARPE, ARPEWeights
from learning.arpe_actor_critic import InferenceCorrection


@pytest.fixture
def arpe_runtime(tmp_path, monkeypatch):
    artifact = ARPEWeights.from_mapping(
        {
            "weights_path": "run",
            "checkpoint_path": "run/checkpoint_p0/selected.pth",
            "base_weights_path": "base",
            "base_checkpoint_path": "base/checkpoint_p0/selected.pth",
        },
        tmp_path,
    )
    artifact.checkpoint_path.parent.mkdir(parents=True)
    config_path = artifact.weights_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "full_config": {
                    "experiment_settings": {"epom_base_weights_path": "saved-base"},
                    "environment": {
                        "tau_radius": 5,
                        "tau_rho": 0.1,
                        "grid_memory_obs_radius": 7,
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    model = torch.nn.Linear(2, 5)
    model.model_to_device = Mock(side_effect=model.to)
    torch.save({"model": model.state_dict()}, artifact.checkpoint_path)
    env = SimpleNamespace(
        observation_space=SimpleNamespace(spaces={"tau": object()}),
        action_space=object(),
        close=Mock(),
    )
    register = Mock()
    validate = Mock(
        side_effect=lambda config: (
            None,
            SimpleNamespace(env="POMAPF-EPOM-ST-v0", full_config=config),
        )
    )
    create_env = Mock(return_value=env)
    create_actor = Mock(return_value=model)
    monkeypatch.setattr(arpe_module, "register_custom_components", register)
    monkeypatch.setattr(arpe_module, "validate_config", validate)
    monkeypatch.setattr(arpe_module, "create_env", create_env)
    monkeypatch.setattr(arpe_module, "create_actor_critic", create_actor)
    return SimpleNamespace(
        artifact=artifact,
        model=model,
        env=env,
        register=register,
        validate=validate,
        create_env=create_env,
        create_actor=create_actor,
        config_path=config_path,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("learned_gate_override", "all"),
        ("entropy_threshold_override", 0.9),
        ("checkpoint_kind", "auto"),
        ("action_sampling", "torch"),
    ],
)
def test_retired_inference_overrides_are_not_accepted(field, value):
    with pytest.raises(TypeError, match=field):
        ARPE(None, seed=0, device="cpu", **{field: value})


def test_checkpoint_selection_never_silently_falls_back_to_best(arpe_runtime):
    runtime = arpe_runtime
    torch.save(
        {"model": runtime.model.state_dict()},
        runtime.artifact.checkpoint_path.parent / "best_0001.pth",
    )
    artifact = replace(
        runtime.artifact,
        checkpoint_path=runtime.artifact.checkpoint_path.parent / "missing.pth",
    )
    with pytest.raises(FileNotFoundError):
        ARPE(artifact, seed=0, device="cpu")
    runtime.env.close.assert_called_once_with()


def test_checkpoint_path_is_required(tmp_path):
    with pytest.raises(KeyError, match="checkpoint_path"):
        ARPEWeights.from_mapping(
            {
                "weights_path": "unused",
                "base_weights_path": "base",
                "base_checkpoint_path": "base.pth",
            },
            tmp_path,
        )


def test_an_explicit_checkpoint_can_be_stored_outside_its_config_directory(
    arpe_runtime, tmp_path
):
    runtime = arpe_runtime
    candidate = tmp_path / "downloaded" / "checkpoint.pth"
    candidate.parent.mkdir(parents=True)
    state = {
        key: torch.ones_like(value) for key, value in runtime.model.state_dict().items()
    }
    torch.save({"model": state}, candidate)
    artifact = replace(runtime.artifact, checkpoint_path=candidate.resolve())
    adapter = ARPE(artifact, seed=0, device="cpu")
    for key, expected in state.items():
        torch.testing.assert_close(adapter.ppo.state_dict()[key], expected)
    assert adapter.checkpoint_path == candidate.resolve()
    assert adapter.artifact.weights_path == runtime.artifact.weights_path


def test_direct_loading_freezes_model_and_closes_temporary_environment(arpe_runtime):
    runtime = arpe_runtime
    original_config = runtime.config_path.read_bytes()
    adapter = ARPE(runtime.artifact, seed=17, device="cpu")
    assert adapter.ppo is runtime.model
    assert adapter.device == torch.device("cpu")
    assert adapter.seed == 17
    assert not adapter.ppo.training
    assert all(not parameter.requires_grad for parameter in adapter.ppo.parameters())
    assert isinstance(adapter.ppo.inference_correction, InferenceCorrection)
    assert adapter.cfg.base_checkpoint_path == str(runtime.artifact.base_checkpoint_path)
    assert adapter.cfg.full_config["experiment_settings"]["epom_base_weights_path"] == str(
        runtime.artifact.base_weights_path
    )
    assert runtime.config_path.read_bytes() == original_config
    runtime.register.assert_called_once_with()
    runtime.create_env.assert_called_once_with(
        adapter.cfg.env, cfg=adapter.cfg, env_config={}
    )
    runtime.create_actor.assert_called_once_with(
        adapter.cfg, runtime.env.observation_space, runtime.env.action_space
    )
    runtime.model.model_to_device.assert_called_once_with(torch.device("cpu"))
    runtime.env.close.assert_called_once_with()


@pytest.mark.parametrize("failure", ["missing_tau", "model_creation"])
def test_temporary_environment_is_closed_on_initialization_failure(
    arpe_runtime, failure
):
    runtime = arpe_runtime
    if failure == "missing_tau":
        runtime.env.observation_space.spaces = {}
        message = "separate tau observation"
    else:
        runtime.create_actor.side_effect = RuntimeError("model creation failed")
        message = "model creation failed"
    with pytest.raises(RuntimeError, match=message):
        ARPE(runtime.artifact, seed=0, device="cpu")
    runtime.env.close.assert_called_once_with()
    if failure == "missing_tau":
        runtime.create_actor.assert_not_called()


def test_direct_loading_requires_all_checkpoint_parameters(arpe_runtime):
    runtime = arpe_runtime
    torch.save(
        {"model": {"weight": runtime.model.weight.detach().clone()}},
        runtime.artifact.checkpoint_path,
    )
    with pytest.raises(RuntimeError, match="Missing key"):
        ARPE(runtime.artifact, seed=0, device="cpu")
    runtime.env.close.assert_called_once_with()


def test_after_reset_clears_memories_and_replays_seeded_randomness(arpe_runtime):
    adapter = ARPE(arpe_runtime.artifact, seed=17, device="cpu")
    adapter.aco.configure(3, 3)
    adapter.after_reset()
    expected_numpy = adapter._numpy_rng.random(8)
    expected_torch = torch.rand(8)
    adapter.rnn_states = torch.ones(2, 3)
    adapter.aco.tau.fill(2)
    adapter.aco.prev_positions = np.array([[1, 1]])
    adapter.grid_memory.memories = [object()]
    adapter.after_reset()
    assert adapter.rnn_states is None
    assert not adapter.aco.tau.any()
    assert adapter.aco.prev_positions is None
    assert adapter.grid_memory.memories is None
    np.testing.assert_array_equal(adapter._numpy_rng.random(8), expected_numpy)
    torch.testing.assert_close(torch.rand(8), expected_torch, rtol=0, atol=0)


def test_after_step_clears_memories_only_when_all_agents_are_done(arpe_runtime):
    adapter = ARPE(arpe_runtime.artifact, seed=17, device="cpu")
    adapter.rnn_states = torch.ones(2, 3)
    adapter.aco.configure(3, 3).fill(2)
    adapter.aco.prev_positions = np.array([[1, 1]])
    adapter.grid_memory.memories = [object()]
    states = adapter.rnn_states
    memories = adapter.grid_memory.memories
    positions = adapter.aco.prev_positions
    numpy_rng = adapter._numpy_rng
    adapter.after_step([True, False])
    assert adapter.rnn_states is states
    assert adapter.grid_memory.memories is memories
    assert adapter.aco.prev_positions is positions
    assert np.all(adapter.aco.tau == 2)
    adapter.after_step([True, True])
    assert adapter.rnn_states is None
    assert adapter.grid_memory.memories is None
    assert adapter.aco.prev_positions is None
    assert not adapter.aco.tau.any()
    assert adapter._numpy_rng is numpy_rng
