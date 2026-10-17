from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from agents import switcher as switcher_module
from agents.hrss import HRSS, HRSSConfig
from agents.switcher import Switcher, SwitcherConfig
from agents.controller import SWITCHER_FIELD_SHAPES


class _FrozenPolicy:
    def __init__(self):
        self.weight = torch.nn.Parameter(torch.ones(()))

    def parameters(self):
        return [self.weight]


class _FakeCandidate:
    def __init__(self, _artifact, *, seed, device):
        self.artifact = _artifact
        self.seed = seed
        self.device = torch.device("cpu")
        self.ppo = _FrozenPolicy()

    def after_reset(self):
        pass


class _FakeSwitcher:
    candidate_artifact = SimpleNamespace(
        project_root=None,
    )

    def __init__(self, _cfg):
        self.candidate_artifact = SimpleNamespace(
            project_root=Path(__file__).resolve().parents[1],
        )


class _MissingCandidateSwitcher(_FakeSwitcher):
    def __init__(self, _cfg):
        self.candidate_artifact = None


class _FakePlanner:
    def __init__(self, **_kwargs):
        pass

    def reset(self):
        pass


def _config():
    return HRSSConfig(
        switcher=SwitcherConfig(path_to_weights="unused"),
    )


def test_hrss_loads_the_arpe_paths_saved_with_switcher():
    algorithm = HRSS(
        _config(),
        candidate_factory=_FakeCandidate,
        switcher_factory=_FakeSwitcher,
        planner_factory=_FakePlanner,
    )

    assert algorithm.candidate.seed == 0
    assert algorithm.candidate.device == torch.device("cpu")


def test_hrss_rejects_a_switcher_without_candidate_paths():
    with pytest.raises(RuntimeError, match="does not contain"):
        HRSS(
            _config(),
            candidate_factory=_FakeCandidate,
            switcher_factory=_MissingCandidateSwitcher,
            planner_factory=_FakePlanner,
        )


def test_hrss_uses_explicit_arpe_paths_without_a_configuration_roundtrip():
    artifact = object()
    algorithm = HRSS(
        _config(),
        candidate=artifact,
        candidate_factory=_FakeCandidate,
        switcher_factory=_MissingCandidateSwitcher,
        planner_factory=_FakePlanner,
    )
    assert algorithm.candidate.artifact is artifact


@pytest.mark.parametrize("option,value", [("deterministic", True), ("checkpoint_kind", "best")])
def test_switcher_has_no_variant_switches(option, value):
    with pytest.raises(ValueError):
        SwitcherConfig(**{option: value})


def test_switcher_loads_latest_checkpoint_before_best(tmp_path):
    best = tmp_path / "best_000001.pth"
    best.touch()
    assert Switcher._resolve_checkpoint(tmp_path) == best.resolve()
    latest = tmp_path / "checkpoint_000002_000000200.pth"
    latest.touch()
    assert Switcher._resolve_checkpoint(tmp_path) == latest.resolve()


def test_switcher_uses_sampled_actor_actions_and_preserves_reset_seed(monkeypatch):
    count = 16
    state = {
        key: np.zeros((count, *shape), dtype=np.float32)
        for key, shape in SWITCHER_FIELD_SHAPES.items()
    }
    state["caar_action"][:, 1] = 1.0
    state["aoreplan_action"][:, 4] = 1.0

    def actor(observations, rnn_states):
        assert torch.all(observations["switch_allowed"] == 1.0)
        assert rnn_states.shape == (count, 0)
        return {"actions": torch.randint(2, (count,))}

    switcher = Switcher.__new__(Switcher)
    switcher.cfg = SwitcherConfig(seed=42)
    switcher.device = torch.device("cpu")
    switcher.rnn_state_size = 0
    switcher.ppo = actor
    monkeypatch.setattr(
        switcher_module, "prepare_and_normalize_obs", lambda _actor, obs: obs
    )

    torch.manual_seed(42)
    expected_first = torch.randint(2, (count,)).numpy()
    expected_second = torch.randint(2, (count,)).numpy()
    switcher.after_reset()
    np.testing.assert_array_equal(switcher.choose(state), expected_first)
    np.testing.assert_array_equal(switcher.choose(state), expected_second)
    switcher.after_reset()
    np.testing.assert_array_equal(switcher.choose(state), expected_first)


@pytest.mark.parametrize("allowed", [(False, True, False), (False, False, False)])
def test_hrss_only_sends_nonwait_rows_to_switcher(allowed):
    features = {"obs": np.arange(3, dtype=np.float32)[:, None]}
    prepared = SimpleNamespace(switch_allowed_mask=allowed, switcher_state=features)
    selected_batches = []
    resolved_branches = []

    def choose(state):
        selected_batches.append(state["obs"])
        return np.ones(len(state["obs"]), dtype=np.int64)

    def resolve(branches):
        resolved_branches.append(tuple(branches))
        return (1, 4, 3)

    algorithm = HRSS.__new__(HRSS)
    algorithm.switcher = SimpleNamespace(choose=choose)
    algorithm.controller = SimpleNamespace(
        prepare_actions=lambda *_args: prepared,
        resolve_actions=resolve,
    )
    assert algorithm.act([{}, {}, {}]) == [1, 4, 3]
    assert resolved_branches == [(1,) if any(allowed) else ()]
    if any(allowed):
        assert len(selected_batches) == 1
        np.testing.assert_array_equal(selected_batches[0], features["obs"][[1]])
    else:
        assert selected_batches == []
