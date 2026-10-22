import numpy as np
import pytest

from agents.controller import (
    AO_BRANCH,
    SwitcherController,
)
from planning.branch import AORePlanStep


class FakeARPE:
    def __init__(self, actions):
        self.actions = actions

    def act(self, *_args):
        return list(self.actions)

    def after_reset(self):
        pass

    def after_step(self, _dones):
        pass

    def set_grid_config(self, _cfg):
        pass

    def set_env(self, _env):
        pass


class FakeAORePlan:
    def __init__(self, actions):
        self.actions = tuple(actions)
        self.commits = []

    def reset(self):
        pass

    def propose(self, observations):
        count = len(observations)
        return AORePlanStep(
            actions=self.actions,
            planned_mask=(True,) * count,
        )

    def commit(self, mask):
        self.commits.append(tuple(mask))


def observations(count):
    return [
        {
            "obstacles": np.zeros((11, 11), dtype=np.float32),
            "agents": np.zeros((11, 11), dtype=np.float32),
            "xy": (i, 0),
            "target_xy": (i, 5),
        }
        for i in range(count)
    ]


def test_wait_actions_bypass_switcher_and_directly_use_arpe():
    planner = FakeAORePlan([0, 4, 0])
    controller = SwitcherController(FakeARPE([1, 2, 3]), planner)
    prepared = controller.prepare_actions(observations(3))
    assert prepared.switch_allowed_mask == (False, True, False)
    result = controller.resolve_actions([AO_BRANCH])
    assert result == (1, 4, 3)
    assert planner.commits == [(False, True, False)]


def test_switcher_choice_count_must_equal_nonwait_count():
    controller = SwitcherController(FakeARPE([1, 2]), FakeAORePlan([4, 3]))
    controller.prepare_actions(observations(2))
    with pytest.raises(ValueError, match="non-wait"):
        controller.resolve_actions([AO_BRANCH])


def test_wait_bypass_never_commits_aoreplan_wait():
    planner = FakeAORePlan([0])
    controller = SwitcherController(FakeARPE([1]), planner)
    controller.prepare_actions(observations(1))
    result = controller.resolve_actions([])
    assert result == (1,)
    assert planner.commits == [(False,)]


def test_same_actions_still_enter_switcher_and_commit_executed_planner_action():
    planner = FakeAORePlan([4, 0])
    controller = SwitcherController(FakeARPE([4, 0]), planner)
    prepared = controller.prepare_actions(observations(2))
    assert prepared.switch_allowed_mask == (True, False)
    assert controller.resolve_actions([0]) == (4, 0)
    assert planner.commits == [(True, True)]


def test_agent_count_cannot_change_without_reset():
    candidate = FakeARPE([1])
    planner = FakeAORePlan([4])
    controller = SwitcherController(candidate, planner)
    controller.prepare_actions(observations(1))
    controller.resolve_actions([AO_BRANCH])
    candidate.actions = [1, 1]
    planner.actions = (4, 4)
    controller.prepare_actions(observations(2))
    with pytest.raises(RuntimeError, match="Agent count changed"):
        controller.resolve_actions([AO_BRANCH, AO_BRANCH])

    controller.after_reset()
    controller.prepare_actions(observations(2))
    assert controller.resolve_actions([AO_BRANCH, AO_BRANCH]) == (4, 4)


def test_done_mask_must_match_resolved_agent_count():
    controller = SwitcherController(FakeARPE([1]), FakeAORePlan([4]))
    controller.prepare_actions(observations(1))
    controller.resolve_actions([AO_BRANCH])
    with pytest.raises(RuntimeError, match="Done mask and agent count differ"):
        controller.after_step([False, False])
    controller.after_step([True])


def test_pending_decision_must_be_resolved_before_next_proposal():
    controller = SwitcherController(FakeARPE([1]), FakeAORePlan([4]))
    with pytest.raises(RuntimeError, match="prepare_actions"):
        controller.resolve_actions([AO_BRANCH])
    controller.prepare_actions(observations(1))
    with pytest.raises(RuntimeError, match="previous switch decision"):
        controller.prepare_actions(observations(1))
    assert controller.resolve_actions([AO_BRANCH]) == (4,)


@pytest.mark.parametrize("source", ["ARPE", "AORePlan"])
@pytest.mark.parametrize("action", [None, -1, 5, 1.5, float("nan"), float("inf")])
def test_both_candidates_reject_invalid_primitive_actions(source, action):
    arpe = FakeARPE([action if source == "ARPE" else 1])
    planner = FakeAORePlan([action if source == "AORePlan" else 4])
    controller = SwitcherController(arpe, planner)
    with pytest.raises(RuntimeError, match=source):
        controller.prepare_actions(observations(1))
    assert controller._pending is None
    assert planner.commits == ([(False,)] if source == "AORePlan" else [])


@pytest.mark.parametrize("mask", [(), (False,)])
def test_incomplete_planner_batch_is_cancelled(mask):
    planner = FakeAORePlan([4])
    planner.propose = lambda _observations: AORePlanStep(actions=(4,), planned_mask=mask)
    controller = SwitcherController(FakeARPE([1]), planner)
    with pytest.raises(RuntimeError, match="AORePlan"):
        controller.prepare_actions(observations(1))
    assert planner.commits == [(False,)]
    assert controller._pending is None
