from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from pogema import GridConfig, pogema_v0

from agents.aoreplan import AORePlan, AORePlanConfig
from agents.hrss import HRSS
from agents.controller import SwitcherController
from planning.aoreplan import AORePlanWrapper
from planning.branch import AORePlanBranch
from pomapf_env.switcher import SwitcherEnv


class SequenceBase:
    def __init__(self, **_kwargs):
        self.actions = iter([1, 2, 2])
        self.rnd = np.random.default_rng(0)

    def act(self, observations, skip_agents=None):
        return [next(self.actions)]

    def commit_proposals(self, executed_mask):
        self.committed = tuple(executed_mask)


class FixedStatic:
    def __init__(self, action):
        self.action = action
        self.calls = 0

    def observe(self, observations):
        pass

    def get_action(self, _index, _observation):
        self.calls += 1
        return self.action


@pytest.fixture(autouse=True)
def branch_base(monkeypatch):
    monkeypatch.setattr("planning.branch.AORePlanBase", SequenceBase)


def observation(position=(5, 5), occupied=False):
    agents = np.zeros((11, 11), dtype=np.int8)
    agents[5, 6] = int(occupied)
    return {
        "obstacles": np.zeros((11, 11), dtype=np.int8),
        "agents": agents,
        "xy": position,
        "target_xy": (8, 8),
    }


def configured_policy(collision, policy_type=AORePlan):
    policy = policy_type(AORePlanConfig())
    policy.set_grid_config(SimpleNamespace(collision_system=collision))
    with patch("agents.aoreplan.AORePlanBase", SequenceBase):
        policy.after_reset()
    return policy


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_default_static_action_into_visible_agent_waits_under_both_rules(collision):
    wrapper = configured_policy(collision)._ao_wrapper
    assert type(wrapper) is AORePlanWrapper
    wrapper.static_astar = FixedStatic(4)
    assert wrapper.act([observation()]) == [1]
    assert wrapper.act([observation((4, 5), occupied=True)]) == [0]
    assert wrapper.static_astar.calls == 1


@pytest.mark.parametrize(
    "policy_type,collision",
    [
        (AORePlan, "block_both"),
        (AORePlan, "soft"),
    ],
)
def test_static_astar_without_action_still_waits(policy_type, collision):
    wrapper = configured_policy(collision, policy_type)._ao_wrapper
    wrapper.static_astar = FixedStatic(None)
    assert wrapper.act([observation()]) == [1]
    assert wrapper.act([observation((4, 5))]) == [0]


def test_default_wrapper_always_retains_local_check():
    wrapper = AORePlanWrapper(SequenceBase())
    wrapper.static_astar = FixedStatic(4)
    wrapper.act([observation()])
    assert wrapper.act([observation((4, 5), occupied=True)]) == [0]


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_branch_conservative_behavior_survives_reset(collision):
    branch = AORePlanBranch()
    controller = SwitcherController(Candidate(), branch)
    controller.set_grid_config(SimpleNamespace(collision_system=collision))
    for _ in range(2):
        branch.reset()
        assert type(branch._wrapper) is AORePlanWrapper
        branch._wrapper.static_astar = FixedStatic(4)
        branch.propose([observation()])
        branch.commit([True])
        step = branch.propose([observation((4, 5), occupied=True)])
        assert step.actions == (0,)
        branch.commit([True])


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_standalone_config_survives_episode_and_wrapper_recreation(collision):
    policy = configured_policy(collision)
    for _ in range(2):
        with patch("agents.aoreplan.AORePlanBase", SequenceBase):
            policy.after_reset()
        assert type(policy._ao_wrapper) is AORePlanWrapper
        policy._ao_wrapper.static_astar = FixedStatic(4)
        assert policy.act([observation()]) == [1]
        assert policy.act([observation((4, 5), occupied=True)]) == [0]
        policy.after_step([True])
        assert policy._ao_wrapper is None


class Candidate:
    def after_reset(self):
        pass

    def set_grid_config(self, config):
        self.grid_config = config

    def set_env(self, env):
        self.env = env

    def act(self, observations, *_args):
        return [3] * len(observations)


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_hrss_default_static_collision_uses_wait_bypass_under_both_rules(collision):
    candidate = Candidate()
    planner = AORePlanBranch()
    policy = HRSS.__new__(HRSS)
    policy.controller = SwitcherController(candidate, planner)
    config = SimpleNamespace(collision_system=collision)
    policy.set_grid_config(config)
    policy.controller.after_reset()
    assert candidate.grid_config is config
    assert type(planner._wrapper) is AORePlanWrapper
    planner._wrapper.static_astar = FixedStatic(4)
    policy.controller.prepare_actions([observation()])
    policy.controller.resolve_actions([1])
    prepared = policy.controller.prepare_actions([observation((4, 5), occupied=True)])
    assert prepared.switch_allowed_mask == (False,)
    resolved = policy.controller.resolve_actions([])
    assert resolved == (3,)


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_training_env_reset_uses_same_planner_rule_as_deployment(collision):
    config = SimpleNamespace(collision_system=collision, seed=0, num_agents=1)
    base_env = SimpleNamespace(
        grid_config=config,
        reset=lambda: ([observation()], [{}]),
    )
    env = SwitcherEnv(
        grid_config=config,
        candidate_artifact=SimpleNamespace(),
        candidate_factory=lambda *_args, **_kwargs: Candidate(),
        planner_factory=AORePlanBranch,
        base_env_factory=lambda **_kwargs: base_env,
    )
    for _ in range(2):
        env.reset()
        assert type(env.controller.aoreplan._wrapper) is AORePlanWrapper
        env.controller.aoreplan._wrapper.static_astar = FixedStatic(4)
        env.controller.resolve_actions([1])
        prepared = env.controller.prepare_actions([observation((4, 5), occupied=True)])
        assert prepared.switch_allowed_mask == (False,)
        assert env.controller.resolve_actions([]) == (3,)


def test_supported_pogema_collision_rules():
    policy = AORePlan(AORePlanConfig())
    branch = AORePlanBranch()
    controller = SwitcherController(Candidate(), branch)
    for owner in (policy, controller):
        for collision in ("block_both", "soft", "priority"):
            owner.set_grid_config(SimpleNamespace(collision_system=collision))
        for config in (SimpleNamespace(), SimpleNamespace(collision_system="unknown")):
            with pytest.raises(ValueError, match="collision"):
                owner.set_grid_config(config)
    assert not hasattr(branch, "set_grid_config")


@pytest.mark.parametrize(
    "collision,moved", [("block_both", [False, True]), ("soft", [True, True])]
)
def test_pogema_resolves_following_after_actions_are_submitted(collision, moved):
    env = pogema_v0(
        GridConfig(
            map="\n".join(["......."] * 7),
            agents_xy=[(2, 1), (2, 2)],
            targets_xy=[[(5, 5), (5, 4)], [(5, 4), (5, 5)]],
            num_agents=2,
            obs_radius=1,
            max_episode_steps=8,
            on_target="restart",
            collision_system=collision,
        )
    )
    try:
        env.reset()
        before = np.asarray(env.unwrapped.grid.get_agents_xy()).copy()
        env.step([4, 4])
        after = np.asarray(env.unwrapped.grid.get_agents_xy())
        assert np.any(after != before, axis=1).tolist() == moved
        assert (after - before).tolist() == [[0, int(value)] for value in moved]
    finally:
        env.close()
