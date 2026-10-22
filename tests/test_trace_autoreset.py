from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

import train_arpe
from pomapf_env.maps import MAPS_REGISTRY
from pomapf_env.env import make_pomapf
from pomapf_env.config import POMAPFConfig
from pomapf_env.trace import SharedTrace
from pomapf_env.wrappers import TauObservationWrapper


def _map(size=16, obstacle=None):
    cells = [["."] * size for _ in range(size)]
    if obstacle is not None:
        cells[obstacle[0]][obstacle[1]] = "#"
    return "\n".join("".join(row) for row in cells)


def _grid(**changes):
    values = dict(
        map=_map(),
        map_name=None,
        width=16,
        height=16,
        num_agents=2,
        agents_xy=[[4, 4], [8, 8]],
        targets_xy=[[[10, 10], [3, 4]], [[3, 3], [8, 9]]],
        seed=1,
        obs_radius=5,
        max_episode_steps=3,
        on_target="restart",
        collision_system="block_both",
    )
    values.update(changes)
    return POMAPFConfig(**values)


@pytest.fixture
def environment_factory(monkeypatch):
    created = []

    def factory(grid=None, *, auto_reset=True):
        grid = _grid() if grid is None else grid

        monkeypatch.setattr(
            train_arpe,
            "make_pomapf",
            lambda *, grid_config: make_pomapf(
                grid_config=grid_config, auto_reset=auto_reset
            ),
        )
        cfg = SimpleNamespace(
            encoder_custom="epom_trace_context",
            seed=1,
            full_config={
                "environment": {
                    "grid_config": grid.dict(),
                    "grid_memory_obs_radius": 7,
                    "tau_rho": 0.1,
                    "tau_radius": 5,
                }
            },
        )
        env = train_arpe.create_pogema_env("POMAPF-EPOM-ST-v0", cfg)
        created.append(env)
        current = env
        while not isinstance(current, TauObservationWrapper):
            current = current.env
        return env, current

    yield factory
    for env in created:
        env.close()


def _assert_first_frame(trace, observations):
    grid = trace._grid()
    obstacles = np.asarray(grid.obstacles, dtype=bool)
    positions = trace._global_positions()
    expected = np.zeros(obstacles.shape, dtype=np.float32)
    for xy in {tuple(position) for position in positions}:
        assert not obstacles[xy]
        expected[xy] = 1.0
    np.testing.assert_array_equal(trace.aco.tau, expected)
    np.testing.assert_array_equal(trace.aco._obstacle_mask, obstacles)
    np.testing.assert_array_equal(trace.aco.prev_positions, positions)

    reference = SharedTrace(rho=0.1)
    reference.configure_from_obstacle_mask(obstacles, clear=True)
    reference.tau[:] = expected
    for observation, (row, col) in zip(observations, positions):
        expected_crop = reference.extract_local_tau(row, col, 5)
        np.testing.assert_array_equal(observation["tau"][0], expected_crop)
        assert set(observation) == {"obs", "xy", "target_xy", "tau"}


@pytest.mark.parametrize("collision_system", ["block_both", "soft"])
def test_same_map_autoreset_starts_fresh_and_deposits_once(
    environment_factory, collision_system
):
    env, trace = environment_factory(_grid(collision_system=collision_system))
    obs, _ = env.reset()
    _assert_first_frame(trace, obs)
    previous_grid = trace._grid()
    previous_obstacles = np.asarray(previous_grid.obstacles).copy()
    for step in range(1, 7):
        obs, _, terminated, truncated, _ = env.step([0, 0])
        if step % 3 == 0:
            assert all(truncated) and not any(terminated)
            assert trace._grid() is not previous_grid
            np.testing.assert_array_equal(trace._grid().obstacles, previous_obstacles)
            _assert_first_frame(trace, obs)
            previous_grid = trace._grid()
        else:
            assert not any(terminated) and not any(truncated)
            assert trace._grid() is previous_grid


@pytest.mark.parametrize("second_size", [16, 20])
def test_autoreset_rebinds_changed_map_mask_and_shape(
    environment_factory, monkeypatch, second_size
):
    first_name = "trace-autoreset-test-map-0"
    second_name = "trace-autoreset-test-map-1"
    monkeypatch.setitem(MAPS_REGISTRY, first_name, _map(16, (5, 5)))
    monkeypatch.setitem(MAPS_REGISTRY, second_name, _map(second_size, (6, 6)))
    env, trace = environment_factory(
        _grid(map=None, map_name="trace-autoreset-test-map-")
    )
    obs, _ = env.reset()
    _assert_first_frame(trace, obs)
    old_grid = trace._grid()
    assert old_grid.config.map_name == first_name
    old_mask = trace.aco._obstacle_mask.copy()
    for _ in range(3):
        obs, _, _, truncated, _ = env.step([0, 0])
    assert all(truncated)
    assert trace._grid() is not old_grid
    assert trace._grid().config.map_name == second_name
    assert not np.array_equal(trace._grid().obstacles, old_mask)
    _assert_first_frame(trace, obs)

    new_grid = trace._grid()
    before = trace.aco.tau.copy()
    obs, _, _, truncated, _ = env.step([0, 0])
    assert not any(truncated)
    assert trace._grid() is new_grid
    expected = before * 0.9
    for xy in {tuple(position) for position in trace._global_positions()}:
        expected[xy] += 1.0
    np.testing.assert_allclose(trace.aco.tau, expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(trace.aco._obstacle_mask, new_grid.obstacles)


def test_terminal_without_autoreset_preserves_final_trace(environment_factory):
    env, trace = environment_factory(auto_reset=False)
    obs, _ = env.reset()
    grid = trace._grid()
    initial = trace.aco.tau.copy()
    for step in range(1, 4):
        obs, _, terminated, truncated, _ = env.step([0, 0])
        assert trace._grid() is grid
        expected_count = sum(0.9**power for power in range(step + 1))
        np.testing.assert_allclose(trace.aco.tau, initial * expected_count, atol=1e-6)
    assert all(truncated) and not any(terminated)
    assert float(trace.aco.tau.max()) == pytest.approx(3.439)
    obs, _ = env.reset()
    assert trace._grid() is not grid
    _assert_first_frame(trace, obs)


def test_lifelong_target_change_keeps_trace_and_grid(environment_factory):
    env, trace = environment_factory(
        _grid(targets_xy=[[[4, 5], [10, 10]], [[3, 3], [8, 9]]], max_episode_steps=10)
    )
    obs, _ = env.reset()
    grid = trace._grid()
    before = trace.aco.tau.copy()
    old_target = np.asarray(obs[0]["target_xy"]).copy()
    obs, _, terminated, truncated, _ = env.step([4, 0])
    assert not any(terminated) and not any(truncated)
    assert trace._grid() is grid
    assert not np.array_equal(obs[0]["target_xy"], old_target)
    expected = before * 0.9
    for xy in {tuple(position) for position in trace._global_positions()}:
        expected[xy] += 1.0
    np.testing.assert_allclose(trace.aco.tau, expected, atol=1e-6)
    assert np.count_nonzero(trace.aco.tau) == 3


def test_partial_termination_does_not_reset_trace(environment_factory):
    env, trace = environment_factory(
        _grid(on_target="finish", targets_xy=[[4, 5], [3, 3]], max_episode_steps=10)
    )
    obs, _ = env.reset()
    grid = trace._grid()
    initial_position = tuple(trace._global_positions()[0])
    obs, _, terminated, truncated, _ = env.step([4, 0])
    assert list(terminated) == [True, False]
    assert not any(truncated)
    assert trace._grid() is grid
    assert float(trace.aco.tau[initial_position]) == pytest.approx(0.9)


def test_all_terminated_autoreset_starts_fresh(environment_factory):
    env, trace = environment_factory(
        _grid(on_target="finish", targets_xy=[[4, 5], [8, 9]], max_episode_steps=10)
    )
    obs, _ = env.reset()
    grid = trace._grid()
    obs, _, terminated, truncated, _ = env.step([4, 4])
    assert all(terminated) and not any(truncated)
    assert trace._grid() is not grid
    _assert_first_frame(trace, obs)
