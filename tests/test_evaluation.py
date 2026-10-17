import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

import evaluate as runner
from agents.aoreplan import AORePlan, AORePlanConfig
from planning.aoreplan import AORePlanWrapper


def test_public_names_are_explicit_and_retired_names_are_rejected():
    assert runner.SUPPORTED_ALGORITHMS == ("AORePlan", "ARPE", "HRSS")
    assert set(runner.ALGORITHM_ALIASES.values()) == set(runner.SUPPORTED_ALGORITHMS)
    for name in (
        "DCC",
        "DHC",
        "Follower",
        "NoReweight",
        "HRSS-NoWaitDetect",
        "HRSS-WaitDetectOnly",
        "v8b",
        "RePlan",
        "EPOM-L",
        "EPOM-Lifelong-FT",
        "Direct",
        "HRSS-NoRule",
        "HRSS-OnlyRule",
        "HRSS-NoWait",
        "HRSS-OnlyWait",
        "AORePlan-SoftNoCheck",
    ):
        with pytest.raises(Exception):
            runner.parse_algorithms(name)


def test_public_cli_defaults_to_aoreplan_only():
    with patch.object(sys, "argv", ["evaluate.py"]):
        args = runner.parse_args()
    assert args.algorithms == ["AORePlan"]
    assert runner.parse_agent_counts(args) == list(range(50, 501, 50))
    assert runner.parse_seeds(args) == [0]
    assert args.map_list is None
    assert args.map_file is None
    assert runner.parse_algorithms("all") == ["AORePlan"]
    assert runner.parse_algorithms("AORePlan,ARPE,HRSS") == [
        "AORePlan",
        "ARPE",
        "HRSS",
    ]


@pytest.mark.parametrize("algorithm", runner.SUPPORTED_ALGORITHMS)
def test_policies_are_always_episode_fresh(algorithm):
    task = {
        "algorithm": algorithm,
        "main_dir": ".",
        "seed": 0,
        "map_name": "test-map",
        "max_steps": 16,
        "num_agents": 4,
        "animate": False,
    }
    first, second = object(), object()
    with (
        patch.object(runner, "build_algorithm", side_effect=[first, second]) as build,
        patch.object(runner, "run_algorithm", return_value={}) as run,
    ):
        for _ in range(2):
            result = runner.run_single_experiment(task)
            assert "error" not in result
    assert build.call_count == 2
    assert [call.args[0] for call in run.call_args_list] == [first, second]


@pytest.mark.parametrize(
    "options",
    [
        ["--cache-algorithms"],
        ["--no-save"],
        ["--works", "1"],
        ["--agent-start", "50"],
        ["--agent-stop", "100"],
        ["--agent-step", "50"],
        ["--seed", "0"],
        ["--map-types", "random"],
        ["--map", "random=test-map"],
        ["--trim-border"],
    ],
)
def test_retired_runner_switches_are_rejected(options):
    with patch.object(sys, "argv", ["evaluate.py", *options]):
        with pytest.raises(SystemExit):
            runner.parse_args()


def test_agent_and_seed_lists_are_explicit_sorted_and_unique():
    assert runner.parse_agent_counts(SimpleNamespace(agents="100, 50,100")) == [50, 100]
    assert runner.parse_seeds(SimpleNamespace(seeds="42, 0,42")) == [0, 42]


@pytest.mark.parametrize("value", ["", ",,", "0", "-1", "50,invalid"])
def test_invalid_agent_lists_are_rejected(value):
    with pytest.raises(ValueError):
        runner.parse_agent_counts(SimpleNamespace(agents=value))


@pytest.mark.parametrize("value", ["", ",,", "0,invalid"])
def test_invalid_seed_lists_are_rejected(value):
    with pytest.raises(ValueError):
        runner.parse_seeds(SimpleNamespace(seeds=value))


@pytest.mark.parametrize(
    "options",
    [
        ["--workers", "0"],
        ["--max-steps", "0"],
        ["--obs-radius", "0"],
        ["--map-file", "tiny.map", "--map-list", "maps.yaml"],
    ],
)
def test_invalid_run_protocol_is_rejected_before_execution(monkeypatch, options):
    monkeypatch.setattr(sys, "argv", ["evaluate.py", *options])
    with patch.object(runner, "run_experiments") as run:
        with pytest.raises((ValueError, SystemExit)):
            runner.main()
    run.assert_not_called()


def test_retired_direct_cli_variants_are_rejected():
    with (
        patch.object(
            sys,
            "argv",
            [
                "evaluate.py",
                "--algorithms",
                "Direct",
                "--direct-transform",
                "clipped_relu",
            ],
        ),
        pytest.raises(SystemExit),
    ):
        runner.parse_args()


def _artifact():
    return SimpleNamespace(
        weights_path=Path("weights/arpe").resolve(),
        checkpoint_path=Path("weights/arpe/checkpoint_p0/model.pth").resolve(),
    )


def test_arpe_loads_exact_frozen_candidate():
    artifact = _artifact()
    with (
        patch.object(runner, "_load_arpe_candidate_artifact", return_value=artifact),
        patch("agents.arpe.ARPE") as load,
    ):
        runner.build_algorithm("ARPE", ".", 42, arpe_candidate_manifest="manifest.json")
    assert load.call_args.args[0] is artifact
    assert load.call_args.kwargs["seed"] == 42
    assert load.call_args.kwargs["device"] == "auto"


@pytest.mark.parametrize("collision", ["block_both", "soft"])
def test_default_aoreplan_keeps_static_occupancy_check(collision):
    agent = AORePlan(AORePlanConfig())
    agent.set_grid_config(SimpleNamespace(collision_system=collision))
    wrapper = object.__new__(AORePlanWrapper)
    wrapper.static_astar = SimpleNamespace(get_action=lambda _index, _observation: 4)
    wrapper.moves = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
    agents = np.zeros((11, 11), dtype=int)
    agents[5, 6] = 1
    obs = {"obstacles": np.zeros((11, 11)), "agents": agents}
    assert wrapper._static_astar_action(0, obs) == 0


def test_public_manifest_uses_portable_paths_without_required_hashes():
    root = Path(__file__).resolve().parents[1]
    data = json.loads((root / "configs/arpe.json").read_text())
    from agents.arpe import ARPEWeights

    artifact = ARPEWeights.from_mapping(data, root)
    assert artifact.weights_path == (root / data["weights_path"]).resolve()
    assert artifact.checkpoint_path == (root / data["checkpoint_path"]).resolve()
    for key in ("checkpoint_sha256", "base_checkpoint_sha256"):
        assert key not in data


def test_map_snapshot_returns_selected_grids_without_archival_hashes(tmp_path):
    import yaml

    registry = tmp_path / "maps.yaml"
    registry.write_text(yaml.safe_dump({"tiny": "...\n.#.\n..."}))
    names, grids = runner.load_map_list_snapshot(registry)
    assert names == {"tiny": "tiny"}
    assert grids == {"tiny": "...\n.#.\n..."}


@pytest.mark.parametrize(
    "payload",
    [{}, [], {"tiny": None}, {"tiny": ""}, {"": "..."}, {"tiny": "...\n.."}],
)
def test_map_lists_require_named_rectangular_grid_contents(tmp_path, payload):
    import yaml

    source = tmp_path / "maps.yaml"
    source.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError):
        runner.load_map_list_snapshot(source)


def test_custom_map_keeps_its_border(tmp_path):
    source = tmp_path / "tiny.map"
    source.write_text("#####\n#...#\n#####\n", encoding="utf-8")
    assert runner.load_map_text(source) == {
        "map_name": "tiny.map",
        "map_text": "#####\n#...#\n#####",
    }


def test_saved_metadata_keeps_protocol_without_weight_or_host_reports(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "evaluate.py",
            "--algorithms",
            "HRSS",
            "--agents",
            "4",
            "--seeds",
            "42",
            "--max-steps",
            "16",
            "--workers",
            "1",
            "--collision-system",
            "soft",
        ],
    )
    monkeypatch.setattr(runner, "run_experiments", lambda *_a, **_k: ([], 0.0))
    monkeypatch.setattr(
        runner, "save_results", lambda _rows, meta, *_a: captured.update(meta)
    )
    runner.main()
    assert captured["algorithms"] == ["HRSS"]
    assert captured["agent_counts"] == [4]
    assert captured["seeds"] == [42]
    assert captured["max_steps"] == 16
    assert captured["collision_system"] == "soft"
    assert captured["maps"] == runner.DEFAULT_MAPS
    assert len(captured["maps"]) == 5
    assert not (
        {"runtime_provenance", "switcher_weights_path", "main_dir", "map_list_sha256"}
        & captured.keys()
    )


@pytest.mark.parametrize("on_target", ["restart", "nothing"])
def test_progress_output_omits_retired_labels_without_changing_results(
    monkeypatch, capsys, on_target
):
    from concurrent.futures import ThreadPoolExecutor

    row = {
        "algorithm": "ARPE",
        "map_name": "test-map",
        "num_agents": 100,
        "seed": 0,
        "error": None,
        "on_target": on_target,
        "avg_throughput": 1.2,
        "ISR": 1.0,
        "CSR": 0.5,
        "ep_length": 512,
    }
    monkeypatch.setattr(runner, "ProcessPoolExecutor", ThreadPoolExecutor)
    monkeypatch.setattr(runner, "run_single_experiment", lambda _task: dict(row))
    results, _elapsed = runner.run_experiments([{}], workers=1)
    output = capsys.readouterr().out
    for label in (
        "caar=", "planner=", "caar_actions=", "guided=", "congestion=", "reverse=",
    ):
        assert label not in output
    assert (
        "throughput=1.2000" if on_target == "restart" else "isr=100.0% csr=50.0%"
    ) in output
    for key, value in row.items():
        assert results[0][key] == value


@pytest.mark.parametrize("on_target", ["restart", "finish"])
def test_single_episode_preserves_actions_seed_and_final_core_metrics(on_target):
    class MoveRightPolicy:
        def after_reset(self):
            self.actions = []
            self.done_masks = []

        def set_grid_config(self, config):
            self.grid_config = config

        def set_env(self, env):
            self.env = env

        def act(self, observations, rewards=None, dones=None, infos=None):
            self.actions.append([4] * len(observations))
            return self.actions[-1]

        def after_step(self, dones):
            self.done_masks.append(list(dones))

    policy = MoveRightPolicy()
    result = runner.run_algorithm(
        policy,
        map_name="single-step",
        max_episode_steps=1,
        seed=42,
        num_agents=1,
        obs_radius=1,
        animate=False,
        on_target=on_target,
        collision_system="block_both",
        map_text="\n".join(["......."] * 5),
        agents_xy=[[2, 1]],
        targets_xy=[[[2, 2], [2, 3]]] if on_target == "restart" else [[2, 2]],
    )
    assert policy.grid_config.seed == 42
    assert policy.actions == [[4]]
    assert policy.done_masks == [[True]]
    assert np.asarray(policy.env.grid.get_agents_xy(ignore_borders=True)).tolist() == [
        [2, 2]
    ]
    if on_target == "restart":
        assert result["avg_throughput"] == 1.0
    else:
        assert result["ISR"] == 1.0
        assert result["CSR"] == 1.0
        assert result["ep_length"] == 1
    assert not any(
        key.startswith(("policy_decision_", "vertex_flow_", "contention_"))
        or key in {"congestion_rate", "move_failure_count", "throughput_segments"}
        for key in result
    )


def test_failed_experiment_does_not_fabricate_measurements():
    task = {
        "algorithm": "AORePlan",
        "main_dir": ".",
        "seed": 42,
        "map_name": "test-map",
        "max_steps": 16,
        "num_agents": 4,
        "animate": False,
        "on_target": "restart",
        "collision_system": "block_both",
    }
    with patch.object(runner, "build_algorithm", side_effect=RuntimeError("load failed")):
        result = runner.run_single_experiment(task)
    assert result["error"] == "load failed"
    assert "RuntimeError: load failed" in result["traceback"]
    for key in ("algorithm", "map_name", "num_agents", "max_steps", "seed", "on_target"):
        assert result[key] == task[key]
    assert set(result) <= {
        "algorithm",
        "map_name",
        "num_agents",
        "max_steps",
        "seed",
        "on_target",
        "collision_system",
        "obs_radius",
        "error",
        "traceback",
    }


def test_saved_result_file_keeps_metadata_and_core_results(tmp_path):
    metadata = {"algorithms": ["AORePlan"], "seeds": [42]}
    rows = [{"algorithm": "AORePlan", "seed": 42, "avg_throughput": 1.0}]
    output = runner.save_results(rows, metadata, tmp_path, "results.json")
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "metadata": metadata,
        "results": rows,
    }
