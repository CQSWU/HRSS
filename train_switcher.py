from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import yaml
from sample_factory.algo.utils.context import global_env_registry
from sample_factory.train import run_rl

import train_arpe as base_train
from agents.arpe import ARPEWeights
from learning.config import Environment
from pomapf_env.switcher import SwitcherEnv

PROJECT_ROOT = Path(__file__).resolve().parent

ENV_NAME = "POMAPF-HRSS-v0"


def create_switcher_env(
    full_env_name,
    cfg=None,
    env_config=None,
    render_mode=None,
):
    del render_mode
    if full_env_name != ENV_NAME:
        raise ValueError(f"Switcher entrypoint cannot construct {full_env_name!r}.")
    environment = Environment(**cfg.full_config["environment"]).for_worker(env_config)
    declaration = cfg.full_config.get("candidate_policy")
    if not isinstance(declaration, dict):
        raise RuntimeError("Saved Switcher config has no candidate_policy paths.")
    artifact = ARPEWeights.from_mapping(declaration, PROJECT_ROOT)
    return SwitcherEnv(
        grid_config=environment.grid_config,
        candidate_artifact=artifact,
        candidate_device=environment.switcher_caar_device,
        max_planning_steps=environment.switcher_max_planning_steps,
        team_reward_coefficient=environment.switcher_team_reward_coefficient,
    )


def register_switcher_components() -> None:
    base_train.register_custom_components()
    global_env_registry()[ENV_NAME] = create_switcher_env


def prepare_switcher_config(config: dict):
    payload = deepcopy(config)
    curriculum = payload.pop("population_curriculum", None)
    declaration = payload.pop("candidate_policy", None)
    if not isinstance(declaration, dict):
        raise ValueError("Switcher config requires candidate_policy.")
    artifact = ARPEWeights.from_mapping(declaration, PROJECT_ROOT)

    _, flat_config = base_train.validate_config(payload)
    if flat_config.encoder_custom != "switcher":
        raise ValueError("Switcher requires encoder_custom='switcher'.")
    if flat_config.env != ENV_NAME:
        raise ValueError(f"Switcher requires environment {ENV_NAME!r}.")
    if bool(flat_config.use_rnn):
        raise ValueError("Switcher must remain feed-forward.")
    flat_config.full_config["candidate_policy"] = artifact.as_dict()
    flat_config.candidate_policy = artifact.as_dict()
    flat_config.population_curriculum = curriculum
    return flat_config


def population_stages(curriculum, total_steps):
    populations = curriculum["populations"]
    cycles = curriculum["cycles"]
    if not populations or any(type(n) is not int or n < 1 for n in populations):
        raise ValueError("Curriculum populations must be positive integers.")
    if type(cycles) is not int or cycles < 1 or total_steps < 1:
        raise ValueError("Curriculum cycles and total steps must be positive.")
    count = cycles * len(populations)
    return [
        (
            populations[index % len(populations)],
            (total_steps * (index + 1) + count - 1) // count,
        )
        for index in range(count)
    ]


def checkpoint_steps(run_dir):
    return max(
        (
            int(path.stem.rsplit("_", 1)[-1])
            for path in (run_dir / "checkpoint_p0").glob("checkpoint_*_*.pth")
        ),
        default=0,
    )


def run_curriculum(config_path, flat_config):
    run_dir = Path(flat_config.train_dir) / flat_config.experiment
    stages = population_stages(
        flat_config.population_curriculum, int(flat_config.train_for_env_steps)
    )
    for population, target in stages:
        if checkpoint_steps(run_dir) >= target:
            continue
        print(
            f"Switcher: {population} agents, cumulative target {target} steps",
            flush=True,
        )
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--config_path",
                str(config_path),
                "--run_name",
                flat_config.experiment,
                "--train_dir",
                str(flat_config.train_dir),
                "--train_for_env_steps",
                str(target),
                "--training_population",
                str(population),
            ]
        )
        if result.returncode:
            return result.returncode
        if checkpoint_steps(run_dir) < target:
            raise RuntimeError(
                "Training stopped before the stage target; rerun to resume."
            )
    return 0


def _apply_overrides(config: dict, args) -> set[str]:
    explicit: set[str] = set()
    if args.run_name is not None:
        config["name"] = args.run_name
        config.setdefault("global_settings", {})["experiment"] = args.run_name
        explicit.add("experiment")
    if args.train_dir is not None:
        config.setdefault("global_settings", {})["train_dir"] = args.train_dir
        explicit.add("train_dir")
    if args.train_for_env_steps is not None:
        config.setdefault("experiment_settings", {})["train_for_env_steps"] = int(
            args.train_for_env_steps
        )
        explicit.add("train_for_env_steps")
    return explicit


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Train Switcher")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--run_name")
    parser.add_argument("--train_dir")
    parser.add_argument("--train_for_env_steps", type=int)
    parser.add_argument("--training_population", type=int)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    register_switcher_components()
    config_path = Path(args.config_path).resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    explicit = _apply_overrides(config, args)
    if args.training_population is not None:
        if args.training_population < 1:
            raise ValueError("--training_population must be positive.")
        environment = config.setdefault("environment", {})
        environment.setdefault("grid_config", {})["num_agents"] = (
            args.training_population
        )
        environment["training_num_agents_by_worker"] = [args.training_population] * int(
            config["async_ppo"]["num_workers"]
        )
    flat_config = prepare_switcher_config(config)
    flat_config.use_env_info_cache = False
    flat_config.restart_behavior = "resume"
    if args.validate_only:
        stages = (
            population_stages(
                flat_config.population_curriculum, int(flat_config.train_for_env_steps)
            )
            if flat_config.population_curriculum and args.training_population is None
            else [
                (
                    flat_config.full_config["environment"]["grid_config"]["num_agents"],
                    int(flat_config.train_for_env_steps),
                )
            ]
        )
        print(
            json.dumps(
                {
                    "validated": True,
                    "experiment": flat_config.experiment,
                    "target_frames": int(flat_config.train_for_env_steps),
                    "workers": int(flat_config.num_workers),
                    "candidate_policy": flat_config.candidate_policy,
                    "population_stages": stages,
                },
                sort_keys=True,
            )
        )
        return 0
    if flat_config.population_curriculum and args.training_population is None:
        return run_curriculum(config_path, flat_config)
    explicit.update({"use_env_info_cache", "restart_behavior"})
    if args.training_population is not None:
        flat_config.training_population = args.training_population
        explicit.update({"training_population", "full_config", "population_curriculum"})
    base_train._sync_resume_cli_overrides(flat_config, explicit)
    return int(run_rl(flat_config))


if __name__ == "__main__":
    raise SystemExit(main())
