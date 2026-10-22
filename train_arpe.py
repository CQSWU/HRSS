import json
import os
import sys

from pathlib import Path

_LOCAL_PPU_SDK = Path("/usr/local/PPU_SDK")
if "PPU_SDK" not in os.environ and _LOCAL_PPU_SDK.is_dir():
    os.environ["PPU_SDK"] = str(_LOCAL_PPU_SDK)

import torch

import yaml

from sample_factory.algo.utils.context import global_env_registry

from sample_factory.cfg.arguments import default_cfg

from sample_factory.train import run_rl

from sample_factory.utils.utils import log

from learning.config import Environment, Experiment

from learning.ppo_losses import patch_switcher_learner_losses

import learning.encoder  # noqa: F401 -- registers the Sample Factory model factories

from pomapf_env.env import make_pomapf
from pomapf_env.rewards import FailedMoveCredit

from pomapf_env.wrappers import (
    GridMemoryObservationWrapper,
    MatrixObservationWrapper,
    TauObservationWrapper,
)

_CUSTOM_COMPONENTS_REGISTERED = False


def _patch_checkpoint_loading():

    from sample_factory.algo.learning.learner import Learner

    if getattr(Learner.load_checkpoint, "_arpe_checkpoint_patch", False):
        return

    def patched(checkpoints, device):

        if not checkpoints:
            log.warning("No checkpoints found")

            return None

        latest_checkpoint = checkpoints[-1]

        device_type = getattr(device, "type", str(device))

        load_device = torch.device("cpu") if device_type == "mps" else device

        last_error = None
        for attempt in range(3):
            try:
                log.warning("Loading state from checkpoint %s...", latest_checkpoint)

                return torch.load(
                    latest_checkpoint,
                    map_location=load_device,
                    weights_only=False,
                )

            except Exception as error:
                log.exception(
                    "Could not load from checkpoint, attempt %d of 3",
                    attempt + 1,
                )
                last_error = error

        raise RuntimeError(
            f"Could not load checkpoint after 3 attempts: {latest_checkpoint}"
        ) from last_error

    patched._arpe_checkpoint_patch = True

    Learner.load_checkpoint = staticmethod(patched)


_patch_checkpoint_loading()

patch_switcher_learner_losses()


def create_pogema_env(full_env_name, cfg=None, env_config=None, render_mode=None):
    del render_mode

    if full_env_name == "POMAPF-HRSS-v0":
        raise RuntimeError(
            "Switcher training uses its own ARPE path configuration. "
            "Use train_switcher.py for Switcher training, not generic train_arpe.py."
        )

    _ensure_patched()

    environment_config = Environment(**cfg.full_config["environment"]).for_worker(
        env_config
    )

    env = make_pomapf(grid_config=environment_config.grid_config)

    if full_env_name == "POMAPF-EPOM-ST-v0":
        environment = cfg.full_config["environment"]
        env = GridMemoryObservationWrapper(
            env,
            memory_radius=environment["grid_memory_obs_radius"],
        )

    env = MatrixObservationWrapper(env)

    if full_env_name == "POMAPF-EPOM-ST-v0":
        env = TauObservationWrapper(
            env,
            rho=environment_config.tau_rho,
            tau_radius=environment_config.tau_radius,
        )

        env = FailedMoveCredit(env)

    return env


def _ensure_patched():

    import sample_factory.algo.utils.make_env as make_env_mod

    if getattr(
        make_env_mod.get_multiagent_info,
        "_pomapf_multiagent_patch",
        False,
    ):
        return

    def patched(env):

        current = env

        while True:
            attrs = getattr(current, "__dict__", {})

            if "num_agents" in attrs and "is_multiagent" in attrs:
                return attrs["is_multiagent"], attrs["num_agents"]

            if hasattr(current, "env") and current.env is not current:
                current = current.env

            else:
                break

        return getattr(env, "is_multiagent", False), getattr(env, "num_agents", 1)

    patched._pomapf_multiagent_patch = True
    make_env_mod.get_multiagent_info = patched

    make_env_mod.is_multiagent_env = lambda env: patched(env)[0]


def register_custom_components():

    global _CUSTOM_COMPONENTS_REGISTERED

    patch_switcher_learner_losses()

    if _CUSTOM_COMPONENTS_REGISTERED:
        return

    for name in (
        "POMAPF-v0",
        "POMAPF-EPOM-ST-v0",
        "POMAPF-HRSS-v0",
    ):
        global_env_registry()[name] = create_pogema_env

    _ensure_patched()

    _patch_checkpoint_loading()

    _CUSTOM_COMPONENTS_REGISTERED = True


def validate_config(config):
    if not isinstance(config, dict):
        raise ValueError("Training config must be a mapping")
    exp = Experiment(**config)
    train_dir = Path(exp.global_settings.train_dir).expanduser()
    if not train_dir.is_absolute():
        train_dir = Path(__file__).resolve().parent / train_dir
    exp.global_settings.train_dir = str(train_dir.resolve())

    flat_config = default_cfg(
        algo=exp.global_settings.algo,
        env=exp.environment.name,
        experiment=exp.name or "",
    )

    for settings in (
        exp.async_ppo,
        exp.experiment_settings,
        exp.global_settings,
        exp.evaluation,
    ):
        for key, value in settings.dict().items():
            setattr(flat_config, key, value)

    flat_config.num_batches_per_epoch = exp.async_ppo.num_batches_per_iteration

    flat_config.rnn_size = exp.experiment_settings.hidden_size

    flat_config.encoder_conv_architecture = exp.experiment_settings.encoder_subtype

    flat_config.encoder_conv_mlp_layers = [exp.experiment_settings.hidden_size]

    flat_config.full_config = exp.dict()

    return exp, flat_config


def _sync_resume_cli_overrides(flat_config, override_keys):

    override_keys = set(override_keys)
    if not override_keys:
        return

    cli_args = dict(getattr(flat_config, "cli_args", {}) or {})
    for key in override_keys:
        cli_args[key] = getattr(flat_config, key)
    flat_config.cli_args = cli_args

    resume_config = Path(flat_config.train_dir) / flat_config.experiment / "config.json"
    if not resume_config.is_file():
        return

    saved = json.loads(resume_config.read_text(encoding="utf-8"))
    changed = False

    for key in override_keys:
        value = getattr(flat_config, key)
        if saved.get(key) != value:
            saved[key] = value
            changed = True

    full_config = saved.get("full_config")
    if isinstance(full_config, dict):
        nested_paths = {
            "experiment": (("name",), ("global_settings", "experiment")),
            "seed": (("global_settings", "seed"),),
            "train_dir": (("global_settings", "train_dir"),),
            "train_for_env_steps": (("experiment_settings", "train_for_env_steps"),),
        }
        nested_updates = [
            (path, getattr(flat_config, key))
            for key, paths in nested_paths.items()
            if key in override_keys
            for path in paths
        ]
        if "training_population" in override_keys:
            nested_updates.append(
                (
                    ("environment", "training_num_agents_by_worker"),
                    list(
                        flat_config.full_config["environment"][
                            "training_num_agents_by_worker"
                        ]
                    ),
                )
            )

        for path, value in nested_updates:
            target = full_config
            for key in path[:-1]:
                child = target.get(key)
                if not isinstance(child, dict):
                    child = {}
                    target[key] = child
                target = child
            if target.get(path[-1]) != value:
                target[path[-1]] = value
                changed = True

    if changed:
        resume_config.write_text(
            json.dumps(saved, indent=2),
            encoding="utf-8",
        )
        log.info("Updated resume overrides in %s", resume_config)


def main():

    import argparse

    parser = argparse.ArgumentParser(description="Process training config.")

    parser.add_argument(
        "--config_path",
        type=str,
        action="store",
        help="path to yaml file with single run configuration",
        required=True,
    )

    parser.add_argument("--run_name", type=str)

    parser.add_argument("--seed", type=int)

    parser.add_argument("--train_dir", type=str)

    parser.add_argument("--train_for_env_steps", type=int)

    parser.add_argument("--training_population", type=int)

    params = parser.parse_args()

    register_custom_components()

    with open(params.config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    explicit_overrides = set()

    if params.run_name is not None:
        config["name"] = params.run_name
        config.setdefault("global_settings", {})["experiment"] = params.run_name
        explicit_overrides.add("experiment")
    if params.seed is not None:
        config.setdefault("global_settings", {})["seed"] = params.seed
        explicit_overrides.add("seed")
    if params.train_dir is not None:
        config.setdefault("global_settings", {})["train_dir"] = params.train_dir
        explicit_overrides.add("train_dir")
    if params.train_for_env_steps is not None:
        config.setdefault("experiment_settings", {})["train_for_env_steps"] = (
            params.train_for_env_steps
        )
        explicit_overrides.add("train_for_env_steps")

    if params.training_population is not None:
        if params.training_population < 1:
            raise ValueError("--training_population must be positive.")
        worker_count = int(config["async_ppo"]["num_workers"])
        config.setdefault("environment", {})["training_num_agents_by_worker"] = [
            params.training_population
        ] * worker_count

    _, flat_config = validate_config(config)

    if params.training_population is not None:
        flat_config.training_population = params.training_population
        explicit_overrides.add("training_population")

    _sync_resume_cli_overrides(flat_config, explicit_overrides)

    status = run_rl(flat_config)

    return status


if __name__ == "__main__":
    sys.exit(main())
