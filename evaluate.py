import argparse
import json
import logging
import multiprocessing
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import suppress
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

from pogema.svg_animation.animation_wrapper import (
    AnimationConfig,
    AnimationMonitor,
)
from pomapf_env.env import make_pomapf
from pomapf_env.config import POMAPFConfig

DEFAULT_MAPS = {
    "mazes": "mazes-s0_wc8_od55",
    "random": "random-s0_d0.15",
    "sc1": "sc1-AcrosstheCape",
    "street": "street-Berlin_0",
    "wc3": "wc3-Battleground",
}

SUPPORTED_ALGORITHMS = ("AORePlan", "ARPE", "HRSS")
DEFAULT_ALGORITHMS = ("AORePlan",)
ALGORITHM_ALIASES = {name.lower(): name for name in SUPPORTED_ALGORITHMS}

def canonical_algorithm_name(value):

    return ALGORITHM_ALIASES.get(value.strip().lower())


def quiet_model_logs():

    logging.getLogger("rl").setLevel(logging.ERROR)

    with suppress(Exception):
        from sample_factory.utils.utils import log

        log.setLevel(logging.ERROR)


def _project_path(main_dir, value):
    path = Path(value)
    if not path.is_absolute():
        path = Path(main_dir) / path
    return path.resolve()


def _load_arpe_candidate_artifact(main_dir, manifest_path):

    if manifest_path is None:
        manifest_path = str(
            Path(main_dir).resolve() / "configs" / "arpe.json"
        )
    path = _project_path(main_dir, manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"ARPE candidate manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ARPE candidate manifest must be a JSON object.")
    from agents.arpe import ARPEWeights

    return ARPEWeights.from_mapping(
        payload,
        Path(main_dir).resolve(),
    )


def build_algorithm(
    algo_name,
    main_dir,
    seed,
    arpe_candidate_manifest=None,
    switcher_weights_path=None,
):

    algo_name = canonical_algorithm_name(algo_name) or algo_name
    if algo_name not in SUPPORTED_ALGORITHMS:
        raise ValueError(f"Unsupported public algorithm: {algo_name}")

    if algo_name == "AORePlan":
        from agents.aoreplan import AORePlan, AORePlanConfig

        return AORePlan(AORePlanConfig(seed=seed))

    if algo_name == "HRSS":
        from agents.hrss import HRSS, HRSSConfig
        from agents.switcher import SwitcherConfig

        artifact = _load_arpe_candidate_artifact(main_dir, arpe_candidate_manifest)

        return HRSS(
            HRSSConfig(
                switcher=SwitcherConfig(
                    path_to_weights=str(
                        _project_path(
                            main_dir,
                            switcher_weights_path or "weights/HRSS-Switcher-Final-1B",
                        )
                    ),
                    device="auto",
                ),
                seed=seed,
            ),
            candidate=artifact,
        )

    if algo_name == "ARPE":
        from agents.arpe import ARPE

        artifact = _load_arpe_candidate_artifact(main_dir, arpe_candidate_manifest)
        return ARPE(
            artifact,
            seed=int(seed),
            device="auto",
        )


def run_algorithm(
    algo,
    *,
    map_name,
    max_episode_steps,
    seed,
    num_agents,
    obs_radius,
    animate,
    on_target,
    collision_system,
    map_text,
    agents_xy=None,
    targets_xy=None,
):

    gc_kwargs = {
        "max_episode_steps": max_episode_steps,
        "seed": seed,
        "num_agents": num_agents,
        "on_target": on_target,
    }
    if obs_radius is not None:
        gc_kwargs["obs_radius"] = obs_radius
    if collision_system is not None:
        gc_kwargs["collision_system"] = collision_system
    if map_text is not None:
        gc_kwargs["map"] = map_text
        gc_kwargs["map_name"] = None
    else:
        gc_kwargs["map_name"] = map_name

    if (agents_xy is None) != (targets_xy is None):
        raise ValueError("Explicit placements require both agents_xy and targets_xy")
    if agents_xy is not None:
        if len(agents_xy) != num_agents or len(targets_xy) != num_agents:
            raise ValueError("Explicit placement count must equal num_agents")
        gc_kwargs["agents_xy"] = [list(position) for position in agents_xy]
        gc_kwargs["targets_xy"] = [list(position) for position in targets_xy]

    grid_config = POMAPFConfig(**gc_kwargs)

    env = make_pomapf(
        grid_config=grid_config,
        with_animations=False,
        auto_reset=False,
    )
    if animate:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        map_short = map_name.split("-")[-1] if "-" in map_name else map_name
        directory = Path("renders") / (
            f"{map_short}_{type(algo).__name__}_{num_agents}agents_{timestamp}"
        )
        env = AnimationMonitor(
            env,
            AnimationConfig(directory=str(directory)),
        )

    try:
        observations, _ = env.reset()
        algo.after_reset()
        if hasattr(algo, "set_grid_config"):
            algo.set_grid_config(env.grid_config)
        if hasattr(algo, "set_env"):
            algo.set_env(env)
        results = {}
        dones = [False for _ in observations]
        infos = [{"is_active": True} for _ in observations]
        rewards = [0 for _ in observations]
        with torch.no_grad():
            while not all(dones):
                actions = algo.act(observations, rewards, dones, infos)
                observations, rewards, terminated, truncated, infos = env.step(actions)
                dones = [done or limit for done, limit in zip(terminated, truncated)]
                results.update(infos[0].get("metrics", {}))
                algo.after_step(dones)
        results["algorithm"] = type(algo).__name__
        return results
    finally:
        env.close()


def run_single_experiment(task):
    quiet_model_logs()
    seed = task["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    record = {
        "algorithm": canonical_algorithm_name(task["algorithm"]) or task["algorithm"],
        "map_name": task["map_name"],
        "num_agents": task["num_agents"],
        "max_steps": task["max_steps"],
        "seed": seed,
        "on_target": task.get("on_target", "restart"),
    }
    try:
        algo = build_algorithm(
            record["algorithm"], task["main_dir"], seed,
            arpe_candidate_manifest=task.get("arpe_candidate_manifest"),
            switcher_weights_path=task.get("switcher_weights_path"),
        )
        result = run_algorithm(
            algo, map_name=task["map_name"], max_episode_steps=task["max_steps"],
            seed=seed, num_agents=task["num_agents"], obs_radius=task.get("obs_radius"),
            animate=task["animate"], on_target=record["on_target"],
            collision_system=task.get("collision_system"), map_text=task.get("map_text"),
            agents_xy=task.get("agents_xy"), targets_xy=task.get("targets_xy"),
        )
        record.update({key: value for key, value in result.items() if key != "algorithm"})
    except Exception as error:
        import traceback
        record.update(error=str(error), traceback=traceback.format_exc())
    return record


def parse_algorithms(value):

    if value.strip().lower() == "all":
        return list(DEFAULT_ALGORITHMS)

    raw_algorithms = [item.strip() for item in value.split(",") if item.strip()]

    algorithms = []

    seen = set()

    unknown = []

    for item in raw_algorithms:
        canonical = canonical_algorithm_name(item)

        if canonical is None:
            unknown.append(item)

        else:
            if canonical not in seen:
                algorithms.append(canonical)

                seen.add(canonical)

    if unknown:
        choices = ", ".join(SUPPORTED_ALGORITHMS)

        raise argparse.ArgumentTypeError(
            f"Unknown algorithm(s): {unknown}. Choices: {choices}"
        )

    if not algorithms:
        raise argparse.ArgumentTypeError("No algorithms selected")

    return algorithms


def _integers(value):
    numbers = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not numbers:
        raise ValueError("At least one value is required")
    return numbers


def parse_agent_counts(args):
    counts = _integers(args.agents)
    if counts[0] <= 0:
        raise ValueError("Agent counts must be positive")
    return counts


def parse_seeds(args):
    return _integers(args.seeds)


def _looks_like_movingai_map(lines):

    if not lines:
        return False

    head = [line.strip().lower() for line in lines[:4]]

    return "map" in head and any(line.startswith("type ") for line in head)


def _translate_map_rows(rows):

    trans = {".": ".", "G": ".", "S": ".", "W": "#", "T": "#", "@": "#", "O": "#"}

    return [
        "".join(trans.get(ch, "#") for ch in row.rstrip())
        for row in rows
        if row.strip()
    ]


def load_map_text(path):

    path = Path(path)

    raw_text = path.read_text()

    lines = raw_text.splitlines()

    if _looks_like_movingai_map(lines):
        map_start = (
            next(i for i, line in enumerate(lines) if line.strip().lower() == "map") + 1
        )

        rows = _translate_map_rows(lines[map_start:])

    else:
        rows = [line.rstrip() for line in lines if line.strip()]

    if not rows:
        raise ValueError(f"Map source produced no rows: {path}")

    width = len(rows[0])

    if width == 0 or any(len(row) != width for row in rows):
        raise ValueError(
            f"Map source must contain a non-empty rectangular grid: {path}"
        )

    return {
        "map_name": path.name or "custom-map",
        "map_text": "\n".join(rows),
    }


def load_map_list_snapshot(path):
    maps = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(maps, dict) or not maps:
        raise ValueError("--map-list requires a non-empty YAML mapping")
    for name, text in maps.items():
        if not isinstance(name, str) or not name or not isinstance(text, str) or not text.strip():
            raise ValueError("Each map requires a name and grid text")
        rows = text.splitlines()
        if not rows[0] or any(len(row) != len(rows[0]) or set(row) - {".", "#"} for row in rows):
            raise ValueError(f"Map {name!r} must be a rectangular .# grid")
    return {name: name for name in maps}, maps


def build_tasks(
    algorithms,
    maps,
    agent_counts,
    seeds,
    args,
    custom_map=None,
    map_texts=None,
):

    map_items = (
        [
            (
                custom_map["map_name"],
                custom_map["map_text"],
            )
        ]
        if custom_map is not None
        else [
            (
                map_name,
                map_texts.get(map_name) if map_texts is not None else None,
            )
            for map_name in maps.values()
        ]
    )

    return [
        {
            "algorithm": algorithm,
            "map_name": map_name,
            "map_text": map_text,
            "num_agents": num_agents,
            "obs_radius": args.obs_radius,
            "max_steps": args.max_steps,
            "seed": seed,
            "animate": args.animate,
            "main_dir": args.main_dir,
            "on_target": args.on_target,
            "collision_system": args.collision_system,
            "arpe_candidate_manifest": args.arpe_candidate_manifest,
            "switcher_weights_path": args.switcher_weights_path,
        }
        for algorithm in algorithms
        for map_name, map_text in map_items
        for num_agents in agent_counts
        for seed in seeds
    ]


def format_duration(seconds):

    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes, rem = divmod(seconds, 60)

    if minutes < 60:
        return f"{int(minutes)}m{int(rem):02d}s"

    hours, minutes = divmod(minutes, 60)

    return f"{int(hours)}h{int(minutes):02d}m"


def run_experiments(tasks, workers):
    results = []
    start = time.monotonic()
    print(f"Experiments: {len(tasks)} | Workers: {workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(run_single_experiment, task) for task in tasks]
        for index, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if result.get("error"):
                status = f"ERROR: {result['error']}"
            elif result.get("on_target", "restart") == "restart":
                status = f"throughput={result['avg_throughput']:.4f}"
            else:
                status = f"isr={result['ISR']:.1%} csr={result['CSR']:.1%}"
            eta = (time.monotonic() - start) / index * (len(tasks) - index)
            print(
                f"[{index}/{len(tasks)}] {result['algorithm']} | {result['map_name']} | "
                f"{result['num_agents']} agents | {status} | ETA {format_duration(eta)}",
                flush=True,
            )
    return results, time.monotonic() - start


def save_results(results, metadata, output_dir, filename=None):

    os.makedirs(output_dir, exist_ok=True)

    if filename:
        output_path = Path(output_dir) / filename

    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_path = Path(output_dir) / f"experiments_{timestamp}.json"

    payload = {
        "metadata": metadata,
        "results": results,
    }

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")

    with temporary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

        f.flush()

        os.fsync(f.fileno())

    temporary_path.replace(output_path)

    print(f"\nResults saved: {output_path}")

    return output_path


def parse_args():
    parser = argparse.ArgumentParser(description="Lifelong MAPF evaluation", allow_abbrev=False)
    parser.add_argument("--algorithms", type=parse_algorithms, default=list(DEFAULT_ALGORITHMS))
    parser.add_argument("--agents", default=",".join(str(n) for n in range(50, 501, 50)))
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--obs-radius", type=int, default=None)
    parser.add_argument("--animate", action="store_true")
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--main-dir", default="./")
    maps = parser.add_mutually_exclusive_group()
    maps.add_argument("--map-file", help="Custom text or MovingAI map")
    maps.add_argument("--map-list", help="YAML map collection, e.g. maps/test.yaml")
    parser.add_argument("--on-target", choices=("restart", "finish", "nothing"), default="restart")
    parser.add_argument("--collision-system", choices=("soft", "block_both", "priority"), default="block_both")
    parser.add_argument("--arpe-candidate-manifest", default=None)
    parser.add_argument("--switcher-weights-path", default=None)
    parser.add_argument("--output-dir", default="exp_result")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def main():
    multiprocessing.set_start_method("spawn", force=True)
    quiet_model_logs()
    args = parse_args()
    if args.workers < 1 or args.max_steps < 1:
        raise ValueError("Workers and max-steps must be positive")
    if args.obs_radius is not None and args.obs_radius < 1:
        raise ValueError("Observation radius must be positive")
    counts, seeds = parse_agent_counts(args), parse_seeds(args)
    custom_map, map_texts = None, None
    if args.map_file:
        custom_map = load_map_text(_project_path(args.main_dir, args.map_file))
        maps = {"custom": custom_map["map_name"]}
    elif args.map_list:
        maps, map_texts = load_map_list_snapshot(_project_path(args.main_dir, args.map_list))
    else:
        maps = dict(DEFAULT_MAPS)
    tasks = build_tasks(args.algorithms, maps, counts, seeds, args, custom_map, map_texts)
    metadata = dict(
        algorithms=args.algorithms, agent_counts=counts, seeds=seeds, maps=maps,
        workers=args.workers, obs_radius=args.obs_radius, max_steps=args.max_steps,
        on_target=args.on_target, collision_system=args.collision_system,
    )
    results, elapsed = run_experiments(tasks, args.workers)
    save_results(results, metadata, args.output_dir, args.output)
    print(f"Total elapsed: {format_duration(elapsed)}")
    failures = [row for row in results if row.get("error")]
    if failures:
        raise RuntimeError(f"{len(failures)} experiments failed; first error: {failures[0]['error']}")


if __name__ == "__main__":
    main()
