# HRSS

**Switch and Reweight with Shared Trace for Lifelong Partially Observable
Multi-Agent Pathfinding.**

HRSS combines AORePlan and ARPE with a learned Switcher. When AORePlan
returns wait, ARPE is used directly; otherwise, the Switcher selects a branch.

| RePlan | AORePlan |
| :---: | :---: |
| <img src="docs/assets/replan.svg" alt="RePlan animation" width="420"> | <img src="docs/assets/aoreplan.svg" alt="AORePlan animation" width="420"> |

## Install

Linux, Python 3.10 or 3.11, and a C++ compiler:

```bash
git clone https://github.com/CQSWU/HRSS.git
cd HRSS
uv sync
```

## Run

```bash
uv run python evaluate.py --algorithms AORePlan
```

Download the [model weights](https://github.com/CQSWU/HRSS/releases/tag/weights-v1)
and extract them into `weights/`. ARPE needs `EPOM-L` and `ARPE-Final-1B`;
HRSS also needs `HRSS-Switcher-Final-1B`. Experiment data is not included.

```bash
uv run python evaluate.py --algorithms HRSS
```

## Train

```bash
uv run python train_arpe.py --config_path configs/train_arpe.yaml
uv run python train_switcher.py --config_path configs/train_switcher.yaml
```

ARPE trains with 200 agents. Switcher alternates between 50, 100 and 200 agents,
resuming the same model and optimizer between stages. Settings are in the YAML files.
Training and test maps are in `maps/train.yaml` and `maps/test.yaml`.

## License

[MIT](LICENSE). AORePlan builds on RePlan, and ARPE uses EPOM as its base policy;
the original authors' notices are preserved in the license.
