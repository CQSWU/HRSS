import multiprocessing
from copy import deepcopy
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Extra, Field, validator

from pomapf_env.config import POMAPFConfig


def checkpoint_experiment_config(config):
    normalized = deepcopy(config)
    sections = {
        "experiment_settings": ExperimentSettings,
        "async_ppo": AsyncPPO,
        "environment": Environment,
        "global_settings": GlobalSettings,
        "evaluation": Evaluation,
    }
    for section, model in sections.items():
        if section in normalized:
            normalized[section] = {
                key: value
                for key, value in normalized[section].items()
                if key in model.__fields__
            }
    grid = normalized.get("environment", {}).get("grid_config", {})
    if grid.get("map_name") == "maps/train_capacity_n600.yaml":
        grid["map_name"] = "maps/train.yaml"
    return normalized


class AsyncPPO(BaseModel, extra=Extra.forbid):
    async_rl: bool = True

    experiment_summaries_interval: int = 20

    adam_eps: float = 1e-6

    adam_beta1: float = 0.9

    adam_beta2: float = 0.999

    gae_lambda: float = 0.95

    rollout: int = 32

    num_workers: int = multiprocessing.cpu_count()

    recurrence: int = 32

    use_rnn: bool = True

    rnn_type: str = "gru"

    rnn_num_layers: int = 1

    ppo_clip_ratio: float = 0.1

    ppo_clip_value: float = 1.0

    batch_size: int = 1024

    num_batches_per_iteration: int = 1

    num_epochs: int = 1

    max_grad_norm: float = 4.0

    exploration_loss_coeff: float = 0.003

    value_loss_coeff: float = 0.5

    kl_loss_coeff: float = 0.0

    exploration_loss: str = "entropy"

    num_envs_per_worker: int = 2

    worker_num_splits: int = 2

    num_policies: int = 1

    policy_workers_per_policy: int = 1

    max_policy_lag: int = 10000

    decorrelate_experience_max_seconds: int = 10

    decorrelate_envs_on_one_worker: bool = True

    with_vtrace: bool = True

    vtrace_rho: float = 1.0

    vtrace_c: float = 1.0

    set_workers_cpu_affinity: bool = True

    force_envs_single_thread: bool = True

    default_niceness: int = 0

    actor_worker_gpus: List[int] = Field(default_factory=list)


class ExperimentSettings(BaseModel, extra=Extra.forbid):
    save_every_sec: int = 120

    save_best_every_sec: int = 5

    save_best_after: int = 100000

    save_best_metric: str = "reward"

    keep_checkpoints: int = 1

    save_milestones_sec: int = -1

    stats_avg: int = 100

    learning_rate: float = 1e-4

    train_for_env_steps: int = 10_000_000_000

    train_for_seconds: int = 10_000_000_000

    obs_subtract_mean: float = 0.0

    obs_scale: float = 1.0

    normalize_input: bool = True

    normalize_input_keys: Optional[List[str]] = None

    gamma: float = 0.99

    reward_scale: float = 1.0

    reward_clip: float = 10.0

    encoder_custom: Optional[
        Literal[
            "pogema_residual",
            "epom_trace_context",
            "switcher",
        ]
    ] = None

    encoder_subtype: str = "resnet_impala"

    encoder_extra_fc_layers: int = 1

    encoder_mlp_layers: List[int] = Field(default_factory=lambda: [512, 512])

    decoder_mlp_layers: List[int] = Field(default_factory=list)

    pogema_encoder_num_filters: int = Field(64, ge=1)

    pogema_encoder_num_res_blocks: int = Field(3, ge=0)

    epom_base_weights_path: str = "weights/EPOM-L"

    hidden_size: int = 512

    nonlinearity: str = "relu"

    policy_initialization: str = "orthogonal"

    policy_init_gain: float = 1.0

    switcher_initial_ao_probability: float = Field(0.1, gt=0.0, lt=1.0)

    actor_critic_share_weights: bool = True

    adaptive_stddev: bool = True

    initial_stddev: float = 1.0

    lr_schedule: str = "kl_adaptive_minibatch"

    lr_schedule_kl_threshold: Optional[float] = None


class GlobalSettings(BaseModel, extra=Extra.forbid):
    algo: str = "APPO"

    env: Optional[str] = None

    experiment: Optional[str] = None

    train_dir: str = "weights/train_dir"

    device: str = "gpu"

    serial_mode: bool = False

    seed: Optional[int] = None

    cli_args: Dict[str, Any] = Field(default_factory=dict)

    with_wandb: bool = False


class Evaluation(BaseModel, extra=Extra.forbid):
    fps: int = 0

    no_render: bool = True

    policy_index: int = 0

    env_frameskip: int = Field(1, ge=1)


class Environment(BaseModel, extra=Extra.forbid):
    grid_config: POMAPFConfig = Field(default_factory=POMAPFConfig)

    training_num_agents_by_worker: Optional[List[int]] = Field(
        None,
        min_items=1,
    )

    name: Literal[
        "POMAPF-v0",
        "POMAPF-EPOM-ST-v0",
        "POMAPF-HRSS-v0",
    ] = "POMAPF-v0"

    tau_rho: float = Field(0.1, gt=0.0, le=1.0)

    tau_radius: Optional[int] = Field(None, ge=1)

    grid_memory_obs_radius: int = Field(7, ge=1)

    switcher_caar_device: str = "auto"

    switcher_max_planning_steps: int = Field(10_000, gt=0)

    switcher_team_reward_coefficient: float = 1.0

    def for_worker(self, env_config=None):
        if self.training_num_agents_by_worker is None:
            return self
        index = (
            env_config.get("worker_index", 0)
            if isinstance(env_config, dict)
            else getattr(env_config, "worker_index", 0)
        )
        try:
            index = int(index or 0)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "Sample Factory worker index must be an integer."
            ) from error
        if index < 0:
            raise ValueError("Sample Factory worker index must be non-negative.")
        grid = deepcopy(self.grid_config)
        grid.num_agents = self.training_num_agents_by_worker[
            index % len(self.training_num_agents_by_worker)
        ]
        return self.copy(update={"grid_config": grid})


class Experiment(BaseModel, extra=Extra.forbid):
    name: Optional[str] = None

    environment: Environment = Field(default_factory=Environment)

    async_ppo: AsyncPPO = Field(default_factory=AsyncPPO)

    experiment_settings: ExperimentSettings = Field(default_factory=ExperimentSettings)

    global_settings: GlobalSettings = Field(default_factory=GlobalSettings)

    evaluation: Evaluation = Field(default_factory=Evaluation)

    @validator("global_settings")
    def seed_initialization(cls, v, values):

        environment = values.get("environment")
        if v.env is None and environment is not None:
            v.env = environment.name

        if v.experiment is None:
            v.experiment = values["name"]

        return v
