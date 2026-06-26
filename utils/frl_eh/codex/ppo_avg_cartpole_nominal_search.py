from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "hyperparams" / "ppo_avg.yml"
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "rl_zoo3" / "train.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_cartpole_nominal_search"
TARGET_ENV_ID = "PerturbCartPole-v1"


@dataclass(frozen=True)
class CartPoleBaseConfig:
    env_id: str
    n_timesteps: int
    num_clients: int
    server_update_weight: float
    n_envs: int
    n_steps: int
    batch_size: int
    local_steps: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    learning_rate_expr: str
    clip_range_expr: str

    @property
    def rollout_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def local_rollouts(self) -> int:
        return max(1, math.ceil(self.local_steps / max(self.rollout_size, 1)))


@dataclass(frozen=True)
class TrialSpec:
    env_id: str
    n_timesteps: int
    num_clients: int
    server_update_weight: float
    n_envs: int
    n_steps: int
    batch_size: int
    local_rollouts: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    learning_rate_expr: str
    clip_range_expr: str
    is_anchor: bool

    @property
    def rollout_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def local_steps(self) -> int:
        return self.rollout_size * self.local_rollouts

    @property
    def approx_minibatches(self) -> int:
        minibatches_per_rollout = max(1, math.ceil(self.rollout_size / max(self.batch_size, 1)))
        return self.local_rollouts * self.n_epochs * minibatches_per_rollout

    @property
    def compute_sort_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.local_steps,
            self.approx_minibatches,
            self.rollout_size,
            self.n_envs,
            self.batch_size,
        )

    @property
    def label(self) -> str:
        prefix = "anchor" if self.is_anchor else "search"
        return (
            f"{prefix}"
            f"_tm{self.n_timesteps}"
            f"_nc{self.num_clients}"
            f"_sw{compact_number(self.server_update_weight)}"
            f"_envs{self.n_envs}"
            f"_steps{self.n_steps}"
            f"_localr{self.local_rollouts}"
            f"_batch{self.batch_size}"
            f"_epochs{self.n_epochs}"
            f"_gamma{compact_number(self.gamma)}"
            f"_gae{compact_number(self.gae_lambda)}"
            f"_lr{sanitize_expr(self.learning_rate_expr)}"
            f"_clip{sanitize_expr(self.clip_range_expr)}"
        )


@dataclass
class TrialResult:
    env_id: str
    label: str
    is_anchor: bool
    status: str
    returncode: int
    duration_sec: float
    n_timesteps: int
    num_clients: int
    server_update_weight: float
    n_envs: int
    n_steps: int
    batch_size: int
    local_rollouts: int
    local_steps: int
    rollout_size: int
    approx_minibatches: int
    n_epochs: int
    gamma: float
    gae_lambda: float
    learning_rate_expr: str
    clip_range_expr: str
    final_local_mean: float | None
    final_local_min: float | None
    final_local_max: float | None
    final_nominal_mean: float | None
    final_nominal_min: float | None
    ranking_score: float | None
    solved: bool
    run_dir: str | None
    log_path: str
    error: str | None = None

    @property
    def compute_sort_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.local_steps,
            self.approx_minibatches,
            self.rollout_size,
            self.n_envs,
            self.batch_size,
        )

    @property
    def is_success(self) -> bool:
        return self.status == "ok" and self.ranking_score is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search for CartPole hyperparameters that let PPOAvg learn under FRL "
            "when every local client uses the same nominal environment."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to ppo_avg.yml.")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT, help="Path to rl_zoo3/train.py.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for sweep logs.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed used for candidate sampling and training.")
    parser.add_argument("--max-workers", type=int, default=16, help="Maximum number of parallel training jobs.")
    parser.add_argument(
        "--num-threads-per-task",
        type=int,
        default=1,
        help="Value forwarded to train.py --num-threads for each worker.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device forwarded to train.py.")
    parser.add_argument("--train-verbose", type=int, default=0, help="Verbose level forwarded to train.py.")
    parser.add_argument(
        "--timesteps-multipliers",
        nargs="*",
        type=float,
        default=[1.0, 2.0],
        help="Multipliers applied to the CartPole n_timesteps from ppo_avg.yml.",
    )
    parser.add_argument(
        "--num-client-values",
        nargs="*",
        type=int,
        default=None,
        help="Optional num_clients values to search. Default: keep the YAML value only.",
    )
    parser.add_argument(
        "--server-update-weights",
        nargs="*",
        type=float,
        default=[1.0, 0.5],
        help="server_update_weight values to search.",
    )
    parser.add_argument(
        "--n-env-values",
        nargs="*",
        type=int,
        default=[1, 2, 4, 8],
        help="n_envs values to search.",
    )
    parser.add_argument(
        "--n-step-values",
        nargs="*",
        type=int,
        default=[4, 8, 16, 32, 64],
        help="n_steps values to search.",
    )
    parser.add_argument(
        "--local-rollout-values",
        nargs="*",
        type=int,
        default=[1, 2, 4, 8, 16],
        help="Number of PPO rollout/update cycles per local FRL round.",
    )
    parser.add_argument(
        "--n-epoch-values",
        nargs="*",
        type=int,
        default=[4, 8, 10, 16, 20],
        help="PPO n_epochs values to search.",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="*",
        type=float,
        default=[0.98, 0.99, 0.995],
        help="Gamma values to search.",
    )
    parser.add_argument(
        "--gae-lambda-values",
        nargs="*",
        type=float,
        default=[0.8, 0.9, 0.95, 0.98],
        help="GAE lambda values to search.",
    )
    parser.add_argument(
        "--learning-rate-values",
        nargs="*",
        type=str,
        default=["3e-4", "1e-3", "'lin_0.001'"],
        help=(
            "Learning-rate expressions forwarded through StoreDict. "
            "Examples: 3e-4 1e-3 '\"lin_0.001\"'"
        ),
    )
    parser.add_argument(
        "--clip-range-values",
        nargs="*",
        type=str,
        default=["0.1", "0.2", "'lin_0.2'"],
        help=(
            "clip_range expressions forwarded through StoreDict. "
            "Examples: 0.1 0.2 '\"lin_0.2\"'"
        ),
    )
    parser.add_argument("--min-batch-size", type=int, default=4, help="Minimum batch_size candidate.")
    parser.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch_size candidate.")
    parser.add_argument(
        "--max-random-trials",
        type=int,
        default=160,
        help="Maximum number of sampled random search trials (anchors are always added separately).",
    )
    parser.add_argument(
        "--eval-local-episodes",
        type=int,
        default=10,
        help="Final FRL local-eval episodes per client.",
    )
    parser.add_argument(
        "--eval-nominal-episodes",
        type=int,
        default=10,
        help="Final FRL nominal-eval episodes per client.",
    )
    parser.add_argument(
        "--eval-round-freq",
        type=int,
        default=10**9,
        help="Intermediate FRL eval frequency. Use a huge value to rely on final eval only.",
    )
    parser.add_argument(
        "--ranking-metric",
        choices=("local_mean", "local_min", "nominal_mean", "mean_and_min"),
        default="mean_and_min",
        help="Metric used to rank CartPole trials.",
    )
    parser.add_argument(
        "--success-mean-threshold",
        type=float,
        default=475.0,
        help="Mean local return required to mark a trial as solved.",
    )
    parser.add_argument(
        "--success-min-threshold",
        type=float,
        default=450.0,
        help="Minimum per-client local return required to mark a trial as solved.",
    )
    parser.add_argument(
        "--near-best-fraction",
        type=float,
        default=0.95,
        help="Compute-light recommendation uses the smallest trial whose score reaches this fraction of the best score.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top trials to keep in the markdown summary.")
    parser.add_argument("--dry-run", action="store_true", default=False, help="Only materialize the trial plan.")
    return parser.parse_args()


def compact_number(value: float) -> str:
    text = f"{value:.6g}"
    return text.replace("-", "m").replace(".", "p")


def sanitize_expr(expr: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in expr).strip("_")


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def coerce_int(value: Any, key: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer-like value for {key}, got {value!r}") from exc


def load_cartpole_base_config(config_path: Path) -> CartPoleBaseConfig:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    if TARGET_ENV_ID not in raw:
        raise KeyError(f"{TARGET_ENV_ID} not found in {config_path}")
    values = raw[TARGET_ENV_ID]
    if not isinstance(values, dict):
        raise ValueError(f"{TARGET_ENV_ID} config must be a mapping.")

    learning_rate_value = values.get("learning_rate", 3e-4)
    clip_range_value = values.get("clip_range", 0.2)
    return CartPoleBaseConfig(
        env_id=TARGET_ENV_ID,
        n_timesteps=coerce_int(values["n_timesteps"], "n_timesteps"),
        num_clients=coerce_int(values.get("num_clients", 5), "num_clients"),
        server_update_weight=float(values.get("server_update_weight", 1.0)),
        n_envs=coerce_int(values["n_envs"], "n_envs"),
        n_steps=coerce_int(values["n_steps"], "n_steps"),
        batch_size=coerce_int(values["batch_size"], "batch_size"),
        local_steps=coerce_int(values["local_steps"], "local_steps"),
        n_epochs=coerce_int(values["n_epochs"], "n_epochs"),
        gamma=float(values.get("gamma", 0.99)),
        gae_lambda=float(values.get("gae_lambda", 0.95)),
        learning_rate_expr=repr(learning_rate_value) if isinstance(learning_rate_value, str) else str(learning_rate_value),
        clip_range_expr=repr(clip_range_value) if isinstance(clip_range_value, str) else str(clip_range_value),
    )


def batch_candidates(rollout_size: int, min_batch_size: int, max_batch_size: int) -> list[int]:
    min_batch_size = max(2, min_batch_size)
    max_batch_size = max(min_batch_size, max_batch_size)
    values: set[int] = set()
    candidate = min(rollout_size, max_batch_size)
    while candidate >= min_batch_size:
        if rollout_size % candidate == 0:
            values.add(candidate)
        candidate //= 2
    if rollout_size <= max_batch_size:
        values.add(rollout_size)
    return sorted(value for value in values if min_batch_size <= value <= rollout_size)


def candidate_pairs(n_env_values: list[int], n_step_values: list[int]) -> list[tuple[int, int]]:
    pairs = [(n_envs, n_steps) for n_envs in n_env_values for n_steps in n_step_values if n_envs * n_steps > 1]
    pairs.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return pairs


def add_unique_spec(store: dict[tuple[Any, ...], TrialSpec], spec: TrialSpec) -> None:
    key = (
        spec.n_timesteps,
        spec.num_clients,
        spec.server_update_weight,
        spec.n_envs,
        spec.n_steps,
        spec.batch_size,
        spec.local_rollouts,
        spec.n_epochs,
        spec.gamma,
        spec.gae_lambda,
        spec.learning_rate_expr,
        spec.clip_range_expr,
    )
    store[key] = spec


def build_anchor_specs(base: CartPoleBaseConfig, args: argparse.Namespace) -> list[TrialSpec]:
    specs: dict[tuple[Any, ...], TrialSpec] = {}
    timesteps_values = [max(1, int(round(base.n_timesteps * multiplier))) for multiplier in args.timesteps_multipliers]
    num_clients_values = args.num_client_values if args.num_client_values else [base.num_clients]
    anchor_local_rollouts = sorted({1, 2, 4, base.local_rollouts})
    anchor_learning_rates = sorted({base.learning_rate_expr, "3e-4"})
    anchor_n_epochs = sorted({base.n_epochs, 10})

    preferred_pairs = [
        (base.n_envs, base.n_steps),
        (1, 8),
        (1, 16),
        (1, 32),
        (2, 8),
        (2, 16),
        (4, 8),
        (8, 4),
    ]

    for n_envs, n_steps in preferred_pairs:
        if n_envs not in args.n_env_values or n_steps not in args.n_step_values:
            continue
        rollout_size = n_envs * n_steps
        batches = batch_candidates(rollout_size, args.min_batch_size, args.max_batch_size)
        if not batches:
            continue
        preferred_batch = batches[-1]
        for n_timesteps in timesteps_values:
            for num_clients in num_clients_values:
                for server_update_weight in args.server_update_weights:
                    for local_rollouts in anchor_local_rollouts:
                        for n_epochs in anchor_n_epochs:
                            for learning_rate_expr in anchor_learning_rates:
                                spec = TrialSpec(
                                    env_id=base.env_id,
                                    n_timesteps=n_timesteps,
                                    num_clients=num_clients,
                                    server_update_weight=float(server_update_weight),
                                    n_envs=n_envs,
                                    n_steps=n_steps,
                                    batch_size=preferred_batch,
                                    local_rollouts=local_rollouts,
                                    n_epochs=n_epochs,
                                    gamma=base.gamma,
                                    gae_lambda=base.gae_lambda,
                                    learning_rate_expr=learning_rate_expr,
                                    clip_range_expr=base.clip_range_expr,
                                    is_anchor=True,
                                )
                                add_unique_spec(specs, spec)

    base_spec = TrialSpec(
        env_id=base.env_id,
        n_timesteps=base.n_timesteps,
        num_clients=base.num_clients,
        server_update_weight=base.server_update_weight,
        n_envs=base.n_envs,
        n_steps=base.n_steps,
        batch_size=base.batch_size,
        local_rollouts=base.local_rollouts,
        n_epochs=base.n_epochs,
        gamma=base.gamma,
        gae_lambda=base.gae_lambda,
        learning_rate_expr=base.learning_rate_expr,
        clip_range_expr=base.clip_range_expr,
        is_anchor=True,
    )
    add_unique_spec(specs, base_spec)
    return sorted(specs.values(), key=lambda item: (item.compute_sort_key, item.label))


def build_random_specs(base: CartPoleBaseConfig, args: argparse.Namespace) -> list[TrialSpec]:
    rng = random.Random(args.seed)
    specs: dict[tuple[Any, ...], TrialSpec] = {}
    timesteps_values = [max(1, int(round(base.n_timesteps * multiplier))) for multiplier in args.timesteps_multipliers]
    num_clients_values = args.num_client_values if args.num_client_values else [base.num_clients]
    pairs = candidate_pairs(args.n_env_values, args.n_step_values)
    if not pairs:
        return []

    max_attempts = max(args.max_random_trials * 50, 1000)
    attempts = 0
    while len(specs) < args.max_random_trials and attempts < max_attempts:
        attempts += 1
        n_envs, n_steps = rng.choice(pairs)
        rollout_size = n_envs * n_steps
        batches = batch_candidates(rollout_size, args.min_batch_size, args.max_batch_size)
        if not batches:
            continue
        spec = TrialSpec(
            env_id=base.env_id,
            n_timesteps=rng.choice(timesteps_values),
            num_clients=rng.choice(num_clients_values),
            server_update_weight=float(rng.choice(args.server_update_weights)),
            n_envs=n_envs,
            n_steps=n_steps,
            batch_size=rng.choice(batches),
            local_rollouts=rng.choice(args.local_rollout_values),
            n_epochs=rng.choice(args.n_epoch_values),
            gamma=float(rng.choice(args.gamma_values)),
            gae_lambda=float(rng.choice(args.gae_lambda_values)),
            learning_rate_expr=rng.choice(args.learning_rate_values),
            clip_range_expr=rng.choice(args.clip_range_values),
            is_anchor=False,
        )
        add_unique_spec(specs, spec)
    return sorted(specs.values(), key=lambda item: (item.compute_sort_key, item.label))


def build_trial_specs(base: CartPoleBaseConfig, args: argparse.Namespace) -> list[TrialSpec]:
    specs: dict[tuple[Any, ...], TrialSpec] = {}
    for spec in build_anchor_specs(base, args):
        add_unique_spec(specs, spec)
    for spec in build_random_specs(base, args):
        add_unique_spec(specs, spec)
    return sorted(specs.values(), key=lambda item: (item.compute_sort_key, item.label))


def write_trial_plan(trials: list[TrialSpec], output_root: Path) -> None:
    payload = []
    for spec in trials:
        payload.append(
            {
                "label": spec.label,
                "is_anchor": spec.is_anchor,
                "n_timesteps": spec.n_timesteps,
                "num_clients": spec.num_clients,
                "server_update_weight": spec.server_update_weight,
                "n_envs": spec.n_envs,
                "n_steps": spec.n_steps,
                "batch_size": spec.batch_size,
                "local_rollouts": spec.local_rollouts,
                "local_steps": spec.local_steps,
                "rollout_size": spec.rollout_size,
                "approx_minibatches": spec.approx_minibatches,
                "n_epochs": spec.n_epochs,
                "gamma": spec.gamma,
                "gae_lambda": spec.gae_lambda,
                "learning_rate_expr": spec.learning_rate_expr,
                "clip_range_expr": spec.clip_range_expr,
            }
        )
    with (output_root / "trial_plan.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def build_train_command(spec: TrialSpec, args: argparse.Namespace, log_root: Path) -> list[str]:
    return [
        args.python,
        str(args.train_script),
        "--algo",
        "ppo_avg",
        "--env",
        spec.env_id,
        "--frl",
        "--conf-file",
        str(args.config),
        "--log-folder",
        str(log_root),
        "--seed",
        str(args.seed),
        "--num-threads",
        str(args.num_threads_per_task),
        "--device",
        str(args.device),
        "--vec-env",
        "dummy",
        "--eval-freq",
        "-1",
        "--save-freq",
        "-1",
        "--log-interval",
        "-1",
        "--verbose",
        str(args.train_verbose),
        "-n",
        str(spec.n_timesteps),
        "--env-kwargs",
        "noise_type:None",
        "noise:0.0",
        "--eval-env-kwargs",
        "noise_type:None",
        "noise:0.0",
        "--hyperparams",
        f"num_clients:{spec.num_clients}",
        f"server_update_weight:{spec.server_update_weight}",
        f"n_envs:{spec.n_envs}",
        f"n_steps:{spec.n_steps}",
        f"batch_size:{spec.batch_size}",
        f"local_steps:{spec.local_steps}",
        f"n_epochs:{spec.n_epochs}",
        f"gamma:{spec.gamma}",
        f"gae_lambda:{spec.gae_lambda}",
        f"learning_rate:{spec.learning_rate_expr}",
        f"clip_range:{spec.clip_range_expr}",
        "perturb_noise_type:None",
        "perturb_noise_range:0.0",
        f"eval_round_freq:{args.eval_round_freq}",
        f"eval_local_episodes:{args.eval_local_episodes}",
        f"eval_nominal_episodes:{args.eval_nominal_episodes}",
    ]


def last_finite_value(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def compute_ranking_score(
    local_mean: float | None,
    local_min: float | None,
    nominal_mean: float | None,
    metric: str,
) -> float | None:
    if metric == "local_mean":
        return local_mean
    if metric == "local_min":
        return local_min
    if metric == "nominal_mean":
        return nominal_mean
    if metric == "mean_and_min":
        values = [value for value in (local_mean, local_min) if value is not None and math.isfinite(value)]
        if not values:
            return None
        return float(sum(values) / len(values))
    raise ValueError(f"Unknown ranking metric: {metric}")


def find_run_dir(log_root: Path) -> Path | None:
    algo_dir = log_root / "ppo_avg"
    if not algo_dir.exists():
        return None
    run_dirs = sorted(path for path in algo_dir.iterdir() if path.is_dir())
    return run_dirs[-1] if run_dirs else None


def load_trial_metrics(run_dir: Path, ranking_metric: str) -> tuple[dict[str, float | None], float | None]:
    eval_path = run_dir / "evaluations.npz"
    if not eval_path.exists():
        metrics = {
            "final_local_mean": None,
            "final_local_min": None,
            "final_local_max": None,
            "final_nominal_mean": None,
            "final_nominal_min": None,
        }
        return metrics, None

    with np.load(eval_path, allow_pickle=True) as data:
        metrics = {
            "final_local_mean": last_finite_value(
                np.asarray(data["local_mean_across_clients"], dtype=np.float64)
                if "local_mean_across_clients" in data.files
                else np.asarray([], dtype=np.float64)
            ),
            "final_local_min": last_finite_value(
                np.asarray(data["local_min_across_clients"], dtype=np.float64)
                if "local_min_across_clients" in data.files
                else np.asarray([], dtype=np.float64)
            ),
            "final_local_max": last_finite_value(
                np.asarray(data["local_max_across_clients"], dtype=np.float64)
                if "local_max_across_clients" in data.files
                else np.asarray([], dtype=np.float64)
            ),
            "final_nominal_mean": last_finite_value(
                np.asarray(data["nominal_mean_across_clients"], dtype=np.float64)
                if "nominal_mean_across_clients" in data.files
                else np.asarray([], dtype=np.float64)
            ),
            "final_nominal_min": last_finite_value(
                np.asarray(data["nominal_min_across_clients"], dtype=np.float64)
                if "nominal_min_across_clients" in data.files
                else np.asarray([], dtype=np.float64)
            ),
        }

    ranking_score = compute_ranking_score(
        metrics["final_local_mean"],
        metrics["final_local_min"],
        metrics["final_nominal_mean"],
        ranking_metric,
    )
    return metrics, ranking_score


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return "".join(lines[-max_lines:]).strip()


def run_trial(spec: TrialSpec, args: argparse.Namespace, output_root: Path) -> TrialResult:
    trial_dir = output_root / safe_name(spec.label)
    log_root = trial_dir / "train_logs"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"

    command = build_train_command(spec, args, log_root)
    start_time = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(command))
        log_file.write("\n\nOUTPUT:\n")
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration_sec = time.time() - start_time

    run_dir = find_run_dir(log_root)
    metrics = {
        "final_local_mean": None,
        "final_local_min": None,
        "final_local_max": None,
        "final_nominal_mean": None,
        "final_nominal_min": None,
    }
    ranking_score: float | None = None
    error: str | None = None

    if process.returncode == 0 and run_dir is not None:
        metrics, ranking_score = load_trial_metrics(run_dir, args.ranking_metric)
        if ranking_score is None:
            error = "Training finished but evaluations.npz did not contain a usable final score."
    else:
        error = tail_text(log_path)

    solved = (
        metrics["final_local_mean"] is not None
        and metrics["final_local_min"] is not None
        and metrics["final_local_mean"] >= args.success_mean_threshold
        and metrics["final_local_min"] >= args.success_min_threshold
    )
    status = "ok" if process.returncode == 0 and ranking_score is not None else "failed"

    return TrialResult(
        env_id=spec.env_id,
        label=spec.label,
        is_anchor=spec.is_anchor,
        status=status,
        returncode=process.returncode,
        duration_sec=duration_sec,
        n_timesteps=spec.n_timesteps,
        num_clients=spec.num_clients,
        server_update_weight=spec.server_update_weight,
        n_envs=spec.n_envs,
        n_steps=spec.n_steps,
        batch_size=spec.batch_size,
        local_rollouts=spec.local_rollouts,
        local_steps=spec.local_steps,
        rollout_size=spec.rollout_size,
        approx_minibatches=spec.approx_minibatches,
        n_epochs=spec.n_epochs,
        gamma=spec.gamma,
        gae_lambda=spec.gae_lambda,
        learning_rate_expr=spec.learning_rate_expr,
        clip_range_expr=spec.clip_range_expr,
        final_local_mean=metrics["final_local_mean"],
        final_local_min=metrics["final_local_min"],
        final_local_max=metrics["final_local_max"],
        final_nominal_mean=metrics["final_nominal_mean"],
        final_nominal_min=metrics["final_nominal_min"],
        ranking_score=ranking_score,
        solved=solved,
        run_dir=str(run_dir) if run_dir is not None else None,
        log_path=str(log_path),
        error=error,
    )


def write_results_csv(results: list[TrialResult], output_root: Path) -> None:
    csv_path = output_root / "results.csv"
    fieldnames = [
        "env_id",
        "label",
        "is_anchor",
        "status",
        "returncode",
        "duration_sec",
        "n_timesteps",
        "num_clients",
        "server_update_weight",
        "n_envs",
        "n_steps",
        "batch_size",
        "local_rollouts",
        "local_steps",
        "rollout_size",
        "approx_minibatches",
        "n_epochs",
        "gamma",
        "gae_lambda",
        "learning_rate_expr",
        "clip_range_expr",
        "final_local_mean",
        "final_local_min",
        "final_local_max",
        "final_nominal_mean",
        "final_nominal_min",
        "ranking_score",
        "solved",
        "run_dir",
        "log_path",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def pick_recommendations(results: list[TrialResult], near_best_fraction: float) -> dict[str, Any]:
    successful = [result for result in results if result.is_success]
    solved = [result for result in successful if result.solved]
    best_score_trial = max(
        successful,
        key=lambda item: (item.ranking_score, tuple(-value for value in item.compute_sort_key)),  # type: ignore[arg-type]
        default=None,
    )
    smallest_solved_trial = min(
        solved,
        key=lambda item: (item.compute_sort_key, -(item.ranking_score or -float("inf"))),
        default=None,
    )

    smallest_near_best_trial = None
    if best_score_trial is not None and best_score_trial.ranking_score is not None:
        threshold = best_score_trial.ranking_score * near_best_fraction
        eligible = [
            result
            for result in successful
            if result.ranking_score is not None and result.ranking_score >= threshold
        ]
        if eligible:
            smallest_near_best_trial = min(
                eligible,
                key=lambda item: (item.compute_sort_key, -(item.ranking_score or -float("inf"))),
            )

    return {
        "best_score_trial": asdict(best_score_trial) if best_score_trial is not None else None,
        "smallest_solved_trial": asdict(smallest_solved_trial) if smallest_solved_trial is not None else None,
        "smallest_near_best_trial": asdict(smallest_near_best_trial) if smallest_near_best_trial is not None else None,
    }


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_summary_markdown(
    base: CartPoleBaseConfig,
    results: list[TrialResult],
    recommendations: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> str:
    successful = [result for result in results if result.is_success]
    solved = [result for result in successful if result.solved]
    failed = [result for result in results if not result.is_success]

    lines = [
        "# PPOAvg CartPole Nominal Search",
        "",
        f"- Output root: `{output_root}`",
        f"- Trials: total={len(results)}, success={len(successful)}, solved={len(solved)}, failed={len(failed)}",
        f"- Ranking metric: `{args.ranking_metric}`",
        f"- Success thresholds: local_mean>={args.success_mean_threshold}, local_min>={args.success_min_threshold}",
        "",
        "## Base Config",
        "",
        make_markdown_table(
            ["field", "value"],
            [
                ["n_timesteps", str(base.n_timesteps)],
                ["num_clients", str(base.num_clients)],
                ["server_update_weight", str(base.server_update_weight)],
                ["n_envs", str(base.n_envs)],
                ["n_steps", str(base.n_steps)],
                ["batch_size", str(base.batch_size)],
                ["local_steps", str(base.local_steps)],
                ["local_rollouts", str(base.local_rollouts)],
                ["n_epochs", str(base.n_epochs)],
                ["gamma", str(base.gamma)],
                ["gae_lambda", str(base.gae_lambda)],
                ["learning_rate", base.learning_rate_expr],
                ["clip_range", base.clip_range_expr],
            ],
        ),
        "",
        "## Recommendations",
        "",
    ]

    for key in ("best_score_trial", "smallest_solved_trial", "smallest_near_best_trial"):
        payload = recommendations.get(key)
        lines.append(f"### {key}")
        if payload is None:
            lines.append("")
            lines.append("No matching trial.")
            lines.append("")
            continue
        lines.append("")
        lines.append(
            make_markdown_table(
                ["field", "value"],
                [
                    ["label", str(payload["label"])],
                    ["ranking_score", format_float(payload["ranking_score"])],
                    ["solved", str(payload["solved"])],
                    ["local_mean", format_float(payload["final_local_mean"])],
                    ["local_min", format_float(payload["final_local_min"])],
                    ["nominal_mean", format_float(payload["final_nominal_mean"])],
                    ["n_timesteps", str(payload["n_timesteps"])],
                    ["num_clients", str(payload["num_clients"])],
                    ["server_update_weight", str(payload["server_update_weight"])],
                    ["n_envs", str(payload["n_envs"])],
                    ["n_steps", str(payload["n_steps"])],
                    ["batch_size", str(payload["batch_size"])],
                    ["local_steps", str(payload["local_steps"])],
                    ["local_rollouts", str(payload["local_rollouts"])],
                    ["rollout_size", str(payload["rollout_size"])],
                    ["approx_minibatches", str(payload["approx_minibatches"])],
                    ["n_epochs", str(payload["n_epochs"])],
                    ["gamma", str(payload["gamma"])],
                    ["gae_lambda", str(payload["gae_lambda"])],
                    ["learning_rate_expr", str(payload["learning_rate_expr"])],
                    ["clip_range_expr", str(payload["clip_range_expr"])],
                    ["run_dir", str(payload["run_dir"])],
                ],
            )
        )
        lines.append("")

    top_rows = []
    for result in sorted(
        successful,
        key=lambda item: (-(item.ranking_score or -float("inf")), item.compute_sort_key),
    )[: args.top_k]:
        tags = []
        if result.is_anchor:
            tags.append("anchor")
        if result.solved:
            tags.append("solved")
        top_rows.append(
            [
                result.label,
                format_float(result.ranking_score),
                format_float(result.final_local_mean),
                format_float(result.final_local_min),
                format_float(result.final_nominal_mean),
                str(result.n_envs),
                str(result.n_steps),
                str(result.batch_size),
                str(result.local_steps),
                str(result.local_rollouts),
                str(result.num_clients),
                compact_number(result.server_update_weight),
                str(result.n_epochs),
                result.learning_rate_expr,
                result.clip_range_expr,
                ",".join(tags),
            ]
        )

    lines.append("## Top Trials")
    lines.append("")
    if top_rows:
        lines.append(
            make_markdown_table(
                [
                    "label",
                    "score",
                    "local_mean",
                    "local_min",
                    "nominal_mean",
                    "n_envs",
                    "n_steps",
                    "batch",
                    "local_steps",
                    "local_rollouts",
                    "num_clients",
                    "server_w",
                    "epochs",
                    "lr",
                    "clip",
                    "tag",
                ],
                top_rows,
            )
        )
    else:
        lines.append("No successful trials.")
    lines.append("")

    return "\n".join(lines)


def print_trial_plan_summary(trials: list[TrialSpec]) -> None:
    anchors = sum(1 for trial in trials if trial.is_anchor)
    searches = len(trials) - anchors
    smallest = min(trials, key=lambda item: item.compute_sort_key)
    largest = max(trials, key=lambda item: item.compute_sort_key)
    print(f"Planned {len(trials)} CartPole trials (anchor={anchors}, random={searches})")
    print(
        f"Local compute range: local_steps {smallest.local_steps}->{largest.local_steps}, "
        f"rollout {smallest.rollout_size}->{largest.rollout_size}, "
        f"approx_minibatches {smallest.approx_minibatches}->{largest.approx_minibatches}"
    )


def print_result_line(result: TrialResult) -> None:
    print(
        f"[{result.status.upper():6}] {result.label} "
        f"score={format_float(result.ranking_score)} "
        f"local_mean={format_float(result.final_local_mean)} "
        f"local_min={format_float(result.final_local_min)} "
        f"solved={result.solved} "
        f"local_steps={result.local_steps} "
        f"rollout={result.rollout_size} "
        f"time={result.duration_sec / 60.0:.1f}m"
    )


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.train_script = args.train_script.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root.resolve() / timestamp
    output_root.mkdir(parents=True, exist_ok=True)

    base = load_cartpole_base_config(args.config)
    trials = build_trial_specs(base, args)
    if not trials:
        raise ValueError("No CartPole trials were generated. Check your search-space arguments.")

    write_trial_plan(trials, output_root)
    print(f"Output root: {output_root}")
    print_trial_plan_summary(trials)

    if args.dry_run:
        print("Dry run only. Trial plan written to trial_plan.json.")
        return

    results: list[TrialResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_trial = {
            executor.submit(run_trial, trial, args, output_root): trial
            for trial in trials
        }
        for future in as_completed(future_to_trial):
            result = future.result()
            results.append(result)
            print_result_line(result)

    results.sort(key=lambda item: (item.compute_sort_key, item.label))
    write_results_csv(results, output_root)
    recommendations = pick_recommendations(results, args.near_best_fraction)
    with (output_root / "recommendations.json").open("w", encoding="utf-8") as file:
        json.dump(recommendations, file, indent=2)

    summary_md = build_summary_markdown(base, results, recommendations, args, output_root)
    summary_path = output_root / "summary.md"
    summary_path.write_text(summary_md + "\n", encoding="utf-8")

    print(f"Results written to: {output_root / 'results.csv'}")
    print(f"Recommendations written to: {output_root / 'recommendations.json'}")
    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
