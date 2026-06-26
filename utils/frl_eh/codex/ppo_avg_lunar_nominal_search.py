from __future__ import annotations

import argparse
import csv
import json
import math
import random
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "hyperparams" / "ppo_avg.yml"
DEFAULT_TRAIN_SCRIPT = REPO_ROOT / "rl_zoo3" / "train.py"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_lunar_nominal_search"
DEFAULT_ENV_IDS = ("PerturbLunarLander-v3", "PerturbLunarLanderContinuous-v3")

# These thresholds are intentionally modest: they are used only to tell apart
# "the run is clearly learning" from "the run is collapsing/noisy".
DEFAULT_SUCCESS_CRITERIA: dict[str, dict[str, float]] = {
    "PerturbLunarLander-v3": {
        "peak_local": 150.0,
        "peak_nominal": 120.0,
        "tail_local": 100.0,
        "tail_nominal": 80.0,
    },
    "PerturbLunarLanderContinuous-v3": {
        "peak_local": 75.0,
        "peak_nominal": 25.0,
        "tail_local": 40.0,
        "tail_nominal": 0.0,
    },
}

# Anchor layouts include the current YAML configuration plus a few layouts that
# looked promising in earlier budget-oriented sweeps.
ENV_ANCHOR_LAYOUTS: dict[str, list[dict[str, int]]] = {
    "PerturbLunarLander-v3": [
        {"n_envs": 4, "n_steps": 8, "batch_size": 8, "local_steps": 32},
        {"n_envs": 1, "n_steps": 8, "batch_size": 4, "local_steps": 8},
        {"n_envs": 2, "n_steps": 8, "batch_size": 16, "local_steps": 16},
        {"n_envs": 4, "n_steps": 4, "batch_size": 16, "local_steps": 16},
        {"n_envs": 4, "n_steps": 16, "batch_size": 32, "local_steps": 64},
        {"n_envs": 8, "n_steps": 4, "batch_size": 8, "local_steps": 32},
    ],
    "PerturbLunarLanderContinuous-v3": [
        {"n_envs": 8, "n_steps": 8, "batch_size": 64, "local_steps": 64},
        {"n_envs": 1, "n_steps": 8, "batch_size": 8, "local_steps": 8},
        {"n_envs": 4, "n_steps": 8, "batch_size": 8, "local_steps": 32},
        {"n_envs": 4, "n_steps": 16, "batch_size": 16, "local_steps": 64},
        {"n_envs": 16, "n_steps": 4, "batch_size": 32, "local_steps": 64},
        {"n_envs": 2, "n_steps": 32, "batch_size": 8, "local_steps": 64},
    ],
}


@dataclass(frozen=True)
class LunarBaseConfig:
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
    ent_coef: float

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
    ent_coef: float
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
            f"_ent{compact_number(self.ent_coef)}"
        )


@dataclass
class SeedTrialResult:
    env_id: str
    label: str
    seed: int
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
    ent_coef: float
    eval_round_freq: int
    final_local_mean: float | None
    final_nominal_mean: float | None
    tail_local_mean: float | None
    tail_nominal_mean: float | None
    best_local_mean: float | None
    best_nominal_mean: float | None
    first_peak_round: int | None
    first_peak_timestep: int | None
    first_stable_round: int | None
    first_stable_timestep: int | None
    learning_success: bool
    stable_success: bool
    ranking_score: float | None
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


@dataclass
class CandidateAggregate:
    env_id: str
    label: str
    is_anchor: bool
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
    ent_coef: float
    num_seeds: int
    ok_seeds: int
    learning_successes: int
    stable_successes: int
    learning_success_rate: float
    stable_success_rate: float
    mean_final_local: float | None
    mean_final_nominal: float | None
    mean_tail_local: float | None
    mean_tail_nominal: float | None
    min_tail_local: float | None
    min_tail_nominal: float | None
    mean_best_local: float | None
    mean_best_nominal: float | None
    mean_first_peak_timestep: float | None
    mean_first_stable_timestep: float | None
    mean_duration_sec: float | None
    ranking_score: float | None
    representative_run_dir: str | None
    seed_log_paths: list[str]

    @property
    def compute_sort_key(self) -> tuple[int, int, int, int, int]:
        return (
            self.local_steps,
            self.approx_minibatches,
            self.rollout_size,
            self.n_envs,
            self.batch_size,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search PPOAvg hyperparameters for nominal FRL learning on "
            "PerturbLunarLander-v3 and PerturbLunarLanderContinuous-v3. "
            "The script runs multi-seed trials, scores full learning curves from "
            "evaluations.npz, and writes ranked recommendations."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to ppo_avg.yml.")
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT, help="Path to rl_zoo3/train.py.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for sweep logs.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument(
        "--envs",
        nargs="*",
        default=list(DEFAULT_ENV_IDS),
        help="Target env ids. Default: both LunarLander variants.",
    )
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2], help="Seeds used for each candidate.")
    parser.add_argument("--max-workers", type=int, default=10, help="Maximum number of parallel training jobs.")
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
        default=[1.0],
        help="Multipliers applied to each env's n_timesteps from ppo_avg.yml.",
    )
    parser.add_argument(
        "--server-update-weights",
        nargs="*",
        type=float,
        default=[1.0, 0.75, 0.5],
        help="server_update_weight values to search.",
    )
    parser.add_argument(
        "--n-env-values",
        nargs="*",
        type=int,
        default=[1, 2, 4, 8, 16],
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
        default=[1, 2, 4, 8],
        help="Number of PPO rollout/update cycles per local FRL round.",
    )
    parser.add_argument(
        "--n-epoch-values",
        nargs="*",
        type=int,
        default=[4, 8, 10],
        help="PPO n_epochs values to search.",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="*",
        type=float,
        default=[0.99, 0.995, 0.999],
        help="Gamma values to search.",
    )
    parser.add_argument(
        "--gae-lambda-values",
        nargs="*",
        type=float,
        default=[0.95, 0.98],
        help="GAE lambda values to search.",
    )
    parser.add_argument(
        "--learning-rate-values",
        nargs="*",
        type=str,
        default=["1e-4", "3e-4", "1e-3", "'lin_0.001'"],
        help="Learning-rate expressions forwarded through StoreDict.",
    )
    parser.add_argument(
        "--clip-range-values",
        nargs="*",
        type=str,
        default=["0.1", "0.2", "'lin_0.2'"],
        help="clip_range expressions forwarded through StoreDict.",
    )
    parser.add_argument(
        "--ent-coef-values",
        nargs="*",
        type=float,
        default=[0.0, 0.005, 0.01, 0.02],
        help="Entropy coefficients to search.",
    )
    parser.add_argument("--min-batch-size", type=int, default=4, help="Minimum batch_size candidate.")
    parser.add_argument("--max-batch-size", type=int, default=256, help="Maximum batch_size candidate.")
    parser.add_argument(
        "--max-random-trials-per-env",
        type=int,
        default=40,
        help="Maximum number of random search trials per environment.",
    )
    parser.add_argument(
        "--eval-local-episodes",
        type=int,
        default=5,
        help="FRL local-eval episodes per client for intermediate/final evaluation.",
    )
    parser.add_argument(
        "--eval-nominal-episodes",
        type=int,
        default=5,
        help="FRL nominal-eval episodes per client for intermediate/final evaluation.",
    )
    parser.add_argument(
        "--desired-evals",
        type=int,
        default=30,
        help="Approximate number of intermediate eval snapshots to keep per run.",
    )
    parser.add_argument(
        "--tail-window",
        type=int,
        default=5,
        help="Number of last evaluation points used for tail-mean stability metrics.",
    )
    parser.add_argument(
        "--stable-success-rate-threshold",
        type=float,
        default=0.67,
        help="Minimum seed-level stable success rate required to call a candidate stable.",
    )
    parser.add_argument(
        "--learning-success-rate-threshold",
        type=float,
        default=0.67,
        help="Minimum seed-level learning success rate required to call a candidate promising.",
    )
    parser.add_argument("--top-k", type=int, default=10, help="Number of top candidates to keep per env in the report.")
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


def coerce_int(value: Any, key: str, env_id: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_id}: expected integer-like value for {key}, got {value!r}") from exc


def load_env_configs(config_path: Path, env_ids: list[str]) -> dict[str, LunarBaseConfig]:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    env_configs: dict[str, LunarBaseConfig] = {}
    for env_id in env_ids:
        if env_id not in raw:
            raise KeyError(f"{env_id} not found in {config_path}")
        values = raw[env_id]
        if not isinstance(values, dict):
            raise ValueError(f"{env_id} config must be a mapping.")

        learning_rate_value = values.get("learning_rate", 3e-4)
        clip_range_value = values.get("clip_range", 0.2)
        env_configs[env_id] = LunarBaseConfig(
            env_id=env_id,
            n_timesteps=coerce_int(values["n_timesteps"], "n_timesteps", env_id),
            num_clients=coerce_int(values.get("num_clients", 5), "num_clients", env_id),
            server_update_weight=float(values.get("server_update_weight", 1.0)),
            n_envs=coerce_int(values["n_envs"], "n_envs", env_id),
            n_steps=coerce_int(values["n_steps"], "n_steps", env_id),
            batch_size=coerce_int(values["batch_size"], "batch_size", env_id),
            local_steps=coerce_int(values["local_steps"], "local_steps", env_id),
            n_epochs=coerce_int(values.get("n_epochs", 10), "n_epochs", env_id),
            gamma=float(values.get("gamma", 0.99)),
            gae_lambda=float(values.get("gae_lambda", 0.95)),
            learning_rate_expr=repr(learning_rate_value) if isinstance(learning_rate_value, str) else str(learning_rate_value),
            clip_range_expr=repr(clip_range_value) if isinstance(clip_range_value, str) else str(clip_range_value),
            ent_coef=float(values.get("ent_coef", 0.0)),
        )
    return env_configs


def batch_candidates(rollout_size: int, min_batch_size: int, max_batch_size: int) -> list[int]:
    min_batch_size = max(2, min_batch_size)
    max_batch_size = max(min_batch_size, max_batch_size)
    values: set[int] = set()
    candidate = min(rollout_size, max_batch_size)
    while candidate >= min_batch_size:
        if rollout_size % candidate == 0:
            values.add(candidate)
        if candidate == min_batch_size:
            break
        candidate = max(min_batch_size, candidate // 2)
        if candidate in values:
            break
    if rollout_size <= max_batch_size:
        values.add(rollout_size)
    return sorted(value for value in values if min_batch_size <= value <= rollout_size)


def candidate_pairs(n_env_values: list[int], n_step_values: list[int]) -> list[tuple[int, int]]:
    pairs = [(n_envs, n_steps) for n_envs in n_env_values for n_steps in n_step_values if n_envs * n_steps > 1]
    pairs.sort(key=lambda item: (item[0] * item[1], item[0], item[1]))
    return pairs


def add_unique_spec(store: dict[tuple[Any, ...], TrialSpec], spec: TrialSpec) -> None:
    key = (
        spec.env_id,
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
        spec.ent_coef,
    )
    store[key] = spec


def anchor_profiles_for_env(base: LunarBaseConfig) -> list[dict[str, Any]]:
    if base.env_id == "PerturbLunarLander-v3":
        profiles = [
            {
                "n_epochs": base.n_epochs,
                "gamma": base.gamma,
                "gae_lambda": base.gae_lambda,
                "learning_rate_expr": base.learning_rate_expr,
                "clip_range_expr": base.clip_range_expr,
                "ent_coef": base.ent_coef,
            },
            {
                "n_epochs": 8,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "learning_rate_expr": "3e-4",
                "clip_range_expr": "0.1",
                "ent_coef": 0.0,
            },
            {
                "n_epochs": 10,
                "gamma": 0.995,
                "gae_lambda": 0.98,
                "learning_rate_expr": "1e-4",
                "clip_range_expr": "0.2",
                "ent_coef": 0.01,
            },
        ]
    else:
        profiles = [
            {
                "n_epochs": base.n_epochs,
                "gamma": base.gamma,
                "gae_lambda": base.gae_lambda,
                "learning_rate_expr": base.learning_rate_expr,
                "clip_range_expr": base.clip_range_expr,
                "ent_coef": base.ent_coef,
            },
            {
                "n_epochs": 8,
                "gamma": 0.995,
                "gae_lambda": 0.95,
                "learning_rate_expr": "1e-4",
                "clip_range_expr": "0.1",
                "ent_coef": 0.0,
            },
            {
                "n_epochs": 10,
                "gamma": 0.999,
                "gae_lambda": 0.98,
                "learning_rate_expr": "3e-4",
                "clip_range_expr": "0.2",
                "ent_coef": 0.005,
            },
        ]

    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for profile in profiles:
        key = (
            profile["n_epochs"],
            profile["gamma"],
            profile["gae_lambda"],
            profile["learning_rate_expr"],
            profile["clip_range_expr"],
            profile["ent_coef"],
        )
        unique[key] = profile
    return list(unique.values())


def build_anchor_specs(base: LunarBaseConfig, args: argparse.Namespace) -> list[TrialSpec]:
    specs: dict[tuple[Any, ...], TrialSpec] = {}
    timesteps_values = [max(1, int(round(base.n_timesteps * multiplier))) for multiplier in args.timesteps_multipliers]
    layouts = ENV_ANCHOR_LAYOUTS[base.env_id]
    profiles = anchor_profiles_for_env(base)

    # Always include the current YAML config as one of the anchor layouts.
    layouts = [
        {
            "n_envs": base.n_envs,
            "n_steps": base.n_steps,
            "batch_size": base.batch_size,
            "local_steps": base.local_steps,
        },
        *layouts,
    ]

    for n_timesteps in timesteps_values:
        for server_update_weight in args.server_update_weights:
            for layout in layouts:
                rollout_size = int(layout["n_envs"]) * int(layout["n_steps"])
                if rollout_size <= 0:
                    continue
                local_steps = int(layout["local_steps"])
                if local_steps < rollout_size:
                    continue
                if local_steps % rollout_size != 0:
                    continue
                local_rollouts = max(1, local_steps // rollout_size)
                for profile in profiles:
                    spec = TrialSpec(
                        env_id=base.env_id,
                        n_timesteps=n_timesteps,
                        num_clients=base.num_clients,
                        server_update_weight=float(server_update_weight),
                        n_envs=int(layout["n_envs"]),
                        n_steps=int(layout["n_steps"]),
                        batch_size=int(layout["batch_size"]),
                        local_rollouts=local_rollouts,
                        n_epochs=int(profile["n_epochs"]),
                        gamma=float(profile["gamma"]),
                        gae_lambda=float(profile["gae_lambda"]),
                        learning_rate_expr=str(profile["learning_rate_expr"]),
                        clip_range_expr=str(profile["clip_range_expr"]),
                        ent_coef=float(profile["ent_coef"]),
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
        ent_coef=base.ent_coef,
        is_anchor=True,
    )
    add_unique_spec(specs, base_spec)
    return sorted(specs.values(), key=lambda item: (item.compute_sort_key, item.label))


def build_random_specs(base: LunarBaseConfig, args: argparse.Namespace) -> list[TrialSpec]:
    rng = random.Random(args.seeds[0] if args.seeds else 0)
    specs: dict[tuple[Any, ...], TrialSpec] = {}
    timesteps_values = [max(1, int(round(base.n_timesteps * multiplier))) for multiplier in args.timesteps_multipliers]
    pairs = candidate_pairs(args.n_env_values, args.n_step_values)
    if not pairs:
        return []

    max_attempts = max(args.max_random_trials_per_env * 80, 1000)
    attempts = 0
    while len(specs) < args.max_random_trials_per_env and attempts < max_attempts:
        attempts += 1
        n_envs, n_steps = rng.choice(pairs)
        rollout_size = n_envs * n_steps
        batches = batch_candidates(rollout_size, args.min_batch_size, args.max_batch_size)
        if not batches:
            continue

        spec = TrialSpec(
            env_id=base.env_id,
            n_timesteps=rng.choice(timesteps_values),
            num_clients=base.num_clients,
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
            ent_coef=float(rng.choice(args.ent_coef_values)),
            is_anchor=False,
        )
        add_unique_spec(specs, spec)
    return sorted(specs.values(), key=lambda item: (item.compute_sort_key, item.label))


def build_trial_specs(env_configs: dict[str, LunarBaseConfig], args: argparse.Namespace) -> list[TrialSpec]:
    specs: dict[tuple[Any, ...], TrialSpec] = {}
    for env_id in args.envs:
        base = env_configs[env_id]
        for spec in build_anchor_specs(base, args):
            add_unique_spec(specs, spec)
        for spec in build_random_specs(base, args):
            add_unique_spec(specs, spec)
    return sorted(specs.values(), key=lambda item: (item.env_id, item.compute_sort_key, item.label))


def write_trial_plan(trials: list[TrialSpec], output_root: Path) -> None:
    payload = []
    for spec in trials:
        payload.append(
            {
                "env_id": spec.env_id,
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
                "ent_coef": spec.ent_coef,
            }
        )
    with (output_root / "trial_plan.json").open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def estimate_eval_round_freq(spec: TrialSpec, desired_evals: int) -> int:
    round_budget = max(1, spec.local_steps * spec.num_clients)
    estimated_rounds = max(1, math.ceil(spec.n_timesteps / round_budget))
    return max(1, estimated_rounds // max(desired_evals, 1))


def build_train_command(spec: TrialSpec, seed: int, args: argparse.Namespace, log_root: Path) -> list[str]:
    eval_round_freq = estimate_eval_round_freq(spec, args.desired_evals)
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
        str(seed),
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
        f"ent_coef:{spec.ent_coef}",
        "perturb_noise_type:None",
        "perturb_noise_range:0.0",
        f"eval_round_freq:{eval_round_freq}",
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


def tail_finite_mean(values: np.ndarray, window: int) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    window = max(1, min(window, finite.size))
    return float(np.mean(finite[-window:]))


def first_threshold_round(
    rounds: np.ndarray,
    local_mean: np.ndarray,
    nominal_mean: np.ndarray,
    local_threshold: float,
    nominal_threshold: float,
) -> tuple[int | None, int | None]:
    if rounds.size == 0 or local_mean.size == 0 or nominal_mean.size == 0:
        return None, None
    size = min(rounds.size, local_mean.size, nominal_mean.size)
    mask = (
        np.isfinite(local_mean[:size])
        & np.isfinite(nominal_mean[:size])
        & (local_mean[:size] >= local_threshold)
        & (nominal_mean[:size] >= nominal_threshold)
    )
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return None, None
    index = int(indices[0])
    return int(index), int(rounds[index])


def get_env_criteria(env_id: str) -> dict[str, float]:
    if env_id not in DEFAULT_SUCCESS_CRITERIA:
        raise KeyError(f"Missing default success criteria for {env_id}")
    return DEFAULT_SUCCESS_CRITERIA[env_id]


def find_run_dir(log_root: Path) -> Path | None:
    algo_dir = log_root / "ppo_avg"
    if not algo_dir.exists():
        return None
    run_dirs = sorted(path for path in algo_dir.iterdir() if path.is_dir())
    if not run_dirs:
        return None
    return run_dirs[-1]


def load_trial_metrics(run_dir: Path, env_id: str, tail_window: int) -> dict[str, Any] | None:
    eval_path = run_dir / "evaluations.npz"
    if not eval_path.exists():
        return None

    criteria = get_env_criteria(env_id)
    with np.load(eval_path, allow_pickle=True) as data:
        rounds = np.asarray(data["rounds"], dtype=np.int64) if "rounds" in data.files else np.asarray([], dtype=np.int64)
        local_mean = (
            np.asarray(data["local_mean_across_clients"], dtype=np.float64)
            if "local_mean_across_clients" in data.files
            else np.asarray([], dtype=np.float64)
        )
        nominal_mean = (
            np.asarray(data["nominal_mean_across_clients"], dtype=np.float64)
            if "nominal_mean_across_clients" in data.files
            else np.asarray([], dtype=np.float64)
        )

    final_local_mean = last_finite_value(local_mean)
    final_nominal_mean = last_finite_value(nominal_mean)
    tail_local_mean = tail_finite_mean(local_mean, tail_window)
    tail_nominal_mean = tail_finite_mean(nominal_mean, tail_window)
    best_local_mean = float(np.max(local_mean[np.isfinite(local_mean)])) if np.isfinite(local_mean).any() else None
    best_nominal_mean = float(np.max(nominal_mean[np.isfinite(nominal_mean)])) if np.isfinite(nominal_mean).any() else None

    _, first_peak_round = first_threshold_round(
        rounds,
        local_mean,
        nominal_mean,
        criteria["peak_local"],
        criteria["peak_nominal"],
    )
    _, first_stable_round = first_threshold_round(
        rounds,
        local_mean,
        nominal_mean,
        criteria["tail_local"],
        criteria["tail_nominal"],
    )

    learning_success = bool(
        best_local_mean is not None
        and best_nominal_mean is not None
        and best_local_mean >= criteria["peak_local"]
        and best_nominal_mean >= criteria["peak_nominal"]
    )
    stable_success = bool(
        tail_local_mean is not None
        and tail_nominal_mean is not None
        and tail_local_mean >= criteria["tail_local"]
        and tail_nominal_mean >= criteria["tail_nominal"]
    )

    ranking_score: float | None = None
    if tail_local_mean is not None and tail_nominal_mean is not None:
        tail_floor = min(tail_local_mean, tail_nominal_mean)
        tail_mean = 0.5 * (tail_local_mean + tail_nominal_mean)
        peak_floor = min(best_local_mean or tail_local_mean, best_nominal_mean or tail_nominal_mean)
        ranking_score = (
            (1_000_000.0 if stable_success else 0.0)
            + (100_000.0 if learning_success else 0.0)
            + 1_000.0 * tail_floor
            + 10.0 * tail_mean
            + peak_floor
        )

    return {
        "final_local_mean": final_local_mean,
        "final_nominal_mean": final_nominal_mean,
        "tail_local_mean": tail_local_mean,
        "tail_nominal_mean": tail_nominal_mean,
        "best_local_mean": best_local_mean,
        "best_nominal_mean": best_nominal_mean,
        "first_peak_round": first_peak_round,
        "first_stable_round": first_stable_round,
        "learning_success": learning_success,
        "stable_success": stable_success,
        "ranking_score": ranking_score,
    }


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return "".join(lines[-max_lines:]).strip()


def run_seed_trial(spec: TrialSpec, seed: int, args: argparse.Namespace, output_root: Path) -> SeedTrialResult:
    trial_dir = output_root / safe_name(spec.env_id) / safe_name(spec.label) / f"seed_{seed}"
    log_root = trial_dir / "train_logs"
    trial_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)
    log_path = trial_dir / "train.log"

    eval_round_freq = estimate_eval_round_freq(spec, args.desired_evals)
    command = build_train_command(spec, seed, args, log_root)
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
    metrics: dict[str, Any] | None = None
    error: str | None = None

    if process.returncode == 0 and run_dir is not None:
        metrics = load_trial_metrics(run_dir, spec.env_id, args.tail_window)
        if metrics is None:
            error = "Training finished but evaluations.npz did not contain usable learning-curve metrics."
    else:
        error = tail_text(log_path)

    status = "ok" if process.returncode == 0 and metrics is not None else "failed"
    metrics = metrics or {}
    first_peak_round = metrics.get("first_peak_round")
    first_stable_round = metrics.get("first_stable_round")
    first_peak_timestep = (
        int(first_peak_round * spec.local_steps * spec.num_clients) if first_peak_round is not None else None
    )
    first_stable_timestep = (
        int(first_stable_round * spec.local_steps * spec.num_clients) if first_stable_round is not None else None
    )

    return SeedTrialResult(
        env_id=spec.env_id,
        label=spec.label,
        seed=seed,
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
        ent_coef=spec.ent_coef,
        eval_round_freq=eval_round_freq,
        final_local_mean=metrics.get("final_local_mean"),
        final_nominal_mean=metrics.get("final_nominal_mean"),
        tail_local_mean=metrics.get("tail_local_mean"),
        tail_nominal_mean=metrics.get("tail_nominal_mean"),
        best_local_mean=metrics.get("best_local_mean"),
        best_nominal_mean=metrics.get("best_nominal_mean"),
        first_peak_round=first_peak_round,
        first_peak_timestep=first_peak_timestep,
        first_stable_round=first_stable_round,
        first_stable_timestep=first_stable_timestep,
        learning_success=bool(metrics.get("learning_success", False)),
        stable_success=bool(metrics.get("stable_success", False)),
        ranking_score=metrics.get("ranking_score"),
        run_dir=str(run_dir) if run_dir is not None else None,
        log_path=str(log_path),
        error=error,
    )


def mean_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def min_or_none(values: list[float | None]) -> float | None:
    finite = [value for value in values if value is not None and math.isfinite(value)]
    if not finite:
        return None
    return float(min(finite))


def aggregate_candidate(results: list[SeedTrialResult]) -> CandidateAggregate:
    first = results[0]
    num_seeds = len(results)
    ok_results = [result for result in results if result.status == "ok"]
    ok_seeds = len(ok_results)
    learning_successes = sum(result.learning_success for result in ok_results)
    stable_successes = sum(result.stable_success for result in ok_results)
    learning_success_rate = learning_successes / float(num_seeds)
    stable_success_rate = stable_successes / float(num_seeds)

    mean_tail_local = mean_or_none([result.tail_local_mean for result in ok_results])
    mean_tail_nominal = mean_or_none([result.tail_nominal_mean for result in ok_results])
    mean_best_local = mean_or_none([result.best_local_mean for result in ok_results])
    mean_best_nominal = mean_or_none([result.best_nominal_mean for result in ok_results])
    mean_final_local = mean_or_none([result.final_local_mean for result in ok_results])
    mean_final_nominal = mean_or_none([result.final_nominal_mean for result in ok_results])
    min_tail_local = min_or_none([result.tail_local_mean for result in ok_results])
    min_tail_nominal = min_or_none([result.tail_nominal_mean for result in ok_results])
    mean_first_peak_timestep = mean_or_none([float(result.first_peak_timestep) if result.first_peak_timestep is not None else None for result in ok_results])
    mean_first_stable_timestep = mean_or_none([float(result.first_stable_timestep) if result.first_stable_timestep is not None else None for result in ok_results])
    mean_duration_sec = mean_or_none([result.duration_sec for result in ok_results])

    ranking_score: float | None = None
    if mean_tail_local is not None and mean_tail_nominal is not None:
        tail_floor = min(mean_tail_local, mean_tail_nominal)
        tail_mean = 0.5 * (mean_tail_local + mean_tail_nominal)
        peak_floor = min(mean_best_local or tail_floor, mean_best_nominal or tail_floor)
        ranking_score = (
            stable_success_rate * 1_000_000.0
            + learning_success_rate * 100_000.0
            + 1_000.0 * tail_floor
            + 10.0 * tail_mean
            + peak_floor
        )

    return CandidateAggregate(
        env_id=first.env_id,
        label=first.label,
        is_anchor=first.is_anchor,
        n_timesteps=first.n_timesteps,
        num_clients=first.num_clients,
        server_update_weight=first.server_update_weight,
        n_envs=first.n_envs,
        n_steps=first.n_steps,
        batch_size=first.batch_size,
        local_rollouts=first.local_rollouts,
        local_steps=first.local_steps,
        rollout_size=first.rollout_size,
        approx_minibatches=first.approx_minibatches,
        n_epochs=first.n_epochs,
        gamma=first.gamma,
        gae_lambda=first.gae_lambda,
        learning_rate_expr=first.learning_rate_expr,
        clip_range_expr=first.clip_range_expr,
        ent_coef=first.ent_coef,
        num_seeds=num_seeds,
        ok_seeds=ok_seeds,
        learning_successes=learning_successes,
        stable_successes=stable_successes,
        learning_success_rate=learning_success_rate,
        stable_success_rate=stable_success_rate,
        mean_final_local=mean_final_local,
        mean_final_nominal=mean_final_nominal,
        mean_tail_local=mean_tail_local,
        mean_tail_nominal=mean_tail_nominal,
        min_tail_local=min_tail_local,
        min_tail_nominal=min_tail_nominal,
        mean_best_local=mean_best_local,
        mean_best_nominal=mean_best_nominal,
        mean_first_peak_timestep=mean_first_peak_timestep,
        mean_first_stable_timestep=mean_first_stable_timestep,
        mean_duration_sec=mean_duration_sec,
        ranking_score=ranking_score,
        representative_run_dir=ok_results[0].run_dir if ok_results else None,
        seed_log_paths=[result.log_path for result in results],
    )


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def write_seed_results_csv(results: list[SeedTrialResult], output_root: Path) -> None:
    fieldnames = [
        "env_id",
        "label",
        "seed",
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
        "ent_coef",
        "eval_round_freq",
        "final_local_mean",
        "final_nominal_mean",
        "tail_local_mean",
        "tail_nominal_mean",
        "best_local_mean",
        "best_nominal_mean",
        "first_peak_round",
        "first_peak_timestep",
        "first_stable_round",
        "first_stable_timestep",
        "learning_success",
        "stable_success",
        "ranking_score",
        "run_dir",
        "log_path",
        "error",
    ]
    with (output_root / "seed_results.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_candidate_summary_csv(aggregates: list[CandidateAggregate], output_root: Path) -> None:
    fieldnames = [
        "env_id",
        "label",
        "is_anchor",
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
        "ent_coef",
        "num_seeds",
        "ok_seeds",
        "learning_successes",
        "stable_successes",
        "learning_success_rate",
        "stable_success_rate",
        "mean_final_local",
        "mean_final_nominal",
        "mean_tail_local",
        "mean_tail_nominal",
        "min_tail_local",
        "min_tail_nominal",
        "mean_best_local",
        "mean_best_nominal",
        "mean_first_peak_timestep",
        "mean_first_stable_timestep",
        "mean_duration_sec",
        "ranking_score",
        "representative_run_dir",
        "seed_log_paths",
    ]
    with (output_root / "candidate_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for aggregate in aggregates:
            row = asdict(aggregate)
            row["seed_log_paths"] = json.dumps(aggregate.seed_log_paths)
            writer.writerow(row)


def rank_candidates(aggregates: list[CandidateAggregate]) -> list[CandidateAggregate]:
    return sorted(
        aggregates,
        key=lambda item: (
            -(item.ranking_score if item.ranking_score is not None else -float("inf")),
            item.compute_sort_key,
            item.label,
        ),
    )


def build_recommendations(
    aggregates: list[CandidateAggregate],
    args: argparse.Namespace,
) -> dict[str, dict[str, CandidateAggregate | None]]:
    grouped: dict[str, list[CandidateAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[aggregate.env_id].append(aggregate)

    recommendations: dict[str, dict[str, CandidateAggregate | None]] = {}
    for env_id, env_aggregates in grouped.items():
        ranked = rank_candidates(env_aggregates)
        stable = [item for item in ranked if item.stable_success_rate >= args.stable_success_rate_threshold]
        promising = [item for item in ranked if item.learning_success_rate >= args.learning_success_rate_threshold]

        smallest_stable = None
        if stable:
            smallest_stable = min(
                stable,
                key=lambda item: (
                    item.compute_sort_key,
                    -(item.stable_success_rate),
                    -(item.mean_tail_local if item.mean_tail_local is not None else -float("inf")),
                    -(item.mean_tail_nominal if item.mean_tail_nominal is not None else -float("inf")),
                ),
            )

        recommendations[env_id] = {
            "best_stable": stable[0] if stable else None,
            "smallest_stable": smallest_stable,
            "best_promising": promising[0] if promising else None,
            "best_overall": ranked[0] if ranked else None,
        }
    return recommendations


def write_recommendations(recommendations: dict[str, dict[str, CandidateAggregate | None]], output_root: Path) -> None:
    json_payload: dict[str, Any] = {}
    best_stable_yaml: dict[str, Any] = {}
    smallest_stable_yaml: dict[str, Any] = {}

    def to_override_payload(aggregate: CandidateAggregate) -> dict[str, Any]:
        return {
            "n_timesteps": aggregate.n_timesteps,
            "num_clients": aggregate.num_clients,
            "server_update_weight": aggregate.server_update_weight,
            "n_envs": aggregate.n_envs,
            "n_steps": aggregate.n_steps,
            "batch_size": aggregate.batch_size,
            "local_steps": aggregate.local_steps,
            "n_epochs": aggregate.n_epochs,
            "gamma": aggregate.gamma,
            "gae_lambda": aggregate.gae_lambda,
            "learning_rate": aggregate.learning_rate_expr,
            "clip_range": aggregate.clip_range_expr,
            "ent_coef": aggregate.ent_coef,
        }

    for env_id, picks in sorted(recommendations.items()):
        json_payload[env_id] = {
            key: (asdict(value) if value is not None else None)
            for key, value in picks.items()
        }
        if picks["best_stable"] is not None:
            best_stable_yaml[env_id] = to_override_payload(picks["best_stable"])
        if picks["smallest_stable"] is not None:
            smallest_stable_yaml[env_id] = to_override_payload(picks["smallest_stable"])

    with (output_root / "recommendations.json").open("w", encoding="utf-8") as file:
        json.dump(json_payload, file, indent=2)
    with (output_root / "best_stable_overrides.yml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(best_stable_yaml, file, sort_keys=False)
    with (output_root / "smallest_stable_overrides.yml").open("w", encoding="utf-8") as file:
        yaml.safe_dump(smallest_stable_yaml, file, sort_keys=False)


def write_summary_markdown(
    aggregates: list[CandidateAggregate],
    recommendations: dict[str, dict[str, CandidateAggregate | None]],
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    grouped: dict[str, list[CandidateAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[aggregate.env_id].append(aggregate)

    lines = [
        "# PPOAvg Lunar Nominal Search",
        "",
        f"- Output dir: `{output_root}`",
        f"- Envs: {', '.join(args.envs)}",
        f"- Seeds per candidate: {args.seeds}",
        f"- Parallel workers: {args.max_workers}",
        f"- Tail window: {args.tail_window}",
        f"- Stable success-rate threshold: {args.stable_success_rate_threshold:.2f}",
        f"- Learning success-rate threshold: {args.learning_success_rate_threshold:.2f}",
        "",
        "## Criteria",
        "",
    ]

    for env_id in args.envs:
        criteria = get_env_criteria(env_id)
        lines.append(
            f"- `{env_id}`: peak(local>={criteria['peak_local']:.1f}, nominal>={criteria['peak_nominal']:.1f}), "
            f"tail(local>={criteria['tail_local']:.1f}, nominal>={criteria['tail_nominal']:.1f})"
        )

    for env_id in args.envs:
        env_aggregates = rank_candidates(grouped.get(env_id, []))
        picks = recommendations.get(env_id, {})
        lines.extend(["", f"## {env_id}", ""])
        lines.append(f"- Candidates: {len(env_aggregates)}")
        if picks.get("best_stable") is not None:
            best_stable = picks["best_stable"]
            assert best_stable is not None
            lines.append(
                f"- Best stable: `{best_stable.label}` stable_rate={best_stable.stable_success_rate:.2f} "
                f"tail_local={format_float(best_stable.mean_tail_local)} tail_nominal={format_float(best_stable.mean_tail_nominal)}"
            )
        else:
            lines.append("- Best stable: none")
        if picks.get("smallest_stable") is not None:
            smallest_stable = picks["smallest_stable"]
            assert smallest_stable is not None
            lines.append(
                f"- Smallest stable: `{smallest_stable.label}` local_steps={smallest_stable.local_steps} "
                f"batch={smallest_stable.batch_size} rollout={smallest_stable.rollout_size}"
            )
        else:
            lines.append("- Smallest stable: none")
        if picks.get("best_overall") is not None:
            best_overall = picks["best_overall"]
            assert best_overall is not None
            lines.append(
                f"- Best overall: `{best_overall.label}` stable_rate={best_overall.stable_success_rate:.2f} "
                f"learning_rate={best_overall.learning_success_rate:.2f}"
            )

        headers = [
            "label",
            "stable_rate",
            "learning_rate",
            "tail_local",
            "tail_nominal",
            "best_local",
            "best_nominal",
            "local_steps",
            "batch",
            "tag",
        ]
        rows: list[list[str]] = []
        for aggregate in env_aggregates[: args.top_k]:
            rows.append(
                [
                    aggregate.label,
                    format_float(aggregate.stable_success_rate, 2),
                    format_float(aggregate.learning_success_rate, 2),
                    format_float(aggregate.mean_tail_local),
                    format_float(aggregate.mean_tail_nominal),
                    format_float(aggregate.mean_best_local),
                    format_float(aggregate.mean_best_nominal),
                    str(aggregate.local_steps),
                    str(aggregate.batch_size),
                    "anchor" if aggregate.is_anchor else "search",
                ]
            )
        if rows:
            lines.extend(["", make_markdown_table(headers, rows)])
        else:
            lines.extend(["", "No candidate results were available."])

    with (output_root / "summary.md").open("w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def print_console_summary(aggregates: list[CandidateAggregate], recommendations: dict[str, dict[str, CandidateAggregate | None]], args: argparse.Namespace) -> None:
    grouped: dict[str, list[CandidateAggregate]] = defaultdict(list)
    for aggregate in aggregates:
        grouped[aggregate.env_id].append(aggregate)

    print(f"Run dir: {args.output_dir}")
    for env_id in args.envs:
        env_aggregates = rank_candidates(grouped.get(env_id, []))
        print(f"\n{env_id}")
        print(f"  candidates={len(env_aggregates)}")
        picks = recommendations.get(env_id, {})
        for key in ("best_stable", "smallest_stable", "best_overall"):
            aggregate = picks.get(key)
            if aggregate is None:
                print(f"  {key}: none")
            else:
                print(
                    f"  {key}: {aggregate.label} "
                    f"stable_rate={aggregate.stable_success_rate:.2f} "
                    f"learning_rate={aggregate.learning_success_rate:.2f} "
                    f"tail_local={format_float(aggregate.mean_tail_local)} "
                    f"tail_nominal={format_float(aggregate.mean_tail_nominal)}"
                )
        for aggregate in env_aggregates[: min(args.top_k, 5)]:
            print(
                f"    top: {aggregate.label} "
                f"stable_rate={aggregate.stable_success_rate:.2f} "
                f"learning_rate={aggregate.learning_success_rate:.2f} "
                f"tail_local={format_float(aggregate.mean_tail_local)} "
                f"tail_nominal={format_float(aggregate.mean_tail_nominal)}"
            )


def main() -> None:
    args = parse_args()
    env_configs = load_env_configs(args.config, args.envs)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_root

    trials = build_trial_specs(env_configs, args)
    write_trial_plan(trials, output_root)

    total_seed_tasks = len(trials) * len(args.seeds)
    print(
        f"Planned {len(trials)} Lunar candidates across {len(args.envs)} envs "
        f"and {len(args.seeds)} seeds ({total_seed_tasks} seed tasks total)."
    )
    print(f"Output dir: {output_root}")
    if args.dry_run:
        return

    seed_results: list[SeedTrialResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_job = {
            executor.submit(run_seed_trial, spec, seed, args, output_root): (spec.env_id, spec.label, seed)
            for spec in trials
            for seed in args.seeds
        }
        for future in as_completed(future_to_job):
            env_id, label, seed = future_to_job[future]
            try:
                result = future.result()
            except Exception as exc:  # pragma: no cover - defensive logging path
                print(f"[FAILED] env={env_id} label={label} seed={seed}: {exc}")
                continue
            seed_results.append(result)
            if result.status == "ok":
                print(
                    f"[OK] env={env_id} label={label} seed={seed} "
                    f"tail_local={format_float(result.tail_local_mean)} "
                    f"tail_nominal={format_float(result.tail_nominal_mean)} "
                    f"stable={result.stable_success} learning={result.learning_success}"
                )
            else:
                print(f"[FAILED] env={env_id} label={label} seed={seed} rc={result.returncode}")

    write_seed_results_csv(seed_results, output_root)

    grouped_seed_results: dict[tuple[str, str], list[SeedTrialResult]] = defaultdict(list)
    for result in seed_results:
        grouped_seed_results[(result.env_id, result.label)].append(result)

    aggregates = [
        aggregate_candidate(results)
        for _, results in sorted(grouped_seed_results.items())
    ]
    aggregates = rank_candidates(aggregates)

    write_candidate_summary_csv(aggregates, output_root)
    recommendations = build_recommendations(aggregates, args)
    write_recommendations(recommendations, output_root)
    write_summary_markdown(aggregates, recommendations, args, output_root)
    print_console_summary(aggregates, recommendations, args)


if __name__ == "__main__":
    main()
