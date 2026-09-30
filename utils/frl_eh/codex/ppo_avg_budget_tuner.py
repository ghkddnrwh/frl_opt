from __future__ import annotations

import argparse
import csv
import json
import math
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
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_budget_tuning"
DEFAULT_SKIP_ENVS = {"PerturbCartPole-v1"}


@dataclass(frozen=True)
class EnvBaseConfig:
    env_id: str
    n_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    local_steps: int
    n_epochs: int

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
    n_envs: int
    n_steps: int
    batch_size: int
    local_steps: int
    n_epochs: int
    is_baseline: bool

    @property
    def rollout_size(self) -> int:
        return self.n_envs * self.n_steps

    @property
    def local_rollouts(self) -> int:
        return max(1, math.ceil(self.local_steps / max(self.rollout_size, 1)))

    @property
    def approx_minibatches(self) -> int:
        batches_per_rollout = max(1, math.ceil(self.rollout_size / max(self.batch_size, 1)))
        return self.local_rollouts * self.n_epochs * batches_per_rollout

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
        prefix = "baseline" if self.is_baseline else "search"
        return (
            f"{prefix}_envs{self.n_envs}_steps{self.n_steps}"
            f"_local{self.local_steps}_batch{self.batch_size}"
        )


@dataclass
class TrialResult:
    env_id: str
    label: str
    is_baseline: bool
    status: str
    returncode: int
    duration_sec: float
    n_timesteps: int
    n_envs: int
    n_steps: int
    batch_size: int
    local_steps: int
    local_rollouts: int
    rollout_size: int
    approx_minibatches: int
    final_local_mean: float | None
    final_nominal_mean: float | None
    score: float | None
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a budget-oriented PPOAvg hyperparameter sweep for FRL. "
            "The script starts from smaller local compute budgets and searches "
            "for configurations that still learn reasonably well."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to ppo_avg YAML config.")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=DEFAULT_TRAIN_SCRIPT,
        help="Path to rl_zoo3/train.py.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Directory where trial artifacts and summaries will be written.",
    )
    parser.add_argument(
        "--envs",
        nargs="*",
        default=None,
        help="Specific env ids to tune. Default: all PPOAvg envs except CartPole.",
    )
    parser.add_argument(
        "--include-cartpole",
        action="store_true",
        default=False,
        help="Include PerturbCartPole-v1 in the sweep.",
    )
    parser.add_argument(
        "--timesteps-multiplier",
        type=float,
        default=2.0,
        help="Multiplier applied to each env's n_timesteps from ppo_avg.yml.",
    )
    parser.add_argument(
        "--min-n-envs",
        type=int,
        default=1,
        help="Minimum n_envs value to consider during search.",
    )
    parser.add_argument(
        "--min-n-steps",
        type=int,
        default=4,
        help="Minimum n_steps value to consider during search.",
    )
    parser.add_argument(
        "--min-batch-size",
        type=int,
        default=4,
        help="Minimum batch_size value to consider during search.",
    )
    parser.add_argument(
        "--max-local-rollouts",
        type=int,
        default=8,
        help=(
            "Maximum number of PPO rollout/update cycles per local FRL round for search candidates. "
            "local_steps is set to n_envs * n_steps * local_rollouts."
        ),
    )
    parser.add_argument(
        "--max-candidates-per-env",
        type=int,
        default=48,
        help="Keep only the lowest-compute search candidates per environment (baseline is always kept).",
    )
    parser.add_argument(
        "--score-metric",
        choices=("local", "nominal", "mean"),
        default="mean",
        help="Which final evaluation metric to use for ranking candidates within each environment.",
    )
    parser.add_argument(
        "--selection-fraction",
        type=float,
        default=0.8,
        help=(
            "Recommend the smallest local-compute config whose score reaches at least this fraction "
            "of the observed score range for the same environment."
        ),
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Maximum number of concurrent training tasks.",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch rl_zoo3/train.py.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used for every trial.")
    parser.add_argument(
        "--num-threads-per-task",
        type=int,
        default=1,
        help="Value forwarded to --num-threads for each training process.",
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device forwarded to rl_zoo3/train.py.")
    parser.add_argument(
        "--eval-local-episodes",
        type=int,
        default=5,
        help="Number of local episodes used by the final FRL evaluation.",
    )
    parser.add_argument(
        "--eval-nominal-episodes",
        type=int,
        default=5,
        help="Number of nominal episodes used by the final FRL evaluation.",
    )
    parser.add_argument(
        "--eval-round-freq",
        type=int,
        default=10**9,
        help=(
            "Intermediate FRL evaluation frequency. "
            "Use a very large value to rely almost entirely on the final guaranteed evaluation."
        ),
    )
    parser.add_argument(
        "--train-verbose",
        type=int,
        default=0,
        help="Verbose level forwarded to rl_zoo3/train.py.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Only generate the candidate plan and summary files without launching training.",
    )
    return parser.parse_args()


def coerce_int(value: Any, key: str, env_id: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{env_id}: expected integer-like value for {key}, got {value!r}") from exc


def load_env_configs(config_path: Path) -> dict[str, EnvBaseConfig]:
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}

    env_configs: dict[str, EnvBaseConfig] = {}
    for env_id, values in raw.items():
        if env_id == "default":
            continue
        if not isinstance(values, dict):
            continue
        required = {"n_timesteps", "n_envs", "n_steps", "batch_size", "local_steps"}
        if not required.issubset(values.keys()):
            continue
        env_configs[env_id] = EnvBaseConfig(
            env_id=env_id,
            n_timesteps=coerce_int(values["n_timesteps"], "n_timesteps", env_id),
            n_envs=coerce_int(values["n_envs"], "n_envs", env_id),
            n_steps=coerce_int(values["n_steps"], "n_steps", env_id),
            batch_size=coerce_int(values["batch_size"], "batch_size", env_id),
            local_steps=coerce_int(values["local_steps"], "local_steps", env_id),
            n_epochs=coerce_int(values.get("n_epochs", 10), "n_epochs", env_id),
        )
    return env_configs


def ascending_halving_sequence(base_value: int, min_value: int) -> list[int]:
    if base_value < 1:
        return [1]
    min_value = max(1, min_value)
    values: set[int] = {base_value}
    current = base_value
    while current > min_value:
        current = max(min_value, current // 2)
        values.add(current)
        if current == min_value:
            break
    return sorted(values)


def batch_size_candidates(base_batch_size: int, rollout_size: int, min_batch_size: int) -> list[int]:
    if rollout_size <= 1:
        return []
    min_batch_size = max(2, min_batch_size)
    start = max(2, min(base_batch_size, rollout_size))
    values: set[int] = {rollout_size}
    current = start
    while current >= min_batch_size:
        values.add(current)
        if current == min_batch_size:
            break
        current = max(min_batch_size, current // 2)
        if current in values:
            break
    valid = sorted(value for value in values if 2 <= value <= rollout_size and rollout_size % value == 0)
    return valid or [rollout_size]


def build_baseline_spec(base: EnvBaseConfig, timesteps_multiplier: float) -> TrialSpec:
    tuned_timesteps = max(1, int(round(base.n_timesteps * timesteps_multiplier)))
    return TrialSpec(
        env_id=base.env_id,
        n_timesteps=tuned_timesteps,
        n_envs=base.n_envs,
        n_steps=base.n_steps,
        batch_size=base.batch_size,
        local_steps=base.local_steps,
        n_epochs=base.n_epochs,
        is_baseline=True,
    )


def build_search_specs(base: EnvBaseConfig, args: argparse.Namespace) -> list[TrialSpec]:
    tuned_timesteps = max(1, int(round(base.n_timesteps * args.timesteps_multiplier)))
    n_env_choices = ascending_halving_sequence(base.n_envs, args.min_n_envs)
    n_step_choices = ascending_halving_sequence(base.n_steps, args.min_n_steps)
    max_local_rollouts = min(args.max_local_rollouts, base.local_rollouts)
    local_rollout_choices = ascending_halving_sequence(max_local_rollouts, 1)

    unique_specs: dict[tuple[int, int, int, int], TrialSpec] = {}
    for n_envs in n_env_choices:
        for n_steps in n_step_choices:
            rollout_size = n_envs * n_steps
            if rollout_size <= 1:
                continue
            for batch_size in batch_size_candidates(base.batch_size, rollout_size, args.min_batch_size):
                for local_rollouts in local_rollout_choices:
                    local_steps = rollout_size * local_rollouts
                    spec = TrialSpec(
                        env_id=base.env_id,
                        n_timesteps=tuned_timesteps,
                        n_envs=n_envs,
                        n_steps=n_steps,
                        batch_size=batch_size,
                        local_steps=local_steps,
                        n_epochs=base.n_epochs,
                        is_baseline=False,
                    )
                    unique_specs[(n_envs, n_steps, batch_size, local_steps)] = spec

    specs = sorted(unique_specs.values(), key=lambda item: item.compute_sort_key)
    if args.max_candidates_per_env > 0:
        specs = specs[: args.max_candidates_per_env]
    return specs


def build_all_trials(env_configs: dict[str, EnvBaseConfig], args: argparse.Namespace) -> list[TrialSpec]:
    trials: list[TrialSpec] = []
    env_ids = sorted(env_configs.keys())
    if args.envs:
        requested = set(args.envs)
        missing = sorted(requested.difference(env_ids))
        if missing:
            raise ValueError(f"Unknown env ids requested: {missing}")
        env_ids = [env_id for env_id in env_ids if env_id in requested]
    elif not args.include_cartpole:
        env_ids = [env_id for env_id in env_ids if env_id not in DEFAULT_SKIP_ENVS]

    for env_id in env_ids:
        base = env_configs[env_id]
        trials.append(build_baseline_spec(base, args.timesteps_multiplier))
        trials.extend(build_search_specs(base, args))
    return trials


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_", ".") else "_" for char in value)


def last_finite_value(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def compute_score(final_local_mean: float | None, final_nominal_mean: float | None, mode: str) -> float | None:
    if mode == "local":
        return final_local_mean
    if mode == "nominal":
        return final_nominal_mean
    values = [value for value in (final_local_mean, final_nominal_mean) if value is not None and math.isfinite(value)]
    if not values:
        return None
    return float(sum(values) / len(values))


def find_run_dir(log_root: Path) -> Path | None:
    algo_dir = log_root / "ppo_avg"
    if not algo_dir.exists():
        return None
    run_dirs = sorted(path for path in algo_dir.iterdir() if path.is_dir())
    if not run_dirs:
        return None
    return run_dirs[-1]


def load_trial_metrics(run_dir: Path, score_metric: str) -> tuple[float | None, float | None, float | None]:
    eval_path = run_dir / "evaluations.npz"
    if not eval_path.exists():
        return None, None, None

    with np.load(eval_path, allow_pickle=True) as data:
        local_series = (
            np.asarray(data["local_mean_across_clients"], dtype=np.float64)
            if "local_mean_across_clients" in data.files
            else np.asarray([], dtype=np.float64)
        )
        nominal_series = (
            np.asarray(data["nominal_mean_across_clients"], dtype=np.float64)
            if "nominal_mean_across_clients" in data.files
            else np.asarray([], dtype=np.float64)
        )

    final_local = last_finite_value(local_series)
    final_nominal = last_finite_value(nominal_series)
    score = compute_score(final_local, final_nominal, score_metric)
    return final_local, final_nominal, score


def tail_text(path: Path, max_lines: int = 40) -> str:
    if not path.exists():
        return ""
    with path.open("r", encoding="utf-8", errors="replace") as file:
        lines = file.readlines()
    return "".join(lines[-max_lines:]).strip()


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
        "--hyperparams",
        f"n_envs:{spec.n_envs}",
        f"n_steps:{spec.n_steps}",
        f"batch_size:{spec.batch_size}",
        f"local_steps:{spec.local_steps}",
        f"eval_round_freq:{args.eval_round_freq}",
        f"eval_local_episodes:{args.eval_local_episodes}",
        f"eval_nominal_episodes:{args.eval_nominal_episodes}",
    ]


def run_trial(spec: TrialSpec, args: argparse.Namespace, output_root: Path) -> TrialResult:
    env_dir = output_root / safe_name(spec.env_id)
    trial_dir = env_dir / safe_name(spec.label)
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
    final_local_mean: float | None = None
    final_nominal_mean: float | None = None
    score: float | None = None
    error: str | None = None

    if process.returncode == 0 and run_dir is not None:
        final_local_mean, final_nominal_mean, score = load_trial_metrics(run_dir, args.score_metric)
        if score is None:
            error = "Training finished but evaluations.npz did not contain a usable final score."
    else:
        error = tail_text(log_path)

    status = "ok" if process.returncode == 0 and score is not None else "failed"
    return TrialResult(
        env_id=spec.env_id,
        label=spec.label,
        is_baseline=spec.is_baseline,
        status=status,
        returncode=process.returncode,
        duration_sec=duration_sec,
        n_timesteps=spec.n_timesteps,
        n_envs=spec.n_envs,
        n_steps=spec.n_steps,
        batch_size=spec.batch_size,
        local_steps=spec.local_steps,
        local_rollouts=spec.local_rollouts,
        rollout_size=spec.rollout_size,
        approx_minibatches=spec.approx_minibatches,
        final_local_mean=final_local_mean,
        final_nominal_mean=final_nominal_mean,
        score=score,
        run_dir=str(run_dir) if run_dir is not None else None,
        log_path=str(log_path),
        error=error,
    )


def recommend_by_env(results: list[TrialResult], selection_fraction: float) -> dict[str, TrialResult]:
    recommendations: dict[str, TrialResult] = {}
    grouped: dict[str, list[TrialResult]] = {}
    for result in results:
        grouped.setdefault(result.env_id, []).append(result)

    for env_id, env_results in grouped.items():
        successful = [result for result in env_results if result.status == "ok" and result.score is not None]
        if not successful:
            continue
        scores = [result.score for result in successful if result.score is not None]
        best_score = max(scores)
        worst_score = min(scores)
        if math.isclose(best_score, worst_score):
            threshold = best_score
        else:
            threshold = worst_score + selection_fraction * (best_score - worst_score)
        eligible = [result for result in successful if result.score is not None and result.score >= threshold]
        recommendations[env_id] = min(eligible, key=lambda item: item.compute_sort_key)
    return recommendations


def write_trial_plan(trials: list[TrialSpec], output_root: Path) -> None:
    plan_path = output_root / "trial_plan.json"
    rows = []
    for spec in trials:
        rows.append(
            {
                "env_id": spec.env_id,
                "label": spec.label,
                "is_baseline": spec.is_baseline,
                "n_timesteps": spec.n_timesteps,
                "n_envs": spec.n_envs,
                "n_steps": spec.n_steps,
                "batch_size": spec.batch_size,
                "local_steps": spec.local_steps,
                "local_rollouts": spec.local_rollouts,
                "rollout_size": spec.rollout_size,
                "approx_minibatches": spec.approx_minibatches,
            }
        )
    with plan_path.open("w", encoding="utf-8") as file:
        json.dump(rows, file, indent=2)


def write_results_csv(results: list[TrialResult], output_root: Path) -> None:
    csv_path = output_root / "results.csv"
    fieldnames = [
        "env_id",
        "label",
        "is_baseline",
        "status",
        "returncode",
        "duration_sec",
        "n_timesteps",
        "n_envs",
        "n_steps",
        "batch_size",
        "local_steps",
        "local_rollouts",
        "rollout_size",
        "approx_minibatches",
        "final_local_mean",
        "final_nominal_mean",
        "score",
        "run_dir",
        "log_path",
        "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))


def write_recommendations(
    recommendations: dict[str, TrialResult],
    env_configs: dict[str, EnvBaseConfig],
    args: argparse.Namespace,
    output_root: Path,
) -> None:
    json_path = output_root / "recommendations.json"
    yaml_path = output_root / "recommended_overrides.yml"

    payload: dict[str, Any] = {}
    yaml_payload: dict[str, Any] = {}
    for env_id, result in sorted(recommendations.items()):
        base = env_configs[env_id]
        payload[env_id] = {
            "recommended": asdict(result),
            "baseline": {
                "n_timesteps": int(round(base.n_timesteps * args.timesteps_multiplier)),
                "n_envs": base.n_envs,
                "n_steps": base.n_steps,
                "batch_size": base.batch_size,
                "local_steps": base.local_steps,
                "rollout_size": base.rollout_size,
                "local_rollouts": base.local_rollouts,
            },
        }
        yaml_payload[env_id] = {
            "n_timesteps": result.n_timesteps,
            "n_envs": result.n_envs,
            "n_steps": result.n_steps,
            "batch_size": result.batch_size,
            "local_steps": result.local_steps,
        }

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
    with yaml_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(yaml_payload, file, sort_keys=False)


def print_search_summary(trials: list[TrialSpec]) -> None:
    grouped: dict[str, list[TrialSpec]] = {}
    for trial in trials:
        grouped.setdefault(trial.env_id, []).append(trial)

    print("Planned trial counts:")
    for env_id, env_trials in sorted(grouped.items()):
        baseline_count = sum(1 for trial in env_trials if trial.is_baseline)
        search_count = len(env_trials) - baseline_count
        smallest = min(env_trials, key=lambda item: item.compute_sort_key)
        largest = max(env_trials, key=lambda item: item.compute_sort_key)
        print(
            f"  {env_id}: {len(env_trials)} trials "
            f"(baseline={baseline_count}, search={search_count}), "
            f"local_steps {smallest.local_steps}->{largest.local_steps}, "
            f"rollout {smallest.rollout_size}->{largest.rollout_size}"
        )


def print_result_line(result: TrialResult) -> None:
    score_text = "nan" if result.score is None else f"{result.score:.3f}"
    print(
        f"[{result.status.upper():6}] {result.env_id} {result.label} "
        f"score={score_text} local_steps={result.local_steps} "
        f"rollout={result.rollout_size} batch={result.batch_size} "
        f"time={result.duration_sec:.1f}s"
    )


def print_recommendations(recommendations: dict[str, TrialResult]) -> None:
    if not recommendations:
        print("No successful recommendations were produced.")
        return
    print("\nRecommended low-budget configs:")
    for env_id, result in sorted(recommendations.items()):
        score_text = "nan" if result.score is None else f"{result.score:.3f}"
        print(
            f"  {env_id}: n_envs={result.n_envs}, n_steps={result.n_steps}, "
            f"local_steps={result.local_steps}, batch_size={result.batch_size}, "
            f"score={score_text}"
        )


def main() -> None:
    args = parse_args()
    args.config = args.config.resolve()
    args.train_script = args.train_script.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    args.output_root = args.output_root.resolve() / timestamp
    args.output_root.mkdir(parents=True, exist_ok=True)

    env_configs = load_env_configs(args.config)
    trials = build_all_trials(env_configs, args)
    if not trials:
        raise ValueError("No trials were generated. Check your env filters and config file.")

    write_trial_plan(trials, args.output_root)
    print(f"Output root: {args.output_root}")
    print_search_summary(trials)

    if args.dry_run:
        print("\nDry run only. Trial plan written to trial_plan.json.")
        return

    results: list[TrialResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_trial = {
            executor.submit(run_trial, trial, args, args.output_root): trial
            for trial in trials
        }
        for future in as_completed(future_to_trial):
            result = future.result()
            results.append(result)
            print_result_line(result)

    results.sort(key=lambda item: (item.env_id, item.compute_sort_key, item.label))
    write_results_csv(results, args.output_root)

    recommendations = recommend_by_env(results, args.selection_fraction)
    write_recommendations(recommendations, env_configs, args, args.output_root)
    print_recommendations(recommendations)


if __name__ == "__main__":
    main()
