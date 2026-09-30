from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
GENERIC_SEARCH_PATH = Path(__file__).with_name("ppo_avg_lunar_nominal_search.py")
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_cartpole_pendulum_nominal_search"
DEFAULT_ENV_IDS = ("PerturbCartPole-v1", "PerturbPendulum-v1")

# Thresholds are environment-specific and intentionally conservative:
# - learning_success: the run clearly learns at some point
# - stable_success: the run does not collapse by the tail of training
DEFAULT_SUCCESS_CRITERIA: dict[str, dict[str, float]] = {
    "PerturbCartPole-v1": {
        "peak_local": 475.0,
        "peak_nominal": 475.0,
        "tail_local": 450.0,
        "tail_nominal": 450.0,
    },
    "PerturbPendulum-v1": {
        "peak_local": -300.0,
        "peak_nominal": -350.0,
        "tail_local": -450.0,
        "tail_nominal": -500.0,
    },
}

# Include the current YAML layout plus nearby layouts that have either looked
# promising in prior searches or are natural low-/mid-compute neighbors.
ENV_ANCHOR_LAYOUTS: dict[str, list[dict[str, int]]] = {
    "PerturbCartPole-v1": [
        {"n_envs": 1, "n_steps": 32, "batch_size": 32, "local_steps": 32},
        {"n_envs": 1, "n_steps": 16, "batch_size": 16, "local_steps": 32},
        {"n_envs": 1, "n_steps": 16, "batch_size": 16, "local_steps": 64},
        {"n_envs": 1, "n_steps": 32, "batch_size": 32, "local_steps": 64},
        {"n_envs": 2, "n_steps": 16, "batch_size": 32, "local_steps": 64},
        {"n_envs": 4, "n_steps": 8, "batch_size": 32, "local_steps": 64},
        {"n_envs": 8, "n_steps": 4, "batch_size": 32, "local_steps": 32},
    ],
    "PerturbPendulum-v1": [
        {"n_envs": 2, "n_steps": 32, "batch_size": 64, "local_steps": 64},
        {"n_envs": 1, "n_steps": 32, "batch_size": 32, "local_steps": 32},
        {"n_envs": 1, "n_steps": 64, "batch_size": 64, "local_steps": 64},
        {"n_envs": 1, "n_steps": 64, "batch_size": 32, "local_steps": 64},
        {"n_envs": 2, "n_steps": 16, "batch_size": 32, "local_steps": 64},
        {"n_envs": 4, "n_steps": 8, "batch_size": 32, "local_steps": 64},
        {"n_envs": 4, "n_steps": 16, "batch_size": 64, "local_steps": 64},
    ],
}


def load_generic_search_module():
    spec = importlib.util.spec_from_file_location("ppo_avg_generic_nominal_search", GENERIC_SEARCH_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load generic search module from {GENERIC_SEARCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search PPOAvg hyperparameters for nominal FRL learning on "
            "PerturbCartPole-v1 and PerturbPendulum-v1. "
            "The script runs multi-seed trials, scores full learning curves from "
            "evaluations.npz, and writes ranked recommendations."
        )
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "hyperparams" / "ppo_avg.yml", help="Path to ppo_avg.yml.")
    parser.add_argument("--train-script", type=Path, default=REPO_ROOT / "rl_zoo3" / "train.py", help="Path to rl_zoo3/train.py.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Root directory for sweep logs.")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable used to launch training.")
    parser.add_argument(
        "--envs",
        nargs="*",
        default=list(DEFAULT_ENV_IDS),
        help="Target env ids. Default: CartPole and Pendulum.",
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
        default=[1.0, 2.0],
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
        default=[1, 2, 4, 8],
        help="n_envs values to search.",
    )
    parser.add_argument(
        "--n-step-values",
        nargs="*",
        type=int,
        default=[4, 8, 16, 32, 64, 128],
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
        default=[4, 8, 10, 16, 20],
        help="PPO n_epochs values to search.",
    )
    parser.add_argument(
        "--gamma-values",
        nargs="*",
        type=float,
        default=[0.9, 0.95, 0.98, 0.99],
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
        default=[0.0, 0.005, 0.01],
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
        default=10,
        help="FRL local-eval episodes per client for intermediate/final evaluation.",
    )
    parser.add_argument(
        "--eval-nominal-episodes",
        type=int,
        default=10,
        help="FRL nominal-eval episodes per client for intermediate/final evaluation.",
    )
    parser.add_argument(
        "--desired-evals",
        type=int,
        default=40,
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


def anchor_profiles_for_env(base: Any) -> list[dict[str, Any]]:
    if base.env_id == "PerturbCartPole-v1":
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
                "n_epochs": 10,
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "learning_rate_expr": "3e-4",
                "clip_range_expr": "'lin_0.2'",
                "ent_coef": 0.0,
            },
            {
                "n_epochs": 20,
                "gamma": 0.98,
                "gae_lambda": 0.8,
                "learning_rate_expr": "3e-4",
                "clip_range_expr": "'lin_0.2'",
                "ent_coef": 0.0,
            },
            {
                "n_epochs": 16,
                "gamma": 0.99,
                "gae_lambda": 0.8,
                "learning_rate_expr": "3e-4",
                "clip_range_expr": "0.1",
                "ent_coef": 0.0,
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
                "n_epochs": 10,
                "gamma": 0.9,
                "gae_lambda": 0.95,
                "learning_rate_expr": "1e-3",
                "clip_range_expr": "0.2",
                "ent_coef": 0.0,
            },
            {
                "n_epochs": 10,
                "gamma": 0.9,
                "gae_lambda": 0.95,
                "learning_rate_expr": "3e-4",
                "clip_range_expr": "0.2",
                "ent_coef": 0.0,
            },
            {
                "n_epochs": 16,
                "gamma": 0.95,
                "gae_lambda": 0.95,
                "learning_rate_expr": "1e-3",
                "clip_range_expr": "0.2",
                "ent_coef": 0.0,
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


def main() -> None:
    generic_search = load_generic_search_module()
    generic_search.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    generic_search.DEFAULT_ENV_IDS = DEFAULT_ENV_IDS
    generic_search.DEFAULT_SUCCESS_CRITERIA = DEFAULT_SUCCESS_CRITERIA
    generic_search.ENV_ANCHOR_LAYOUTS = ENV_ANCHOR_LAYOUTS
    generic_search.anchor_profiles_for_env = anchor_profiles_for_env

    args = parse_args()
    env_configs = generic_search.load_env_configs(args.config, args.envs)

    timestamp = generic_search.time.strftime("%Y%m%d_%H%M%S")
    output_root = args.output_root / timestamp
    output_root.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_root

    trials = generic_search.build_trial_specs(env_configs, args)
    generic_search.write_trial_plan(trials, output_root)

    total_seed_tasks = len(trials) * len(args.seeds)
    print(
        f"Planned {len(trials)} CartPole/Pendulum candidates across {len(args.envs)} envs "
        f"and {len(args.seeds)} seeds ({total_seed_tasks} seed tasks total)."
    )
    print(f"Output dir: {output_root}")
    if args.dry_run:
        return

    seed_results: list[generic_search.SeedTrialResult] = []
    with generic_search.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_job = {
            executor.submit(generic_search.run_seed_trial, spec, seed, args, output_root): (spec.env_id, spec.label, seed)
            for spec in trials
            for seed in args.seeds
        }
        for future in generic_search.as_completed(future_to_job):
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
                    f"tail_local={generic_search.format_float(result.tail_local_mean)} "
                    f"tail_nominal={generic_search.format_float(result.tail_nominal_mean)} "
                    f"stable={result.stable_success} learning={result.learning_success}"
                )
            else:
                print(f"[FAILED] env={env_id} label={label} seed={seed} rc={result.returncode}")

    generic_search.write_seed_results_csv(seed_results, output_root)

    grouped_seed_results: dict[tuple[str, str], list[generic_search.SeedTrialResult]] = generic_search.defaultdict(list)
    for result in seed_results:
        grouped_seed_results[(result.env_id, result.label)].append(result)

    aggregates = [
        generic_search.aggregate_candidate(results)
        for _, results in sorted(grouped_seed_results.items())
    ]
    aggregates = generic_search.rank_candidates(aggregates)

    generic_search.write_candidate_summary_csv(aggregates, output_root)
    recommendations = generic_search.build_recommendations(aggregates, args)
    generic_search.write_recommendations(recommendations, output_root)
    generic_search.write_summary_markdown(aggregates, recommendations, args, output_root)
    generic_search.print_console_summary(aggregates, recommendations, args)


if __name__ == "__main__":
    main()
