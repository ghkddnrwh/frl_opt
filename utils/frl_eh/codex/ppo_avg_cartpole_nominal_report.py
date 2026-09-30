from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_cartpole_nominal_search"


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
    error: str | None

    @property
    def is_success(self) -> bool:
        return self.status == "ok" and self.ranking_score is not None

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
            "Summarize CartPole nominal PPOAvg search results and make it easier "
            "to choose final hyperparameters."
        )
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help="Root directory containing timestamped CartPole nominal search runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific run directory to summarize. Default: latest run with results.csv.",
    )
    parser.add_argument(
        "--near-best-fraction",
        type=float,
        default=0.95,
        help="Threshold for near-best tables when recommendations.json is unavailable.",
    )
    parser.add_argument("--top-score-k", type=int, default=10, help="How many top-score trials to show.")
    parser.add_argument(
        "--top-solved-compute-k",
        type=int,
        default=10,
        help="How many solved trials sorted by smallest compute to show.",
    )
    parser.add_argument(
        "--top-near-best-compute-k",
        type=int,
        default=10,
        help="How many near-best trials sorted by smallest compute to show.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        default=False,
        help="Include failed trials in the markdown report.",
    )
    return parser.parse_args()


def latest_run_dir(sweep_root: Path) -> Path:
    candidates = sorted(
        path
        for path in sweep_root.iterdir()
        if path.is_dir() and (path / "results.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No completed CartPole search run with results.csv was found under {sweep_root}")
    return candidates[-1]


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def parse_optional_float(value: str) -> float | None:
    text = value.strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def parse_optional_str(value: str) -> str | None:
    text = value.strip()
    return None if text == "" else text


def load_results(results_path: Path) -> list[TrialResult]:
    results: list[TrialResult] = []
    with results_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            results.append(
                TrialResult(
                    env_id=row["env_id"],
                    label=row["label"],
                    is_anchor=parse_bool(row["is_anchor"]),
                    status=row["status"],
                    returncode=int(row["returncode"]),
                    duration_sec=float(row["duration_sec"]),
                    n_timesteps=int(float(row["n_timesteps"])),
                    num_clients=int(float(row["num_clients"])),
                    server_update_weight=float(row["server_update_weight"]),
                    n_envs=int(float(row["n_envs"])),
                    n_steps=int(float(row["n_steps"])),
                    batch_size=int(float(row["batch_size"])),
                    local_rollouts=int(float(row["local_rollouts"])),
                    local_steps=int(float(row["local_steps"])),
                    rollout_size=int(float(row["rollout_size"])),
                    approx_minibatches=int(float(row["approx_minibatches"])),
                    n_epochs=int(float(row["n_epochs"])),
                    gamma=float(row["gamma"]),
                    gae_lambda=float(row["gae_lambda"]),
                    learning_rate_expr=row["learning_rate_expr"],
                    clip_range_expr=row["clip_range_expr"],
                    final_local_mean=parse_optional_float(row["final_local_mean"]),
                    final_local_min=parse_optional_float(row["final_local_min"]),
                    final_local_max=parse_optional_float(row["final_local_max"]),
                    final_nominal_mean=parse_optional_float(row["final_nominal_mean"]),
                    final_nominal_min=parse_optional_float(row["final_nominal_min"]),
                    ranking_score=parse_optional_float(row["ranking_score"]),
                    solved=parse_bool(row["solved"]),
                    run_dir=parse_optional_str(row["run_dir"]),
                    log_path=row["log_path"],
                    error=parse_optional_str(row.get("error", "")),
                )
            )
    return results


def load_recommendations(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_ratio(current: int | float, baseline: int | float) -> str:
    if baseline == 0:
        return "n/a"
    return f"{float(current) / float(baseline):.4f}x"


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def top_by_score(results: list[TrialResult], top_k: int) -> list[TrialResult]:
    successful = [result for result in results if result.is_success]
    successful.sort(
        key=lambda item: (
            -(item.ranking_score if item.ranking_score is not None else -float("inf")),
            item.compute_sort_key,
        )
    )
    return successful[:top_k]


def solved_by_compute(results: list[TrialResult], top_k: int) -> list[TrialResult]:
    solved = [result for result in results if result.is_success and result.solved]
    solved.sort(key=lambda item: (item.compute_sort_key, -(item.ranking_score or -float("inf"))))
    return solved[:top_k]


def near_best_by_compute(results: list[TrialResult], near_best_fraction: float, top_k: int) -> list[TrialResult]:
    successful = [result for result in results if result.is_success]
    if not successful:
        return []
    best_score = max(result.ranking_score for result in successful if result.ranking_score is not None)
    threshold = best_score * near_best_fraction
    eligible = [
        result
        for result in successful
        if result.ranking_score is not None and result.ranking_score >= threshold
    ]
    eligible.sort(key=lambda item: (item.compute_sort_key, -(item.ranking_score or -float("inf"))))
    return eligible[:top_k]


def find_baseline(results: list[TrialResult]) -> TrialResult | None:
    baselines = [result for result in results if result.is_anchor and result.label.startswith("anchor")]
    if not baselines:
        return None
    return min(baselines, key=lambda item: item.compute_sort_key)


def pick_reference(recommendations: dict[str, Any], key: str) -> dict[str, Any] | None:
    payload = recommendations.get(key)
    return payload if isinstance(payload, dict) else None


def write_summary_csv(path: Path, results: list[TrialResult]) -> None:
    fieldnames = [
        "label",
        "is_anchor",
        "status",
        "solved",
        "ranking_score",
        "final_local_mean",
        "final_local_min",
        "final_nominal_mean",
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
        "duration_sec",
        "run_dir",
        "log_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "label": result.label,
                    "is_anchor": result.is_anchor,
                    "status": result.status,
                    "solved": result.solved,
                    "ranking_score": result.ranking_score,
                    "final_local_mean": result.final_local_mean,
                    "final_local_min": result.final_local_min,
                    "final_nominal_mean": result.final_nominal_mean,
                    "n_timesteps": result.n_timesteps,
                    "num_clients": result.num_clients,
                    "server_update_weight": result.server_update_weight,
                    "n_envs": result.n_envs,
                    "n_steps": result.n_steps,
                    "batch_size": result.batch_size,
                    "local_rollouts": result.local_rollouts,
                    "local_steps": result.local_steps,
                    "rollout_size": result.rollout_size,
                    "approx_minibatches": result.approx_minibatches,
                    "n_epochs": result.n_epochs,
                    "gamma": result.gamma,
                    "gae_lambda": result.gae_lambda,
                    "learning_rate_expr": result.learning_rate_expr,
                    "clip_range_expr": result.clip_range_expr,
                    "duration_sec": result.duration_sec,
                    "run_dir": result.run_dir,
                    "log_path": result.log_path,
                }
            )


def build_summary_markdown(
    run_dir: Path,
    results: list[TrialResult],
    recommendations: dict[str, Any],
    args: argparse.Namespace,
) -> str:
    successful = [result for result in results if result.is_success]
    solved = [result for result in successful if result.solved]
    failed = [result for result in results if not result.is_success]
    baseline = find_baseline(results)

    lines = [
        "# PPOAvg CartPole Nominal Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Trials: total={len(results)}, success={len(successful)}, solved={len(solved)}, failed={len(failed)}",
        f"- Near-best fraction: {args.near_best_fraction}",
        "",
    ]

    lines.append("## Key Picks")
    lines.append("")
    key_rows: list[list[str]] = []
    for key in ("best_score_trial", "smallest_solved_trial", "smallest_near_best_trial"):
        payload = pick_reference(recommendations, key)
        if payload is None:
            key_rows.append([key, "n/a", "n/a", "n/a", "n/a", "n/a", "n/a"])
            continue
        key_rows.append(
            [
                key,
                str(payload.get("label", "n/a")),
                format_float(payload.get("ranking_score")),
                format_float(payload.get("final_local_mean")),
                format_float(payload.get("final_local_min")),
                str(payload.get("local_steps", "n/a")),
                str(payload.get("approx_minibatches", "n/a")),
            ]
        )
    lines.append(
        make_markdown_table(
            ["pick", "label", "score", "local_mean", "local_min", "local_steps", "minibatches"],
            key_rows,
        )
    )
    lines.append("")

    if baseline is not None:
        lines.append("## Baseline Reference")
        lines.append("")
        lines.append(
            make_markdown_table(
                ["label", "score", "local_mean", "local_min", "local_steps", "rollout", "minibatches"],
                [[
                    baseline.label,
                    format_float(baseline.ranking_score),
                    format_float(baseline.final_local_mean),
                    format_float(baseline.final_local_min),
                    str(baseline.local_steps),
                    str(baseline.rollout_size),
                    str(baseline.approx_minibatches),
                ]],
            )
        )
        lines.append("")

    top_score_rows = [
        [
            result.label,
            format_float(result.ranking_score),
            format_float(result.final_local_mean),
            format_float(result.final_local_min),
            format_float(result.final_nominal_mean),
            str(result.local_steps),
            str(result.rollout_size),
            str(result.batch_size),
            str(result.approx_minibatches),
            "anchor" if result.is_anchor else "search",
        ]
        for result in top_by_score(results, args.top_score_k)
    ]
    lines.append("## Top Score Trials")
    lines.append("")
    if top_score_rows:
        lines.append(
            make_markdown_table(
                [
                    "label",
                    "score",
                    "local_mean",
                    "local_min",
                    "nominal_mean",
                    "local_steps",
                    "rollout",
                    "batch",
                    "minibatches",
                    "tag",
                ],
                top_score_rows,
            )
        )
    else:
        lines.append("No successful trials.")
    lines.append("")

    solved_rows = [
        [
            result.label,
            format_float(result.ranking_score),
            format_float(result.final_local_mean),
            format_float(result.final_local_min),
            str(result.local_steps),
            str(result.rollout_size),
            str(result.batch_size),
            str(result.approx_minibatches),
            format_ratio(result.local_steps, baseline.local_steps) if baseline is not None else "n/a",
        ]
        for result in solved_by_compute(results, args.top_solved_compute_k)
    ]
    lines.append("## Solved Trials Sorted By Compute")
    lines.append("")
    if solved_rows:
        lines.append(
            make_markdown_table(
                [
                    "label",
                    "score",
                    "local_mean",
                    "local_min",
                    "local_steps",
                    "rollout",
                    "batch",
                    "minibatches",
                    "steps_vs_baseline",
                ],
                solved_rows,
            )
        )
    else:
        lines.append("No solved trial was found.")
    lines.append("")

    near_best_rows = [
        [
            result.label,
            format_float(result.ranking_score),
            format_float(result.final_local_mean),
            format_float(result.final_local_min),
            str(result.local_steps),
            str(result.rollout_size),
            str(result.batch_size),
            str(result.approx_minibatches),
            "anchor" if result.is_anchor else "search",
        ]
        for result in near_best_by_compute(results, args.near_best_fraction, args.top_near_best_compute_k)
    ]
    lines.append("## Near-Best Trials Sorted By Compute")
    lines.append("")
    if near_best_rows:
        lines.append(
            make_markdown_table(
                [
                    "label",
                    "score",
                    "local_mean",
                    "local_min",
                    "local_steps",
                    "rollout",
                    "batch",
                    "minibatches",
                    "tag",
                ],
                near_best_rows,
            )
        )
    else:
        lines.append("No near-best trial was found.")
    lines.append("")

    if args.show_failures and failed:
        lines.append("## Failed Trials")
        lines.append("")
        for result in failed:
            lines.append(f"- `{result.label}` returncode={result.returncode} log={result.log_path}")
        lines.append("")

    return "\n".join(lines)


def print_console_summary(run_dir: Path, results: list[TrialResult], recommendations: dict[str, Any], args: argparse.Namespace) -> None:
    successful = [result for result in results if result.is_success]
    solved = [result for result in successful if result.solved]
    print(f"Run dir: {run_dir}")
    print(f"Trials: total={len(results)}, success={len(successful)}, solved={len(solved)}")
    print("")
    for key in ("best_score_trial", "smallest_solved_trial", "smallest_near_best_trial"):
        payload = pick_reference(recommendations, key)
        if payload is None:
            print(f"{key}: n/a")
            continue
        print(
            f"{key}: score={format_float(payload.get('ranking_score'))} "
            f"local_mean={format_float(payload.get('final_local_mean'))} "
            f"local_min={format_float(payload.get('final_local_min'))} "
            f"local_steps={payload.get('local_steps')} "
            f"rollout={payload.get('rollout_size')} "
            f"batch={payload.get('batch_size')} "
            f"label={payload.get('label')}"
        )
    print("")
    print("Top score trials:")
    for result in top_by_score(results, args.top_score_k)[: min(5, args.top_score_k)]:
        print(
            f"  {result.label} score={format_float(result.ranking_score)} "
            f"local_mean={format_float(result.final_local_mean)} "
            f"local_min={format_float(result.final_local_min)} "
            f"local_steps={result.local_steps}"
        )
    print("")
    print("Smallest solved trials:")
    solved_list = solved_by_compute(results, args.top_solved_compute_k)
    if solved_list:
        for result in solved_list[: min(5, args.top_solved_compute_k)]:
            print(
                f"  {result.label} score={format_float(result.ranking_score)} "
                f"local_steps={result.local_steps} rollout={result.rollout_size} "
                f"batch={result.batch_size} minibatches={result.approx_minibatches}"
            )
    else:
        print("  No solved trial.")


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else latest_run_dir(sweep_root)
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found in {run_dir}")

    results = load_results(results_path)
    recommendations = load_recommendations(run_dir / "recommendations.json")

    summary_md_path = run_dir / "report.md"
    summary_csv_path = run_dir / "report_sorted_by_score.csv"
    summary_md = build_summary_markdown(run_dir, results, recommendations, args)
    summary_md_path.write_text(summary_md + "\n", encoding="utf-8")

    sorted_results = top_by_score(results, len(results))
    write_summary_csv(summary_csv_path, sorted_results)

    print_console_summary(run_dir, results, recommendations, args)
    print(f"Markdown report written to: {summary_md_path}")
    print(f"CSV report written to: {summary_csv_path}")


if __name__ == "__main__":
    main()
