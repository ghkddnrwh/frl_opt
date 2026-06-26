from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_budget_tuning"


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
    error: str | None

    @property
    def is_success(self) -> bool:
        return self.status == "ok" and self.score is not None

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
            "Summarize PPOAvg budget tuning results. "
            "By default, the latest run under logs/codex/ppo_avg_budget_tuning is used."
        )
    )
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=DEFAULT_SWEEP_ROOT,
        help="Root directory containing timestamped budget tuning runs.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Specific timestamped run directory to summarize.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of top successful candidates to include per environment.",
    )
    parser.add_argument(
        "--show-failures",
        action="store_true",
        default=False,
        help="Include failed trial counts and a short failed-trial list in the markdown report.",
    )
    parser.add_argument(
        "--print-top-k",
        type=int,
        default=10,
        help="Number of top candidates to print per environment in the console summary.",
    )
    return parser.parse_args()


def latest_run_dir(sweep_root: Path) -> Path:
    candidates = sorted(
        path
        for path in sweep_root.iterdir()
        if path.is_dir() and (path / "results.csv").exists()
    )
    if not candidates:
        raise FileNotFoundError(f"No completed sweep run with results.csv was found under {sweep_root}")
    return candidates[-1]


def parse_optional_float(value: str) -> float | None:
    text = value.strip()
    if text == "" or text.lower() == "none":
        return None
    return float(text)


def parse_optional_str(value: str) -> str | None:
    text = value.strip()
    return None if text == "" else text


def parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def load_results(results_path: Path) -> list[TrialResult]:
    rows: list[TrialResult] = []
    with results_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append(
                TrialResult(
                    env_id=row["env_id"],
                    label=row["label"],
                    is_baseline=parse_bool(row["is_baseline"]),
                    status=row["status"],
                    returncode=int(row["returncode"]),
                    duration_sec=float(row["duration_sec"]),
                    n_timesteps=int(float(row["n_timesteps"])),
                    n_envs=int(float(row["n_envs"])),
                    n_steps=int(float(row["n_steps"])),
                    batch_size=int(float(row["batch_size"])),
                    local_steps=int(float(row["local_steps"])),
                    local_rollouts=int(float(row["local_rollouts"])),
                    rollout_size=int(float(row["rollout_size"])),
                    approx_minibatches=int(float(row["approx_minibatches"])),
                    final_local_mean=parse_optional_float(row["final_local_mean"]),
                    final_nominal_mean=parse_optional_float(row["final_nominal_mean"]),
                    score=parse_optional_float(row["score"]),
                    run_dir=parse_optional_str(row["run_dir"]),
                    log_path=row["log_path"],
                    error=parse_optional_str(row.get("error", "")),
                )
            )
    return rows


def load_recommendations(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None or not math.isfinite(value):
        return "n/a"
    return f"{value:.{digits}f}"


def format_ratio(numerator: int | float, denominator: int | float) -> str:
    if denominator == 0:
        return "n/a"
    ratio = float(numerator) / float(denominator)
    return f"{ratio:.4f}x"


def format_reduction(current: int | float, baseline: int | float) -> str:
    if baseline == 0:
        return "n/a"
    ratio = float(current) / float(baseline)
    reduction = 100.0 * (1.0 - ratio)
    return f"{reduction:.1f}%"


def best_by_score(results: list[TrialResult]) -> TrialResult | None:
    successful = [result for result in results if result.is_success]
    if not successful:
        return None
    return max(successful, key=lambda item: (item.score, tuple(-value for value in item.compute_sort_key)))  # type: ignore[arg-type]


def sort_top_candidates(results: list[TrialResult], top_k: int) -> list[TrialResult]:
    successful = [result for result in results if result.is_success]
    successful.sort(
        key=lambda item: (
            -(item.score if item.score is not None else -float("inf")),
            item.compute_sort_key,
        )
    )
    return successful[:top_k]


def lookup_recommended(env_id: str, env_results: list[TrialResult], recommendations: dict[str, Any]) -> TrialResult | None:
    payload = recommendations.get(env_id)
    if not payload:
        return None
    recommended_label = payload.get("recommended", {}).get("label")
    if recommended_label is None:
        return None
    for result in env_results:
        if result.label == recommended_label:
            return result
    return None


def group_by_env(results: list[TrialResult]) -> dict[str, list[TrialResult]]:
    grouped: dict[str, list[TrialResult]] = {}
    for result in results:
        grouped.setdefault(result.env_id, []).append(result)
    return grouped


def make_markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def build_env_summary_rows(
    env_id: str,
    env_results: list[TrialResult],
    recommendations: dict[str, Any],
    top_k: int,
    show_failures: bool,
) -> tuple[list[str], dict[str, str]]:
    baseline = next((result for result in env_results if result.is_baseline), None)
    recommended = lookup_recommended(env_id, env_results, recommendations)
    best = best_by_score(env_results)
    successful = [result for result in env_results if result.is_success]
    failed = [result for result in env_results if not result.is_success]

    lines = [f"## {env_id}"]
    lines.append(f"- Trials: total={len(env_results)}, success={len(successful)}, failed={len(failed)}")

    if baseline is not None:
        lines.append(
            "- Baseline: "
            f"`{baseline.label}` score={format_float(baseline.score)} "
            f"local_steps={baseline.local_steps} rollout={baseline.rollout_size} "
            f"batch={baseline.batch_size} minibatches={baseline.approx_minibatches}"
        )
    else:
        lines.append("- Baseline: not found in results.csv")

    if best is not None:
        lines.append(
            "- Best score: "
            f"`{best.label}` score={format_float(best.score)} "
            f"local={format_float(best.final_local_mean)} nominal={format_float(best.final_nominal_mean)} "
            f"local_steps={best.local_steps} rollout={best.rollout_size} batch={best.batch_size}"
        )
    else:
        lines.append("- Best score: no successful trial")

    if recommended is not None and baseline is not None:
        lines.append(
            "- Recommended low-budget: "
            f"`{recommended.label}` score={format_float(recommended.score)} "
            f"score_delta_vs_baseline={format_float((recommended.score or 0.0) - (baseline.score or 0.0))} "
            f"local_steps_reduction={format_reduction(recommended.local_steps, baseline.local_steps)} "
            f"rollout_reduction={format_reduction(recommended.rollout_size, baseline.rollout_size)} "
            f"minibatch_reduction={format_reduction(recommended.approx_minibatches, baseline.approx_minibatches)}"
        )
    elif recommended is not None:
        lines.append(
            "- Recommended low-budget: "
            f"`{recommended.label}` score={format_float(recommended.score)} "
            f"local_steps={recommended.local_steps} rollout={recommended.rollout_size}"
        )
    else:
        lines.append("- Recommended low-budget: not found")

    top_rows: list[list[str]] = []
    for result in sort_top_candidates(env_results, top_k):
        marker = []
        if result.is_baseline:
            marker.append("baseline")
        if recommended is not None and result.label == recommended.label:
            marker.append("recommended")
        tag = ",".join(marker) if marker else ""
        top_rows.append(
            [
                result.label,
                format_float(result.score),
                format_float(result.final_local_mean),
                format_float(result.final_nominal_mean),
                str(result.local_steps),
                str(result.rollout_size),
                str(result.batch_size),
                str(result.approx_minibatches),
                f"{result.duration_sec / 60.0:.1f}",
                tag,
            ]
        )

    if top_rows:
        lines.append("")
        lines.append(
            make_markdown_table(
                [
                    "label",
                    "score",
                    "local",
                    "nominal",
                    "local_steps",
                    "rollout",
                    "batch",
                    "minibatches",
                    "minutes",
                    "tag",
                ],
                top_rows,
            )
        )

    if show_failures and failed:
        lines.append("")
        lines.append("Failed trials:")
        for result in failed[: min(5, len(failed))]:
            lines.append(f"- `{result.label}` returncode={result.returncode} log={result.log_path}")

    summary_row = {
        "env_id": env_id,
        "num_trials": str(len(env_results)),
        "num_success": str(len(successful)),
        "num_failed": str(len(failed)),
        "baseline_label": baseline.label if baseline is not None else "",
        "baseline_score": format_float(baseline.score) if baseline is not None else "n/a",
        "baseline_local_steps": str(baseline.local_steps) if baseline is not None else "",
        "baseline_rollout": str(baseline.rollout_size) if baseline is not None else "",
        "baseline_minibatches": str(baseline.approx_minibatches) if baseline is not None else "",
        "best_label": best.label if best is not None else "",
        "best_score": format_float(best.score) if best is not None else "n/a",
        "best_local_steps": str(best.local_steps) if best is not None else "",
        "best_rollout": str(best.rollout_size) if best is not None else "",
        "best_minibatches": str(best.approx_minibatches) if best is not None else "",
        "recommended_label": recommended.label if recommended is not None else "",
        "recommended_score": format_float(recommended.score) if recommended is not None else "n/a",
        "recommended_local_steps": str(recommended.local_steps) if recommended is not None else "",
        "recommended_rollout": str(recommended.rollout_size) if recommended is not None else "",
        "recommended_minibatches": str(recommended.approx_minibatches) if recommended is not None else "",
        "recommended_vs_baseline_score_delta": (
            format_float((recommended.score or 0.0) - (baseline.score or 0.0))
            if recommended is not None and baseline is not None
            else "n/a"
        ),
        "recommended_vs_baseline_local_steps_ratio": (
            format_ratio(recommended.local_steps, baseline.local_steps)
            if recommended is not None and baseline is not None
            else "n/a"
        ),
        "recommended_vs_baseline_rollout_ratio": (
            format_ratio(recommended.rollout_size, baseline.rollout_size)
            if recommended is not None and baseline is not None
            else "n/a"
        ),
        "recommended_vs_baseline_minibatches_ratio": (
            format_ratio(recommended.approx_minibatches, baseline.approx_minibatches)
            if recommended is not None and baseline is not None
            else "n/a"
        ),
    }
    return lines, summary_row


def write_env_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_console_summary(
    run_dir: Path,
    grouped: dict[str, list[TrialResult]],
    recommendations: dict[str, Any],
    print_top_k: int,
) -> None:
    print(f"Run dir: {run_dir}")
    total_trials = sum(len(results) for results in grouped.values())
    total_success = sum(sum(1 for result in results if result.is_success) for results in grouped.values())
    print(f"Envs: {len(grouped)}, trials: {total_trials}, successful trials: {total_success}")
    print("")
    for env_id in sorted(grouped.keys()):
        env_results = grouped[env_id]
        baseline = next((result for result in env_results if result.is_baseline), None)
        recommended = lookup_recommended(env_id, env_results, recommendations)
        best = best_by_score(env_results)
        print(env_id)
        if baseline is not None:
            print(
                f"  baseline    score={format_float(baseline.score):>8} "
                f"local_steps={baseline.local_steps:>6} rollout={baseline.rollout_size:>6} "
                f"batch={baseline.batch_size:>4}"
            )
        if recommended is not None:
            print(
                f"  recommended score={format_float(recommended.score):>8} "
                f"local_steps={recommended.local_steps:>6} rollout={recommended.rollout_size:>6} "
                f"batch={recommended.batch_size:>4}"
            )
        if best is not None:
            print(
                f"  best        score={format_float(best.score):>8} "
                f"local_steps={best.local_steps:>6} rollout={best.rollout_size:>6} "
                f"batch={best.batch_size:>4}"
            )
        top_results = sort_top_candidates(env_results, print_top_k)
        for idx, result in enumerate(top_results, start=1):
            print(
                f"    top{idx}: {result.label} score={format_float(result.score)} "
                f"local={format_float(result.final_local_mean)} nominal={format_float(result.final_nominal_mean)}"
            )
        print("")


def main() -> None:
    args = parse_args()
    sweep_root = args.sweep_root.resolve()
    run_dir = args.run_dir.resolve() if args.run_dir is not None else latest_run_dir(sweep_root)
    results_path = run_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found in {run_dir}")

    results = load_results(results_path)
    recommendations = load_recommendations(run_dir / "recommendations.json")
    grouped = group_by_env(results)

    markdown_lines = [
        "# PPOAvg Budget Tuning Summary",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Environments: {len(grouped)}",
        f"- Total trials: {len(results)}",
        f"- Successful trials: {sum(1 for result in results if result.is_success)}",
        f"- Failed trials: {sum(1 for result in results if not result.is_success)}",
        "",
    ]

    summary_rows: list[dict[str, str]] = []
    for env_id in sorted(grouped.keys()):
        env_lines, summary_row = build_env_summary_rows(
            env_id=env_id,
            env_results=grouped[env_id],
            recommendations=recommendations,
            top_k=args.top_k,
            show_failures=args.show_failures,
        )
        markdown_lines.extend(env_lines)
        markdown_lines.append("")
        summary_rows.append(summary_row)

    summary_md_path = run_dir / "summary.md"
    summary_csv_path = run_dir / "env_summary.csv"
    summary_md_path.write_text("\n".join(markdown_lines).strip() + "\n", encoding="utf-8")
    write_env_summary_csv(summary_csv_path, summary_rows)

    print_console_summary(run_dir, grouped, recommendations, args.print_top_k)
    print(f"Markdown summary written to: {summary_md_path}")
    print(f"CSV summary written to: {summary_csv_path}")


if __name__ == "__main__":
    main()
