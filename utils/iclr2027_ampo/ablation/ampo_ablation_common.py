from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Experiment layout
# -----------------------------------------------------------------------------
DEFAULT_LOG_ROOT = Path("logs/wandb_logs_final_160")
DEFAULT_OUT_ROOT = Path("logs/iclr2027_ampo/ablation")

ENV_ORDER = ("ant", "halfcheetah", "hopper", "walker2d")
PERTURBATION_ORDER = ("friction", "gravity")
ADAPTIVE_VARIANTS = ("ampo_adaptive_dual_lr_1e-4", "ampo_adaptive_dual_lr_3e-4")
ALL_VARIANTS = ("ppo_avg", "ampo_uniform", *ADAPTIVE_VARIANTS)

DISPLAY_ENV = {
    "ant": "Ant",
    "halfcheetah": "HalfCheetah",
    "hopper": "Hopper",
    "walker2d": "Walker2d",
}
DISPLAY_VARIANT = {
    "ppo_avg": "PPOAvg",
    "ampo_uniform": "AMPO-Uniform",
    "ampo_adaptive_dual_lr_1e-4": r"AMPO-Adaptive ($10^{-4}$)",
    "ampo_adaptive_dual_lr_3e-4": r"AMPO-Adaptive ($3\\times10^{-4}$)",
}
SHORT_VARIANT = {
    "ppo_avg": "PPOAvg",
    "ampo_uniform": "Uniform",
    "ampo_adaptive_dual_lr_1e-4": "Adaptive 1e-4",
    "ampo_adaptive_dual_lr_3e-4": "Adaptive 3e-4",
}

SERVER_SCALAR_KEYS = (
    "effective_clients",
    "lambda_entropy",
    "lambda_max",
    "lambda_min",
    "lambda_delta_norm",
    "lambda_num_at_cap",
    "lambda_support_size",
    "pairwise_grad_conflict_rate",
    "pairwise_grad_cosine_mean",
    "pairwise_grad_cosine_min",
    "high_lambda_conflict_rate",
    "top_lambda_grad_cosine_to_aggregate",
    "worst_return_grad_cosine_to_aggregate",
    "lambda_influence_l1_gap",
    "max_influence_is_worst_return",
    "max_influence_share",
    "dual_update_applied",
    "dual_update_due",
    "dual_scale_value",
    "dual_current_return_spread",
    "worst_group_cap_active",
    "worst_group_lambda_cap",
    "worst_group_min_support",
)


# -----------------------------------------------------------------------------
# Publication plotting style: matched to the user's reference plotting code.
# -----------------------------------------------------------------------------
def apply_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 15,
            "axes.labelsize": 15,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 11,
            "axes.unicode_minus": False,
        }
    )


def save_figure(fig: plt.Figure, stem: Path, dpi: int = 300) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    print(f"[Saved] {stem.with_suffix('.png')}")


# -----------------------------------------------------------------------------
# Run discovery and compact cache
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class RunSpec:
    variant: str
    env: str
    perturbation: str
    seed: int
    run_dir: Path


def _parse_seed(run_dir: Path) -> int:
    match = re.match(r"seed_(\d+)(?:__.*)?$", run_dir.name)
    if not match:
        raise ValueError(f"Cannot parse seed from run directory: {run_dir}")
    return int(match.group(1))


def discover_runs(log_root: Path, variants: Sequence[str] | None = None) -> list[RunSpec]:
    log_root = Path(log_root)
    if not log_root.exists():
        raise FileNotFoundError(
            f"Log root does not exist: {log_root}\n"
            "Expected the extracted directory logs/wandb_logs_final_160."
        )
    variants = tuple(variants or ALL_VARIANTS)
    runs: list[RunSpec] = []
    for variant in variants:
        variant_dir = log_root / variant
        if not variant_dir.exists():
            continue
        for env in ENV_ORDER:
            for perturbation in PERTURBATION_ORDER:
                parent = variant_dir / env / perturbation
                if not parent.exists():
                    continue
                for run_dir in sorted(parent.glob("seed_*")):
                    if not run_dir.is_dir() or not (run_dir / "history.jsonl").exists():
                        continue
                    runs.append(
                        RunSpec(
                            variant=variant,
                            env=env,
                            perturbation=perturbation,
                            seed=_parse_seed(run_dir),
                            run_dir=run_dir,
                        )
                    )
    return sorted(runs, key=lambda r: (ALL_VARIANTS.index(r.variant) if r.variant in ALL_VARIANTS else 99,
                                       ENV_ORDER.index(r.env), PERTURBATION_ORDER.index(r.perturbation), r.seed))


def validate_expected_layout(log_root: Path) -> None:
    runs = discover_runs(log_root)
    counts = {variant: 0 for variant in ALL_VARIANTS}
    for r in runs:
        counts[r.variant] += 1
    print("[Run counts]", ", ".join(f"{k}={v}" for k, v in counts.items()))
    expected = 4 * 2 * 5
    missing = [f"{k}: {v}/{expected}" for k, v in counts.items() if v != expected]
    if missing:
        print("[Warning] Expected 40 runs per method but found: " + ", ".join(missing))


def _config_info(run_dir: Path) -> dict:
    path = run_dir / "config.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)
    hp = cfg.get("hyperparams", {}) or {}
    saved = cfg.get("saved_hyperparams", {}) or {}
    merged = {**saved, **hp}
    return {
        "num_clients": int(merged.get("num_clients", len(merged.get("client_noise_values", [])) or 5)),
        "client_noise_values": merged.get("client_noise_values", None),
        "dual_lr": merged.get("dual_lr", None),
        "dual_lambda_cap": merged.get("dual_lambda_cap", None),
        "dual_update_mode": merged.get("dual_update_mode", None),
        "eval_round_freq": merged.get("eval_round_freq", None),
    }


def _to_float(value, default=np.nan) -> float:
    if value is None:
        return float(default)
    try:
        x = float(value)
    except (TypeError, ValueError):
        return float(default)
    return x if np.isfinite(x) else float(default)


def _row_vector(row: dict, template: str, k: int) -> np.ndarray:
    return np.asarray([_to_float(row.get(template.format(i=i))) for i in range(k)], dtype=np.float64)


def _cache_path(cache_root: Path, spec: RunSpec) -> Path:
    return Path(cache_root) / spec.variant / spec.env / spec.perturbation / f"seed_{spec.seed}.npz"


def _cache_fresh(cache_path: Path, run_dir: Path) -> bool:
    if not cache_path.exists():
        return False
    source_paths = [run_dir / "history.jsonl", run_dir / "config.json"]
    source_mtime = max((p.stat().st_mtime for p in source_paths if p.exists()), default=0.0)
    return cache_path.stat().st_mtime >= source_mtime


def build_run_cache(spec: RunSpec, cache_root: Path, force: bool = False) -> Path:
    """Parse one W&B history once and store only ablation-relevant arrays."""
    cache_path = _cache_path(cache_root, spec)
    if not force and _cache_fresh(cache_path, spec.run_dir):
        return cache_path

    info = _config_info(spec.run_dir)
    k = int(info.get("num_clients", 5))
    noises_cfg = info.get("client_noise_values")
    noises = (
        np.asarray(noises_cfg, dtype=np.float64)
        if noises_cfg is not None and len(noises_cfg) == k
        else np.full(k, np.nan, dtype=np.float64)
    )

    server_rounds: list[int] = []
    server_returns: list[np.ndarray] = []
    server_dual_signal: list[np.ndarray] = []
    lambda_actor: list[np.ndarray] = []
    lambda_after: list[np.ndarray] = []
    grad_norms: list[np.ndarray] = []
    influence_shares: list[np.ndarray] = []
    grad_cos_to_agg: list[np.ndarray] = []
    server_scalars: dict[str, list[float]] = {key: [] for key in SERVER_SCALAR_KEYS}

    eval_rounds: list[int] = []
    eval_local: list[np.ndarray] = []
    eval_nominal: list[np.ndarray] = []

    history_path = spec.run_dir / "history.jsonl"
    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            round_value = row.get("frl/round")
            if round_value is None:
                continue
            round_int = int(round(float(round_value)))

            # Populate noise from history if config did not provide it.
            if np.any(~np.isfinite(noises)):
                candidate = _row_vector(row, "frl/client_{i}/noise", k)
                mask = np.isfinite(candidate)
                noises[mask] = candidate[mask]

            # Eval rows are separate W&B rows but share frl/round with the preceding server row.
            local = _row_vector(row, "eval/client_{i}/local_mean", k)
            if np.any(np.isfinite(local)):
                eval_rounds.append(round_int)
                eval_local.append(local)
                eval_nominal.append(_row_vector(row, "eval/client_{i}/nominal_mean", k))

            # A server row is identified by at least one AMPO server metric.
            # PPOAvg has no AMPO server row and does not need the server arrays.
            server_marker = row.get("server/ampo/dual_round")
            if server_marker is None:
                server_marker = row.get("server/ampo/lambda_max")
            if server_marker is None:
                continue

            server_rounds.append(round_int)
            server_returns.append(_row_vector(row, "server/ampo/client_{i}/return", k))
            server_dual_signal.append(_row_vector(row, "server/ampo/client_{i}/dual_signal", k))
            lambda_actor.append(_row_vector(row, "server/ampo/client_{i}/lambda_actor", k))
            lambda_after.append(_row_vector(row, "server/ampo/client_{i}/lambda", k))
            grad_norms.append(_row_vector(row, "server/ampo/client_{i}/grad_norm", k))
            influence_shares.append(_row_vector(row, "server/ampo/client_{i}/gradient_influence_share", k))
            grad_cos_to_agg.append(_row_vector(row, "server/ampo/client_{i}/grad_cosine_to_aggregate", k))
            for key in SERVER_SCALAR_KEYS:
                server_scalars[key].append(_to_float(row.get(f"server/ampo/{key}")))

    def stack_or_empty(rows: list[np.ndarray]) -> np.ndarray:
        return np.vstack(rows) if rows else np.empty((0, k), dtype=np.float64)

    payload: dict[str, np.ndarray] = {
        "server_rounds": np.asarray(server_rounds, dtype=np.int64),
        "server_returns": stack_or_empty(server_returns),
        "server_dual_signal": stack_or_empty(server_dual_signal),
        "lambda_actor": stack_or_empty(lambda_actor),
        "lambda_after": stack_or_empty(lambda_after),
        "grad_norms": stack_or_empty(grad_norms),
        "influence_shares": stack_or_empty(influence_shares),
        "grad_cos_to_agg": stack_or_empty(grad_cos_to_agg),
        "eval_rounds": np.asarray(eval_rounds, dtype=np.int64),
        "eval_local": stack_or_empty(eval_local),
        "eval_nominal": stack_or_empty(eval_nominal),
        "noises": noises,
        "num_clients": np.asarray(k, dtype=np.int64),
        "dual_lr": np.asarray(_to_float(info.get("dual_lr")), dtype=np.float64),
        "dual_lambda_cap": np.asarray(_to_float(info.get("dual_lambda_cap")), dtype=np.float64),
    }
    for key, values in server_scalars.items():
        payload[f"server_{key}"] = np.asarray(values, dtype=np.float64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **payload)
    print(f"[Cache built] {spec.variant}/{spec.env}/{spec.perturbation}/seed={spec.seed}")
    return cache_path


def load_run(spec: RunSpec, cache_root: Path, force_cache: bool = False) -> dict[str, np.ndarray]:
    path = build_run_cache(spec, cache_root, force=force_cache)
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def group_specs(log_root: Path, variants: Sequence[str] | None = None) -> dict[tuple[str, str, str], list[RunSpec]]:
    groups: dict[tuple[str, str, str], list[RunSpec]] = {}
    for spec in discover_runs(log_root, variants=variants):
        groups.setdefault((spec.variant, spec.env, spec.perturbation), []).append(spec)
    for key in groups:
        groups[key] = sorted(groups[key], key=lambda x: x.seed)
    return groups


# -----------------------------------------------------------------------------
# Numerical helpers
# -----------------------------------------------------------------------------
def finite_mean(x: np.ndarray, axis=None) -> float | np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    with np.errstate(all="ignore"):
        return np.nanmean(x, axis=axis)


def finite_std(x: np.ndarray, axis=None, ddof: int = 1) -> float | np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return np.nan
    if axis is None:
        n = int(np.sum(np.isfinite(x)))
        if n <= ddof:
            ddof = 0
    with np.errstate(all="ignore"):
        return np.nanstd(x, axis=axis, ddof=ddof)


def _average_rank_1d(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rank input must be 1D")
    n = len(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    i = 0
    while i < n:
        j = i + 1
        while j < n and values[order[j]] == values[order[i]]:
            j += 1
        # ranks are 1-based; ties get average rank.
        avg_rank = 0.5 * ((i + 1) + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return np.nan
    x = x - np.mean(x)
    y = y - np.mean(y)
    denom = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denom <= 0.0:
        return np.nan
    return float(np.dot(x, y) / denom)


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2:
        return np.nan
    return pearson_corr(_average_rank_1d(x), _average_rank_1d(y))


def late_indices(n: int, late_evals: int) -> np.ndarray:
    if n <= 0:
        return np.empty(0, dtype=int)
    start = max(0, n - int(late_evals))
    return np.arange(start, n, dtype=int)


def late_eval_metrics(run: dict[str, np.ndarray], late_evals: int = 10) -> dict[str, float]:
    local = np.asarray(run["eval_local"], dtype=np.float64)
    nominal = np.asarray(run["eval_nominal"], dtype=np.float64)
    idx = late_indices(len(local), late_evals)
    local = local[idx]
    nominal = nominal[idx]
    if local.size == 0:
        return {"worst": np.nan, "average": np.nan, "nominal": np.nan, "gap": np.nan, "bottom2": np.nan}
    per_eval_worst = np.nanmin(local, axis=1)
    per_eval_avg = np.nanmean(local, axis=1)
    sorted_local = np.sort(local, axis=1)
    bottom2 = np.nanmean(sorted_local[:, : min(2, local.shape[1])], axis=1)
    nominal_mean = np.nanmean(nominal, axis=1) if nominal.size else np.full(len(local), np.nan)
    return {
        "worst": float(np.nanmean(per_eval_worst)),
        "average": float(np.nanmean(per_eval_avg)),
        "nominal": float(np.nanmean(nominal_mean)),
        "gap": float(np.nanmean(per_eval_avg - per_eval_worst)),
        "bottom2": float(np.nanmean(bottom2)),
    }


def summarize_seed_values(values: Iterable[float]) -> tuple[float, float, int]:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return np.nan, np.nan, 0
    return float(np.mean(arr)), float(np.std(arr, ddof=1 if len(arr) > 1 else 0)), int(len(arr))


def find_server_indices_for_eval_rounds(run: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return (eval_indices, server_indices) for exact round matches."""
    er = np.asarray(run["eval_rounds"], dtype=np.int64)
    sr = np.asarray(run["server_rounds"], dtype=np.int64)
    if len(er) == 0 or len(sr) == 0:
        return np.empty(0, dtype=int), np.empty(0, dtype=int)
    mapping = {int(r): i for i, r in enumerate(sr)}
    e_idx, s_idx = [], []
    for i, r in enumerate(er):
        j = mapping.get(int(r))
        if j is not None:
            e_idx.append(i)
            s_idx.append(j)
    return np.asarray(e_idx, dtype=int), np.asarray(s_idx, dtype=int)


def select_late_eval_server_pairs(run: dict[str, np.ndarray], late_evals: int = 10) -> tuple[np.ndarray, np.ndarray]:
    e_idx, s_idx = find_server_indices_for_eval_rounds(run)
    if len(e_idx) == 0:
        return e_idx, s_idx
    keep = late_indices(len(e_idx), late_evals)
    return e_idx[keep], s_idx[keep]


def normalized_entropy(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64)
    w = w[np.isfinite(w) & (w > 0)]
    if len(w) <= 1:
        return 0.0
    h = -float(np.sum(w * np.log(w)))
    return h / math.log(len(weights))


def moving_average_matrix(x: np.ndarray, window: int = 3) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if window <= 1 or len(x) == 0:
        return x.copy()
    out = np.empty_like(x)
    half = window // 2
    for i in range(len(x)):
        lo = max(0, i - half)
        hi = min(len(x), i + (window - half))
        out[i] = np.nanmean(x[lo:hi], axis=0)
    return out


def switch_rate(ids: np.ndarray) -> float:
    ids = np.asarray(ids)
    if len(ids) <= 1:
        return np.nan
    return float(np.mean(ids[1:] != ids[:-1]))


def dominant_share(ids: np.ndarray) -> float:
    ids = np.asarray(ids)
    if len(ids) == 0:
        return np.nan
    _, counts = np.unique(ids, return_counts=True)
    return float(np.max(counts) / len(ids))


def format_pm(mean: float, std: float, digits: int = 2) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return f"{mean:.{digits}f}"
    return rf"${mean:.{digits}f} \pm {std:.{digits}f}$"


def make_latex_table_rows(
    column_names: Sequence[str],
    rows: Sequence[str],
    comments: Sequence[str] | None = None,
) -> str:
    """Return copy-ready LaTeX tabular rows with a visible header and ``\\midrule``.

    The first printed line is the actual column-name row, e.g.
    ``Environment & Perturbation & Metric \\``.  This makes each generated
    ``latex_*_rows.txt`` block directly pasteable inside an existing tabular.
    Optional comments are emitted after ``\\midrule`` so they do not hide the
    table header when the block is printed in the terminal.
    """
    header_row = " & ".join(str(name) for name in column_names) + r" \\"
    lines = [header_row, r"\midrule"]
    if comments:
        for comment in comments:
            comment = str(comment)
            lines.append(comment if comment.startswith("%") else "% " + comment)
    lines.extend(str(row) for row in rows)
    return "\n".join(lines) + "\n"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"[Saved] {path}")


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"[Saved] {path}")


def task_label(env: str, perturbation: str) -> str:
    return f"{DISPLAY_ENV.get(env, env)}-{perturbation.capitalize()}"


def common_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--late-evals", type=int, default=10,
                   help="Number of final evaluation points used for late-training summaries.")
    p.add_argument("--force-cache", action="store_true",
                   help="Rebuild reduced caches even if they are up to date.")
    return p


def get_group_runs(
    groups: dict[tuple[str, str, str], list[RunSpec]],
    variant: str,
    env: str,
    perturbation: str,
) -> list[RunSpec]:
    return groups.get((variant, env, perturbation), [])


def load_group(
    specs: Sequence[RunSpec],
    cache_root: Path,
    force_cache: bool = False,
) -> list[tuple[RunSpec, dict[str, np.ndarray]]]:
    return [(spec, load_run(spec, cache_root, force_cache=force_cache)) for spec in specs]


def require_five_seeds(specs: Sequence[RunSpec], label: str) -> None:
    seeds = sorted(s.seed for s in specs)
    if seeds != [1, 2, 3, 4, 5]:
        print(f"[Warning] {label}: expected seeds [1,2,3,4,5], found {seeds}")


if __name__ == "__main__":
    parser = common_parser("Build/validate compact ablation caches.")
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)
    cache_root = args.out_root / "_cache"
    for spec in discover_runs(args.log_root):
        build_run_cache(spec, cache_root, force=args.force_cache)
    print(f"[Done] cache root: {cache_root}")
