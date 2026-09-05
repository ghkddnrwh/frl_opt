from __future__ import annotations

import numpy as np

from ampo_ablation_common import (
    ADAPTIVE_VARIANTS,
    ENV_ORDER,
    PERTURBATION_ORDER,
    DISPLAY_ENV,
    SHORT_VARIANT,
    apply_publication_style,
    common_parser,
    format_pm,
    get_group_runs,
    group_specs,
    load_group,
    make_latex_table_rows,
    spearman_corr,
    summarize_seed_values,
    validate_expected_layout,
    write_csv,
    write_text,
)


def per_seed_focus_response(run: dict[str, np.ndarray], late_evals: int) -> dict[str, float]:
    er = np.asarray(run["eval_rounds"], dtype=int)
    eval_local = np.asarray(run["eval_local"], dtype=float)
    sr = np.asarray(run["server_rounds"], dtype=int)
    la = np.asarray(run["lambda_actor"], dtype=float)
    if len(er) < 2 or len(sr) == 0:
        return {}

    # Use the last late_evals evaluation points, which create late_evals-1 intervals.
    start = max(0, len(er) - late_evals)
    interval_scores = []
    positive_flags = []
    rank_corrs = []
    focus_was_worst = []

    for i in range(start, len(er) - 1):
        r0, r1 = int(er[i]), int(er[i + 1])
        j0 = eval_local[i]
        j1 = eval_local[i + 1]
        mask = (sr > r0) & (sr <= r1)
        if not np.any(mask):
            continue
        interval_lambda = np.nanmean(la[mask], axis=0)
        delta_j = j1 - j0
        valid = np.isfinite(interval_lambda) & np.isfinite(delta_j) & np.isfinite(j0)
        if np.sum(valid) < 2:
            continue

        lam_v = interval_lambda[valid]
        dj_v = delta_j[valid]
        j0_v = j0[valid]
        focus = int(np.argmax(lam_v))
        others = np.arange(len(lam_v)) != focus
        response = float(dj_v[focus] - np.mean(dj_v[others]))
        interval_scores.append(response)
        positive_flags.append(float(response > 0.0))
        rank_corrs.append(spearman_corr(lam_v, dj_v))
        focus_was_worst.append(float(focus == int(np.argmin(j0_v))))

    if not interval_scores:
        return {}
    return {
        "focus_response": float(np.mean(interval_scores)),
        "focus_response_positive_rate": float(np.mean(positive_flags)),
        "lambda_improvement_rank_corr": float(np.nanmean(rank_corrs)) if np.any(np.isfinite(rank_corrs)) else np.nan,
        "focused_client_was_interval_start_worst": float(np.mean(focus_was_worst)),
        "num_intervals": float(len(interval_scores)),
    }


def main() -> None:
    parser = common_parser(
        "Ablation 07: test whether clients receiving larger lambda during an eval interval improve more by the next evaluation."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "07_focus_response"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)

    metrics = (
        "focus_response",
        "focus_response_positive_rate",
        "lambda_improvement_rank_corr",
        "focused_client_was_interval_start_worst",
    )
    csv_rows, latex_rows = [], []

    for variant in ADAPTIVE_VARIANTS:
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                seed_metrics = [(spec.seed, per_seed_focus_response(run, args.late_evals)) for spec, run in loaded]
                seed_metrics = [(s, m) for s, m in seed_metrics if m]
                cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
                for key in metrics:
                    vals = [m[key] for _, m in seed_metrics]
                    mean, std, n = summarize_seed_values(vals)
                    cells.append(format_pm(mean, std, digits=2 if "response" not in key or key.endswith("rate") else 1))
                    csv_rows.append([variant, env, pert, key, mean, std, n])
                latex_rows.append(" & ".join(cells) + r" \\")

    write_csv(
        out_dir / "focus_response_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    text = make_latex_table_rows(
        [
            "Method", "Environment", "Perturbation", "Focus Response", "Positive Rate",
            r"$\rho(\bar{\lambda},\Delta J)$", "Focused Client Was Start-Worst",
        ],
        latex_rows,
        comments=[
            "For eval interval (t,t+1], the focused client is argmax of mean lambda_actor over all server rounds in that interval.",
            "focus-response = DeltaJ_focused - mean(DeltaJ_other clients). This is temporal association, not a causal estimate.",
        ],
    )
    write_text(out_dir / "latex_focus_response_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    print(f"\n[Done] Ablation 07 outputs: {out_dir}")
    print("[Paper recommendation] Use this as an appendix/mechanism table. It directly tests bottleneck-resolution dynamics but should be described as temporal association, not causality.")


if __name__ == "__main__":
    main()
