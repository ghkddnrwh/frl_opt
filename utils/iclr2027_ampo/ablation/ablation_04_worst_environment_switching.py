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
    dominant_share,
    find_server_indices_for_eval_rounds,
    format_pm,
    get_group_runs,
    group_specs,
    late_indices,
    load_group,
    make_latex_table_rows,
    moving_average_matrix,
    summarize_seed_values,
    switch_rate,
    validate_expected_layout,
    write_csv,
    write_text,
)


def per_seed_switching(run: dict[str, np.ndarray], late_evals: int, smooth_window: int = 3) -> dict[str, float]:
    e_idx, s_idx = find_server_indices_for_eval_rounds(run)
    if len(e_idx) < 2:
        return {}

    eval_local = np.asarray(run["eval_local"], dtype=float)[e_idx]
    lambda_actor = np.asarray(run["lambda_actor"], dtype=float)[s_idx]
    valid_rows = np.all(np.isfinite(eval_local), axis=1) & np.all(np.isfinite(lambda_actor), axis=1)
    eval_local = eval_local[valid_rows]
    lambda_actor = lambda_actor[valid_rows]
    if len(eval_local) < 2:
        return {}

    worst_ids = np.argmin(eval_local, axis=1)
    top_lambda_ids = np.argmax(lambda_actor, axis=1)
    smooth_eval = moving_average_matrix(eval_local, window=smooth_window)
    smooth_worst_ids = np.argmin(smooth_eval, axis=1)

    late = late_indices(len(eval_local), late_evals)
    late_worst = worst_ids[late]
    late_top = top_lambda_ids[late]
    late_smooth = smooth_worst_ids[late]

    return {
        "worst_switch_full": switch_rate(worst_ids),
        "worst_switch_late": switch_rate(late_worst),
        "smoothed_worst_switch_late": switch_rate(late_smooth),
        "top_lambda_switch_late": switch_rate(late_top),
        "worst_dominant_share_late": dominant_share(late_worst),
        "top_lambda_dominant_share_late": dominant_share(late_top),
        "top1_match_late": float(np.mean(late_worst == late_top)) if len(late_worst) else np.nan,
    }


def main() -> None:
    parser = common_parser(
        "Ablation 04: quantify policy-dependent worst-environment switching and compare it with lambda switching."
    )
    parser.add_argument("--smooth-window", type=int, default=3,
                        help="Centered evaluation-point window used to define a smoothed worst environment.")
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "04_worst_environment_switching"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)

    metrics = (
        "worst_switch_full",
        "worst_switch_late",
        "smoothed_worst_switch_late",
        "top_lambda_switch_late",
        "worst_dominant_share_late",
        "top_lambda_dominant_share_late",
        "top1_match_late",
    )
    csv_rows = []
    latex_rows = []

    for variant in ADAPTIVE_VARIANTS:
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                seed_metrics = []
                for spec, run in loaded:
                    m = per_seed_switching(run, args.late_evals, smooth_window=args.smooth_window)
                    if m:
                        seed_metrics.append((spec.seed, m))

                cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
                for key in metrics:
                    vals = [m[key] for _, m in seed_metrics]
                    mean, std, n = summarize_seed_values(vals)
                    cells.append(format_pm(mean, std, digits=2))
                    csv_rows.append([variant, env, pert, key, mean, std, n])
                latex_rows.append(" & ".join(cells) + r" \\")

    write_csv(
        out_dir / "worst_environment_switching_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    text = make_latex_table_rows(
        [
            "Method", "Environment", "Perturbation", "Worst Switch (Full)", "Worst Switch (Late)",
            f"Smoothed Worst Switch (Late, w={args.smooth_window})", r"Top-$\lambda$ Switch (Late)",
            "Dominant Worst Share", r"Dominant Top-$\lambda$ Share", "Top-1 Match",
        ],
        latex_rows,
        comments=[
            "Switch rate = fraction of consecutive evaluation points whose identity changes.",
            "Reported as mean +- std across five seeds.",
        ],
    )
    write_text(out_dir / "latex_worst_switching_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    print(f"\n[Done] Ablation 04 outputs: {out_dir}")
    print("[Paper recommendation] Use a compact table: exact switch rates directly support the policy-dependent-worst claim, while lambda switching shows the slower dual timescale.")


if __name__ == "__main__":
    main()
