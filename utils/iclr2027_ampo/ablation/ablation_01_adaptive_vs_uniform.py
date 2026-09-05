from __future__ import annotations

from pathlib import Path

import numpy as np

from ampo_ablation_common import (
    ALL_VARIANTS,
    ENV_ORDER,
    PERTURBATION_ORDER,
    DISPLAY_ENV,
    SHORT_VARIANT,
    apply_publication_style,
    common_parser,
    format_pm,
    get_group_runs,
    group_specs,
    late_eval_metrics,
    load_group,
    make_latex_table_rows,
    summarize_seed_values,
    validate_expected_layout,
    write_csv,
    write_text,
)


def main() -> None:
    parser = common_parser(
        "Ablation 01: isolate the effect of adaptive dual reweighting using late-training performance."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "01_adaptive_vs_uniform"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ALL_VARIANTS)

    # Seed-level summaries. Every reported std is across seeds, not across eval episodes.
    per_setting = {}
    csv_rows = []
    for env in ENV_ORDER:
        for pert in PERTURBATION_ORDER:
            for variant in ALL_VARIANTS:
                loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                seed_metrics = []
                for spec, run in loaded:
                    m = late_eval_metrics(run, late_evals=args.late_evals)
                    seed_metrics.append((spec.seed, m))
                per_setting[(variant, env, pert)] = seed_metrics
                for seed, m in seed_metrics:
                    csv_rows.append(
                        [
                            variant,
                            env,
                            pert,
                            seed,
                            m["worst"],
                            m["average"],
                            m["nominal"],
                            m["gap"],
                            m["bottom2"],
                        ]
                    )

    write_csv(
        out_dir / "seed_level_late_metrics.csv",
        ["variant", "environment", "perturbation", "seed", "worst", "average", "nominal", "avg_minus_worst", "bottom2"],
        csv_rows,
    )

    # Primary paper tables: AMPO-Uniform vs each adaptive dual LR.
    #
    # The reported Uniform/Adaptive values are first summarized within each seed over
    # the requested late-evaluation window and then reported as mean +- std across seeds.
    # Improvement is intentionally computed from the two across-seed means:
    #
    #   100 * (Adaptive_mean - Uniform_mean) / abs(Uniform_mean)
    #
    # It is NOT the average of seed-wise percentage changes.  This keeps the sign and
    # interpretation consistent with the reported mean returns.
    comparison_csv = []
    adaptive_variants = (
        ("ampo_adaptive_dual_lr_1e-4", "1e-4"),
        ("ampo_adaptive_dual_lr_3e-4", "3e-4"),
    )

    for variant, lr_label in adaptive_variants:
        rows = []
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                uniform_vals = [m["worst"] for _, m in per_setting[("ampo_uniform", env, pert)]]
                adaptive_vals = [m["worst"] for _, m in per_setting[(variant, env, pert)]]

                uniform_mean, uniform_std, uniform_n = summarize_seed_values(uniform_vals)
                adaptive_mean, adaptive_std, adaptive_n = summarize_seed_values(adaptive_vals)

                if np.isfinite(uniform_mean) and np.isfinite(adaptive_mean) and abs(uniform_mean) > 1e-12:
                    improvement_pct = 100.0 * (adaptive_mean - uniform_mean) / abs(uniform_mean)
                else:
                    improvement_pct = np.nan

                improvement_cell = (
                    rf"$\mathbf{{{improvement_pct:+.1f}\%}}$"
                    if np.isfinite(improvement_pct)
                    else r"$\mathrm{nan}$"
                )

                rows.append(
                    " & ".join(
                        [
                            DISPLAY_ENV[env],
                            pert.capitalize(),
                            format_pm(uniform_mean, uniform_std, digits=1),
                            format_pm(adaptive_mean, adaptive_std, digits=1),
                            improvement_cell,
                        ]
                    )
                    + r" \\"
                )

                comparison_csv.append(
                    [
                        env,
                        pert,
                        variant,
                        uniform_mean,
                        uniform_std,
                        uniform_n,
                        adaptive_mean,
                        adaptive_std,
                        adaptive_n,
                        improvement_pct,
                    ]
                )

        text = make_latex_table_rows(
            ["Environment", "Perturbation", "Uniform Worst", "Adaptive Worst", "Improvement"],
            rows,
            comments=[
                f"Adaptive dual LR = {lr_label}.",
                f"Uniform/Adaptive: late-{args.late_evals}-evaluation worst return, mean +- seed std.",
                r"Improvement = 100 * (Adaptive mean - Uniform mean) / |Uniform mean|.",
            ],
        )
        write_text(out_dir / f"latex_adaptive_vs_uniform_{lr_label}_rows.txt", text)
        print(f"\n[LaTeX rows: AMPO-Uniform vs AMPO-Adaptive({lr_label})]\n" + text)

    write_csv(
        out_dir / "adaptive_vs_uniform_summary.csv",
        [
            "environment",
            "perturbation",
            "variant",
            "uniform_mean",
            "uniform_std",
            "uniform_num_seeds",
            "adaptive_mean",
            "adaptive_std",
            "adaptive_num_seeds",
            "improvement_percent_from_means",
        ],
        comparison_csv,
    )

    # Supplementary all-method worst-return tables.  These are saved for convenience
    # but are not the primary console output for Ablation 01.
    for pert in PERTURBATION_ORDER:
        rows = []
        for env in ENV_ORDER:
            cells = [DISPLAY_ENV[env]]
            summaries = []
            for variant in ALL_VARIANTS:
                vals = [m["worst"] for _, m in per_setting[(variant, env, pert)]]
                summaries.append(summarize_seed_values(vals))
            finite_means = [x[0] for x in summaries if np.isfinite(x[0])]
            best_mean = max(finite_means) if finite_means else np.nan
            for mean, std, _ in summaries:
                if np.isfinite(mean) and np.isclose(mean, best_mean):
                    cells.append(rf"$\mathbf{{{mean:.1f} \pm {std:.1f}}}$")
                else:
                    cells.append(format_pm(mean, std, digits=1))
            rows.append(" & ".join(cells) + r" \\")
        text = make_latex_table_rows(
            ["Environment", "PPOAvg", "AMPO-Uniform", "AMPO-Adaptive(1e-4)", "AMPO-Adaptive(3e-4)"],
            rows,
            comments=[f"Metric: mean late-{args.late_evals}-evaluation worst-case return; mean +- seed std."],
        )
        write_text(out_dir / f"latex_all_methods_worst_return_{pert}_rows.txt", text)

    # Also provide average/local and robustness-gap rows for appendix use.
    for metric_name in ("average", "gap", "bottom2", "nominal"):
        rows = []
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                cells = [DISPLAY_ENV[env], pert.capitalize()]
                for variant in ALL_VARIANTS:
                    vals = [m[metric_name] for _, m in per_setting[(variant, env, pert)]]
                    mean, std, _ = summarize_seed_values(vals)
                    cells.append(format_pm(mean, std, digits=1))
                rows.append(" & ".join(cells) + r" \\")
        text = make_latex_table_rows(
            ["Environment", "Perturbation", "PPOAvg", "AMPO-Uniform", "AMPO-Adaptive(1e-4)", "AMPO-Adaptive(3e-4)"],
            rows,
            comments=[f"Metric: {metric_name}; late {args.late_evals} evaluations; mean +- seed std."],
        )
        write_text(out_dir / f"latex_{metric_name}_rows.txt", text)
        print(f"\n[LaTeX rows: {metric_name}]\n" + text)

    print(f"\n[Done] Ablation 01 outputs: {out_dir}")
    print("[Paper recommendation] Use the worst-return table in the main ablation; keep average/nominal/bottom-2 rows for appendix or robustness discussion.")


if __name__ == "__main__":
    main()
