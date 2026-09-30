from __future__ import annotations

import numpy as np

from ampo_ablation_common import (
    ALL_VARIANTS,
    ENV_ORDER,
    PERTURBATION_ORDER,
    DISPLAY_ENV,
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


def normalized_robustness_gap(j_avg: float, j_worst: float) -> float:
    """
    Normalized robustness gap:

        Gap = (J_avg - J_worst) / J_avg

    where:
        J_avg   = average return over all evaluation environments
        J_worst = worst-environment return

    Returns NaN if the values are invalid or J_avg is too close to zero.

    NOTE:
    If J_avg is negative, this definition can produce a negative gap.
    This intentionally follows the requested formula exactly.
    """
    if (
        np.isfinite(j_avg)
        and np.isfinite(j_worst)
        and abs(j_avg) > 1e-12
    ):
        return float((j_avg - j_worst) / j_avg)

    return np.nan


def main() -> None:
    parser = common_parser(
        "Ablation 01: isolate the effect of adaptive dual reweighting "
        "using late-training performance."
    )
    args = parser.parse_args()

    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "01_adaptive_vs_uniform"
    cache_root = args.out_root / "_cache"

    groups = group_specs(
        args.log_root,
        variants=ALL_VARIANTS,
    )

    # ================================================================
    # Seed-level summaries
    #
    # Every reported std is across seeds, not across evaluation episodes.
    #
    # m["worst"]:
    #   Worst evaluation-environment return.
    #
    # m["average"]:
    #   Return averaged over all evaluation environments.
    #
    # m["gap"]:
    #   Existing unnormalized gap:
    #       J_avg - J_worst
    #
    # normalized_gap:
    #   New requested normalized gap:
    #       (J_avg - J_worst) / J_avg
    # ================================================================

    per_setting = {}
    csv_rows = []

    for env in ENV_ORDER:
        for pert in PERTURBATION_ORDER:
            for variant in ALL_VARIANTS:
                loaded = load_group(
                    get_group_runs(
                        groups,
                        variant,
                        env,
                        pert,
                    ),
                    cache_root,
                    force_cache=args.force_cache,
                )

                seed_metrics = []

                for spec, run in loaded:
                    m = late_eval_metrics(
                        run,
                        late_evals=args.late_evals,
                    )

                    seed_metrics.append(
                        (
                            spec.seed,
                            m,
                        )
                    )

                per_setting[
                    (
                        variant,
                        env,
                        pert,
                    )
                ] = seed_metrics

                for seed, m in seed_metrics:
                    norm_gap = normalized_robustness_gap(
                        m["average"],
                        m["worst"],
                    )

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
                            norm_gap,
                            m["bottom2"],
                        ]
                    )

    write_csv(
        out_dir / "seed_level_late_metrics.csv",
        [
            "variant",
            "environment",
            "perturbation",
            "seed",
            "worst",
            "average",
            "nominal",
            "avg_minus_worst",
            "normalized_gap",
            "bottom2",
        ],
        csv_rows,
    )

    # ================================================================
    # Adaptive variants to compare against AMPO-Uniform
    # ================================================================

    adaptive_variants = (
        (
            "ampo_adaptive_dual_lr_1e-4",
            "1e-4",
        ),
        (
            "ampo_adaptive_dual_lr_3e-4",
            "3e-4",
        ),
    )

    # ================================================================
    # 1. PRIMARY TABLE:
    #    WORST-ENVIRONMENT RETURN
    #
    # Uniform/Adaptive values:
    #   1. summarize late evaluations within each seed
    #   2. compute mean +- std across seeds
    #
    # Improvement:
    #
    #   100 * (Adaptive_mean - Uniform_mean) / |Uniform_mean|
    #
    # This is NOT the average of seed-wise percentage changes.
    # ================================================================

    worst_comparison_csv = []

    for variant, lr_label in adaptive_variants:
        rows = []

        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                uniform_vals = [
                    m["worst"]
                    for _, m in per_setting[
                        (
                            "ampo_uniform",
                            env,
                            pert,
                        )
                    ]
                ]

                adaptive_vals = [
                    m["worst"]
                    for _, m in per_setting[
                        (
                            variant,
                            env,
                            pert,
                        )
                    ]
                ]

                (
                    uniform_mean,
                    uniform_std,
                    uniform_n,
                ) = summarize_seed_values(
                    uniform_vals
                )

                (
                    adaptive_mean,
                    adaptive_std,
                    adaptive_n,
                ) = summarize_seed_values(
                    adaptive_vals
                )

                if (
                    np.isfinite(uniform_mean)
                    and np.isfinite(adaptive_mean)
                    and abs(uniform_mean) > 1e-12
                ):
                    improvement_pct = (
                        100.0
                        * (
                            adaptive_mean
                            - uniform_mean
                        )
                        / abs(uniform_mean)
                    )
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
                            format_pm(
                                uniform_mean,
                                uniform_std,
                                digits=1,
                            ),
                            format_pm(
                                adaptive_mean,
                                adaptive_std,
                                digits=1,
                            ),
                            improvement_cell,
                        ]
                    )
                    + r" \\"
                )

                worst_comparison_csv.append(
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
            [
                "Environment",
                "Perturbation",
                "Uniform Worst",
                "Adaptive Worst",
                "Improvement",
            ],
            rows,
            comments=[
                f"Adaptive dual LR = {lr_label}.",
                (
                    f"Uniform/Adaptive: late-{args.late_evals}-evaluation "
                    "worst-environment return; mean +- seed std."
                ),
                (
                    r"Improvement = 100 * "
                    r"(Adaptive mean - Uniform mean) / |Uniform mean|."
                ),
            ],
        )

        write_text(
            out_dir
            / f"latex_adaptive_vs_uniform_{lr_label}_rows.txt",
            text,
        )

        print(
            f"\n"
            f"[LaTeX rows: Worst Return, "
            f"AMPO-Uniform vs AMPO-Adaptive({lr_label})]"
            f"\n{text}"
        )

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
        worst_comparison_csv,
    )

    # ================================================================
    # 2. PRIMARY / ADDITIONAL TABLE:
    #    AVERAGE RETURN OVER ALL EVALUATION ENVIRONMENTS
    #
    # J_avg:
    #   Average return over all evaluation environments.
    #
    # The procedure is the same as for the worst-return table:
    #
    #   1. obtain J_avg for each seed
    #   2. compute mean +- std across seeds
    #   3. compare Adaptive mean against Uniform mean
    #
    # Improvement:
    #
    #   100 * (Adaptive_mean - Uniform_mean) / |Uniform_mean|
    # ================================================================

    average_comparison_csv = []

    for variant, lr_label in adaptive_variants:
        rows = []

        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                uniform_vals = [
                    m["average"]
                    for _, m in per_setting[
                        (
                            "ampo_uniform",
                            env,
                            pert,
                        )
                    ]
                ]

                adaptive_vals = [
                    m["average"]
                    for _, m in per_setting[
                        (
                            variant,
                            env,
                            pert,
                        )
                    ]
                ]

                (
                    uniform_mean,
                    uniform_std,
                    uniform_n,
                ) = summarize_seed_values(
                    uniform_vals
                )

                (
                    adaptive_mean,
                    adaptive_std,
                    adaptive_n,
                ) = summarize_seed_values(
                    adaptive_vals
                )

                if (
                    np.isfinite(uniform_mean)
                    and np.isfinite(adaptive_mean)
                    and abs(uniform_mean) > 1e-12
                ):
                    improvement_pct = (
                        100.0
                        * (
                            adaptive_mean
                            - uniform_mean
                        )
                        / abs(uniform_mean)
                    )
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
                            format_pm(
                                uniform_mean,
                                uniform_std,
                                digits=1,
                            ),
                            format_pm(
                                adaptive_mean,
                                adaptive_std,
                                digits=1,
                            ),
                            improvement_cell,
                        ]
                    )
                    + r" \\"
                )

                average_comparison_csv.append(
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
            [
                "Environment",
                "Perturbation",
                "Uniform Average",
                "Adaptive Average",
                "Improvement",
            ],
            rows,
            comments=[
                f"Adaptive dual LR = {lr_label}.",
                (
                    f"Uniform/Adaptive: late-{args.late_evals}-evaluation "
                    "return averaged over all evaluation environments; "
                    "mean +- seed std."
                ),
                (
                    r"Improvement = 100 * "
                    r"(Adaptive mean - Uniform mean) / |Uniform mean|."
                ),
            ],
        )

        write_text(
            out_dir
            / f"latex_adaptive_vs_uniform_average_{lr_label}_rows.txt",
            text,
        )

        print(
            f"\n"
            f"[LaTeX rows: Average Return, "
            f"AMPO-Uniform vs AMPO-Adaptive({lr_label})]"
            f"\n{text}"
        )

    write_csv(
        out_dir / "adaptive_vs_uniform_average_summary.csv",
        [
            "environment",
            "perturbation",
            "variant",
            "uniform_average_mean",
            "uniform_average_std",
            "uniform_num_seeds",
            "adaptive_average_mean",
            "adaptive_average_std",
            "adaptive_num_seeds",
            "improvement_percent_from_means",
        ],
        average_comparison_csv,
    )

    # ================================================================
    # 3. PRIMARY / ADDITIONAL TABLE:
    #    NORMALIZED ROBUSTNESS GAP
    #
    # For EACH seed:
    #
    #                   J_avg - J_worst
    #       Gap = ---------------------------
    #                         J_avg
    #
    # We first calculate this quantity independently for every seed.
    # We then report mean +- std across seeds.
    #
    # IMPORTANT:
    # This is different from m["gap"].
    #
    #   m["gap"] = J_avg - J_worst
    #
    # whereas:
    #
    #   normalized gap =
    #       (J_avg - J_worst) / J_avg
    #
    # Smaller absolute separation between average and worst performance
    # indicates more uniform performance across evaluation environments.
    # ================================================================

    normalized_gap_comparison_csv = []

    for variant, lr_label in adaptive_variants:
        rows = []

        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                uniform_gap_vals = []

                for _, m in per_setting[
                    (
                        "ampo_uniform",
                        env,
                        pert,
                    )
                ]:
                    gap = normalized_robustness_gap(
                        m["average"],
                        m["worst"],
                    )

                    uniform_gap_vals.append(
                        gap
                    )

                adaptive_gap_vals = []

                for _, m in per_setting[
                    (
                        variant,
                        env,
                        pert,
                    )
                ]:
                    gap = normalized_robustness_gap(
                        m["average"],
                        m["worst"],
                    )

                    adaptive_gap_vals.append(
                        gap
                    )

                (
                    uniform_gap_mean,
                    uniform_gap_std,
                    uniform_n,
                ) = summarize_seed_values(
                    uniform_gap_vals
                )

                (
                    adaptive_gap_mean,
                    adaptive_gap_std,
                    adaptive_n,
                ) = summarize_seed_values(
                    adaptive_gap_vals
                )

                rows.append(
                    " & ".join(
                        [
                            DISPLAY_ENV[env],
                            pert.capitalize(),
                            format_pm(
                                uniform_gap_mean,
                                uniform_gap_std,
                                digits=3,
                            ),
                            format_pm(
                                adaptive_gap_mean,
                                adaptive_gap_std,
                                digits=3,
                            ),
                        ]
                    )
                    + r" \\"
                )

                normalized_gap_comparison_csv.append(
                    [
                        env,
                        pert,
                        variant,
                        uniform_gap_mean,
                        uniform_gap_std,
                        uniform_n,
                        adaptive_gap_mean,
                        adaptive_gap_std,
                        adaptive_n,
                    ]
                )

        text = make_latex_table_rows(
            [
                "Environment",
                "Perturbation",
                "Uniform Gap",
                "Adaptive Gap",
            ],
            rows,
            comments=[
                f"Adaptive dual LR = {lr_label}.",
                (
                    r"Gap = "
                    r"$(J_{\mathrm{avg}} - J_{\mathrm{worst}})"
                    r"/J_{\mathrm{avg}}$."
                ),
                (
                    f"Gap is computed independently for each seed "
                    f"using late-{args.late_evals}-evaluation summaries, "
                    "then reported as mean +- seed std."
                ),
            ],
        )

        write_text(
            out_dir
            / f"latex_adaptive_vs_uniform_gap_{lr_label}_rows.txt",
            text,
        )

        print(
            f"\n"
            f"[LaTeX rows: Normalized Robustness Gap, "
            f"AMPO-Uniform vs AMPO-Adaptive({lr_label})]"
            f"\n{text}"
        )

    write_csv(
        out_dir / "adaptive_vs_uniform_gap_summary.csv",
        [
            "environment",
            "perturbation",
            "variant",
            "uniform_gap_mean",
            "uniform_gap_std",
            "uniform_num_seeds",
            "adaptive_gap_mean",
            "adaptive_gap_std",
            "adaptive_num_seeds",
        ],
        normalized_gap_comparison_csv,
    )

    # ================================================================
    # 4. SUPPLEMENTARY:
    #    ALL-METHOD WORST-RETURN TABLES
    #
    # These are kept for convenience and are not the main
    # Uniform-vs-Adaptive comparison tables.
    # ================================================================

    for pert in PERTURBATION_ORDER:
        rows = []

        for env in ENV_ORDER:
            cells = [
                DISPLAY_ENV[env]
            ]

            summaries = []

            for variant in ALL_VARIANTS:
                vals = [
                    m["worst"]
                    for _, m in per_setting[
                        (
                            variant,
                            env,
                            pert,
                        )
                    ]
                ]

                summaries.append(
                    summarize_seed_values(vals)
                )

            finite_means = [
                summary[0]
                for summary in summaries
                if np.isfinite(summary[0])
            ]

            best_mean = (
                max(finite_means)
                if finite_means
                else np.nan
            )

            for mean, std, _ in summaries:
                if (
                    np.isfinite(mean)
                    and np.isclose(
                        mean,
                        best_mean,
                    )
                ):
                    cells.append(
                        rf"$\mathbf{{{mean:.1f} \pm {std:.1f}}}$"
                    )
                else:
                    cells.append(
                        format_pm(
                            mean,
                            std,
                            digits=1,
                        )
                    )

            rows.append(
                " & ".join(cells)
                + r" \\"
            )

        text = make_latex_table_rows(
            [
                "Environment",
                "PPOAvg",
                "AMPO-Uniform",
                "AMPO-Adaptive(1e-4)",
                "AMPO-Adaptive(3e-4)",
            ],
            rows,
            comments=[
                (
                    f"Metric: mean late-{args.late_evals}-evaluation "
                    "worst-case return; mean +- seed std."
                ),
            ],
        )

        write_text(
            out_dir
            / f"latex_all_methods_worst_return_{pert}_rows.txt",
            text,
        )

    # ================================================================
    # 5. APPENDIX:
    #    Average / raw gap / bottom-2 / nominal
    #
    # IMPORTANT:
    # The "gap" below is the ORIGINAL UNNORMALIZED metric:
    #
    #       J_avg - J_worst
    #
    # It is not the normalized gap requested above.
    # ================================================================

    appendix_metrics = (
        (
            "average",
            "average",
        ),
        (
            "gap",
            "raw_gap",
        ),
        (
            "bottom2",
            "bottom2",
        ),
        (
            "nominal",
            "nominal",
        ),
    )

    for metric_key, output_name in appendix_metrics:
        rows = []

        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                cells = [
                    DISPLAY_ENV[env],
                    pert.capitalize(),
                ]

                for variant in ALL_VARIANTS:
                    vals = [
                        m[metric_key]
                        for _, m in per_setting[
                            (
                                variant,
                                env,
                                pert,
                            )
                        ]
                    ]

                    mean, std, _ = summarize_seed_values(
                        vals
                    )

                    cells.append(
                        format_pm(
                            mean,
                            std,
                            digits=1,
                        )
                    )

                rows.append(
                    " & ".join(cells)
                    + r" \\"
                )

        if metric_key == "gap":
            metric_comment = (
                "Metric: raw gap = "
                "J_avg - J_worst "
                "(unnormalized)"
            )
        elif metric_key == "average":
            metric_comment = (
                "Metric: average return over all "
                "evaluation environments"
            )
        else:
            metric_comment = (
                f"Metric: {metric_key}"
            )

        text = make_latex_table_rows(
            [
                "Environment",
                "Perturbation",
                "PPOAvg",
                "AMPO-Uniform",
                "AMPO-Adaptive(1e-4)",
                "AMPO-Adaptive(3e-4)",
            ],
            rows,
            comments=[
                (
                    f"{metric_comment}; "
                    f"late {args.late_evals} evaluations; "
                    "mean +- seed std."
                ),
            ],
        )

        write_text(
            out_dir
            / f"latex_{output_name}_rows.txt",
            text,
        )

        print(
            f"\n"
            f"[LaTeX rows: {output_name}]"
            f"\n{text}"
        )

    # ================================================================
    # Final output summary
    # ================================================================

    print(
        f"\n[Done] Ablation 01 outputs: {out_dir}"
    )

    print(
        "\n[Primary paper outputs]"
    )

    print(
        "  Worst return:"
    )
    print(
        "    latex_adaptive_vs_uniform_1e-4_rows.txt"
    )
    print(
        "    latex_adaptive_vs_uniform_3e-4_rows.txt"
    )

    print(
        "\n  Average return over all evaluation environments:"
    )
    print(
        "    latex_adaptive_vs_uniform_average_1e-4_rows.txt"
    )
    print(
        "    latex_adaptive_vs_uniform_average_3e-4_rows.txt"
    )

    print(
        "\n  Normalized robustness gap:"
    )
    print(
        "    latex_adaptive_vs_uniform_gap_1e-4_rows.txt"
    )
    print(
        "    latex_adaptive_vs_uniform_gap_3e-4_rows.txt"
    )

    print(
        "\n[Paper recommendation] "
        "Report worst return together with average return and "
        "normalized robustness gap. "
        "Keep nominal/bottom-2/raw-gap tables for appendix use."
    )


if __name__ == "__main__":
    main()