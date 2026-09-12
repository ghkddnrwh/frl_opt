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
    load_group,
    make_latex_table_rows,
    summarize_seed_values,
    switch_rate,
    validate_expected_layout,
    write_csv,
    write_text,
)


# =============================================================================
# Variants
# =============================================================================
#
# 실제 로그 디렉토리 이름:
#
#   ampo_uniform
#   ampo_adaptive_dual_lr_1e-4
#   ampo_adaptive_dual_lr_3e-4
#
# 따라서 Uniform variant key는 "uniform"이 아니라 "ampo_uniform"이다.
# =============================================================================

UNIFORM_VARIANT = "ampo_uniform"

TRACKING_VARIANTS = (
    UNIFORM_VARIANT,
    *ADAPTIVE_VARIANTS,
)


def variant_short_name(
    variant: str,
) -> str:
    """
    LaTeX table에 사용할 method 이름.
    """
    if variant == UNIFORM_VARIANT:
        return "AMPO-PPO(U)"

    if variant in SHORT_VARIANT:
        return SHORT_VARIANT[
            variant
        ]

    return variant


# =============================================================================
# Lambda metrics
# =============================================================================


def effective_num_clients(
    lambda_weights: np.ndarray,
) -> np.ndarray:
    """
    Compute

        K_eff(t) = 1 / sum_k lambda_k(t)^2

    for each row.

    Uniform over K clients:
        lambda_k = 1/K
        -> K_eff = K

    Fully concentrated:
        lambda = [1, 0, ..., 0]
        -> K_eff = 1
    """
    lambda_weights = np.asarray(
        lambda_weights,
        dtype=float,
    )

    if lambda_weights.ndim != 2:
        raise ValueError(
            "lambda_weights must be 2D, "
            f"got {lambda_weights.shape}"
        )

    denom = np.sum(
        np.square(
            lambda_weights
        ),
        axis=1,
    )

    out = np.full(
        len(denom),
        np.nan,
        dtype=float,
    )

    valid = (
        np.isfinite(
            denom
        )
        & (denom > 0.0)
    )

    out[
        valid
    ] = (
        1.0
        / denom[
            valid
        ]
    )

    return out


def tie_aware_top2_match(
    lambda_weights: np.ndarray,
    worst_ids: np.ndarray,
) -> float:
    """
    Tie-aware Top-2 Match.

    각 evaluation point에서 eval-worst client가
    lambda 기준 상위 2개 client에 포함되는지를 측정한다.

    lambda 값이 모두 distinct하면:

        worst client in Top-2:
            score = 1

        otherwise:
            score = 0

    그러나 Top-2 cutoff에서 tie가 존재하면
    deterministic client-index tie breaking을 사용하지 않고
    fractional score를 부여한다.

    -------------------------------------------------------------------------
    Example 1: no tie
    -------------------------------------------------------------------------

        lambda = [0.50, 0.25, 0.15, 0.07, 0.03]

    Top-2 = client 0, client 1.

    worst = client 0 or 1:
        score = 1

    worst = client 2, 3, or 4:
        score = 0


    -------------------------------------------------------------------------
    Example 2: tie at Top-2 boundary
    -------------------------------------------------------------------------

        lambda = [0.50, 0.20, 0.20, 0.10, 0.00]

    client 0은 확실하게 첫 번째 slot을 차지한다.

    client 1과 client 2가 두 번째 slot 하나를 공유하므로:

        score(client 1) = 1/2
        score(client 2) = 1/2


    -------------------------------------------------------------------------
    Exact Uniform
    -------------------------------------------------------------------------

        lambda = [0.2, 0.2, 0.2, 0.2, 0.2]

    5 clients가 Top-2의 두 slot을 공유하므로
    각 client의 expected membership score는

        2 / 5 = 0.4

    따라서 K=5 exact Uniform이면:

        Top-2 Match = 0.4

    일반적으로 exact Uniform over K clients:

        Top-2 Match = min(2, K) / K
    """
    lambda_weights = np.asarray(
        lambda_weights,
        dtype=float,
    )

    worst_ids = np.asarray(
        worst_ids,
        dtype=int,
    ).reshape(-1)

    if lambda_weights.ndim != 2:
        raise ValueError(
            "lambda_weights must be 2D, "
            f"got {lambda_weights.shape}"
        )

    if len(lambda_weights) != len(worst_ids):
        raise ValueError(
            "lambda_weights and worst_ids must "
            "have the same number of rows."
        )

    if len(worst_ids) == 0:
        return np.nan

    num_clients = lambda_weights.shape[1]

    if num_clients <= 0:
        return np.nan

    top_k = min(
        2,
        num_clients,
    )

    scores = np.full(
        len(worst_ids),
        np.nan,
        dtype=float,
    )

    for t, worst_id in enumerate(
        worst_ids
    ):
        if (
            worst_id < 0
            or worst_id >= num_clients
        ):
            continue

        row = lambda_weights[
            t
        ]

        worst_weight = row[
            worst_id
        ]

        # -------------------------------------------------------------
        # Identify clients effectively tied with the worst client's
        # lambda weight.
        # -------------------------------------------------------------

        tied_mask = np.isclose(
            row,
            worst_weight,
            rtol=1e-7,
            atol=1e-10,
        )

        # -------------------------------------------------------------
        # Clients strictly above the worst client's lambda.
        #
        # Values regarded as tied by np.isclose are deliberately
        # excluded here so that numerical noise does not create
        # arbitrary Top-2 membership.
        # -------------------------------------------------------------

        strictly_higher_mask = (
            (row > worst_weight)
            & (~tied_mask)
        )

        num_higher = int(
            np.sum(
                strictly_higher_mask
            )
        )

        num_tied = int(
            np.sum(
                tied_mask
            )
        )

        if num_tied <= 0:
            continue

        remaining_slots = (
            top_k
            - num_higher
        )

        # -------------------------------------------------------------
        # No Top-k slot remains for this tie group.
        # -------------------------------------------------------------

        if remaining_slots <= 0:
            scores[
                t
            ] = 0.0
            continue

        # -------------------------------------------------------------
        # All members of this tie group fit inside Top-k.
        # -------------------------------------------------------------

        if remaining_slots >= num_tied:
            scores[
                t
            ] = 1.0
            continue

        # -------------------------------------------------------------
        # Only part of the tie group fits.
        #
        # Fractional expected membership under symmetric
        # tie-breaking.
        # -------------------------------------------------------------

        scores[
            t
        ] = (
            float(
                remaining_slots
            )
            / float(
                num_tied
            )
        )

    valid = np.isfinite(
        scores
    )

    if not np.any(
        valid
    ):
        return np.nan

    return float(
        np.mean(
            scores[
                valid
            ]
        )
    )


# =============================================================================
# Per-seed metrics
# =============================================================================


def per_seed_compact_metrics(
    run: dict[str, np.ndarray],
    late_evals: int,
) -> dict[str, float]:
    """
    Compute worst-environment diagnostics.

    Metrics
    -------
    1. Worst Switch (Full)

       Fraction of consecutive evaluation points where
       the worst client identity changes.


    2. Dominant Worst Share (Full)

       Fraction of evaluations occupied by the client
       that is worst most frequently.


    3. Top-2 Match (Full)

       Tie-aware alignment between the eval-defined worst client
       and the two largest lambda weights.

       With unique lambda values:

           score = 1 if worst client is in lambda Top-2
                   0 otherwise

       Ties at the Top-2 cutoff are handled fractionally.

       For exact Uniform weighting with K clients:

           Top-2 Match = min(2, K) / K

       Thus K=5 exact Uniform gives:

           Top-2 Match = 0.4


    4. K_eff (Last)

       First compute

           K_eff(t)
               = 1 / sum_k lambda_k(t)^2

       at every matched evaluation/server point.

       Then average only over the final `late_evals`
       matched points:

           K_eff(Last)
               = mean(
                   K_eff[-late_evals:]
                 )

       This follows the common ablation convention for "Last".

       If fewer than `late_evals` valid matched points exist,
       all available valid matched points are used.

       For exact Uniform weighting:

           K_eff = K


    Each seed is reduced to one scalar per metric first.

    Final table reports mean +- std across seeds.
    """
    late_evals = int(
        late_evals
    )

    if late_evals <= 0:
        raise ValueError(
            "late_evals must be positive, "
            f"got {late_evals}."
        )

    (
        e_idx,
        s_idx,
    ) = find_server_indices_for_eval_rounds(
        run
    )

    if len(
        e_idx
    ) < 2:
        return {}

    # -------------------------------------------------------------------------
    # Evaluation returns
    # -------------------------------------------------------------------------

    eval_local = np.asarray(
        run[
            "eval_local"
        ],
        dtype=float,
    )[
        e_idx
    ]

    # -------------------------------------------------------------------------
    # Lambda used by actor/server aggregation
    #
    # IMPORTANT:
    # ampo_uniform cache에도 lambda_actor가 실제 저장되어 있다.
    #
    # 확인된 값 예:
    #
    #     [0.2, 0.2, 0.2, 0.2, 0.2]
    #
    # 따라서 Uniform에 대해서도 동일한 Top-2 / K_eff 계산을
    # 수행할 수 있다.
    # -------------------------------------------------------------------------

    lambda_actor = np.asarray(
        run[
            "lambda_actor"
        ],
        dtype=float,
    )[
        s_idx
    ]

    # -------------------------------------------------------------------------
    # Shape validation
    # -------------------------------------------------------------------------

    if eval_local.ndim != 2:
        raise ValueError(
            "eval_local must be 2D, "
            f"got {eval_local.shape}"
        )

    if lambda_actor.ndim != 2:
        raise ValueError(
            "lambda_actor must be 2D, "
            f"got {lambda_actor.shape}"
        )

    if (
        eval_local.shape[
            1
        ]
        != lambda_actor.shape[
            1
        ]
    ):
        raise ValueError(
            "Client-count mismatch: "
            f"eval_local has {eval_local.shape[1]}, "
            f"lambda_actor has {lambda_actor.shape[1]}."
        )

    # -------------------------------------------------------------------------
    # Keep only fully finite matched rows
    #
    # Important:
    # "Last" is defined after exact eval/server matching and finite filtering.
    # Therefore the final `late_evals` rows here are exactly the final usable
    # matched evaluation/server points.
    # -------------------------------------------------------------------------

    valid_rows = (
        np.all(
            np.isfinite(
                eval_local
            ),
            axis=1,
        )
        & np.all(
            np.isfinite(
                lambda_actor
            ),
            axis=1,
        )
    )

    eval_local = eval_local[
        valid_rows
    ]

    lambda_actor = lambda_actor[
        valid_rows
    ]

    if len(
        eval_local
    ) < 2:
        return {}

    # -------------------------------------------------------------------------
    # Worst-client identity
    # -------------------------------------------------------------------------

    worst_ids = np.argmin(
        eval_local,
        axis=1,
    )

    # -------------------------------------------------------------------------
    # Effective number of clients over all matched points
    # -------------------------------------------------------------------------

    keff = effective_num_clients(
        lambda_actor
    )

    # -------------------------------------------------------------------------
    # Last-window K_eff
    #
    # Uses exactly the same late_evals argument supplied by common_parser.
    #
    # Default in the common ablation setup is normally:
    #
    #     --late-evals 10
    #
    # so by default this averages the final 10 matched points.
    # -------------------------------------------------------------------------

    late_count = min(
        late_evals,
        len(
            keff
        ),
    )

    keff_last = keff[
        -late_count:
    ]

    # -------------------------------------------------------------------------
    # Scalar metrics for this seed
    # -------------------------------------------------------------------------

    return {
        "worst_switch_full": switch_rate(
            worst_ids
        ),

        "worst_dominant_share_full": dominant_share(
            worst_ids
        ),

        "top2_match_full": tie_aware_top2_match(
            lambda_weights=lambda_actor,
            worst_ids=worst_ids,
        ),

        "keff_last": (
            float(
                np.nanmean(
                    keff_last
                )
            )
            if np.any(
                np.isfinite(
                    keff_last
                )
            )
            else np.nan
        ),
    }


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = common_parser(
        "Ablation 04b: compact main-paper table for "
        "worst-environment switching, persistence, "
        "Top-2 lambda alignment, and late-stage effective "
        "client count for AMPO-PPO Uniform and Adaptive variants."
    )

    args = parser.parse_args()

    apply_publication_style()

    validate_expected_layout(
        args.log_root
    )

    # -------------------------------------------------------------------------
    # Validate Last-window argument from common parser
    # -------------------------------------------------------------------------

    late_evals = int(
        args.late_evals
    )

    if late_evals <= 0:
        raise ValueError(
            "--late-evals must be positive, "
            f"got {args.late_evals}."
        )

    # -------------------------------------------------------------------------
    # Output
    # -------------------------------------------------------------------------

    out_dir = (
        args.out_root
        / "04_worst_environment_switching"
    )

    cache_root = (
        args.out_root
        / "_cache"
    )

    # -------------------------------------------------------------------------
    # IMPORTANT
    #
    # 실제 variant names:
    #
    #   ampo_uniform
    #   ampo_adaptive_dual_lr_1e-4
    #   ampo_adaptive_dual_lr_3e-4
    # -------------------------------------------------------------------------

    groups = group_specs(
        args.log_root,
        variants=TRACKING_VARIANTS,
    )

    metric_keys = (
        "worst_switch_full",
        "worst_dominant_share_full",
        "top2_match_full",
        "keff_last",
    )

    csv_rows = []
    latex_rows = []

    # =========================================================================
    # Variant
    # =========================================================================

    for variant in TRACKING_VARIANTS:

        # =====================================================================
        # Environment
        # =====================================================================

        for env in ENV_ORDER:

            # =================================================================
            # Perturbation
            # =================================================================

            for pert in PERTURBATION_ORDER:

                run_specs = get_group_runs(
                    groups,
                    variant,
                    env,
                    pert,
                )

                # -------------------------------------------------------------
                # Useful diagnostic:
                # Uniform이 실제로 발견되는지 바로 확인할 수 있다.
                # -------------------------------------------------------------

                print(
                    f"[Group] "
                    f"variant={variant}, "
                    f"env={env}, "
                    f"pert={pert}, "
                    f"runs={len(run_specs)}"
                )

                loaded = load_group(
                    run_specs,
                    cache_root,
                    force_cache=args.force_cache,
                )

                print(
                    f"[Loaded] "
                    f"variant={variant}, "
                    f"env={env}, "
                    f"pert={pert}, "
                    f"loaded={len(loaded)}"
                )

                seed_metrics = []

                for (
                    spec,
                    run,
                ) in loaded:

                    metrics = per_seed_compact_metrics(
                        run,
                        late_evals=late_evals,
                    )

                    if metrics:
                        seed_metrics.append(
                            (
                                spec.seed,
                                metrics,
                            )
                        )

                        print(
                            f"  seed={spec.seed}: "
                            f"switch="
                            f"{metrics['worst_switch_full']:.4f}, "
                            f"dominance="
                            f"{metrics['worst_dominant_share_full']:.4f}, "
                            f"top2="
                            f"{metrics['top2_match_full']:.4f}, "
                            f"Keff(last {late_evals})="
                            f"{metrics['keff_last']:.4f}"
                        )

                # -------------------------------------------------------------
                # Summary across seeds
                # -------------------------------------------------------------

                summaries = {}

                for key in metric_keys:
                    values = [
                        metrics[
                            key
                        ]
                        for (
                            _,
                            metrics,
                        ) in seed_metrics
                    ]

                    summaries[
                        key
                    ] = summarize_seed_values(
                        values
                    )

                # -------------------------------------------------------------
                # LaTeX
                # -------------------------------------------------------------

                cells = [
                    variant_short_name(
                        variant
                    ),

                    DISPLAY_ENV[
                        env
                    ],

                    pert.capitalize(),

                    format_pm(
                        *summaries[
                            "worst_switch_full"
                        ][
                            :2
                        ],
                        digits=2,
                    ),

                    format_pm(
                        *summaries[
                            "worst_dominant_share_full"
                        ][
                            :2
                        ],
                        digits=2,
                    ),

                    format_pm(
                        *summaries[
                            "top2_match_full"
                        ][
                            :2
                        ],
                        digits=2,
                    ),

                    format_pm(
                        *summaries[
                            "keff_last"
                        ][
                            :2
                        ],
                        digits=2,
                    ),
                ]

                latex_rows.append(
                    " & ".join(
                        cells
                    )
                    + r" \\"
                )

                # -------------------------------------------------------------
                # CSV
                # -------------------------------------------------------------

                csv_rows.append(
                    [
                        variant,
                        env,
                        pert,

                        summaries[
                            "worst_switch_full"
                        ][0],

                        summaries[
                            "worst_switch_full"
                        ][1],

                        summaries[
                            "worst_dominant_share_full"
                        ][0],

                        summaries[
                            "worst_dominant_share_full"
                        ][1],

                        summaries[
                            "top2_match_full"
                        ][0],

                        summaries[
                            "top2_match_full"
                        ][1],

                        summaries[
                            "keff_last"
                        ][0],

                        summaries[
                            "keff_last"
                        ][1],

                        summaries[
                            "keff_last"
                        ][2],
                    ]
                )

    # =========================================================================
    # CSV output
    # =========================================================================

    write_csv(
        out_dir
        / (
            "worst_tracking_compact_full_top2_"
            "with_keff_last_uniform_adaptive_summary.csv"
        ),

        [
            "variant",
            "environment",
            "perturbation",

            "worst_switch_full_mean",
            "worst_switch_full_std",

            "dominant_worst_share_full_mean",
            "dominant_worst_share_full_std",

            "top2_match_full_mean",
            "top2_match_full_std",

            "keff_last_mean",
            "keff_last_std",

            "num_seeds",
        ],

        csv_rows,
    )

    # =========================================================================
    # LaTeX
    # =========================================================================

    text = make_latex_table_rows(
        [
            "Method",
            "Environment",
            "Perturbation",
            "Worst Switch (Full)",
            "Dominant Worst Share (Full)",
            "Top-2 Match (Full)",
            r"$K_{\mathrm{eff}}$ (Last)",
        ],

        latex_rows,

        comments=[
            (
                "The table includes AMPO-PPO(U) and "
                "all requested AMPO-PPO(A) variants."
            ),

            (
                "Worst Switch, Dominant Worst Share, and Top-2 Match "
                "use the full set of valid matched evaluation/server "
                "points within each seed."
            ),

            (
                rf"$K_{{\mathrm{{eff}}}}$ (Last) uses only the final "
                rf"{late_evals} valid matched evaluation/server points "
                rf"within each seed."
            ),

            (
                "If a seed has fewer than the requested number of "
                "late evaluation points, all available valid matched "
                "points are used for K_eff (Last)."
            ),

            (
                "Worst Switch = fraction of consecutive "
                "evaluation points whose eval-worst "
                "client identity changes."
            ),

            (
                "Dominant Worst Share = fraction of evaluations "
                "occupied by the most frequently worst client."
            ),

            (
                "Worst Switch and Dominant Worst Share depend "
                "only on evaluation returns."
            ),

            (
                "Top-2 Match measures whether the eval-worst client "
                "belongs to the two largest actor-lambda weights."
            ),

            (
                "Top-2 cutoff ties are handled fractionally rather "
                "than by deterministic client-index tie breaking."
            ),

            (
                "If h clients have lambda strictly larger than the "
                "worst client's lambda, and m clients share the "
                "worst client's tied lambda value, the contribution "
                "is clipped according to the remaining Top-2 slots."
            ),

            (
                "Therefore exact Uniform weighting over K clients "
                "gives Top-2 Match = min(2, K) / K; "
                "for K=5 the value is 0.4."
            ),

            (
                r"$K_{\mathrm{eff}}(t)"
                r"=1/\sum_k\lambda_{\mathrm{actor},k}(t)^2$."
            ),

            (
                r"Exact Uniform weighting gives "
                r"$K_{\mathrm{eff}}=K$."
            ),

            "Reported as mean +- std across seeds.",
        ],
    )

    write_text(
        out_dir
        / (
            "latex_worst_tracking_compact_full_top2_"
            "with_keff_last_uniform_adaptive_rows.txt"
        ),
        text,
    )

    print(
        "\n[LaTeX rows]\n"
        + text
    )

    print(
        "\n[Done] Ablation 04b "
        "Uniform + Adaptive outputs: "
        f"{out_dir}"
    )

    print(
        "[Config] "
        f"Top-2 Match = full trajectory, "
        f"K_eff Last = final {late_evals} matched points."
    )


if __name__ == "__main__":
    main()