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


def effective_num_clients(lambda_weights: np.ndarray) -> np.ndarray:
    """Compute K_eff = 1 / sum_k lambda_k^2 for each row."""
    lambda_weights = np.asarray(lambda_weights, dtype=float)
    denom = np.sum(np.square(lambda_weights), axis=1)
    out = np.full(len(denom), np.nan, dtype=float)
    valid = np.isfinite(denom) & (denom > 0.0)
    out[valid] = 1.0 / denom[valid]
    return out


def per_seed_compact_metrics(run: dict[str, np.ndarray]) -> dict[str, float]:
    """Compute the compact full-trajectory worst-environment diagnostics.

    All metrics use the full set of matched evaluation/server points:

      1. Worst Switch (Full): fraction of consecutive eval points where the
         eval-defined worst client identity changes.
      2. Dominant Worst Share (Full): fraction of matched evaluations occupied
         by the client that is worst most often.
      3. Top-1 Match (Full): fraction of matched evaluations where
         argmax(lambda_actor) equals argmin(eval return).
      4. K_eff (Full): time-average of
         K_eff(t) = 1 / sum_k lambda_actor,k(t)^2 over all matched evaluations.

    The per-seed scalar for each metric is computed first; the final table then
    reports mean +- std across seeds.
    """
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
    keff = effective_num_clients(lambda_actor)

    return {
        "worst_switch_full": switch_rate(worst_ids),
        "worst_dominant_share_full": dominant_share(worst_ids),
        "top1_match_full": float(np.mean(worst_ids == top_lambda_ids)) if len(worst_ids) else np.nan,
        "keff_full": float(np.nanmean(keff)) if np.any(np.isfinite(keff)) else np.nan,
    }


def main() -> None:
    parser = common_parser(
        "Ablation 04b: compact main-paper table for worst-environment switching, persistence, lambda alignment, and effective client count."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    # Keep the compact output in the existing Ablation 04 directory.
    out_dir = args.out_root / "04_worst_environment_switching"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)

    metric_keys = (
        "worst_switch_full",
        "worst_dominant_share_full",
        "top1_match_full",
        "keff_full",
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
                    metrics = per_seed_compact_metrics(run)
                    if metrics:
                        seed_metrics.append((spec.seed, metrics))

                summaries = {}
                for key in metric_keys:
                    values = [metrics[key] for _, metrics in seed_metrics]
                    summaries[key] = summarize_seed_values(values)

                cells = [
                    SHORT_VARIANT[variant],
                    DISPLAY_ENV[env],
                    pert.capitalize(),
                    format_pm(*summaries["worst_switch_full"][:2], digits=2),
                    format_pm(*summaries["worst_dominant_share_full"][:2], digits=2),
                    format_pm(*summaries["top1_match_full"][:2], digits=2),
                    format_pm(*summaries["keff_full"][:2], digits=2),
                ]
                latex_rows.append(" & ".join(cells) + r" \\")

                csv_rows.append([
                    variant,
                    env,
                    pert,
                    summaries["worst_switch_full"][0],
                    summaries["worst_switch_full"][1],
                    summaries["worst_dominant_share_full"][0],
                    summaries["worst_dominant_share_full"][1],
                    summaries["top1_match_full"][0],
                    summaries["top1_match_full"][1],
                    summaries["keff_full"][0],
                    summaries["keff_full"][1],
                    summaries["keff_full"][2],
                ])

    write_csv(
        out_dir / "worst_tracking_compact_full_with_keff_summary.csv",
        [
            "variant",
            "environment",
            "perturbation",
            "worst_switch_full_mean",
            "worst_switch_full_std",
            "dominant_worst_share_full_mean",
            "dominant_worst_share_full_std",
            "top1_match_full_mean",
            "top1_match_full_std",
            "keff_full_mean",
            "keff_full_std",
            "num_seeds",
        ],
        csv_rows,
    )

    text = make_latex_table_rows(
        [
            "Method",
            "Environment",
            "Perturbation",
            "Worst Switch (Full)",
            "Dominant Worst Share (Full)",
            "Top-1 Match (Full)",
            r"$K_{\mathrm{eff}}$ (Full)",
        ],
        latex_rows,
        comments=[
            "All metrics use the full set of matched evaluation/server points within each seed.",
            "Worst Switch = fraction of consecutive evaluation points whose eval-worst client identity changes.",
            "Dominant Worst Share = fraction of evaluations occupied by the most frequently worst client.",
            r"Top-1 Match = fraction of evaluations where argmax(lambda_actor) equals argmin(eval return).",
            r"K_eff = mean over matched evaluations of 1 / sum_k lambda_actor,k^2.",
            "Reported as mean +- std across seeds.",
        ],
    )

    write_text(out_dir / "latex_worst_tracking_compact_full_with_keff_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)
    print(f"\n[Done] Ablation 04b compact-full + K_eff outputs: {out_dir}")
    print("[Paper recommendation] Use the compact table to jointly show worst-environment non-stationarity, persistence, lambda alignment, and concentration.")


if __name__ == "__main__":
    main()
