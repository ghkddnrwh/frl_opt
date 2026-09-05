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
    normalized_entropy,
    select_late_eval_server_pairs,
    summarize_seed_values,
    validate_expected_layout,
    write_csv,
    write_text,
)


def format_sci_pm(mean: float, std: float) -> str:
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(std):
        return rf"${mean:.2e}$"
    return rf"${mean:.2e} \pm {std:.2e}$"


def per_seed_timescale(run: dict[str, np.ndarray], late_evals: int) -> dict[str, float]:
    e_idx, s_idx = select_late_eval_server_pairs(run, late_evals=late_evals)
    if len(e_idx) == 0:
        return {}

    eval_rounds = np.asarray(run["eval_rounds"], dtype=int)[e_idx]
    threshold_round = int(np.min(eval_rounds))
    server_rounds = np.asarray(run["server_rounds"], dtype=int)
    late_server = np.flatnonzero(server_rounds >= threshold_round)
    if len(late_server) == 0:
        return {}

    lactor_all = np.asarray(run["lambda_actor"], dtype=float)[late_server]
    finite_rows = np.all(np.isfinite(lactor_all), axis=1)
    lactor_all = lactor_all[finite_rows]
    if len(lactor_all) == 0:
        return {}

    lambda_max = np.max(lactor_all, axis=1)
    k_eff = 1.0 / np.sum(np.square(lactor_all), axis=1)
    entropy = np.asarray([normalized_entropy(row) for row in lactor_all], dtype=float)

    delta_norm = np.asarray(run.get("server_lambda_delta_norm", np.array([])), dtype=float)
    update_applied = np.asarray(run.get("server_dual_update_applied", np.array([])), dtype=float)
    if len(delta_norm) == len(server_rounds):
        dn = delta_norm[late_server]
        if len(update_applied) == len(server_rounds):
            ua = update_applied[late_server]
            dn = dn[np.isfinite(dn) & (ua > 0.5)]
        else:
            dn = dn[np.isfinite(dn)]
        mean_update_norm = float(np.mean(dn)) if len(dn) else np.nan
    else:
        mean_update_norm = np.nan

    cap_count = np.asarray(run.get("server_lambda_num_at_cap", np.array([])), dtype=float)
    if len(cap_count) == len(server_rounds):
        cc = cap_count[late_server]
        cap_hit_rate = float(np.mean(cc[np.isfinite(cc)] > 0.0)) if np.any(np.isfinite(cc)) else np.nan
    else:
        cap_hit_rate = np.nan

    eval_local = np.asarray(run["eval_local"], dtype=float)[e_idx]
    lactor_eval = np.asarray(run["lambda_actor"], dtype=float)[s_idx]
    worst_mass, bottom2_mass = [], []
    for jeval, lam in zip(eval_local, lactor_eval):
        if not (np.all(np.isfinite(jeval)) and np.all(np.isfinite(lam))):
            continue
        worst = int(np.argmin(jeval))
        bottom2 = np.argsort(jeval)[: min(2, len(jeval))]
        worst_mass.append(float(lam[worst]))
        bottom2_mass.append(float(np.sum(lam[bottom2])))

    return {
        "lambda_max": float(np.mean(lambda_max)),
        "k_eff": float(np.mean(k_eff)),
        "normalized_entropy": float(np.mean(entropy)),
        "worst_mass": float(np.mean(worst_mass)) if worst_mass else np.nan,
        "bottom2_mass": float(np.mean(bottom2_mass)) if bottom2_mass else np.nan,
        "dual_update_norm": mean_update_norm,
        "cap_hit_rate": cap_hit_rate,
    }


def main() -> None:
    parser = common_parser(
        "Ablation 05: dual learning-rate timescale and concentration/effective-parallelism trade-off."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "05_dual_lr_timescale"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)
    metric_order = (
        "lambda_max",
        "k_eff",
        "normalized_entropy",
        "worst_mass",
        "bottom2_mass",
        "dual_update_norm",
        "cap_hit_rate",
    )

    csv_rows, latex_rows = [], []
    global_seed_values = {variant: {metric: [] for metric in metric_order} for variant in ADAPTIVE_VARIANTS}

    for env in ENV_ORDER:
        for pert in PERTURBATION_ORDER:
            for variant in ADAPTIVE_VARIANTS:
                loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                seed_metrics = [(spec.seed, per_seed_timescale(run, args.late_evals)) for spec, run in loaded]
                cells = [DISPLAY_ENV[env], pert.capitalize(), SHORT_VARIANT[variant]]
                for metric in metric_order:
                    vals = [m.get(metric, np.nan) for _, m in seed_metrics]
                    mean, std, n = summarize_seed_values(vals)
                    cells.append(format_sci_pm(mean, std) if metric == "dual_update_norm" else format_pm(mean, std, digits=2))
                    csv_rows.append([variant, env, pert, metric, mean, std, n])
                    global_seed_values[variant][metric].extend([v for v in vals if np.isfinite(v)])
                latex_rows.append(" & ".join(cells) + r" \\")

    write_csv(
        out_dir / "dual_lr_timescale_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    text = make_latex_table_rows(
        [
            "Environment", "Perturbation", "Dual LR", r"$\lambda_{\max}$", r"$K_{\mathrm{eff}}$",
            "Normalized Entropy", "Worst Mass", "Bottom-2 Mass", "Dual-Update Norm", "Cap-Hit Rate",
        ],
        latex_rows,
        comments=[f"Late-training window starts at the first of the final {args.late_evals} evaluation points."],
    )
    write_text(out_dir / "latex_dual_lr_timescale_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    global_rows = []
    for variant in ADAPTIVE_VARIANTS:
        cells = [SHORT_VARIANT[variant]]
        for metric in metric_order:
            mean, std, _ = summarize_seed_values(global_seed_values[variant][metric])
            cells.append(format_sci_pm(mean, std) if metric == "dual_update_norm" else format_pm(mean, std, digits=2))
        global_rows.append(" & ".join(cells) + r" \\")
    global_text = make_latex_table_rows(
        [
            "Dual LR", r"$\lambda_{\max}$", r"$K_{\mathrm{eff}}$", "Normalized Entropy",
            "Worst Mass", "Bottom-2 Mass", "Dual-Update Norm", "Cap-Hit Rate",
        ],
        global_rows,
        comments=["Descriptive aggregate across all task/perturbation/seed units; do not treat this as a cross-task return average."],
    )
    write_text(out_dir / "latex_dual_lr_global_descriptive_rows.txt", global_text)
    print("\n[LaTeX rows: global descriptive]\n" + global_text)

    print(f"\n[Done] Ablation 05 outputs: {out_dir}")
    print("[Paper recommendation] Use a table: with only two dual learning rates, exact concentration/K_eff values are more informative than a line plot.")


if __name__ == "__main__":
    main()
