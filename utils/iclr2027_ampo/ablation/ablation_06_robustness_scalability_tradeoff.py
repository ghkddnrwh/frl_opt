from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

from ampo_ablation_common import (
    ADAPTIVE_VARIANTS,
    ENV_ORDER,
    PERTURBATION_ORDER,
    SHORT_VARIANT,
    apply_publication_style,
    common_parser,
    format_pm,
    get_group_runs,
    group_specs,
    late_eval_metrics,
    load_group,
    make_latex_table_rows,
    spearman_corr,
    summarize_seed_values,
    task_label,
    validate_expected_layout,
    write_csv,
    write_text,
    save_figure,
)


def late_k_eff(run: dict[str, np.ndarray], late_evals: int) -> float:
    eval_rounds = np.asarray(run["eval_rounds"], dtype=int)
    if len(eval_rounds) == 0:
        return np.nan
    threshold = int(eval_rounds[max(0, len(eval_rounds) - late_evals)])
    sr = np.asarray(run["server_rounds"], dtype=int)
    la = np.asarray(run["lambda_actor"], dtype=float)
    idx = np.flatnonzero(sr >= threshold)
    if len(idx) == 0:
        return np.nan
    la = la[idx]
    valid = np.all(np.isfinite(la), axis=1)
    la = la[valid]
    if len(la) == 0:
        return np.nan
    return float(np.mean(1.0 / np.sum(np.square(la), axis=1)))


def main() -> None:
    parser = common_parser(
        "Ablation 06: empirical robustness-scalability trade-off using K_eff and paired worst-return gain."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "06_robustness_scalability_tradeoff"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=("ampo_uniform", *ADAPTIVE_VARIANTS))

    setting_points = []
    seed_rows = []
    all_seed_x, all_seed_y = [], []

    for variant in ADAPTIVE_VARIANTS:
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                uniform_loaded = load_group(
                    get_group_runs(groups, "ampo_uniform", env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                adaptive_loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                uniform_by_seed = {
                    spec.seed: late_eval_metrics(run, args.late_evals)["worst"]
                    for spec, run in uniform_loaded
                }
                adaptive_by_seed = {
                    spec.seed: (late_eval_metrics(run, args.late_evals)["worst"], late_k_eff(run, args.late_evals))
                    for spec, run in adaptive_loaded
                }

                xs, ys = [], []
                for seed in sorted(set(uniform_by_seed) & set(adaptive_by_seed)):
                    u = uniform_by_seed[seed]
                    a, keff = adaptive_by_seed[seed]
                    if not (np.isfinite(u) and np.isfinite(a) and np.isfinite(keff)):
                        continue
                    gain_pct = 100.0 * (a - u) / max(abs(u), 1e-12)
                    xs.append(keff)
                    ys.append(gain_pct)
                    all_seed_x.append(keff)
                    all_seed_y.append(gain_pct)
                    seed_rows.append([variant, env, pert, seed, keff, u, a, a - u, gain_pct])

                x_mean, x_std, n = summarize_seed_values(xs)
                y_mean, y_std, _ = summarize_seed_values(ys)
                setting_points.append((variant, env, pert, x_mean, x_std, y_mean, y_std, n))

    write_csv(
        out_dir / "seed_level_keff_vs_gain.csv",
        ["variant", "environment", "perturbation", "seed", "k_eff", "uniform_worst", "adaptive_worst", "absolute_gain", "percent_gain"],
        seed_rows,
    )

    # Primary paper figure: each marker is a task/perturbation mean over five paired seeds.
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    marker_by_variant = {
        "ampo_adaptive_dual_lr_1e-4": "o",
        "ampo_adaptive_dual_lr_3e-4": "s",
    }
    for variant in ADAPTIVE_VARIANTS:
        pts = [p for p in setting_points if p[0] == variant]
        xs = np.asarray([p[3] for p in pts], dtype=float)
        ys = np.asarray([p[5] for p in pts], dtype=float)
        xerr = np.asarray([p[4] for p in pts], dtype=float)
        yerr = np.asarray([p[6] for p in pts], dtype=float)
        container = ax.errorbar(
            xs,
            ys,
            xerr=xerr,
            yerr=yerr,
            fmt=marker_by_variant[variant],
            markersize=7,
            capsize=3,
            linewidth=1.4,
            label=SHORT_VARIANT[variant],
        )
        # annotate every task while preserving the method's matplotlib-assigned color
        for p in pts:
            label = f"{p[1][:2].upper()}-{p[2][0].upper()}"
            ax.annotate(label, (p[3], p[5]), xytext=(4, 4), textcoords="offset points", fontsize=9)

    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.axvline(5.0, linestyle=":", linewidth=1.2)
    ax.set_xlabel(r"Effective number of environments $K_{\mathrm{eff}}$")
    ax.set_ylabel("Worst-return gain over AMPO-Uniform (%)")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    fig.tight_layout(pad=0.2)
    save_figure(fig, out_dir / "keff_vs_worst_gain")
    plt.close(fig)

    seed_rho = spearman_corr(np.asarray(all_seed_x), np.asarray(all_seed_y))
    mean_x = np.asarray([p[3] for p in setting_points], dtype=float)
    mean_y = np.asarray([p[5] for p in setting_points], dtype=float)
    setting_rho = spearman_corr(mean_x, mean_y)
    summary = (
        "Empirical association only; this plot is not a causal test of the theoretical speedup bound.\n"
        f"Seed-level Spearman rho(K_eff, paired worst-gain%) = {seed_rho:.4f}\n"
        f"Setting-mean Spearman rho(K_eff, paired worst-gain%) = {setting_rho:.4f}\n"
        "Lower K_eff means stronger lambda concentration and less effective averaging.\n"
    )
    write_text(out_dir / "correlation_summary.txt", summary)

    latex_rows = []
    for variant, env, pert, xm, xs, ym, ys, n in setting_points:
        latex_rows.append(
            " & ".join(
                [SHORT_VARIANT[variant], task_label(env, pert), format_pm(xm, xs, 2), format_pm(ym, ys, 1)]
            )
            + r" \\"
        )
    latex_text = make_latex_table_rows(
        ["Method", "Setting", r"$K_{\mathrm{eff}}$", r"Paired Worst-Return Gain (\%)"],
        latex_rows,
        comments=["Gain is paired by seed against AMPO-Uniform."],
    )
    write_text(out_dir / "latex_keff_gain_rows.txt", latex_text)
    print("\n[LaTeX rows]\n" + latex_text)

    print(f"\n[Done] Ablation 06 outputs: {out_dir}")
    print("[Paper recommendation] Use the scatter/error-bar plot in the main paper. It is the direct empirical counterpart of the K_eff robustness-scalability discussion.")


if __name__ == "__main__":
    main()
