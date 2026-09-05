from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
    save_figure,
    select_late_eval_server_pairs,
    spearman_corr,
    summarize_seed_values,
    task_label,
    validate_expected_layout,
    write_csv,
    write_text,
)


CORR_KEYS = (
    "signal_fidelity",
    "dual_responsiveness",
    "actor_eval_alignment",
    "postdual_eval_alignment",
)
CORR_LABELS = (
    r"$\rho(J^{dual},J^{eval})$",
    r"$\rho(\lambda^{+},-J^{dual})$",
    r"$\rho(\lambda^{actor},-J^{eval})$",
    r"$\rho(\lambda^{+},-J^{eval})$",
)
FOCUS_KEYS = ("top1_match", "worst_mass", "bottom2_mass")
FOCUS_LABELS = ("Top-1 match", r"Worst $\lambda$ mass", r"Bottom-2 $\lambda$ mass")


def per_seed_alignment(run: dict[str, np.ndarray], late_evals: int) -> dict[str, float]:
    e_idx, s_idx = select_late_eval_server_pairs(run, late_evals=late_evals)
    if len(e_idx) == 0:
        return {key: np.nan for key in (*CORR_KEYS, *FOCUS_KEYS)}

    values = {key: [] for key in (*CORR_KEYS, *FOCUS_KEYS)}
    for e, s in zip(e_idx, s_idx):
        jeval = np.asarray(run["eval_local"][e], dtype=float)
        jdual = np.asarray(run["server_returns"][s], dtype=float)
        lactor = np.asarray(run["lambda_actor"][s], dtype=float)
        lafter = np.asarray(run["lambda_after"][s], dtype=float)
        valid = np.isfinite(jeval) & np.isfinite(jdual) & np.isfinite(lactor) & np.isfinite(lafter)
        if np.sum(valid) < 2:
            continue
        jeval = jeval[valid]
        jdual = jdual[valid]
        lactor = lactor[valid]
        lafter = lafter[valid]

        values["signal_fidelity"].append(spearman_corr(jdual, jeval))
        values["dual_responsiveness"].append(spearman_corr(lafter, -jdual))
        values["actor_eval_alignment"].append(spearman_corr(lactor, -jeval))
        values["postdual_eval_alignment"].append(spearman_corr(lafter, -jeval))

        worst = int(np.argmin(jeval))
        top = int(np.argmax(lactor))
        bottom2 = np.argsort(jeval)[: min(2, len(jeval))]
        values["top1_match"].append(float(top == worst))
        values["worst_mass"].append(float(lactor[worst]))
        values["bottom2_mass"].append(float(np.sum(lactor[bottom2])))

    return {
        key: float(np.nanmean(v)) if len(v) and np.any(np.isfinite(v)) else np.nan
        for key, v in values.items()
    }


def draw_heatmap(matrix, row_labels, col_labels, title, stem, vmin, vmax, fmt=".2f") -> None:
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", rotation=18)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "--" if not np.isfinite(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=11)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout(pad=0.2)
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    parser = common_parser(
        "Ablation 02: quantify whether the AMPO dual tracks training and evaluation-defined bottlenecks."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "02_dual_tracking_alignment"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)

    csv_rows = []
    latex_rows = []
    task_rows = [(env, pert) for env in ENV_ORDER for pert in PERTURBATION_ORDER]

    for variant in ADAPTIVE_VARIANTS:
        setting_stats = {}
        for env, pert in task_rows:
            loaded = load_group(
                get_group_runs(groups, variant, env, pert),
                cache_root,
                force_cache=args.force_cache,
            )
            seed_metrics = [(spec.seed, per_seed_alignment(run, args.late_evals)) for spec, run in loaded]
            setting_stats[(env, pert)] = seed_metrics

            cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
            for key in (*CORR_KEYS, *FOCUS_KEYS):
                vals = [m[key] for _, m in seed_metrics]
                mean, std, n = summarize_seed_values(vals)
                cells.append(format_pm(mean, std, digits=2))
                csv_rows.append([variant, env, pert, key, mean, std, n])
            latex_rows.append(" & ".join(cells) + r" \\")

        corr_matrix = np.empty((len(task_rows), len(CORR_KEYS)), dtype=float)
        focus_matrix = np.empty((len(task_rows), len(FOCUS_KEYS)), dtype=float)
        for i, setting in enumerate(task_rows):
            seed_metrics = setting_stats[setting]
            for j, key in enumerate(CORR_KEYS):
                corr_matrix[i, j] = summarize_seed_values([m[key] for _, m in seed_metrics])[0]
            for j, key in enumerate(FOCUS_KEYS):
                focus_matrix[i, j] = summarize_seed_values([m[key] for _, m in seed_metrics])[0]

        row_labels = [task_label(env, pert) for env, pert in task_rows]
        suffix = "1e-4" if variant.endswith("1e-4") else "3e-4"
        draw_heatmap(
            corr_matrix,
            row_labels,
            CORR_LABELS,
            f"Dual tracking correlations ({SHORT_VARIANT[variant]})",
            out_dir / f"dual_tracking_correlations_{suffix}",
            vmin=-1.0,
            vmax=1.0,
        )
        draw_heatmap(
            focus_matrix,
            row_labels,
            FOCUS_LABELS,
            f"Worst-environment focusing ({SHORT_VARIANT[variant]})",
            out_dir / f"worst_focusing_{suffix}",
            vmin=0.0,
            vmax=1.0,
        )

    write_csv(
        out_dir / "dual_tracking_alignment_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    text = make_latex_table_rows(
        [
            "Method", "Environment", "Perturbation", "Signal Fidelity", "Dual Responsiveness",
            "Actor/Eval Alignment", "Post-Dual/Eval Alignment", "Top-1 Match", "Worst Mass", "Bottom-2 Mass",
        ],
        latex_rows,
        comments=[
            f"Each cell averages the final {args.late_evals} matched eval points within each seed, then reports mean +- std across seeds.",
            "Same-round associations are diagnostics, not causal estimates.",
        ],
    )
    write_text(out_dir / "latex_dual_tracking_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    print(f"\n[Done] Ablation 02 outputs: {out_dir}")
    print("[Paper recommendation] Use the two heatmaps in the main mechanistic ablation. The LaTeX rows provide exact values for appendix/table support.")


if __name__ == "__main__":
    main()
