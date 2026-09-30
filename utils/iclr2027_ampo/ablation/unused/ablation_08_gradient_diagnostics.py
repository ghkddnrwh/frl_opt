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
    summarize_seed_values,
    task_label,
    validate_expected_layout,
    write_csv,
    write_text,
)

METRICS = (
    "pairwise_grad_conflict_rate",
    "high_lambda_conflict_rate",
    "top_lambda_grad_cosine_to_aggregate",
    "worst_return_grad_cosine_to_aggregate",
    "lambda_influence_l1_gap",
    "max_influence_is_worst_return",
)
LABELS = (
    "Pairwise conflict",
    "High-$\\lambda$ conflict",
    r"$\cos(g_{top\lambda},g_\lambda)$",
    r"$\cos(g_{worst},g_\lambda)$",
    r"$\|p-\lambda\|_1$",
    "Max-influence = worst",
)


def per_seed_diag(run: dict[str, np.ndarray], late_evals: int) -> dict[str, float]:
    er = np.asarray(run["eval_rounds"], dtype=int)
    sr = np.asarray(run["server_rounds"], dtype=int)
    if len(er) == 0 or len(sr) == 0:
        return {}
    threshold = int(er[max(0, len(er) - late_evals)])
    idx = np.flatnonzero(sr >= threshold)
    if len(idx) == 0:
        return {}
    out = {}
    for key in METRICS:
        arr = np.asarray(run.get(f"server_{key}", np.array([])), dtype=float)
        if len(arr) != len(sr):
            out[key] = np.nan
            continue
        vals = arr[idx]
        vals = vals[np.isfinite(vals)]
        out[key] = float(np.mean(vals)) if len(vals) else np.nan
    return out


def draw_heatmap(matrix, row_labels, col_labels, title, stem):
    fig, ax = plt.subplots(figsize=(10.2, 5.6))
    im = ax.imshow(matrix, aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", rotation=18)
    ax.set_title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            ax.text(j, i, "--" if not np.isfinite(v) else f"{v:.2f}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.03)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout(pad=0.2)
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    parser = common_parser(
        "Ablation 08: summarize AMPO gradient-geometry diagnostics already present in the adaptive runs."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "08_gradient_diagnostics"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)
    task_rows = [(env, pert) for env in ENV_ORDER for pert in PERTURBATION_ORDER]
    csv_rows, latex_rows = [], []

    for variant in ADAPTIVE_VARIANTS:
        matrix = np.full((len(task_rows), len(METRICS)), np.nan, dtype=float)
        for i, (env, pert) in enumerate(task_rows):
            loaded = load_group(
                get_group_runs(groups, variant, env, pert),
                cache_root,
                force_cache=args.force_cache,
            )
            seed_metrics = [(spec.seed, per_seed_diag(run, args.late_evals)) for spec, run in loaded]
            cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
            for j, key in enumerate(METRICS):
                vals = [m.get(key, np.nan) for _, m in seed_metrics]
                mean, std, n = summarize_seed_values(vals)
                matrix[i, j] = mean
                cells.append(format_pm(mean, std, digits=2))
                csv_rows.append([variant, env, pert, key, mean, std, n])
            latex_rows.append(" & ".join(cells) + r" \\")

        suffix = "1e-4" if variant.endswith("1e-4") else "3e-4"
        draw_heatmap(
            matrix,
            [task_label(e, p) for e, p in task_rows],
            LABELS,
            f"Gradient diagnostics ({SHORT_VARIANT[variant]})",
            out_dir / f"gradient_diagnostics_{suffix}",
        )

    write_csv(
        out_dir / "gradient_diagnostics_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    text = make_latex_table_rows(
        [
            "Method", "Environment", "Perturbation", "Pairwise Conflict", r"High-$\lambda$ Conflict",
            r"Top-$\lambda$ Cosine", "Worst-Return Cosine", r"$\lambda$-Influence $L_1$ Gap",
            "Max Influence = Worst",
        ],
        latex_rows,
    )
    write_text(out_dir / "latex_gradient_diagnostics_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    print(f"\n[Done] Ablation 08 outputs: {out_dir}")
    print("[Paper recommendation] Put this in the appendix unless gradient conflict becomes a central claim. The key comparison is pairwise conflict vs high-lambda/aggregate alignment.")


if __name__ == "__main__":
    main()
