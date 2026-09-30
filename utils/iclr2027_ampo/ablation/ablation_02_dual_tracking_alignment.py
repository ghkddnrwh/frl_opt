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
    find_server_indices_for_eval_rounds,
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


# Only these two quantities are visualized in the main heatmap / learning curves.
# The definitions intentionally preserve the first two metrics from the original
# analysis:
#   1) signal fidelity:       rho(J_dual, J_eval)
#   2) dual responsiveness:   rho(lambda_after, -J_dual)
TRACK_KEYS = (
    "signal_fidelity",
    "dual_responsiveness",
)
TRACK_LABELS = (
    r"$\rho(J^{dual},J^{eval})$",
    r"$\rho(\lambda^{+},-J^{dual})$",
)

# Keep the remaining metrics in the exact-value CSV / LaTeX table for appendix
# support, but do not draw them in the heatmap.
CORR_KEYS = (
    "signal_fidelity",
    "dual_responsiveness",
    "actor_eval_alignment",
    "postdual_eval_alignment",
)
FOCUS_KEYS = ("top1_match", "worst_mass", "bottom2_mass")

# Compact paper-table metrics. K_eff is computed from the same post-dual
# weights (lambda^+) used by the second alignment metric, so the three columns
# summarize signal fidelity -> dual targeting -> concentration.
COMPACT_TABLE_KEYS = (
    "signal_fidelity",
    "dual_responsiveness",
    "keff_after",
)


def per_seed_alignment(run: dict[str, np.ndarray], late_evals: int) -> dict[str, float]:
    """Late-training summary used for the CSV / LaTeX table."""
    e_idx, s_idx = select_late_eval_server_pairs(run, late_evals=late_evals)
    if len(e_idx) == 0:
        return {key: np.nan for key in (*CORR_KEYS, *FOCUS_KEYS, "keff_after")}

    values = {key: [] for key in (*CORR_KEYS, *FOCUS_KEYS, "keff_after")}
    for e, s in zip(e_idx, s_idx):
        jeval = np.asarray(run["eval_local"][e], dtype=float)
        jdual = np.asarray(run["server_returns"][s], dtype=float)
        lactor = np.asarray(run["lambda_actor"][s], dtype=float)
        lafter = np.asarray(run["lambda_after"][s], dtype=float)

        # Effective number of clients from the post-dual weights:
        #   K_eff = 1 / sum_k (lambda_k^+)^2.
        # Use the complete lambda vector so missing clients never change the
        # normalization implicitly.
        if len(lafter) and np.all(np.isfinite(lafter)):
            denom = float(np.sum(np.square(lafter)))
            if denom > 0.0:
                values["keff_after"].append(1.0 / denom)

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


def per_seed_tracking_curve(run: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Compute the two requested Spearman similarities at every matched evaluation
    round over the *full training trajectory*.

    Each point is a correlation across the clients at one communication round.
    """
    e_idx, s_idx = find_server_indices_for_eval_rounds(run)
    if len(e_idx) == 0:
        return {
            "round": np.empty(0, dtype=np.int64),
            "signal_fidelity": np.empty(0, dtype=float),
            "dual_responsiveness": np.empty(0, dtype=float),
        }

    rounds: list[int] = []
    signal_fidelity: list[float] = []
    dual_responsiveness: list[float] = []

    eval_rounds = np.asarray(run["eval_rounds"], dtype=np.int64)

    for e, s in zip(e_idx, s_idx):
        jeval = np.asarray(run["eval_local"][e], dtype=float)
        jdual = np.asarray(run["server_returns"][s], dtype=float)
        lafter = np.asarray(run["lambda_after"][s], dtype=float)

        valid = np.isfinite(jeval) & np.isfinite(jdual) & np.isfinite(lafter)
        if np.sum(valid) < 2:
            continue

        jeval = jeval[valid]
        jdual = jdual[valid]
        lafter = lafter[valid]

        rounds.append(int(eval_rounds[e]))
        signal_fidelity.append(spearman_corr(jdual, jeval))
        dual_responsiveness.append(spearman_corr(lafter, -jdual))

    return {
        "round": np.asarray(rounds, dtype=np.int64),
        "signal_fidelity": np.asarray(signal_fidelity, dtype=float),
        "dual_responsiveness": np.asarray(dual_responsiveness, dtype=float),
    }


def aggregate_seed_curves(
    seed_curves: list[tuple[int, dict[str, np.ndarray]]],
    key: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate exact-round curve values across seeds; no interpolation."""
    all_rounds = sorted(
        {
            int(r)
            for _, curve in seed_curves
            for r in np.asarray(curve["round"], dtype=np.int64)
        }
    )

    means: list[float] = []
    stds: list[float] = []
    counts: list[int] = []

    round_maps = []
    for _, curve in seed_curves:
        round_maps.append(
            {
                int(r): float(v)
                for r, v in zip(curve["round"], curve[key])
                if np.isfinite(v)
            }
        )

    for r in all_rounds:
        vals = [m[r] for m in round_maps if r in m and np.isfinite(m[r])]
        mean, std, n = summarize_seed_values(vals)
        means.append(mean)
        stds.append(std)
        counts.append(n)

    return (
        np.asarray(all_rounds, dtype=np.int64),
        np.asarray(means, dtype=float),
        np.asarray(stds, dtype=float),
        np.asarray(counts, dtype=np.int64),
    )


def draw_heatmap(matrix, row_labels, col_labels, title, stem, vmin=-1.0, vmax=1.0, fmt=".2f") -> None:
    """Draw one compact heatmap containing only the two requested metrics."""
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    im = ax.imshow(matrix, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(col_labels)), labels=col_labels)
    ax.set_yticks(np.arange(len(row_labels)), labels=row_labels)
    ax.tick_params(axis="x", rotation=10)
    # ax.set_title(title)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix[i, j]
            text = "--" if not np.isfinite(value) else format(value, fmt)
            ax.text(j, i, text, ha="center", va="center", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04)
    cbar.ax.tick_params(labelsize=11)
    fig.tight_layout(pad=0.2)
    save_figure(fig, stem)
    plt.close(fig)


def draw_tracking_learning_curve(
    seed_curves: list[tuple[int, dict[str, np.ndarray]]],
    title: str,
    stem,
) -> None:
    """Plot seed mean +/- seed std of both similarities over communication rounds."""
    fig, ax = plt.subplots(figsize=(7.8, 4.8))

    for key, label in zip(TRACK_KEYS, TRACK_LABELS):
        rounds, mean, std, _ = aggregate_seed_curves(seed_curves, key)
        valid = np.isfinite(rounds) & np.isfinite(mean)
        if not np.any(valid):
            continue

        x = rounds[valid]
        y = mean[valid]
        s = std[valid]
        line, = ax.plot(x, y, linewidth=2.0, label=label)
        finite_std = np.where(np.isfinite(s), s, 0.0)
        ax.fill_between(
            x,
            np.clip(y - finite_std, -1.0, 1.0),
            np.clip(y + finite_std, -1.0, 1.0),
            alpha=0.18,
            color=line.get_color(),
            linewidth=0,
        )

    ax.axhline(0.0, linewidth=1.0, linestyle="--", alpha=0.6)
    ax.set_xlabel("Communication Round")
    ax.set_ylabel("Spearman Correlation")
    ax.set_ylim(-1.05, 1.05)
    # ax.set_title(title)
    ax.grid(True, alpha=0.22)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout(pad=0.2)
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    parser = common_parser(
        "Ablation 02: track dual/evaluation signal fidelity and lambda/dual-return alignment."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "02_dual_tracking_alignment"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)

    csv_rows = []
    curve_csv_rows = []
    latex_rows = []
    compact_csv_rows = []
    compact_latex_rows = []
    task_rows = [(env, pert) for env in ENV_ORDER for pert in PERTURBATION_ORDER]

    for variant in ADAPTIVE_VARIANTS:
        setting_stats = {}
        setting_curves = {}

        for env, pert in task_rows:
            loaded = load_group(
                get_group_runs(groups, variant, env, pert),
                cache_root,
                force_cache=args.force_cache,
            )

            # Late-window exact values for the existing summary table.
            seed_metrics = [(spec.seed, per_seed_alignment(run, args.late_evals)) for spec, run in loaded]
            setting_stats[(env, pert)] = seed_metrics

            cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
            for key in (*CORR_KEYS, *FOCUS_KEYS):
                vals = [m[key] for _, m in seed_metrics]
                mean, std, n = summarize_seed_values(vals)
                cells.append(format_pm(mean, std, digits=2))
                csv_rows.append([variant, env, pert, key, mean, std, n])
            latex_rows.append(" & ".join(cells) + r" \\")

            # Compact paper table: signal fidelity, lambda/dual-return
            # alignment, and post-dual concentration K_eff.
            compact_cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
            compact_values = {}
            for key in COMPACT_TABLE_KEYS:
                vals = [m[key] for _, m in seed_metrics]
                mean, std, n = summarize_seed_values(vals)
                compact_values[key] = (mean, std, n)
                compact_cells.append(format_pm(mean, std, digits=2))
            compact_latex_rows.append(" & ".join(compact_cells) + r" \\")
            compact_csv_rows.append([
                variant,
                env,
                pert,
                compact_values["signal_fidelity"][0],
                compact_values["signal_fidelity"][1],
                compact_values["dual_responsiveness"][0],
                compact_values["dual_responsiveness"][1],
                compact_values["keff_after"][0],
                compact_values["keff_after"][1],
                min(v[2] for v in compact_values.values()),
            ])

            # Full-trajectory curves for the two requested similarities.
            seed_curves = [(spec.seed, per_seed_tracking_curve(run)) for spec, run in loaded]
            setting_curves[(env, pert)] = seed_curves

            for seed, curve in seed_curves:
                for r, sig, dual in zip(
                    curve["round"],
                    curve["signal_fidelity"],
                    curve["dual_responsiveness"],
                ):
                    curve_csv_rows.append([
                        variant,
                        env,
                        pert,
                        seed,
                        int(r),
                        float(sig),
                        float(dual),
                    ])

            suffix = "1e-4" if variant.endswith("1e-4") else "3e-4"
            draw_tracking_learning_curve(
                seed_curves,
                title=f"{DISPLAY_ENV[env]}-{pert.capitalize()} ({SHORT_VARIANT[variant]})",
                stem=(
                    out_dir
                    / "learning_curves"
                    / suffix
                    / f"{env}_{pert}_dual_tracking_similarity"
                ),
            )

        # One heatmap per dual LR, with only two columns.
        tracking_matrix = np.empty((len(task_rows), len(TRACK_KEYS)), dtype=float)
        for i, setting in enumerate(task_rows):
            seed_metrics = setting_stats[setting]
            for j, key in enumerate(TRACK_KEYS):
                tracking_matrix[i, j] = summarize_seed_values([m[key] for _, m in seed_metrics])[0]

        row_labels = [task_label(env, pert) for env, pert in task_rows]
        suffix = "1e-4" if variant.endswith("1e-4") else "3e-4"
        draw_heatmap(
            tracking_matrix,
            row_labels,
            TRACK_LABELS,
            f"Dual tracking ({SHORT_VARIANT[variant]})",
            out_dir / f"dual_tracking_two_metrics_{suffix}",
            vmin=-1.0,
            vmax=1.0,
        )

    write_csv(
        out_dir / "dual_tracking_alignment_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    write_csv(
        out_dir / "dual_tracking_learning_curves.csv",
        [
            "variant",
            "environment",
            "perturbation",
            "seed",
            "round",
            "signal_fidelity",
            "dual_responsiveness",
        ],
        curve_csv_rows,
    )

    write_csv(
        out_dir / "dual_tracking_keff_compact_summary.csv",
        [
            "variant",
            "environment",
            "perturbation",
            "signal_fidelity_mean",
            "signal_fidelity_std",
            "lambda_dual_alignment_mean",
            "lambda_dual_alignment_std",
            "keff_after_mean",
            "keff_after_std",
            "num_seeds",
        ],
        compact_csv_rows,
    )

    compact_text = make_latex_table_rows(
        [
            "Method",
            "Environment",
            "Perturbation",
            r"$\rho(J^{dual},J^{eval})$",
            r"$\rho(\lambda^{+},-J^{dual})$",
            r"$K_{\mathrm{eff}}$",
        ],
        compact_latex_rows,
        comments=[
            f"Each metric is averaged over the final {args.late_evals} exact matched evaluation rounds within each seed, then reported as mean +- std across seeds.",
            r"$K_{\mathrm{eff}}=1/\sum_k(\lambda_k^{+})^2$ uses the post-dual weights, matching the lambda in the second column.",
            "The first two columns are Spearman rank correlations; K_eff is a concentration statistic, not a correlation.",
        ],
    )
    write_text(out_dir / "latex_dual_tracking_keff_rows.txt", compact_text)
    print("\n[Compact LaTeX table: tracking + K_eff]\n" + compact_text)

    text = make_latex_table_rows(
        [
            "Method", "Environment", "Perturbation", "Signal Fidelity", "Dual Responsiveness",
            "Actor/Eval Alignment", "Post-Dual/Eval Alignment", "Top-1 Match", "Worst Mass", "Bottom-2 Mass",
        ],
        latex_rows,
        comments=[
            f"Each cell averages the final {args.late_evals} matched eval points within each seed, then reports mean +- std across seeds.",
            "Heatmaps visualize only Signal Fidelity and Dual Responsiveness.",
            "Learning curves use all exact matched evaluation rounds; no interpolation is used.",
            "Dual Responsiveness here preserves the original definition rho(lambda^+, -J_dual).",
            "Same-round associations are diagnostics, not causal estimates.",
        ],
    )
    write_text(out_dir / "latex_dual_tracking_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    print(f"\n[Done] Ablation 02 outputs: {out_dir}")
    print(
        "[Paper recommendation] Use the 2-column heatmap plus selected learning-curve examples. "
        "The compact Signal-Fidelity / Lambda-Dual-Alignment / K_eff table provides a concise "
        "numerical mechanism summary; the full exact-value table remains available for appendix support."
    )


if __name__ == "__main__":
    main()
