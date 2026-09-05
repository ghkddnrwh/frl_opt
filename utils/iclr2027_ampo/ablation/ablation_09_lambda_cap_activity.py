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
    summarize_seed_values,
    validate_expected_layout,
    write_csv,
    write_text,
)


def per_seed_cap_activity(run: dict[str, np.ndarray], late_evals: int) -> dict[str, float]:
    er = np.asarray(run["eval_rounds"], dtype=int)
    sr = np.asarray(run["server_rounds"], dtype=int)
    if len(er) == 0 or len(sr) == 0:
        return {}
    threshold = int(er[max(0, len(er) - late_evals)])
    idx = np.flatnonzero(sr >= threshold)
    if len(idx) == 0:
        return {}

    la = np.asarray(run["lambda_after"], dtype=float)[idx]
    valid = np.all(np.isfinite(la), axis=1)
    la = la[valid]
    if len(la) == 0:
        return {}

    cap_value = float(np.asarray(run.get("dual_lambda_cap", np.nan)).reshape(-1)[0])
    if not np.isfinite(cap_value):
        return {
            "cap": np.nan,
            "cap_hit_rate": 0.0,
            "lambda_max": float(np.mean(np.max(la, axis=1))),
            "support_size": float(np.mean(np.sum(la > 1e-10, axis=1))),
            "k_eff": float(np.mean(1.0 / np.sum(np.square(la), axis=1))),
        }

    hit = np.any(np.isclose(la, cap_value, atol=1e-7, rtol=1e-6), axis=1)
    return {
        "cap": cap_value,
        "cap_hit_rate": float(np.mean(hit)),
        "lambda_max": float(np.mean(np.max(la, axis=1))),
        "support_size": float(np.mean(np.sum(la > 1e-10, axis=1))),
        "k_eff": float(np.mean(1.0 / np.sum(np.square(la), axis=1))),
    }


def main() -> None:
    parser = common_parser(
        "Ablation 09: report where the currently configured lambda cap is actually active. This is not a causal cap ablation."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "09_lambda_cap_activity"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)
    metrics = ("cap", "cap_hit_rate", "lambda_max", "support_size", "k_eff")

    csv_rows, latex_rows = [], []
    for variant in ADAPTIVE_VARIANTS:
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                seed_metrics = [(spec.seed, per_seed_cap_activity(run, args.late_evals)) for spec, run in loaded]
                cells = [SHORT_VARIANT[variant], DISPLAY_ENV[env], pert.capitalize()]
                for key in metrics:
                    vals = [m.get(key, np.nan) for _, m in seed_metrics]
                    mean, std, n = summarize_seed_values(vals)
                    cells.append(format_pm(mean, std, digits=2))
                    csv_rows.append([variant, env, pert, key, mean, std, n])
                latex_rows.append(" & ".join(cells) + r" \\")

    write_csv(
        out_dir / "lambda_cap_activity_summary.csv",
        ["variant", "environment", "perturbation", "metric", "seed_mean", "seed_std", "num_seeds"],
        csv_rows,
    )
    text = make_latex_table_rows(
        [
            "Method", "Environment", "Perturbation", "Cap", "Late Cap-Hit Rate",
            r"$\lambda_{\max}$", "Support Size", r"$K_{\mathrm{eff}}$",
        ],
        latex_rows,
        comments=[
            "IMPORTANT: current archive contains capped adaptive runs, so this table measures cap activity only.",
            "It does NOT estimate the causal effect of the cap. For that, add matched cap=None runs.",
        ],
    )
    write_text(out_dir / "latex_lambda_cap_activity_rows.txt", text)
    print("\n[LaTeX rows]\n" + text)

    print(f"\n[Done] Ablation 09 outputs: {out_dir}")
    print("[Paper recommendation] Use this as a sanity/activity diagnostic only. A real cap ablation requires matched uncapped runs with the same seeds/settings.")


if __name__ == "__main__":
    main()
