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
    get_group_runs,
    group_specs,
    load_group,
    save_figure,
    validate_expected_layout,
)


def collect_seed_eval_and_lambda(loaded):
    records = []
    for spec, run in loaded:
        e_idx, s_idx = find_server_indices_for_eval_rounds(run)
        if len(e_idx) == 0:
            continue
        rounds = np.asarray(run["eval_rounds"], dtype=int)[e_idx]
        eval_local = np.asarray(run["eval_local"], dtype=float)[e_idx]
        lambda_actor = np.asarray(run["lambda_actor"], dtype=float)[s_idx]
        records.append((spec.seed, rounds, eval_local, lambda_actor, np.asarray(run["noises"], dtype=float)))
    if not records:
        return None

    common = records[0][1]
    for _, rounds, _, _, _ in records[1:]:
        common = np.intersect1d(common, rounds)
    if len(common) == 0:
        return None

    eval_seeds, lambda_seeds = [], []
    for _, rounds, eval_local, lambda_actor, _ in records:
        pos = {int(r): i for i, r in enumerate(rounds)}
        idx = np.asarray([pos[int(r)] for r in common], dtype=int)
        eval_seeds.append(eval_local[idx])
        lambda_seeds.append(lambda_actor[idx])

    noises = records[0][4]
    return common, np.stack(eval_seeds), np.stack(lambda_seeds), noises


def plot_one(rounds, eval_seeds, lambda_seeds, noises, title, stem):
    k = eval_seeds.shape[-1]
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    eval_mean = np.nanmean(eval_seeds, axis=0)
    eval_std = np.nanstd(eval_seeds, axis=0, ddof=1 if eval_seeds.shape[0] > 1 else 0)
    lam_mean = np.nanmean(lambda_seeds, axis=0)
    lam_std = np.nanstd(lambda_seeds, axis=0, ddof=1 if lambda_seeds.shape[0] > 1 else 0)

    for client in range(k):
        noise = noises[client] if client < len(noises) else np.nan
        label = rf"Client {client} ($\delta={noise:+.2f}$)" if np.isfinite(noise) else f"Client {client}"
        line, = axes[0].plot(rounds, eval_mean[:, client], linewidth=2.2, label=label)
        axes[0].fill_between(
            rounds,
            eval_mean[:, client] - eval_std[:, client],
            eval_mean[:, client] + eval_std[:, client],
            color=line.get_color(),
            alpha=0.10,
            linewidth=0.0,
        )
        axes[1].plot(rounds, lam_mean[:, client], linewidth=2.2)
        axes[1].fill_between(
            rounds,
            lam_mean[:, client] - lam_std[:, client],
            lam_mean[:, client] + lam_std[:, client],
            color=line.get_color(),
            alpha=0.10,
            linewidth=0.0,
        )

    axes[0].set_ylabel("Local evaluation return")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(frameon=False, ncol=2)
    axes[0].set_title(title)

    axes[1].axhline(1.0 / k, linestyle="--", linewidth=1.5, label=r"Uniform $1/K$")
    axes[1].set_xlabel("Global Communication Rounds")
    axes[1].set_ylabel(r"Actor weight $\lambda_k$")
    axes[1].set_ylim(bottom=-0.02)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(frameon=False)

    fig.tight_layout(pad=0.25, h_pad=0.4)
    save_figure(fig, stem)
    plt.close(fig)


def main() -> None:
    parser = common_parser(
        "Ablation 03: visualize per-client evaluation returns and AMPO lambda dynamics."
    )
    args = parser.parse_args()
    apply_publication_style()
    validate_expected_layout(args.log_root)

    out_dir = args.out_root / "03_lambda_dynamics"
    cache_root = args.out_root / "_cache"
    groups = group_specs(args.log_root, variants=ADAPTIVE_VARIANTS)

    for variant in ADAPTIVE_VARIANTS:
        suffix = "1e-4" if variant.endswith("1e-4") else "3e-4"
        for env in ENV_ORDER:
            for pert in PERTURBATION_ORDER:
                loaded = load_group(
                    get_group_runs(groups, variant, env, pert),
                    cache_root,
                    force_cache=args.force_cache,
                )
                data = collect_seed_eval_and_lambda(loaded)
                if data is None:
                    print(f"[Skip] no matched eval/lambda data: {variant}, {env}, {pert}")
                    continue
                rounds, eval_seeds, lambda_seeds, noises = data
                title = f"{DISPLAY_ENV[env]}: {pert.capitalize()} ({SHORT_VARIANT[variant]})"
                stem = out_dir / suffix / f"{env}_{pert}_returns_and_lambda"
                plot_one(rounds, eval_seeds, lambda_seeds, noises, title, stem)

    note = out_dir / "PAPER_FIGURE_RECOMMENDATION.txt"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        "Primary mechanistic figure recommendation:\n"
        "  1) 3e-4/ant_friction_returns_and_lambda.png\n"
        "  2) 3e-4/ant_gravity_returns_and_lambda.png\n"
        "These two settings usually show the cleanest reversal of the bottleneck direction under friction vs gravity.\n"
        "Use other task figures in the appendix to show that worst identity can be noisier/policy-dependent.\n",
        encoding="utf-8",
    )
    print(f"[Saved] {note}")
    print(f"\n[Done] Ablation 03 outputs: {out_dir}")
    print("[Paper recommendation] This ablation should be a plot, because the paper's mechanism is temporal: returns define the bottleneck and lambda should move with it.")


if __name__ == "__main__":
    main()
