import os
import numpy as np


# ============================================================================
# Config
# ============================================================================

cache_root = "logs/iclr2027_ampo/ablation/_cache"

envs = [
    "ant",
    "halfcheetah",
    "hopper",
    "walker2d",
]

perts = [
    "friction",
    "gravity",
]


# 기본: 논문에서 사용하는 Adaptive 3e-4
methods = [
    (
        "ppo_avg",
        "PPOAvg",
        False,
    ),
    (
        "ampo_uniform",
        "AMPO-Uniform",
        False,
    ),
    (
        "ampo_adaptive_dual_lr_3e-4",
        r"AMPO-Adaptive ($3\times10^{-4}$)",
        True,
    ),

    # 필요하면 이것도 추가
    # (
    #     "ampo_adaptive_dual_lr_1e-4",
    #     r"AMPO-Adaptive ($10^{-4}$)",
    #     True,
    # ),
]


DISPLAY_ENV = {
    "ant": "Ant",
    "halfcheetah": "HalfCheetah",
    "hopper": "Hopper",
    "walker2d": "Walker2d",
}

DISPLAY_PERT = {
    "friction": "Friction",
    "gravity": "Gravity",
}


LATE_EVALS = 10


# ============================================================================
# Helper:
# last-N eval rounds와 정확히 대응되는 lambda_actor 선택
# ============================================================================

def late_eval_matched_lambda_actor(
    data,
    late_evals=10,
):
    """
    Adaptive run 하나에 대해:

      1. 마지막 late_evals evaluation round 선택
      2. 같은 round의 server row를 정확히 매칭
      3. 그때 actor update에 실제 사용된 lambda_actor 반환

    return shape:
        (# matched late evals, num_clients)
    """

    eval_rounds = np.asarray(
        data["eval_rounds"],
        dtype=np.int64,
    )

    server_rounds = np.asarray(
        data["server_rounds"],
        dtype=np.int64,
    )

    lambda_actor = np.asarray(
        data["lambda_actor"],
        dtype=float,
    )

    if (
        len(eval_rounds) == 0
        or len(server_rounds) == 0
        or len(lambda_actor) == 0
    ):
        return None

    # 마지막 N개 evaluation
    late_eval_rounds = eval_rounds[
        max(
            0,
            len(eval_rounds) - late_evals,
        ):
    ]

    # server round -> index
    server_map = {
        int(round_value): idx
        for idx, round_value
        in enumerate(server_rounds)
    }

    matched_indices = []

    for round_value in late_eval_rounds:

        idx = server_map.get(
            int(round_value)
        )

        if idx is not None:
            matched_indices.append(
                idx
            )

    if len(matched_indices) == 0:
        return None

    return lambda_actor[
        np.asarray(
            matched_indices,
            dtype=int,
        )
    ]


# ============================================================================
# Build LaTeX rows
# ============================================================================

latex_rows = []


for env in envs:

    for pert in perts:

        for (
            method,
            method_label,
            has_lambda,
        ) in methods:

            seed_profiles = []
            seed_lambdas = []

            # ================================================================
            # Load 5 seeds
            # ================================================================

            for seed in range(
                1,
                6,
            ):

                path = os.path.join(
                    cache_root,
                    method,
                    env,
                    pert,
                    f"seed_{seed}.npz",
                )

                if not os.path.exists(
                    path
                ):

                    print(
                        f"[Missing] {path}"
                    )

                    continue

                with np.load(
                    path,
                    allow_pickle=True,
                ) as d:

                    # --------------------------------------------------------
                    # Return profile:
                    #
                    # seed-wise:
                    #   last 10 evals -> client-wise mean
                    # --------------------------------------------------------

                    eval_local = np.asarray(
                        d["eval_local"],
                        dtype=float,
                    )

                    if len(
                        eval_local
                    ) == 0:
                        continue

                    profile_seed = np.mean(
                        eval_local[
                            -LATE_EVALS:
                        ],
                        axis=0,
                    )

                    seed_profiles.append(
                        profile_seed
                    )

                    # --------------------------------------------------------
                    # Adaptive lambda:
                    #
                    # same final evaluation rounds에서의 lambda_actor
                    # --------------------------------------------------------

                    if has_lambda:

                        matched_lambda = (
                            late_eval_matched_lambda_actor(
                                d,
                                late_evals=LATE_EVALS,
                            )
                        )

                        if matched_lambda is None:

                            print(
                                f"[Warning] "
                                f"No matched lambda: "
                                f"{method}/"
                                f"{env}/"
                                f"{pert}/"
                                f"seed={seed}"
                            )

                        else:

                            lambda_seed = np.mean(
                                matched_lambda,
                                axis=0,
                            )

                            seed_lambdas.append(
                                lambda_seed
                            )

            # ================================================================
            # Return summary
            # ================================================================

            if len(
                seed_profiles
            ) == 0:

                print(
                    f"[Skip] "
                    f"{method}/"
                    f"{env}/"
                    f"{pert}: "
                    f"no valid seeds"
                )

                continue

            # First average late evals within seed,
            # then average across seeds.
            profile = np.mean(
                np.asarray(
                    seed_profiles
                ),
                axis=0,
            )

            mean_return = float(
                np.mean(
                    profile
                )
            )

            spread = float(
                np.max(
                    profile
                )
                -
                np.min(
                    profile
                )
            )

            if abs(
                mean_return
            ) > 1e-12:

                rel_spread = (
                    100.0
                    * spread
                    / abs(
                        mean_return
                    )
                )

            else:

                rel_spread = np.nan

            # ================================================================
            # Return LaTeX row
            # ================================================================

            cells = [
                DISPLAY_ENV[
                    env
                ],

                DISPLAY_PERT[
                    pert
                ],

                method_label,
            ]

            cells.extend(
                [
                    f"{value:.1f}"
                    for value
                    in profile
                ]
            )

            cells.extend(
                [
                    f"{mean_return:.1f}",
                    f"{spread:.1f}",
                    (
                        f"{rel_spread:.1f}"
                        if np.isfinite(
                            rel_spread
                        )
                        else "--"
                    ),
                ]
            )

            latex_rows.append(
                " & ".join(
                    cells
                )
                + r" \\"
            )

            # ================================================================
            # Adaptive lambda row
            # ================================================================

            if has_lambda:

                if len(
                    seed_lambdas
                ) > 0:

                    lambda_profile = np.mean(
                        np.asarray(
                            seed_lambdas
                        ),
                        axis=0,
                    )

                    # lambda sum sanity check
                    lambda_sum = float(
                        np.sum(
                            lambda_profile
                        )
                    )

                    print(
                        f"[Lambda] "
                        f"{env}/{pert}/"
                        f"{method}: "
                        f"{np.round(lambda_profile, 3)} "
                        f"sum={lambda_sum:.4f}"
                    )

                    lambda_cells = [
                        "",
                        "",
                        r"\quad $\lambda_{\mathrm{actor}}$",
                    ]

                    lambda_cells.extend(
                        [
                            f"{value:.3f}"
                            for value
                            in lambda_profile
                        ]
                    )

                    # Mean / Spread / Relative Spread columns
                    # are return metrics, so lambda row에는 비워둠.
                    lambda_cells.extend(
                        [
                            "--",
                            "--",
                            "--",
                        ]
                    )

                    latex_rows.append(
                        " & ".join(
                            lambda_cells
                        )
                        + r" \\"
                    )

                else:

                    print(
                        f"[Warning] "
                        f"No lambda values for "
                        f"{method}/{env}/{pert}"
                    )


# ============================================================================
# Print copy-ready LaTeX
# ============================================================================

print(
    "\n"
    "============================================================"
)

print(
    "LaTeX rows: "
    "PPOAvg / AMPO-Uniform / AMPO-Adaptive"
)

print(
    "============================================================"
)

print(
    r"Environment & Perturbation & Method & "
    r"$-0.30$ & $-0.15$ & $0$ & $+0.15$ & $+0.30$ & "
    r"Mean & Spread & Rel. Spread (\%) \\"
)

print(
    r"\midrule"
)

for row in latex_rows:
    print(
        row
    )