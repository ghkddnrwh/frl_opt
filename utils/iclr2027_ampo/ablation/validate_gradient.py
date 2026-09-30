from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# PPOAvg vs AMPO-Uniform
# Trajectory stability / failure-suppression analysis
#
# NO RETRAINING REQUIRED.
#
# Tests:
#   H1. Are initially hard clients more unstable under PPOAvg?
#   H2. Does AMPO-Uniform reduce that instability?
#   H3. Does instability reduction predict the final Uniform gain?
#   H4. Does PPOAvg spread grow during training while Uniform suppresses it?
#   H5. Does Uniform reduce across-seed variability, especially on hard clients?
#
# Important:
#   This can support "trajectory instability suppression".
#   It does NOT directly measure parameter drift ||theta_k - theta_server||.
# =============================================================================


ENV_ORDER = (
    "ant",
    "halfcheetah",
    "hopper",
    "walker2d",
)

PERTURBATION_ORDER = (
    "friction",
    "gravity",
)

METHODS = (
    "ppo_avg",
    "ampo_uniform",
)

DEFAULT_LOG_ROOT = Path(
    "logs/wandb_logs_final_160"
)

DEFAULT_OUT_DIR = Path(
    "logs/iclr2027_ampo/ablation/"
    "11_uniform_trajectory_stability"
)


# =============================================================================
# Run discovery
# =============================================================================


@dataclass(frozen=True)
class RunSpec:
    variant: str
    env: str
    perturbation: str
    seed: int
    run_dir: Path


def parse_seed(
    run_dir: Path,
) -> int:

    match = re.match(
        r"seed_(\d+)(?:__.*)?$",
        run_dir.name,
    )

    if not match:
        raise ValueError(
            f"Cannot parse seed from: {run_dir}"
        )

    return int(
        match.group(1)
    )


def discover_runs(
    log_root: Path,
    variants: Sequence[str] = METHODS,
) -> list[RunSpec]:

    log_root = Path(
        log_root
    )

    if not log_root.exists():
        raise FileNotFoundError(
            f"Log root does not exist: {log_root}"
        )

    runs = []

    for variant in variants:

        for env in ENV_ORDER:

            for pert in PERTURBATION_ORDER:

                parent = (
                    log_root
                    / variant
                    / env
                    / pert
                )

                if not parent.exists():
                    continue

                for run_dir in sorted(
                    parent.glob("seed_*")
                ):

                    if not run_dir.is_dir():
                        continue

                    history_path = (
                        run_dir
                        / "history.jsonl"
                    )

                    if not history_path.exists():
                        continue

                    runs.append(
                        RunSpec(
                            variant=variant,
                            env=env,
                            perturbation=pert,
                            seed=parse_seed(
                                run_dir
                            ),
                            run_dir=run_dir,
                        )
                    )

    return runs


# =============================================================================
# Config / parsing helpers
# =============================================================================


def config_info(
    run_dir: Path,
) -> tuple[int, np.ndarray]:

    config_path = (
        run_dir
        / "config.json"
    )

    if not config_path.exists():

        return (
            5,
            np.full(
                5,
                np.nan,
                dtype=float,
            ),
        )

    with config_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        config = json.load(
            f
        )

    hp = (
        config.get(
            "hyperparams",
            {},
        )
        or {}
    )

    saved = (
        config.get(
            "saved_hyperparams",
            {},
        )
        or {}
    )

    merged = {
        **saved,
        **hp,
    }

    num_clients = int(
        merged.get(
            "num_clients",
            len(
                merged.get(
                    "client_noise_values",
                    [],
                )
            )
            or 5,
        )
    )

    noise_values = merged.get(
        "client_noise_values",
        None,
    )

    if (
        noise_values is not None
        and len(
            noise_values
        )
        == num_clients
    ):

        noises = np.asarray(
            noise_values,
            dtype=float,
        )

    else:

        noises = np.full(
            num_clients,
            np.nan,
            dtype=float,
        )

    return (
        num_clients,
        noises,
    )


def to_float(
    value,
    default=np.nan,
) -> float:

    if value is None:
        return float(
            default
        )

    try:
        value = float(
            value
        )

    except (
        TypeError,
        ValueError,
    ):

        return float(
            default
        )

    if not np.isfinite(
        value
    ):
        return float(
            default
        )

    return value


def row_vector(
    row: dict,
    template: str,
    num_clients: int,
) -> np.ndarray:

    return np.asarray(
        [
            to_float(
                row.get(
                    template.format(
                        i=i
                    )
                )
            )
            for i in range(
                num_clients
            )
        ],
        dtype=float,
    )


# =============================================================================
# Evaluation-history loading
# =============================================================================


def load_eval_history(
    spec: RunSpec,
) -> dict[str, np.ndarray]:

    num_clients, noises = (
        config_info(
            spec.run_dir
        )
    )

    # round -> client vector
    #
    # dict form also removes any accidental duplicate eval row
    # at the same round.
    eval_by_round = {}

    history_path = (
        spec.run_dir
        / "history.jsonl"
    )

    with history_path.open(
        "r",
        encoding="utf-8",
    ) as f:

        for line in f:

            if not line.strip():
                continue

            row = json.loads(
                line
            )

            round_value = row.get(
                "frl/round"
            )

            if round_value is None:
                continue

            round_int = int(
                round(
                    float(
                        round_value
                    )
                )
            )

            local = row_vector(
                row,
                "eval/client_{i}/local_mean",
                num_clients,
            )

            if np.any(
                np.isfinite(
                    local
                )
            ):

                eval_by_round[
                    round_int
                ] = local

            # fallback noise extraction
            if np.any(
                ~np.isfinite(
                    noises
                )
            ):

                candidate = row_vector(
                    row,
                    "frl/client_{i}/noise",
                    num_clients,
                )

                mask = np.isfinite(
                    candidate
                )

                noises[
                    mask
                ] = candidate[
                    mask
                ]

    rounds = np.asarray(
        sorted(
            eval_by_round.keys()
        ),
        dtype=int,
    )

    if len(
        rounds
    ):

        local = np.vstack(
            [
                eval_by_round[
                    int(r)
                ]
                for r in rounds
            ]
        )

    else:

        local = np.empty(
            (
                0,
                num_clients,
            ),
            dtype=float,
        )

    return {
        "eval_rounds":
            rounds,

        "eval_local":
            local,

        "noises":
            noises,
    }


# =============================================================================
# Exact PPOAvg / Uniform round matching
# =============================================================================


def pair_exact_rounds(
    ppo_run: dict,
    uniform_run: dict,
):

    ppo_rounds = np.asarray(
        ppo_run[
            "eval_rounds"
        ],
        dtype=int,
    )

    uniform_rounds = np.asarray(
        uniform_run[
            "eval_rounds"
        ],
        dtype=int,
    )

    common_rounds = np.intersect1d(
        ppo_rounds,
        uniform_rounds,
    )

    ppo_map = {
        int(r): i
        for i, r
        in enumerate(
            ppo_rounds
        )
    }

    uniform_map = {
        int(r): i
        for i, r
        in enumerate(
            uniform_rounds
        )
    }

    ppo_idx = np.asarray(
        [
            ppo_map[
                int(r)
            ]
            for r
            in common_rounds
        ],
        dtype=int,
    )

    uniform_idx = np.asarray(
        [
            uniform_map[
                int(r)
            ]
            for r
            in common_rounds
        ],
        dtype=int,
    )

    return (
        common_rounds,
        ppo_idx,
        uniform_idx,
    )


# =============================================================================
# Correlation helpers
# =============================================================================


def average_rank(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    order = np.argsort(
        values,
        kind="mergesort",
    )

    ranks = np.empty(
        len(
            values
        ),
        dtype=float,
    )

    i = 0

    while i < len(
        values
    ):

        j = i + 1

        while (
            j
            < len(
                values
            )
            and
            values[
                order[j]
            ]
            ==
            values[
                order[i]
            ]
        ):

            j += 1

        ranks[
            order[i:j]
        ] = 0.5 * (
            (i + 1)
            + j
        )

        i = j

    return ranks


def pearson_corr(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(
        x
    ) < 2:
        return np.nan

    if (
        np.allclose(
            x,
            x[0],
        )
        or
        np.allclose(
            y,
            y[0],
        )
    ):

        return np.nan

    x = (
        x
        - np.mean(
            x
        )
    )

    y = (
        y
        - np.mean(
            y
        )
    )

    denom = (
        np.linalg.norm(
            x
        )
        *
        np.linalg.norm(
            y
        )
    )

    if denom <= 0:
        return np.nan

    return float(
        np.dot(
            x,
            y,
        )
        /
        denom
    )


def spearman_corr(
    x,
    y,
):

    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    y = np.asarray(
        y,
        dtype=float,
    ).reshape(-1)

    mask = (
        np.isfinite(x)
        & np.isfinite(y)
    )

    x = x[
        mask
    ]

    y = y[
        mask
    ]

    if len(
        x
    ) < 2:
        return np.nan

    return pearson_corr(
        average_rank(
            x
        ),
        average_rank(
            y
        ),
    )


def finite_mean(
    values,
):

    values = np.asarray(
        values,
        dtype=float,
    )

    if not np.any(
        np.isfinite(
            values
        )
    ):

        return np.nan

    return float(
        np.nanmean(
            values
        )
    )


# =============================================================================
# Trajectory-instability metrics
# =============================================================================


TRAJECTORY_METRICS = (
    "downside_semidev",
    "negative_jump_rate",
    "large_drop_rate_10pct",
    "max_drawdown_ratio",
    "collapse_fraction_70pct_peak",
    "normalized_total_variation",
    "temporal_cv",
)


def trajectory_metrics(
    returns,
    start_idx,
):

    returns = np.asarray(
        returns,
        dtype=float,
    )

    returns = returns[
        start_idx:
    ]

    returns = returns[
        np.isfinite(
            returns
        )
    ]

    if len(
        returns
    ) < 3:

        return {
            metric:
                np.nan
            for metric
            in TRAJECTORY_METRICS
        }

    delta = np.diff(
        returns
    )

    scale = max(
        abs(
            float(
                np.mean(
                    returns
                )
            )
        ),
        1e-8,
    )

    # -----------------------------------------------------------------
    # 1. Downside semideviation
    #
    # Large when the trajectory repeatedly suffers strong negative jumps.
    # Scale-normalized for comparison across tasks.
    # -----------------------------------------------------------------

    negative_delta = np.minimum(
        delta,
        0.0,
    )

    downside_semidev = (
        np.sqrt(
            np.mean(
                negative_delta
                ** 2
            )
        )
        /
        scale
    )

    # -----------------------------------------------------------------
    # 2. Fraction of transitions with decreasing return
    # -----------------------------------------------------------------

    negative_jump_rate = np.mean(
        delta < 0.0
    )

    # -----------------------------------------------------------------
    # 3. Large drop frequency
    #
    # More than 10% relative drop compared with previous eval.
    # -----------------------------------------------------------------

    previous_scale = np.maximum(
        np.abs(
            returns[:-1]
        ),
        0.1
        * scale,
    )

    relative_delta = (
        delta
        /
        np.maximum(
            previous_scale,
            1e-8,
        )
    )

    large_drop_rate = np.mean(
        relative_delta
        < -0.10
    )

    # -----------------------------------------------------------------
    # 4. Maximum drawdown
    # -----------------------------------------------------------------

    running_peak = np.maximum.accumulate(
        returns
    )

    drawdown_denominator = np.maximum(
        np.abs(
            running_peak
        ),
        0.1
        * scale,
    )

    drawdown_ratio = (
        running_peak
        - returns
    ) / np.maximum(
        drawdown_denominator,
        1e-8,
    )

    max_drawdown_ratio = np.max(
        drawdown_ratio
    )

    # -----------------------------------------------------------------
    # 5. Collapse frequency
    #
    # Current performance < 70% of its previous running peak.
    # -----------------------------------------------------------------

    collapse_fraction = np.mean(
        returns
        <
        0.70
        * running_peak
    )

    # -----------------------------------------------------------------
    # 6. Normalized total variation
    # -----------------------------------------------------------------

    normalized_total_variation = (
        np.mean(
            np.abs(
                delta
            )
        )
        /
        scale
    )

    # -----------------------------------------------------------------
    # 7. Temporal coefficient of variation
    # -----------------------------------------------------------------

    temporal_cv = (
        np.std(
            returns,
            ddof=1,
        )
        /
        scale
    )

    return {
        "downside_semidev":
            float(
                downside_semidev
            ),

        "negative_jump_rate":
            float(
                negative_jump_rate
            ),

        "large_drop_rate_10pct":
            float(
                large_drop_rate
            ),

        "max_drawdown_ratio":
            float(
                max_drawdown_ratio
            ),

        "collapse_fraction_70pct_peak":
            float(
                collapse_fraction
            ),

        "normalized_total_variation":
            float(
                normalized_total_variation
            ),

        "temporal_cv":
            float(
                temporal_cv
            ),
    }


# =============================================================================
# Paired seed analysis
# =============================================================================


def analyze_paired_seed(
    ppo_spec,
    uniform_spec,
    ppo_run,
    uniform_run,
    early_evals,
    late_evals,
):

    (
        common_rounds,
        ppo_idx,
        uniform_idx,
    ) = pair_exact_rounds(
        ppo_run,
        uniform_run,
    )

    ppo = np.asarray(
        ppo_run[
            "eval_local"
        ],
        dtype=float,
    )[
        ppo_idx
    ]

    uniform = np.asarray(
        uniform_run[
            "eval_local"
        ],
        dtype=float,
    )[
        uniform_idx
    ]

    min_required = max(
        early_evals + 3,
        late_evals,
    )

    if len(
        common_rounds
    ) < min_required:

        return (
            [],
            [],
            {},
        )

    num_clients = (
        ppo.shape[1]
    )

    early_n = min(
        early_evals,
        max(
            1,
            len(
                common_rounds
            )
            // 3,
        ),
    )

    late_n = min(
        late_evals,
        len(
            common_rounds
        ),
    )

    # =================================================================
    # IMPORTANT:
    #
    # Hard/easy classification uses ONLY EARLY TRAINING.
    #
    # We use the average of PPOAvg and Uniform early return:
    #
    #   J_early_ref =
    #       0.5 * (J_early_PPO + J_early_Uniform)
    #
    # This avoids defining difficulty using PPOAvg late return and then
    # correlating it with Uniform - PPOAvg, which shares the same term
    # and can create mechanical correlation.
    # =================================================================

    early_reference = (
        0.5
        * (
            np.mean(
                ppo[
                    :early_n
                ],
                axis=0,
            )
            +
            np.mean(
                uniform[
                    :early_n
                ],
                axis=0,
            )
        )
    )

    early_mean = np.mean(
        early_reference
    )

    early_difficulty = (
        early_mean
        - early_reference
    )

    # high difficulty = low early return

    order = np.argsort(
        early_reference
    )

    hard2 = set(
        int(i)
        for i
        in order[
            :min(
                2,
                num_clients,
            )
        ]
    )

    ppo_late = np.mean(
        ppo[
            -late_n:
        ],
        axis=0,
    )

    uniform_late = np.mean(
        uniform[
            -late_n:
        ],
        axis=0,
    )

    uniform_gain = (
        uniform_late
        - ppo_late
    )

    noises = np.asarray(
        ppo_run.get(
            "noises",
            np.full(
                num_clients,
                np.nan,
            ),
        ),
        dtype=float,
    )

    client_rows = []

    # Ignore early reference period for instability metrics.
    analysis_start = early_n

    for client in range(
        num_clients
    ):

        ppo_metrics = (
            trajectory_metrics(
                ppo[
                    :,
                    client,
                ],
                start_idx=analysis_start,
            )
        )

        uniform_metrics = (
            trajectory_metrics(
                uniform[
                    :,
                    client,
                ],
                start_idx=analysis_start,
            )
        )

        row = {
            "environment":
                ppo_spec.env,

            "perturbation":
                ppo_spec.perturbation,

            "seed":
                ppo_spec.seed,

            "client":
                client,

            "noise":
                (
                    float(
                        noises[
                            client
                        ]
                    )
                    if
                    client
                    < len(
                        noises
                    )
                    else
                    np.nan
                ),

            "num_common_evals":
                len(
                    common_rounds
                ),

            "early_reference_return":
                float(
                    early_reference[
                        client
                    ]
                ),

            "early_difficulty":
                float(
                    early_difficulty[
                        client
                    ]
                ),

            "early_is_hard2":
                int(
                    client
                    in hard2
                ),

            "ppoavg_late_return":
                float(
                    ppo_late[
                        client
                    ]
                ),

            "uniform_late_return":
                float(
                    uniform_late[
                        client
                    ]
                ),

            "uniform_gain":
                float(
                    uniform_gain[
                        client
                    ]
                ),
        }

        for metric in (
            TRAJECTORY_METRICS
        ):

            ppo_value = (
                ppo_metrics[
                    metric
                ]
            )

            uniform_value = (
                uniform_metrics[
                    metric
                ]
            )

            row[
                f"ppoavg_{metric}"
            ] = ppo_value

            row[
                f"uniform_{metric}"
            ] = uniform_value

            # Positive value:
            # AMPO-Uniform has LESS instability.
            row[
                f"reduction_{metric}"
            ] = (
                ppo_value
                - uniform_value
            )

        client_rows.append(
            row
        )

    # =================================================================
    # Spread trajectory
    # =================================================================

    spread_rows = []

    for eval_index, round_value in enumerate(
        common_rounds
    ):

        p = ppo[
            eval_index
        ]

        u = uniform[
            eval_index
        ]

        p_mean = float(
            np.mean(
                p
            )
        )

        u_mean = float(
            np.mean(
                u
            )
        )

        p_abs_spread = float(
            np.max(
                p
            )
            -
            np.min(
                p
            )
        )

        u_abs_spread = float(
            np.max(
                u
            )
            -
            np.min(
                u
            )
        )

        p_rel_spread = (
            p_abs_spread
            /
            max(
                abs(
                    p_mean
                ),
                1e-8,
            )
        )

        u_rel_spread = (
            u_abs_spread
            /
            max(
                abs(
                    u_mean
                ),
                1e-8,
            )
        )

        if eval_index < early_n:

            phase = "early"

        elif (
            eval_index
            >=
            len(
                common_rounds
            )
            - late_n
        ):

            phase = "late"

        else:

            phase = "middle"

        spread_rows.append(
            {
                "environment":
                    ppo_spec.env,

                "perturbation":
                    ppo_spec.perturbation,

                "seed":
                    ppo_spec.seed,

                "eval_index":
                    eval_index,

                "round":
                    int(
                        round_value
                    ),

                "phase":
                    phase,

                "ppoavg_mean_return":
                    p_mean,

                "uniform_mean_return":
                    u_mean,

                "ppoavg_abs_spread":
                    p_abs_spread,

                "uniform_abs_spread":
                    u_abs_spread,

                "abs_spread_reduction":
                    p_abs_spread
                    - u_abs_spread,

                "ppoavg_rel_spread":
                    p_rel_spread,

                "uniform_rel_spread":
                    u_rel_spread,

                # positive = Uniform has smaller spread
                "rel_spread_reduction":
                    p_rel_spread
                    - u_rel_spread,
            }
        )

    early_spreads = (
        spread_rows[
            :early_n
        ]
    )

    late_spreads = (
        spread_rows[
            -late_n:
        ]
    )

    seed_summary = {
        "environment":
            ppo_spec.env,

        "perturbation":
            ppo_spec.perturbation,

        "seed":
            ppo_spec.seed,

        "num_common_evals":
            len(
                common_rounds
            ),

        "early_mean_rel_spread_ppoavg":
            finite_mean(
                [
                    r[
                        "ppoavg_rel_spread"
                    ]
                    for r
                    in early_spreads
                ]
            ),

        "early_mean_rel_spread_uniform":
            finite_mean(
                [
                    r[
                        "uniform_rel_spread"
                    ]
                    for r
                    in early_spreads
                ]
            ),

        "late_mean_rel_spread_ppoavg":
            finite_mean(
                [
                    r[
                        "ppoavg_rel_spread"
                    ]
                    for r
                    in late_spreads
                ]
            ),

        "late_mean_rel_spread_uniform":
            finite_mean(
                [
                    r[
                        "uniform_rel_spread"
                    ]
                    for r
                    in late_spreads
                ]
            ),
    }

    seed_summary[
        "ppoavg_rel_spread_growth"
    ] = (
        seed_summary[
            "late_mean_rel_spread_ppoavg"
        ]
        -
        seed_summary[
            "early_mean_rel_spread_ppoavg"
        ]
    )

    seed_summary[
        "uniform_rel_spread_growth"
    ] = (
        seed_summary[
            "late_mean_rel_spread_uniform"
        ]
        -
        seed_summary[
            "early_mean_rel_spread_uniform"
        ]
    )

    seed_summary[
        "late_rel_spread_reduction"
    ] = (
        seed_summary[
            "late_mean_rel_spread_ppoavg"
        ]
        -
        seed_summary[
            "late_mean_rel_spread_uniform"
        ]
    )

    return (
        client_rows,
        spread_rows,
        seed_summary,
    )


# =============================================================================
# Setting-level trajectory summaries
# =============================================================================


def mean_seedwise_rank_corr(
    rows,
    x_key,
    y_key,
):

    seed_values = []

    seeds = sorted(
        {
            int(
                r[
                    "seed"
                ]
            )
            for r
            in rows
        }
    )

    for seed in seeds:

        seed_rows = [
            r
            for r
            in rows
            if int(
                r[
                    "seed"
                ]
            )
            == seed
        ]

        rho = spearman_corr(
            [
                r[
                    x_key
                ]
                for r
                in seed_rows
            ],
            [
                r[
                    y_key
                ]
                for r
                in seed_rows
            ],
        )

        seed_values.append(
            rho
        )

    return finite_mean(
        seed_values
    )


def summarize_client_rows(
    client_rows,
):

    summaries = []

    settings = [
        (
            env,
            pert,
        )
        for env
        in ENV_ORDER
        for pert
        in PERTURBATION_ORDER
    ]

    settings.append(
        (
            "ALL",
            "ALL",
        )
    )

    for env, pert in settings:

        if env == "ALL":

            rows = (
                client_rows
            )

        else:

            rows = [
                r
                for r
                in client_rows
                if (
                    r[
                        "environment"
                    ]
                    == env
                    and
                    r[
                        "perturbation"
                    ]
                    == pert
                )
            ]

        if not rows:
            continue

        hard_rows = [
            r
            for r
            in rows
            if r[
                "early_is_hard2"
            ]
            == 1
        ]

        easy_rows = [
            r
            for r
            in rows
            if r[
                "early_is_hard2"
            ]
            == 0
        ]

        summary = {
            "environment":
                env,

            "perturbation":
                pert,

            "n_seed_client_points":
                len(
                    rows
                ),

            # -------------------------------------------------------------
            # Does EARLY difficulty predict eventual Uniform recovery?
            # -------------------------------------------------------------

            "pooled_spearman_early_difficulty_vs_gain":
                spearman_corr(
                    [
                        r[
                            "early_difficulty"
                        ]
                        for r
                        in rows
                    ],
                    [
                        r[
                            "uniform_gain"
                        ]
                        for r
                        in rows
                    ],
                ),

            "mean_seedwise_spearman_early_difficulty_vs_gain":
                mean_seedwise_rank_corr(
                    rows,
                    "early_difficulty",
                    "uniform_gain",
                ),

            "mean_hard2_gain":
                finite_mean(
                    [
                        r[
                            "uniform_gain"
                        ]
                        for r
                        in hard_rows
                    ]
                ),

            "mean_easy3_gain":
                finite_mean(
                    [
                        r[
                            "uniform_gain"
                        ]
                        for r
                        in easy_rows
                    ]
                ),
        }

        summary[
            "hard2_minus_easy3_gain"
        ] = (
            summary[
                "mean_hard2_gain"
            ]
            -
            summary[
                "mean_easy3_gain"
            ]
        )

        # -------------------------------------------------------------
        # Instability metrics
        # -------------------------------------------------------------

        for metric in TRAJECTORY_METRICS:

            ppo_key = (
                f"ppoavg_{metric}"
            )

            uniform_key = (
                f"uniform_{metric}"
            )

            reduction_key = (
                f"reduction_{metric}"
            )

            summary[
                f"mean_ppoavg_{metric}"
            ] = finite_mean(
                [
                    r[
                        ppo_key
                    ]
                    for r
                    in rows
                ]
            )

            summary[
                f"mean_uniform_{metric}"
            ] = finite_mean(
                [
                    r[
                        uniform_key
                    ]
                    for r
                    in rows
                ]
            )

            summary[
                f"mean_reduction_{metric}"
            ] = finite_mean(
                [
                    r[
                        reduction_key
                    ]
                    for r
                    in rows
                ]
            )

            # Are initially hard clients unstable under PPOAvg?

            summary[
                f"spearman_early_difficulty_vs_ppoavg_{metric}"
            ] = spearman_corr(
                [
                    r[
                        "early_difficulty"
                    ]
                    for r
                    in rows
                ],
                [
                    r[
                        ppo_key
                    ]
                    for r
                    in rows
                ],
            )

            # Does that relation remain under Uniform?

            summary[
                f"spearman_early_difficulty_vs_uniform_{metric}"
            ] = spearman_corr(
                [
                    r[
                        "early_difficulty"
                    ]
                    for r
                    in rows
                ],
                [
                    r[
                        uniform_key
                    ]
                    for r
                    in rows
                ],
            )

            # Most important mechanism test:
            #
            # More instability reduction
            #        ->
            # Larger final Uniform gain?

            summary[
                f"spearman_reduction_{metric}_vs_gain"
            ] = spearman_corr(
                [
                    r[
                        reduction_key
                    ]
                    for r
                    in rows
                ],
                [
                    r[
                        "uniform_gain"
                    ]
                    for r
                    in rows
                ],
            )

            summary[
                f"mean_seedwise_spearman_reduction_{metric}_vs_gain"
            ] = (
                mean_seedwise_rank_corr(
                    rows,
                    reduction_key,
                    "uniform_gain",
                )
            )

            hard_reduction = finite_mean(
                [
                    r[
                        reduction_key
                    ]
                    for r
                    in hard_rows
                ]
            )

            easy_reduction = finite_mean(
                [
                    r[
                        reduction_key
                    ]
                    for r
                    in easy_rows
                ]
            )

            summary[
                f"hard2_mean_reduction_{metric}"
            ] = hard_reduction

            summary[
                f"easy3_mean_reduction_{metric}"
            ] = easy_reduction

            summary[
                f"hard2_minus_easy3_reduction_{metric}"
            ] = (
                hard_reduction
                - easy_reduction
            )

        summaries.append(
            summary
        )

    return summaries


# =============================================================================
# Spread summaries
# =============================================================================


def summarize_spreads(
    seed_summaries,
):

    output = []

    settings = [
        (
            env,
            pert,
        )
        for env
        in ENV_ORDER
        for pert
        in PERTURBATION_ORDER
    ]

    settings.append(
        (
            "ALL",
            "ALL",
        )
    )

    for env, pert in settings:

        if env == "ALL":

            rows = (
                seed_summaries
            )

        else:

            rows = [
                r
                for r
                in seed_summaries
                if (
                    r[
                        "environment"
                    ]
                    == env
                    and
                    r[
                        "perturbation"
                    ]
                    == pert
                )
            ]

        if not rows:
            continue

        output.append(
            {
                "environment":
                    env,

                "perturbation":
                    pert,

                "num_seeds":
                    len(
                        rows
                    ),

                "early_rel_spread_ppoavg":
                    finite_mean(
                        [
                            r[
                                "early_mean_rel_spread_ppoavg"
                            ]
                            for r
                            in rows
                        ]
                    ),

                "early_rel_spread_uniform":
                    finite_mean(
                        [
                            r[
                                "early_mean_rel_spread_uniform"
                            ]
                            for r
                            in rows
                        ]
                    ),

                "late_rel_spread_ppoavg":
                    finite_mean(
                        [
                            r[
                                "late_mean_rel_spread_ppoavg"
                            ]
                            for r
                            in rows
                        ]
                    ),

                "late_rel_spread_uniform":
                    finite_mean(
                        [
                            r[
                                "late_mean_rel_spread_uniform"
                            ]
                            for r
                            in rows
                        ]
                    ),

                "ppoavg_rel_spread_growth":
                    finite_mean(
                        [
                            r[
                                "ppoavg_rel_spread_growth"
                            ]
                            for r
                            in rows
                        ]
                    ),

                "uniform_rel_spread_growth":
                    finite_mean(
                        [
                            r[
                                "uniform_rel_spread_growth"
                            ]
                            for r
                            in rows
                        ]
                    ),

                "late_rel_spread_reduction":
                    finite_mean(
                        [
                            r[
                                "late_rel_spread_reduction"
                            ]
                            for r
                            in rows
                        ]
                    ),
            }
        )

    return output


# =============================================================================
# Across-seed variability
# =============================================================================


def setting_level_hard_clients(
    client_rows,
):

    result = {}

    for env in ENV_ORDER:

        for pert in PERTURBATION_ORDER:

            rows = [
                r
                for r
                in client_rows
                if (
                    r[
                        "environment"
                    ]
                    == env
                    and
                    r[
                        "perturbation"
                    ]
                    == pert
                )
            ]

            if not rows:
                continue

            client_ids = sorted(
                {
                    int(
                        r[
                            "client"
                        ]
                    )
                    for r
                    in rows
                }
            )

            profile = []

            for client in client_ids:

                profile.append(
                    finite_mean(
                        [
                            r[
                                "early_reference_return"
                            ]
                            for r
                            in rows
                            if int(
                                r[
                                    "client"
                                ]
                            )
                            == client
                        ]
                    )
                )

            order = np.argsort(
                profile
            )

            result[
                (
                    env,
                    pert,
                )
            ] = set(
                client_ids[
                    int(index)
                ]
                for index
                in order[
                    :min(
                        2,
                        len(
                            order
                        ),
                    )
                ]
            )

    return result


def compute_seed_variance(
    run_map,
    client_rows,
    early_evals,
    late_evals,
):

    hard_map = (
        setting_level_hard_clients(
            client_rows
        )
    )

    time_rows = []
    summary_rows = []

    for env in ENV_ORDER:

        for pert in PERTURBATION_ORDER:

            hard_clients = (
                hard_map.get(
                    (
                        env,
                        pert,
                    ),
                    set(),
                )
            )

            for method in METHODS:

                runs = []

                for seed in range(
                    1,
                    6,
                ):

                    key = (
                        method,
                        env,
                        pert,
                        seed,
                    )

                    if key in run_map:

                        runs.append(
                            run_map[
                                key
                            ]
                        )

                if len(
                    runs
                ) < 2:
                    continue

                # exact rounds common to all available seeds

                common_rounds = set(
                    int(r)
                    for r
                    in runs[0][
                        "eval_rounds"
                    ]
                )

                for run in runs[
                    1:
                ]:

                    common_rounds &= set(
                        int(r)
                        for r
                        in run[
                            "eval_rounds"
                        ]
                    )

                common_rounds = np.asarray(
                    sorted(
                        common_rounds
                    ),
                    dtype=int,
                )

                if len(
                    common_rounds
                ) == 0:
                    continue

                seed_matrices = []

                for run in runs:

                    round_to_index = {
                        int(r): i
                        for i, r
                        in enumerate(
                            run[
                                "eval_rounds"
                            ]
                        )
                    }

                    indices = [
                        round_to_index[
                            int(r)
                        ]
                        for r
                        in common_rounds
                    ]

                    seed_matrices.append(
                        np.asarray(
                            run[
                                "eval_local"
                            ],
                            dtype=float,
                        )[
                            indices
                        ]
                    )

                # shape:
                #   seeds x eval_points x clients

                data = np.stack(
                    seed_matrices,
                    axis=0,
                )

                seed_mean = np.mean(
                    data,
                    axis=0,
                )

                seed_std = np.std(
                    data,
                    axis=0,
                    ddof=1,
                )

                # normalized across-seed variability

                seed_cv = (
                    seed_std
                    /
                    np.maximum(
                        np.abs(
                            seed_mean
                        ),
                        1e-8,
                    )
                )

                num_clients = (
                    data.shape[
                        2
                    ]
                )

                hard_indices = sorted(
                    hard_clients
                )

                easy_indices = [
                    i
                    for i
                    in range(
                        num_clients
                    )
                    if i
                    not in hard_clients
                ]

                for eval_index, round_value in enumerate(
                    common_rounds
                ):

                    row = {
                        "environment":
                            env,

                        "perturbation":
                            pert,

                        "method":
                            method,

                        "round":
                            int(
                                round_value
                            ),

                        "eval_index":
                            eval_index,

                        "mean_seed_cv_all":
                            float(
                                np.mean(
                                    seed_cv[
                                        eval_index
                                    ]
                                )
                            ),

                        "mean_seed_cv_hard2":
                            (
                                float(
                                    np.mean(
                                        seed_cv[
                                            eval_index,
                                            hard_indices,
                                        ]
                                    )
                                )
                                if
                                hard_indices
                                else
                                np.nan
                            ),

                        "mean_seed_cv_easy3":
                            (
                                float(
                                    np.mean(
                                        seed_cv[
                                            eval_index,
                                            easy_indices,
                                        ]
                                    )
                                )
                                if
                                easy_indices
                                else
                                np.nan
                            ),
                    }

                    for client in range(
                        num_clients
                    ):

                        row[
                            f"client_{client}_seed_cv"
                        ] = float(
                            seed_cv[
                                eval_index,
                                client,
                            ]
                        )

                    time_rows.append(
                        row
                    )

                early_n = min(
                    early_evals,
                    max(
                        1,
                        len(
                            common_rounds
                        )
                        // 3,
                    ),
                )

                late_n = min(
                    late_evals,
                    len(
                        common_rounds
                    ),
                )

                post_early = (
                    seed_cv[
                        early_n:
                    ]
                )

                late = (
                    seed_cv[
                        -late_n:
                    ]
                )

                summary_rows.append(
                    {
                        "environment":
                            env,

                        "perturbation":
                            pert,

                        "method":
                            method,

                        "num_seeds":
                            len(
                                runs
                            ),

                        "num_common_evals":
                            len(
                                common_rounds
                            ),

                        "postearly_seed_cv_all":
                            float(
                                np.mean(
                                    post_early
                                )
                            ),

                        "postearly_seed_cv_hard2":
                            (
                                float(
                                    np.mean(
                                        post_early[
                                            :,
                                            hard_indices,
                                        ]
                                    )
                                )
                                if
                                hard_indices
                                else
                                np.nan
                            ),

                        "postearly_seed_cv_easy3":
                            (
                                float(
                                    np.mean(
                                        post_early[
                                            :,
                                            easy_indices,
                                        ]
                                    )
                                )
                                if
                                easy_indices
                                else
                                np.nan
                            ),

                        "late_seed_cv_all":
                            float(
                                np.mean(
                                    late
                                )
                            ),

                        "late_seed_cv_hard2":
                            (
                                float(
                                    np.mean(
                                        late[
                                            :,
                                            hard_indices,
                                        ]
                                    )
                                )
                                if
                                hard_indices
                                else
                                np.nan
                            ),

                        "late_seed_cv_easy3":
                            (
                                float(
                                    np.mean(
                                        late[
                                            :,
                                            easy_indices,
                                        ]
                                    )
                                )
                                if
                                easy_indices
                                else
                                np.nan
                            ),
                    }
                )

    return (
        time_rows,
        summary_rows,
    )


# =============================================================================
# CSV output
# =============================================================================


def write_csv(
    path,
    rows,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:

        print(
            f"[Skip empty] {path}"
        )

        return

    columns = list(
        rows[
            0
        ].keys()
    )

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=columns,
        )

        writer.writeheader()

        writer.writerows(
            rows
        )

    print(
        f"[Saved] {path}"
    )


# =============================================================================
# Spread plots
# =============================================================================


def plot_spread_trajectories(
    spread_rows,
    out_dir,
):

    for env in ENV_ORDER:

        for pert in PERTURBATION_ORDER:

            rows = [
                r
                for r
                in spread_rows
                if (
                    r[
                        "environment"
                    ]
                    == env
                    and
                    r[
                        "perturbation"
                    ]
                    == pert
                )
            ]

            if not rows:
                continue

            rounds = sorted(
                {
                    int(
                        r[
                            "round"
                        ]
                    )
                    for r
                    in rows
                }
            )

            ppo_mean = []
            ppo_std = []

            uniform_mean = []
            uniform_std = []

            for round_value in rounds:

                round_rows = [
                    r
                    for r
                    in rows
                    if int(
                        r[
                            "round"
                        ]
                    )
                    == round_value
                ]

                p_values = np.asarray(
                    [
                        r[
                            "ppoavg_rel_spread"
                        ]
                        for r
                        in round_rows
                    ],
                    dtype=float,
                )

                u_values = np.asarray(
                    [
                        r[
                            "uniform_rel_spread"
                        ]
                        for r
                        in round_rows
                    ],
                    dtype=float,
                )

                ppo_mean.append(
                    np.mean(
                        p_values
                    )
                )

                ppo_std.append(
                    np.std(
                        p_values,
                        ddof=(
                            1
                            if len(
                                p_values
                            )
                            > 1
                            else
                            0
                        ),
                    )
                )

                uniform_mean.append(
                    np.mean(
                        u_values
                    )
                )

                uniform_std.append(
                    np.std(
                        u_values,
                        ddof=(
                            1
                            if len(
                                u_values
                            )
                            > 1
                            else
                            0
                        ),
                    )
                )

            x = np.asarray(
                rounds,
                dtype=float,
            )

            ppo_mean = np.asarray(
                ppo_mean,
                dtype=float,
            )

            ppo_std = np.asarray(
                ppo_std,
                dtype=float,
            )

            uniform_mean = np.asarray(
                uniform_mean,
                dtype=float,
            )

            uniform_std = np.asarray(
                uniform_std,
                dtype=float,
            )

            fig, ax = plt.subplots(
                figsize=(
                    7.2,
                    4.8,
                )
            )

            ax.plot(
                x,
                ppo_mean,
                label="PPOAvg",
            )

            ax.fill_between(
                x,
                ppo_mean
                - ppo_std,
                ppo_mean
                + ppo_std,
                alpha=0.18,
            )

            ax.plot(
                x,
                uniform_mean,
                label="AMPO-Uniform",
            )

            ax.fill_between(
                x,
                uniform_mean
                - uniform_std,
                uniform_mean
                + uniform_std,
                alpha=0.18,
            )

            ax.set_xlabel(
                "Federated round"
            )

            ax.set_ylabel(
                "Relative client spread"
            )

            ax.set_title(
                f"{env} / {pert}"
            )

            ax.legend()

            fig.tight_layout()

            output_path = (
                out_dir
                / (
                    f"spread_"
                    f"{env}_"
                    f"{pert}.png"
                )
            )

            fig.savefig(
                output_path,
                dpi=250,
                bbox_inches="tight",
            )

            plt.close(
                fig
            )

            print(
                f"[Saved] {output_path}"
            )


# =============================================================================
# Main
# =============================================================================


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Analyze PPOAvg vs AMPO-Uniform "
            "trajectory stability using existing logs only."
        )
    )

    parser.add_argument(
        "--log-root",
        type=Path,
        default=DEFAULT_LOG_ROOT,
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
    )

    parser.add_argument(
        "--early-evals",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--late-evals",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    specs = discover_runs(
        args.log_root
    )

    print(
        f"[Found] {len(specs)} "
        f"PPOAvg/Uniform runs"
    )

    run_map = {}
    spec_map = {}

    for spec in specs:

        key = (
            spec.variant,
            spec.env,
            spec.perturbation,
            spec.seed,
        )

        run_map[
            key
        ] = load_eval_history(
            spec
        )

        spec_map[
            key
        ] = spec

    client_rows = []

    spread_rows = []

    seed_spread_summaries = []

    # -------------------------------------------------------------------------
    # Paired PPOAvg / Uniform seed analysis
    # -------------------------------------------------------------------------

    for env in ENV_ORDER:

        for pert in PERTURBATION_ORDER:

            for seed in range(
                1,
                6,
            ):

                ppo_key = (
                    "ppo_avg",
                    env,
                    pert,
                    seed,
                )

                uniform_key = (
                    "ampo_uniform",
                    env,
                    pert,
                    seed,
                )

                if (
                    ppo_key
                    not in run_map
                    or
                    uniform_key
                    not in run_map
                ):

                    print(
                        f"[Missing pair] "
                        f"{env}/"
                        f"{pert}/"
                        f"seed={seed}"
                    )

                    continue

                (
                    pair_client_rows,
                    pair_spread_rows,
                    pair_seed_summary,
                ) = analyze_paired_seed(
                    spec_map[
                        ppo_key
                    ],
                    spec_map[
                        uniform_key
                    ],
                    run_map[
                        ppo_key
                    ],
                    run_map[
                        uniform_key
                    ],
                    early_evals=args.early_evals,
                    late_evals=args.late_evals,
                )

                client_rows.extend(
                    pair_client_rows
                )

                spread_rows.extend(
                    pair_spread_rows
                )

                if pair_seed_summary:

                    seed_spread_summaries.append(
                        pair_seed_summary
                    )

    # -------------------------------------------------------------------------
    # Summaries
    # -------------------------------------------------------------------------

    trajectory_summary = (
        summarize_client_rows(
            client_rows
        )
    )

    spread_summary = (
        summarize_spreads(
            seed_spread_summaries
        )
    )

    (
        seed_variance_trajectory,
        seed_variance_summary,
    ) = compute_seed_variance(
        run_map,
        client_rows,
        early_evals=args.early_evals,
        late_evals=args.late_evals,
    )

    # -------------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------------

    args.out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    write_csv(
        args.out_dir
        / "client_trajectory_metrics.csv",
        client_rows,
    )

    write_csv(
        args.out_dir
        / "trajectory_mechanism_summary.csv",
        trajectory_summary,
    )

    write_csv(
        args.out_dir
        / "spread_trajectory.csv",
        spread_rows,
    )

    write_csv(
        args.out_dir
        / "spread_summary.csv",
        spread_summary,
    )

    write_csv(
        args.out_dir
        / "seed_variance_trajectory.csv",
        seed_variance_trajectory,
    )

    write_csv(
        args.out_dir
        / "seed_variance_summary.csv",
        seed_variance_summary,
    )

    plot_spread_trajectories(
        spread_rows,
        args.out_dir,
    )

    # =========================================================================
    # Console output
    # =========================================================================

    print(
        "\n"
        "============================================================"
    )

    print(
        "Trajectory mechanism summary"
    )

    print(
        "============================================================"
    )

    for row in trajectory_summary:

        print(
            f"{row['environment']:>12} "
            f"{row['perturbation']:>8} | "

            f"rho(earlyDiff,gain)="
            f"{row['pooled_spearman_early_difficulty_vs_gain']:+.3f} | "

            f"hard2-easy3 gain="
            f"{row['hard2_minus_easy3_gain']:+.1f} | "

            f"rho(MDDred,gain)="
            f"{row['spearman_reduction_max_drawdown_ratio_vs_gain']:+.3f} | "

            f"rho(DownRed,gain)="
            f"{row['spearman_reduction_downside_semidev_vs_gain']:+.3f} | "

            f"rho(CollapseRed,gain)="
            f"{row['spearman_reduction_collapse_fraction_70pct_peak_vs_gain']:+.3f} | "

            f"rho(TVred,gain)="
            f"{row['spearman_reduction_normalized_total_variation_vs_gain']:+.3f}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Spread trajectory summary"
    )

    print(
        "============================================================"
    )

    for row in spread_summary:

        print(
            f"{row['environment']:>12} "
            f"{row['perturbation']:>8} | "

            f"early PPO/U="
            f"{row['early_rel_spread_ppoavg']:.3f}/"
            f"{row['early_rel_spread_uniform']:.3f} | "

            f"late PPO/U="
            f"{row['late_rel_spread_ppoavg']:.3f}/"
            f"{row['late_rel_spread_uniform']:.3f} | "

            f"growth PPO/U="
            f"{row['ppoavg_rel_spread_growth']:+.3f}/"
            f"{row['uniform_rel_spread_growth']:+.3f} | "

            f"late reduction="
            f"{row['late_rel_spread_reduction']:+.3f}"
        )

    print(
        "\n"
        "============================================================"
    )

    print(
        "Across-seed variance summary"
    )

    print(
        "============================================================"
    )

    seed_variance_lookup = {
        (
            r[
                "environment"
            ],
            r[
                "perturbation"
            ],
            r[
                "method"
            ],
        ): r
        for r
        in seed_variance_summary
    }

    for env in ENV_ORDER:

        for pert in PERTURBATION_ORDER:

            ppo = seed_variance_lookup.get(
                (
                    env,
                    pert,
                    "ppo_avg",
                )
            )

            uniform = (
                seed_variance_lookup.get(
                    (
                        env,
                        pert,
                        "ampo_uniform",
                    )
                )
            )

            if (
                ppo is None
                or uniform is None
            ):
                continue

            hard_reduction = (
                ppo[
                    "late_seed_cv_hard2"
                ]
                -
                uniform[
                    "late_seed_cv_hard2"
                ]
            )

            print(
                f"{env:>12} "
                f"{pert:>8} | "

                f"late hard2 CV PPO/U="
                f"{ppo['late_seed_cv_hard2']:.3f}/"
                f"{uniform['late_seed_cv_hard2']:.3f} | "

                f"reduction="
                f"{hard_reduction:+.3f} | "

                f"late easy3 CV PPO/U="
                f"{ppo['late_seed_cv_easy3']:.3f}/"
                f"{uniform['late_seed_cv_easy3']:.3f}"
            )

    print(
        f"\n[Done] "
        f"{args.out_dir}"
    )


if __name__ == "__main__":
    main()