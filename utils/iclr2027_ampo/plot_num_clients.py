import os

import numpy as np
import matplotlib.pyplot as plt


# ============================================================================
# Publication-style plotting defaults
# ============================================================================

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",
        "font.size": 15,
        "axes.labelsize": 15,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "axes.unicode_minus": False,
    }
)


# ============================================================================
# NPZ helpers
# ============================================================================


def read_scalar_from_npz(data, key, default=None):
    """
    evaluations.npz 안의 scalar 값을 안전하게 읽는다.

    예:
        local_steps.npy     -> 128
        num_clients.npy     -> 5
        eval_round_freq.npy -> 40
    """
    if key not in data:
        return default

    value = np.asarray(data[key])

    if value.ndim == 0:
        return value.item()

    if value.size == 1:
        return value.reshape(-1)[0].item()

    return value


def infer_num_clients_from_npz(
    data,
    metric_array=None,
):
    """
    evaluations.npz 내부 값으로 num_clients를 추정한다.

    우선순위:
        1. num_clients
        2. client_noises 길이
        3. metric array가 2D이면 두 번째 차원
        4. 실패하면 1
    """
    num_clients = read_scalar_from_npz(
        data,
        "num_clients",
        default=None,
    )

    if num_clients is not None:
        return int(num_clients)

    if "client_noises" in data:
        return int(
            len(
                np.asarray(
                    data["client_noises"]
                )
            )
        )

    if metric_array is not None:
        metric_array = np.asarray(
            metric_array
        )

        if metric_array.ndim == 2:
            return int(
                metric_array.shape[1]
            )

    return 1


def make_timesteps_from_npz(
    data,
    num_rounds,
    metric_array=None,
):
    """
    evaluations.npz 내부 값만 이용해 timesteps를 만든다.

    우선순위:
        1. timesteps가 있으면 그대로 사용
        2. rounds * local_steps * num_clients
        3. rounds * local_steps
        4. rounds
        5. 0, 1, 2, ...
    """
    if "timesteps" in data:
        return np.asarray(
            data["timesteps"],
            dtype=float,
        )[:num_rounds]

    if "rounds" in data:
        rounds = np.asarray(
            data["rounds"],
            dtype=float,
        )[:num_rounds]

        local_steps = read_scalar_from_npz(
            data,
            "local_steps",
            default=None,
        )

        if local_steps is not None:
            num_clients = infer_num_clients_from_npz(
                data=data,
                metric_array=metric_array,
            )

            return (
                rounds
                * float(local_steps)
                * float(num_clients)
            )

        return rounds

    return np.arange(
        num_rounds,
        dtype=float,
    )


# ============================================================================
# Evaluation-round helpers
# ============================================================================


def normalize_eval_round_freq(
    eval_round_freq,
):
    """
    evaluation round 주기를 양의 정수로 정규화한다.
    """
    if eval_round_freq is None:
        return None

    value = float(
        eval_round_freq
    )

    rounded = int(
        round(value)
    )

    if (
        not np.isfinite(value)
        or value <= 0
        or not np.isclose(
            value,
            rounded,
        )
    ):
        raise ValueError(
            "eval_round_freq must be a positive integer, "
            f"but got {eval_round_freq}"
        )

    return rounded


def get_logged_eval_round_freq(
    data,
    rounds,
):
    """
    한 evaluations.npz가 실제 저장한 evaluation round 주기를 가져온다.

    우선순위:
        1. eval_round_freq
        2. rounds positive diff의 median으로 추정
    """
    logged_freq = read_scalar_from_npz(
        data,
        "eval_round_freq",
        default=None,
    )

    if logged_freq is not None:
        value = float(
            logged_freq
        )

        rounded = int(
            round(value)
        )

        if (
            not np.isfinite(value)
            or value <= 0
            or not np.isclose(
                value,
                rounded,
            )
        ):
            raise ValueError(
                "Invalid eval_round_freq in evaluations.npz: "
                f"{logged_freq}"
            )

        return rounded

    rounds = np.asarray(
        rounds,
        dtype=float,
    ).reshape(-1)

    diffs = np.diff(
        rounds
    )

    positive_diffs = diffs[
        np.isfinite(diffs)
        & (diffs > 0)
    ]

    if len(positive_diffs) == 0:
        raise ValueError(
            "eval_round_freq is missing and cannot be "
            "inferred from rounds."
        )

    value = float(
        np.median(
            positive_diffs
        )
    )

    rounded = int(
        round(value)
    )

    if not np.isclose(
        value,
        rounded,
    ):
        raise ValueError(
            "eval_round_freq is missing and the inferred "
            "round spacing is not an integer: "
            f"{value}"
        )

    return rounded


def select_exact_eval_rounds(
    data,
    x,
    curve,
    requested_eval_round_freq,
):
    """
    requested_eval_round_freq에 정확히 해당하는 실제 저장값만 선택한다.

    예:
        logged freq = 40
        requested   = 80

        -> round 80, 160, 240, ... 만 선택

    interpolation은 사용하지 않는다.
    """
    requested_eval_round_freq = normalize_eval_round_freq(
        requested_eval_round_freq
    )

    x = np.asarray(
        x,
        dtype=float,
    ).reshape(-1)

    curve = np.asarray(
        curve,
        dtype=float,
    ).reshape(-1)

    if "rounds" not in data:
        raise ValueError(
            "Strict eval_round_freq selection requires "
            "'rounds' in evaluations.npz."
        )

    rounds = np.asarray(
        data["rounds"],
        dtype=float,
    ).reshape(-1)

    usable_len = min(
        len(rounds),
        len(x),
        len(curve),
    )

    rounds = rounds[:usable_len]
    x = x[:usable_len]
    curve = curve[:usable_len]

    if usable_len == 0:
        raise ValueError(
            "No evaluation data exists in evaluations.npz."
        )

    if not np.all(
        np.isfinite(rounds)
    ):
        raise ValueError(
            "rounds contains NaN or inf values."
        )

    rounded_rounds = np.rint(
        rounds
    ).astype(
        np.int64
    )

    if not np.allclose(
        rounds,
        rounded_rounds,
    ):
        raise ValueError(
            "rounds must contain integer-valued round numbers."
        )

    logged_eval_round_freq = get_logged_eval_round_freq(
        data,
        rounded_rounds,
    )

    if (
        requested_eval_round_freq
        % logged_eval_round_freq
        != 0
    ):
        raise ValueError(
            "Requested eval_round_freq is incompatible "
            "with this seed: "
            f"requested={requested_eval_round_freq}, "
            f"logged={logged_eval_round_freq}. "
            "The requested frequency must be an integer "
            "multiple of the logged frequency because "
            "interpolation is disabled."
        )

    exact_mask = (
        rounded_rounds
        % requested_eval_round_freq
    ) == 0

    selected_rounds = rounded_rounds[
        exact_mask
    ]

    selected_x = x[
        exact_mask
    ]

    selected_curve = curve[
        exact_mask
    ]

    if len(selected_rounds) == 0:
        raise ValueError(
            "No rounds matching "
            f"eval_round_freq={requested_eval_round_freq} "
            "exist in the logged range "
            f"[{rounded_rounds.min()}, "
            f"{rounded_rounds.max()}]."
        )

    # ------------------------------------------------------------------------
    # 저장 range 안에서 있어야 하는 exact evaluation round가 빠졌는지 검사
    # ------------------------------------------------------------------------

    first_expected = (
        (
            int(
                rounded_rounds.min()
            )
            + requested_eval_round_freq
            - 1
        )
        // requested_eval_round_freq
    ) * requested_eval_round_freq

    last_expected = (
        int(
            rounded_rounds.max()
        )
        // requested_eval_round_freq
    ) * requested_eval_round_freq

    if first_expected <= last_expected:
        expected_rounds = np.arange(
            first_expected,
            last_expected + 1,
            requested_eval_round_freq,
            dtype=np.int64,
        )

        missing_rounds = np.setdiff1d(
            expected_rounds,
            selected_rounds,
        )

        if len(missing_rounds) > 0:
            preview = missing_rounds[
                :10
            ].tolist()

            more = (
                " ..."
                if len(missing_rounds) > 10
                else ""
            )

            raise ValueError(
                "Missing exact evaluation rounds for "
                f"requested eval_round_freq="
                f"{requested_eval_round_freq}: "
                f"{preview}{more}. "
                "Interpolation is disabled."
            )

    return (
        selected_x,
        selected_curve,
        selected_rounds,
        logged_eval_round_freq,
    )


# ============================================================================
# Load one seed
# ============================================================================


def load_single_curve(
    npz_path,
    metric="nominal",
    eval_round_freq=None,
):
    """
    한 seed의 evaluations.npz에서 metric curve를 가져온다.

    metric:
        nominal
        local_mean
        local_min

    local_mean:
        client return들의 평균

    local_min:
        client return들의 minimum
        -> worst-case local performance
    """
    with np.load(
        npz_path,
        allow_pickle=True,
    ) as data:

        if metric == "nominal":
            mean_all = np.array(
                data["nominal_mean"],
                dtype=float,
            )

            if mean_all.ndim == 2:
                mean_curve = np.mean(
                    mean_all,
                    axis=1,
                )
            else:
                mean_curve = mean_all

        elif metric == "local_mean":
            mean_all = np.array(
                data["local_mean"],
                dtype=float,
            )

            if mean_all.ndim == 2:
                mean_curve = np.mean(
                    mean_all,
                    axis=1,
                )
            else:
                mean_curve = mean_all

        elif metric == "local_min":
            mean_all = np.array(
                data["local_mean"],
                dtype=float,
            )

            if mean_all.ndim == 2:
                mean_curve = np.min(
                    mean_all,
                    axis=1,
                )
            else:
                mean_curve = mean_all

        else:
            raise ValueError(
                f"Unknown metric: {metric}"
            )

        num_rounds = len(
            mean_curve
        )

        x = make_timesteps_from_npz(
            data=data,
            num_rounds=num_rounds,
            metric_array=mean_all,
        )

        if "rounds" in data:
            rounds = np.asarray(
                data["rounds"],
                dtype=float,
            )[:num_rounds]
        else:
            rounds = np.arange(
                num_rounds,
                dtype=float,
            )

        if eval_round_freq is None:
            logged_eval_round_freq = (
                get_logged_eval_round_freq(
                    data,
                    rounds,
                )
                if "rounds" in data
                else None
            )

            return (
                x,
                mean_curve,
                rounds,
                logged_eval_round_freq,
            )

        return select_exact_eval_rounds(
            data=data,
            x=x,
            curve=mean_curve,
            requested_eval_round_freq=eval_round_freq,
        )


# ============================================================================
# Exact seed alignment
# ============================================================================


def align_seed_curves_by_round_exact(
    x_list,
    curve_list,
    rounds_list,
):
    """
    여러 seed를 실제 round 번호 기준으로 정확히 정렬한다.

    - interpolation 없음
    - 모든 seed에 실제 존재하는 common round만 사용
    """
    if not (
        len(x_list)
        == len(curve_list)
        == len(rounds_list)
    ):
        raise ValueError(
            "x_list, curve_list, and rounds_list "
            "must have the same length."
        )

    if len(rounds_list) == 0:
        raise ValueError(
            "No seed curves to align."
        )

    normalized = []

    for (
        x,
        curve,
        rounds,
    ) in zip(
        x_list,
        curve_list,
        rounds_list,
    ):
        x = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        curve = np.asarray(
            curve,
            dtype=float,
        ).reshape(-1)

        rounds = np.asarray(
            rounds,
            dtype=np.int64,
        ).reshape(-1)

        usable_len = min(
            len(x),
            len(curve),
            len(rounds),
        )

        x = x[:usable_len]
        curve = curve[:usable_len]
        rounds = rounds[:usable_len]

        if (
            len(
                np.unique(
                    rounds
                )
            )
            != len(rounds)
        ):
            raise ValueError(
                "Duplicate round values exist in a seed."
            )

        order = np.argsort(
            rounds,
            kind="stable",
        )

        normalized.append(
            (
                x[order],
                curve[order],
                rounds[order],
            )
        )

    common_rounds = normalized[
        0
    ][2]

    for (
        _,
        _,
        rounds,
    ) in normalized[1:]:
        common_rounds = np.intersect1d(
            common_rounds,
            rounds,
            assume_unique=True,
        )

    if len(common_rounds) == 0:
        raise ValueError(
            "No exact common evaluation rounds exist "
            "across seeds."
        )

    aligned_x = []
    aligned_curves = []

    for (
        x,
        curve,
        rounds,
    ) in normalized:

        index_by_round = {
            int(r): i
            for i, r in enumerate(
                rounds
            )
        }

        indices = np.array(
            [
                index_by_round[int(r)]
                for r in common_rounds
            ],
            dtype=int,
        )

        aligned_x.append(
            x[indices]
        )

        aligned_curves.append(
            curve[indices]
        )

    reference_x = aligned_x[
        0
    ]

    for seed_idx, seed_x in enumerate(
        aligned_x[1:],
        start=2,
    ):
        if not np.allclose(
            seed_x,
            reference_x,
            rtol=1e-9,
            atol=1e-9,
            equal_nan=False,
        ):
            raise ValueError(
                "The same evaluation rounds map to "
                "different timestep x-values across seeds. "
                f"Mismatch at seed index {seed_idx}. "
                "Check local_steps, num_clients, or timesteps."
            )

    return (
        reference_x,
        np.vstack(
            aligned_curves
        ),
        common_rounds,
    )


# ============================================================================
# Collect seeds for one configuration
# ============================================================================


def collect_seed_curves(
    algo_id,
    env_id,
    result_root_path,
    metric,
    num_trials,
    eval_round_freq=None,
):
    """
    여러 seed의 metric curve를 수집한다.

    expected:
        {result_root_path}/{algo_id}/{env_id}_{seed}/evaluations.npz

    seed 내부에서는 exact common evaluation round만 사용한다.
    """
    eval_round_freq = normalize_eval_round_freq(
        eval_round_freq
    )

    x_list = []
    curve_list = []
    rounds_list = []

    valid_seeds = []
    logged_freqs = []

    for seed in range(
        1,
        num_trials + 1,
    ):
        npz_path = os.path.join(
            result_root_path,
            algo_id,
            f"{env_id}_{seed}",
            "evaluations.npz",
        )

        if not os.path.exists(
            npz_path
        ):
            print(
                f"[Missing] seed {seed}: "
                f"{npz_path}"
            )
            continue

        try:
            (
                x,
                mean_curve,
                rounds,
                logged_freq,
            ) = load_single_curve(
                npz_path=npz_path,
                metric=metric,
                eval_round_freq=eval_round_freq,
            )

        except ValueError as e:
            raise ValueError(
                f"Failed to load seed {seed} strictly: "
                f"{npz_path}\n{e}"
            ) from e

        except Exception as e:
            print(
                f"[Error] seed {seed}: "
                f"{npz_path}"
            )
            print(
                f"        {e}"
            )
            continue

        x_list.append(
            x
        )

        curve_list.append(
            mean_curve
        )

        rounds_list.append(
            np.asarray(
                rounds,
                dtype=np.int64,
            )
        )

        valid_seeds.append(
            seed
        )

        logged_freqs.append(
            logged_freq
        )

    if len(curve_list) == 0:
        return (
            None,
            None,
            [],
        )

    (
        _,
        seed_curves,
        common_rounds,
    ) = align_seed_curves_by_round_exact(
        x_list=x_list,
        curve_list=curve_list,
        rounds_list=rounds_list,
    )

    # ------------------------------------------------------------------------
    # 논문 그림의 x-axis는 Global Communication Rounds로 사용한다.
    # 따라서 timestep이 아니라 exact common rounds를 반환한다.
    # ------------------------------------------------------------------------

    x = common_rounds.astype(
        float
    )

    freq_info = ", ".join(
        f"seed {seed}: {freq}"
        for seed, freq in zip(
            valid_seeds,
            logged_freqs,
        )
    )

    requested_text = (
        "all"
        if eval_round_freq is None
        else str(
            eval_round_freq
        )
    )

    print(
        f"[Exact Eval Round] "
        f"requested={requested_text}; "
        f"logged=({freq_info}); "
        f"common points={len(common_rounds)}; "
        f"round range="
        f"{common_rounds[0]}.."
        f"{common_rounds[-1]}"
    )

    return (
        x,
        seed_curves,
        valid_seeds,
    )


# ============================================================================
# Smoothing
# ============================================================================


def normalize_window_size(
    window_size=None,
):
    """
    smoothing window를 정규화한다.
    """
    if window_size is None:
        return 1

    window_size = int(
        window_size
    )

    if window_size < 1:
        raise ValueError(
            "window_size must be >= 1, "
            f"but got {window_size}"
        )

    return window_size


def make_window_suffix(
    window_size=None,
):
    """
    파일명용 smoothing suffix.
    """
    window_size = normalize_window_size(
        window_size
    )

    if window_size <= 1:
        return ""

    return (
        f"_window{window_size}"
    )


def smooth_curve_with_window(
    curve,
    window_size=None,
):
    """
    centered moving average.

    출력 길이는 입력과 동일하다.
    edge에서는 가능한 범위만 사용한다.
    """
    curve = np.asarray(
        curve,
        dtype=float,
    )

    window_size = normalize_window_size(
        window_size
    )

    if (
        window_size <= 1
        or len(curve) == 0
    ):
        return curve.copy()

    window_size = min(
        window_size,
        len(curve),
    )

    left = (
        window_size // 2
    )

    right = (
        window_size
        - left
        - 1
    )

    smoothed_curve = np.empty_like(
        curve,
        dtype=float,
    )

    for idx in range(
        len(curve)
    ):
        start = max(
            0,
            idx - left,
        )

        end = min(
            len(curve),
            idx + right + 1,
        )

        window_values = curve[
            start:end
        ]

        if np.all(
            np.isnan(
                window_values
            )
        ):
            smoothed_curve[
                idx
            ] = np.nan

        else:
            smoothed_curve[
                idx
            ] = np.nanmean(
                window_values
            )

    return smoothed_curve


def smooth_seed_curves_with_window(
    seed_curves,
    window_size=None,
):
    """
    각 seed curve에 smoothing 적용.

    shape:
        (num_seeds, num_rounds)
    """
    window_size = normalize_window_size(
        window_size
    )

    seed_curves = np.asarray(
        seed_curves,
        dtype=float,
    )

    if window_size <= 1:
        return seed_curves.copy()

    return np.array(
        [
            smooth_curve_with_window(
                curve,
                window_size,
            )
            for curve in seed_curves
        ],
        dtype=float,
    )


# ============================================================================
# NEW:
# Align different N configurations to shortest training horizon
# ============================================================================


def truncate_algo_data_to_shortest_horizon(
    algo_data_list,
):
    """
    모든 client-count 설정을 가장 짧게 학습된 설정의 x horizon에 맞춘다.

    Input:
        [
            (plot_label, x, seed_curves),
            ...
        ]

    예:
        N=3 -> 4000 rounds
        N=5 -> 3920 rounds
        N=7 -> 3200 rounds
        N=9 -> 2800 rounds

    그러면 모든 curve를 2800 round까지만 표시한다.

    IMPORTANT:
        반드시 smoothing 전에 이 함수를 호출해야 한다.

    이유:
        긴 run의 cutoff 이후 값이 centered moving-average를 통해
        마지막 common region의 값에 섞이는 것을 방지한다.
    """
    if len(
        algo_data_list
    ) == 0:
        return (
            [],
            None,
        )

    normalized_data = []
    final_x_values = []

    # ------------------------------------------------------------------------
    # 각 configuration 정리
    # ------------------------------------------------------------------------

    for (
        plot_label,
        x,
        seed_curves,
    ) in algo_data_list:

        x = np.asarray(
            x,
            dtype=float,
        ).reshape(-1)

        seed_curves = np.asarray(
            seed_curves,
            dtype=float,
        )

        if seed_curves.ndim != 2:
            raise ValueError(
                f"{plot_label}: "
                "seed_curves must have shape "
                "(num_seeds, num_points), "
                f"but got {seed_curves.shape}"
            )

        usable_len = min(
            len(x),
            seed_curves.shape[1],
        )

        if usable_len == 0:
            raise ValueError(
                f"{plot_label}: "
                "no usable x/curve points."
            )

        x = x[
            :usable_len
        ]

        seed_curves = seed_curves[
            :,
            :usable_len,
        ]

        finite_x = np.isfinite(
            x
        )

        x = x[
            finite_x
        ]

        seed_curves = seed_curves[
            :,
            finite_x,
        ]

        if len(x) == 0:
            raise ValueError(
                f"{plot_label}: "
                "no finite x values."
            )

        final_x = float(
            np.max(
                x
            )
        )

        final_x_values.append(
            final_x
        )

        normalized_data.append(
            (
                plot_label,
                x,
                seed_curves,
            )
        )

    # ------------------------------------------------------------------------
    # 가장 짧은 training horizon
    # ------------------------------------------------------------------------

    common_x_max = float(
        np.min(
            final_x_values
        )
    )

    truncated_data = []

    for (
        plot_label,
        x,
        seed_curves,
    ) in normalized_data:

        tolerance = (
            1e-9
            * max(
                1.0,
                abs(
                    common_x_max
                ),
            )
        )

        keep_mask = (
            x
            <= common_x_max
            + tolerance
        )

        truncated_x = x[
            keep_mask
        ]

        truncated_seed_curves = (
            seed_curves[
                :,
                keep_mask,
            ]
        )

        if len(
            truncated_x
        ) == 0:
            raise ValueError(
                f"{plot_label}: "
                "no data remains after "
                "common-horizon truncation."
            )

        truncated_data.append(
            (
                plot_label,
                truncated_x,
                truncated_seed_curves,
            )
        )

        print(
            f"[Common Horizon] "
            f"{plot_label}: "
            f"original_end="
            f"{x[-1]:g}, "
            f"used_end="
            f"{truncated_x[-1]:g}, "
            f"points="
            f"{len(truncated_x)}"
        )

    print(
        "[Common Horizon] "
        "All Num Clients curves "
        f"are restricted to <= "
        f"{common_x_max:g} communication rounds."
    )

    return (
        truncated_data,
        common_x_max,
    )


# ============================================================================
# Plot
# ============================================================================


def get_metric_ylabel(
    metric,
):
    """
    논문용 y-axis label.
    """
    labels = {
        "nominal": "Nominal Return",
        "local_mean": "Average Local Return",
        "local_min": "Worst-Case Local Return",
    }

    return labels.get(
        metric,
        "Return",
    )


def plot_multiple_algos(
    algo_data_list,
    env_id,
    metric,
    save_dir,
    filename_prefix,
    window_size=None,
):
    """
    Number-of-Clients ablation plot.

    순서:
        1. N별 seed data 준비
        2. 모든 N을 shortest horizon으로 truncate
        3. 각 seed를 smoothing
        4. seed mean/std 계산
        5. plot

    따라서 더 긴 run의 미래 값은 common horizon의 마지막 부분에
    영향을 주지 않는다.
    """
    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    window_size = normalize_window_size(
        window_size
    )

    window_suffix = make_window_suffix(
        window_size
    )

    # ------------------------------------------------------------------------
    # 가장 짧은 N 설정까지만 사용.
    #
    # IMPORTANT:
    # smoothing 이전에 수행.
    # ------------------------------------------------------------------------

    (
        algo_data_list,
        common_x_max,
    ) = truncate_algo_data_to_shortest_horizon(
        algo_data_list
    )

    plt.figure(
        figsize=(10, 6)
    )

    for (
        plot_label,
        x,
        seed_curves,
    ) in algo_data_list:

        # --------------------------------------------------------------------
        # truncate 이후 seed별 smoothing
        # --------------------------------------------------------------------

        plot_seed_curves = (
            smooth_seed_curves_with_window(
                seed_curves,
                window_size,
            )
        )

        avg_curve = np.nanmean(
            plot_seed_curves,
            axis=0,
        )

        std_curve = np.nanstd(
            plot_seed_curves,
            axis=0,
        )

        # --------------------------------------------------------------------
        # Seed average
        # --------------------------------------------------------------------

        plt.plot(
            x,
            avg_curve,
            linewidth=2.5,
            label=plot_label,
        )

        # --------------------------------------------------------------------
        # Seed std
        # --------------------------------------------------------------------

        plt.fill_between(
            x,
            avg_curve - std_curve,
            avg_curve + std_curve,
            alpha=0.15,
        )

    # ------------------------------------------------------------------------
    # Axis
    # ------------------------------------------------------------------------

    if common_x_max is not None:
        plt.xlim(
            right=common_x_max
        )

    plt.xlabel(
        "Global Communication Rounds"
    )

    plt.ylabel(
        get_metric_ylabel(
            metric
        )
    )

    plt.grid(
        True,
        alpha=0.3,
    )

    # ------------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------------

    handles, labels = (
        plt.gca()
        .get_legend_handles_labels()
    )

    unique_legend = {}

    for (
        handle,
        label,
    ) in zip(
        handles,
        labels,
    ):
        if label not in unique_legend:
            unique_legend[
                label
            ] = handle

    plt.legend(
        unique_legend.values(),
        unique_legend.keys(),
        title=r"Number of Clients ($N$)",
        frameon=False,
    )

    plt.tight_layout(
        pad=0.15,
        h_pad=0.15,
        w_pad=0.15,
    )

    # ------------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------------

    save_file = os.path.join(
        save_dir,
        (
            f"{filename_prefix}_"
            f"{metric}_learning_curve"
            f"{window_suffix}.png"
        ),
    )

    plt.savefig(
        save_file,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()

    print(
        f"[Saved] {save_file}"
    )


# ============================================================================
# Main
# ============================================================================


def main():
    # ------------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------------

    env_id = "PerturbAnt-v4"

    # ------------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------------

    metric_list = [
        "nominal",
        "local_mean",
        "local_min",
    ]

    # ------------------------------------------------------------------------
    # Number of seeds
    # ------------------------------------------------------------------------

    num_trials = 5

    # ------------------------------------------------------------------------
    # Plot smoothing
    #
    # 1 or None:
    #     smoothing 없음
    #
    # 3:
    #     centered moving average window 3
    # ------------------------------------------------------------------------

    plot_window_size = 3

    # ------------------------------------------------------------------------
    # Evaluation round frequency
    #
    # 40:
    #     round 40, 80, 120, ... 의 실제 저장값만 사용
    #
    # interpolation은 하지 않는다.
    # ------------------------------------------------------------------------

    plot_eval_round_freq = 40

    # ------------------------------------------------------------------------
    # Perturbations
    # ------------------------------------------------------------------------

    perturbation_types = [
        "friction",
        "gravity",
    ]

    # ========================================================================
    # Num Clients ablation
    # ========================================================================

    for perturbation_type in perturbation_types:

        # --------------------------------------------------------------------
        # IMPORTANT:
        #
        # default experiment corresponds to N=5.
        #
        # 모든 설정은 동일한 AMPO-PPO(A) configuration이며
        # Num Clients N만 변경한다.
        # --------------------------------------------------------------------

        algo_config_list = [
            (
                "fed_ampo_ppo",
                (
                    "logs/fed_ampo/tuned_mujoco/fixed/"
                    "noise_assignment/"
                    f"{perturbation_type}/"
                    "3/0.3/"
                    "fed_ampo_ppo/adaptive/"
                    "undiscounted/0.0003"
                ),
                r"$N=3$",
            ),
            (
                "fed_ampo_ppo",
                (
                    "logs/fed_ampo/tuned_mujoco/fixed/"
                    "noise_assignment/"
                    f"{perturbation_type}/"
                    "0.3/"
                    "fed_ampo_ppo/adaptive/"
                    "undiscounted/0.0003"
                ),
                r"$N=5$",
            ),
            (
                "fed_ampo_ppo",
                (
                    "logs/fed_ampo/tuned_mujoco/fixed/"
                    "noise_assignment/"
                    f"{perturbation_type}/"
                    "7/0.3/"
                    "fed_ampo_ppo/adaptive/"
                    "undiscounted/0.0003"
                ),
                r"$N=7$",
            ),
            (
                "fed_ampo_ppo",
                (
                    "logs/fed_ampo/tuned_mujoco/fixed/"
                    "noise_assignment/"
                    f"{perturbation_type}/"
                    "9/0.3/"
                    "fed_ampo_ppo/adaptive/"
                    "undiscounted/0.0003"
                ),
                r"$N=9$",
            ),
        ]

        # --------------------------------------------------------------------
        # Plot output
        # --------------------------------------------------------------------

        plot_root_path = (
            "plots/iclr2027_ampo/"
            f"num_clients/{perturbation_type}"
        )

        filename_prefix = (
            "ampo_num_clients"
        )

        # ====================================================================
        # Metric loop
        # ====================================================================

        for metric in metric_list:

            algo_data_list = []

            # ----------------------------------------------------------------
            # Load each N
            # ----------------------------------------------------------------

            for (
                algo_id,
                result_root_path,
                plot_label,
            ) in algo_config_list:

                (
                    x,
                    seed_curves,
                    valid_seeds,
                ) = collect_seed_curves(
                    algo_id=algo_id,
                    env_id=env_id,
                    result_root_path=result_root_path,
                    metric=metric,
                    num_trials=num_trials,
                    eval_round_freq=plot_eval_round_freq,
                )

                if seed_curves is None:
                    print(
                        "[Skip] "
                        "No valid data: "
                        f"{plot_label}, "
                        f"{env_id}, "
                        f"{metric}"
                    )
                    continue

                print(
                    f"[Info] "
                    f"{plot_label}, "
                    f"{env_id}, "
                    f"{metric}: "
                    f"{len(valid_seeds)} "
                    "seeds loaded; "
                    f"x range="
                    f"{x[0]:g}.."
                    f"{x[-1]:g}"
                )

                algo_data_list.append(
                    (
                        plot_label,
                        x,
                        seed_curves,
                    )
                )

            # ----------------------------------------------------------------
            # Plot all N together
            # ----------------------------------------------------------------

            if len(
                algo_data_list
            ) == 0:
                print(
                    "[Skip] "
                    "No client-count configuration "
                    f"available for {metric}."
                )
                continue

            save_dir = os.path.join(
                plot_root_path,
                env_id,
            )

            plot_multiple_algos(
                algo_data_list=algo_data_list,
                env_id=env_id,
                metric=metric,
                save_dir=save_dir,
                filename_prefix=filename_prefix,
                window_size=plot_window_size,
            )


if __name__ == "__main__":
    main()