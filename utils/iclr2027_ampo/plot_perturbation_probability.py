import os

import numpy as np
import matplotlib.pyplot as plt


# =============================================================================
# Publication-style plotting defaults
# =============================================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 15,
    "axes.labelsize": 15,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 12,
    "axes.unicode_minus": False,
})


# =============================================================================
# NPZ helper
# =============================================================================

def read_scalar_from_npz(data, key, default=None):
    """
    evaluations.npz 안의 scalar 값을 안전하게 읽는다.

    예:
        local_steps.npy -> 128
        num_clients.npy -> 5
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


def infer_num_clients_from_npz(data, metric_array=None):
    """
    evaluations.npz 내부 값으로 num_clients를 추정한다.

    우선순위:
        1. num_clients.npy
        2. client_noises.npy 길이
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
    evaluations.npz 내부 값만 이용해서 x축 timesteps를 만든다.

    우선순위:
        1. timesteps.npy가 있으면 그대로 사용
        2. rounds.npy + local_steps.npy + num_clients.npy 사용
           x = rounds * local_steps * num_clients
        3. rounds.npy + local_steps.npy
        4. rounds.npy
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


# =============================================================================
# Evaluation round handling
# =============================================================================

def normalize_eval_round_freq(eval_round_freq):
    """
    사용자가 지정한 evaluation round 주기를 양의 정수로 정규화한다.
    """
    if eval_round_freq is None:
        return None

    value = float(eval_round_freq)
    rounded = int(round(value))

    if (
        not np.isfinite(value)
        or value <= 0
        or not np.isclose(value, rounded)
    ):
        raise ValueError(
            f"eval_round_freq must be a positive integer, "
            f"but got {eval_round_freq}"
        )

    return rounded


def get_logged_eval_round_freq(data, rounds):
    """
    한 evaluations.npz가 실제로 저장한 evaluation round 주기를 가져온다.

    우선순위:
        1. eval_round_freq.npy
        2. 없으면 rounds의 positive diff median으로 추정

    마지막 학습 round에서 추가 evaluation이 있어도
    median을 사용해 정상 evaluation frequency를 추정한다.
    """
    logged_freq = read_scalar_from_npz(
        data,
        "eval_round_freq",
        default=None,
    )

    if logged_freq is not None:
        value = float(logged_freq)
        rounded = int(round(value))

        if (
            not np.isfinite(value)
            or value <= 0
            or not np.isclose(value, rounded)
        ):
            raise ValueError(
                f"Invalid eval_round_freq in evaluations.npz: "
                f"{logged_freq}"
            )

        return rounded

    rounds = np.asarray(
        rounds,
        dtype=float,
    ).reshape(-1)

    diffs = np.diff(rounds)

    positive_diffs = diffs[
        np.isfinite(diffs)
        & (diffs > 0)
    ]

    if len(positive_diffs) == 0:
        raise ValueError(
            "eval_round_freq is missing and it cannot be "
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
        logged freq = 20
        requested   = 40

        -> 40, 80, 120, ... 만 사용

    중요:
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
            f"No rounds matching "
            f"eval_round_freq={requested_eval_round_freq} "
            f"exist in the logged range "
            f"[{rounded_rounds.min()}, "
            f"{rounded_rounds.max()}]."
        )

    # -------------------------------------------------------------------------
    # 필요한 exact evaluation round가 중간에 빠져 있는지 확인
    # -------------------------------------------------------------------------

    first_expected = (
        (
            int(rounded_rounds.min())
            + requested_eval_round_freq
            - 1
        )
        // requested_eval_round_freq
    ) * requested_eval_round_freq

    last_expected = (
        int(rounded_rounds.max())
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
                f"Missing exact evaluation rounds for requested "
                f"eval_round_freq={requested_eval_round_freq}: "
                f"{preview}{more}. "
                "Interpolation is disabled."
            )

    return (
        selected_x,
        selected_curve,
        selected_rounds,
        logged_eval_round_freq,
    )


# =============================================================================
# Load one seed curve
# =============================================================================

def load_single_curve(
    npz_path,
    metric="nominal",
    eval_round_freq=None,
):
    """
    한 seed의 evaluations.npz에서 round별 metric curve를 가져온다.

    metric:
        nominal
        local_mean
        local_min
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


# =============================================================================
# Align seeds using exact rounds
# =============================================================================

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
            len(np.unique(rounds))
            != len(rounds)
        ):
            raise ValueError(
                "Duplicate round values exist in a seed "
                "after filtering."
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

    for (
        seed_idx,
        seed_x,
    ) in enumerate(
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
                "The same evaluation rounds map to different "
                "timestep x-values across seeds. "
                f"Mismatch detected at seed index {seed_idx}. "
                "Check local_steps, num_clients, or saved timesteps."
            )

    return (
        reference_x,
        np.vstack(
            aligned_curves
        ),
        common_rounds,
    )


# =============================================================================
# Collect seeds
# =============================================================================

def collect_seed_curves(
    algo_id,
    env_id,
    result_root_path,
    metric,
    num_trials,
    eval_round_freq=None,
):
    """
    여러 seed의 mean curve를 모은다.

    expected path:

        {result_root_path}/
            {algo_id}/
                {env_id}_{seed}/
                    evaluations.npz
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

    # x-axis는 Global Communication Rounds 사용
    x = common_rounds.astype(
        float
    )

    freq_info = ", ".join(
        f"seed {seed}: {freq}"
        for (
            seed,
            freq,
        ) in zip(
            valid_seeds,
            logged_freqs,
        )
    )

    requested_text = (
        "all"
        if eval_round_freq is None
        else str(eval_round_freq)
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


# =============================================================================
# Extra path args
# =============================================================================

def normalize_extra_args(extra_args=None):
    """
    추가 하위 폴더 인자를 tuple 형태로 정규화한다.
    """
    if extra_args is None:
        return ()

    if isinstance(
        extra_args,
        (list, tuple),
    ):
        return tuple(
            extra_args
        )

    return (
        extra_args,
    )


def normalize_extra_arg_sets(extra_arg_sets=None):
    """
    여러 extra_args 조합을 정규화한다.
    """
    if (
        extra_arg_sets is None
        or len(extra_arg_sets) == 0
    ):
        return [
            ()
        ]

    return [
        normalize_extra_args(
            extra_args
        )
        for extra_args in extra_arg_sets
    ]


def append_extra_args_to_path(
    root_path,
    extra_args=None,
):
    """
    extra_args가 있으면 root_path 아래에 붙인다.
    """
    extra_args = normalize_extra_args(
        extra_args
    )

    if len(extra_args) == 0:
        return root_path

    return os.path.join(
        root_path,
        *[
            str(arg)
            for arg in extra_args
        ],
    )


def make_extra_args_suffix(
    extra_args=None,
):
    """
    파일명에 붙일 extra_args suffix.
    """
    extra_args = normalize_extra_args(
        extra_args
    )

    if len(extra_args) == 0:
        return ""

    return (
        "_"
        + "_".join(
            str(arg)
            for arg in extra_args
        )
    )


# =============================================================================
# Algorithm config
# =============================================================================

def unpack_algo_config(config):
    """
    지원 형식:

        (algo_id, result_root_path)

        (algo_id, result_root_path, plot_label)
    """
    if len(config) == 2:
        (
            algo_id,
            result_root_path,
        ) = config

        plot_label = None

    elif len(config) == 3:
        (
            algo_id,
            result_root_path,
            plot_label,
        ) = config

    else:
        raise ValueError(
            "Each algo config must be "
            "(algo_id, path) or "
            "(algo_id, path, plot_label)."
        )

    return (
        algo_id,
        result_root_path,
        plot_label,
    )


def count_algo_ids(
    algo_config_list,
):
    """
    algo_config_list에서 algo_id 등장 횟수 계산.
    """
    algo_id_counts = {}

    for config in algo_config_list:
        (
            algo_id,
            _,
            _,
        ) = unpack_algo_config(
            config
        )

        algo_id_counts[
            algo_id
        ] = (
            algo_id_counts.get(
                algo_id,
                0,
            )
            + 1
        )

    return algo_id_counts


def make_unique_plot_label(
    algo_id,
    result_root_path,
    algo_id_counts,
    used_plot_labels,
    custom_plot_label=None,
):
    """
    custom plot label이 있으면 그것을 우선 사용한다.
    """
    if custom_plot_label is not None:
        base_label = str(
            custom_plot_label
        )

    elif algo_id == "ppo_avg":
        base_label = "PPOAvg"

    elif algo_id == "fed_ampo_ppo":
        base_label = "AMPO-PPO"

    elif algo_id == "fed_svrpg_m":
        base_label = "FedSVRPG-M-PPO"

    else:
        base_label = algo_id

    if base_label not in used_plot_labels:
        return base_label

    duplicate_idx = 2

    while (
        f"{base_label} ({duplicate_idx})"
        in used_plot_labels
    ):
        duplicate_idx += 1

    return (
        f"{base_label} ({duplicate_idx})"
    )


# =============================================================================
# Smoothing
# =============================================================================

def normalize_window_size(
    window_size=None,
):
    if window_size is None:
        return 1

    window_size = int(
        window_size
    )

    if window_size < 1:
        raise ValueError(
            f"window_size must be >= 1, "
            f"but got {window_size}"
        )

    return window_size


def make_window_suffix(
    window_size=None,
):
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
    seed_curves = np.asarray(
        seed_curves,
        dtype=float,
    )

    window_size = normalize_window_size(
        window_size
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


# =============================================================================
# Common shortest horizon
# =============================================================================

def truncate_algo_data_to_shortest_horizon(
    algo_data_list,
):
    """
    perturbation strength별 학습 길이가 다르면,
    가장 짧은 curve의 마지막 communication round까지만 사용한다.

    IMPORTANT:
        smoothing 전에 truncate한다.
    """
    if len(algo_data_list) == 0:
        return (
            [],
            None,
        )

    normalized = []
    final_x_values = []

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
                f"seed_curves must be 2D, "
                f"got {seed_curves.shape}"
            )

        usable_len = min(
            len(x),
            seed_curves.shape[1],
        )

        x = x[:usable_len]
        seed_curves = seed_curves[
            :,
            :usable_len,
        ]

        if len(x) == 0:
            raise ValueError(
                f"{plot_label}: "
                "no usable data."
            )

        final_x_values.append(
            float(
                x[-1]
            )
        )

        normalized.append(
            (
                plot_label,
                x,
                seed_curves,
            )
        )

    common_x_max = float(
        np.min(
            final_x_values
        )
    )

    truncated = []

    for (
        plot_label,
        x,
        seed_curves,
    ) in normalized:

        keep_mask = (
            x
            <= common_x_max
            + 1e-9
        )

        truncated_x = x[
            keep_mask
        ]

        truncated_seed_curves = seed_curves[
            :,
            keep_mask,
        ]

        truncated.append(
            (
                plot_label,
                truncated_x,
                truncated_seed_curves,
            )
        )

        print(
            f"[Common Horizon] "
            f"{plot_label}: "
            f"original_end={x[-1]:g}, "
            f"used_end={truncated_x[-1]:g}, "
            f"points={len(truncated_x)}"
        )

    print(
        "[Common Horizon] "
        "All perturbation-strength curves are restricted to "
        f"round <= {common_x_max:g}"
    )

    return (
        truncated,
        common_x_max,
    )


# =============================================================================
# Plot
# =============================================================================

def get_metric_ylabel(metric):
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
    서로 다른 perturbation strength의 seed average curve를 비교한다.
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

    # -------------------------------------------------------------------------
    # 가장 짧은 perturbation-strength run까지만 비교
    #
    # 반드시 smoothing 전에 수행한다.
    # -------------------------------------------------------------------------

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

        plt.plot(
            x,
            avg_curve,
            linewidth=2.5,
            label=plot_label,
        )

        plt.fill_between(
            x,
            avg_curve - std_curve,
            avg_curve + std_curve,
            alpha=0.15,
        )

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
        title="Perturbation Strength",
        frameon=False,
    )

    plt.tight_layout(
        pad=0.15,
        h_pad=0.15,
        w_pad=0.15,
    )

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


# =============================================================================
# Main
# =============================================================================

def main():
    # env_id_list = ["PerturbPendulum-v1"]
    env_id = "PerturbAnt-v4"

    metric_list = [
        "nominal",
        "local_mean",
        "local_min",
    ]

    num_trials = 5

    # plot smoothing window
    # - 1 또는 None이면 기존처럼 smoothing 없이 plot
    # - 예: 5, 10, 20 등으로 설정하면 centered moving average 적용
    plot_window_size = 3

    # plot에 사용할 evaluation round 주기
    # 예: 80이면 모든 seed에서 round 80, 160, 240, ... 의 실제 저장값만 사용
    # seed의 원래 eval_round_freq가 80의 약수가 아니면 ValueError 발생
    # (예: logged=30, requested=80 -> error)
    plot_eval_round_freq = 40

    # perturbation_types = ["none", "gravity", "mass", "length"]
    perturbation_types = [
        "friction",
        "gravity",
    ]

    for perturbation_type in perturbation_types:

        # =====================================================================
        # Perturbation-strength ablation
        #
        # 0.1 -> explicit /0.1/ directory
        # 0.3 -> explicit /0.3/ directory
        #
        # 0.5 -> DEFAULT experiment
        #        숫자 directory가 없음
        #
        # 0.7 -> explicit /0.7/ directory
        #
        # Algorithm / dual setting은 모두 동일:
        #
        #   fed_ampo_ppo / adaptive / undiscounted / 0.0003
        # =====================================================================

        algo_config_list = [
            (
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/"
                f"{perturbation_type}/0.1/"
                f"fed_ampo_ppo/adaptive/undiscounted/0.0003",
                "0.1",
            ),

            (
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/"
                f"{perturbation_type}/0.3/"
                f"fed_ampo_ppo/adaptive/undiscounted/0.0003",
                "0.3",
            ),

            # -------------------------------------------------------------
            # Default perturbation strength = 0.5
            #
            # IMPORTANT:
            # /0.5/ directory가 들어가지 않는다.
            # -------------------------------------------------------------
            (
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/"
                f"{perturbation_type}/"
                f"fed_ampo_ppo/adaptive/undiscounted/0.0003",
                "0.5",
            ),

            (
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/"
                f"{perturbation_type}/0.7/"
                f"fed_ampo_ppo/adaptive/undiscounted/0.0003",
                "0.7",
            ),
        ]

        # plot 저장 root
        plot_root_path = (
            f"plots/iclr2027_ampo/"
            f"perturbation_strength/"
            f"{env_id}/"
            f"{perturbation_type}"
        )

        # 추가 하위 폴더 인자 묶음
        #
        # 사용 예시:
        #
        #   extra_arg_sets = []
        #       -> 추가 폴더 없이 실행
        #
        #   extra_arg_sets = [1024, 2048]
        #       -> result_root_path/1024
        #          result_root_path/2048
        #
        #   extra_arg_sets = [
        #       (1024, 64),
        #       (2048, 128),
        #   ]
        #
        #   extra_arg_sets = [
        #       (),
        #       (1024,),
        #       (1024, 64),
        #   ]
        #
        extra_arg_sets = []

        for extra_args in normalize_extra_arg_sets(
            extra_arg_sets
        ):
            # 파일 이름 suffix
            suffix = make_extra_args_suffix(
                extra_args
            )

            filename_prefix = (
                f"ampo_perturbation_strength"
                f"{suffix}"
            )

            # ================================================================
            # Metric별 plot
            # ================================================================

            for metric in metric_list:

                algo_data_list = []

                algo_id_counts = count_algo_ids(
                    algo_config_list
                )

                used_plot_labels = set()

                # ============================================================
                # 각 perturbation strength 데이터 수집
                # ============================================================

                for config in algo_config_list:

                    (
                        algo_id,
                        result_root_path,
                        custom_plot_label,
                    ) = unpack_algo_config(
                        config
                    )

                    result_root_with_args = (
                        append_extra_args_to_path(
                            result_root_path,
                            extra_args,
                        )
                    )

                    plot_label = make_unique_plot_label(
                        algo_id=algo_id,
                        result_root_path=result_root_with_args,
                        algo_id_counts=algo_id_counts,
                        used_plot_labels=used_plot_labels,
                        custom_plot_label=custom_plot_label,
                    )

                    used_plot_labels.add(
                        plot_label
                    )

                    print(
                        f"\n"
                        f"[Load] "
                        f"{perturbation_type}, "
                        f"strength={plot_label}, "
                        f"metric={metric}"
                    )

                    print(
                        f"       path="
                        f"{result_root_with_args}"
                    )

                    (
                        x,
                        seed_curves,
                        valid_seeds,
                    ) = collect_seed_curves(
                        algo_id=algo_id,
                        env_id=env_id,
                        result_root_path=result_root_with_args,
                        metric=metric,
                        num_trials=num_trials,
                        eval_round_freq=plot_eval_round_freq,
                    )

                    if seed_curves is None:
                        print(
                            f"[Skip] No valid data: "
                            f"{plot_label}, "
                            f"{env_id}, "
                            f"{metric}"
                        )

                        continue

                    print(
                        f"[Info] "
                        f"strength={plot_label}, "
                        f"{env_id}, "
                        f"{metric}: "
                        f"{len(valid_seeds)} seeds loaded; "
                        f"round range="
                        f"{x[0]:g}..{x[-1]:g}"
                    )

                    algo_data_list.append(
                        (
                            plot_label,
                            x,
                            seed_curves,
                        )
                    )

                # ============================================================
                # 모든 perturbation strength를 하나의 plot에 표시
                # ============================================================

                if len(
                    algo_data_list
                ) > 0:

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