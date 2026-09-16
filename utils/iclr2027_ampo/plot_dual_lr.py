import os
import numpy as np
import matplotlib.pyplot as plt


# Publication-style plotting defaults
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

    num_clients = read_scalar_from_npz(data, "num_clients", default=None)

    if num_clients is not None:
        return int(num_clients)

    if "client_noises" in data:
        return int(len(np.asarray(data["client_noises"])))

    if metric_array is not None:
        metric_array = np.asarray(metric_array)
        if metric_array.ndim == 2:
            return int(metric_array.shape[1])

    return 1


def make_timesteps_from_npz(data, num_rounds, metric_array=None):
    """
    evaluations.npz 내부 값만 이용해서 x축 timesteps를 만든다.

    우선순위:
        1. timesteps.npy가 있으면 그대로 사용
        2. rounds.npy + local_steps.npy + num_clients.npy 사용
           x = rounds * local_steps * num_clients
        3. rounds.npy + local_steps.npy만 있으면
           x = rounds * local_steps
        4. rounds.npy만 있으면 rounds 그대로 사용
        5. 아무것도 없으면 0, 1, 2, ...
    """

    if "timesteps" in data:
        return np.asarray(data["timesteps"], dtype=float)[:num_rounds]

    if "rounds" in data:
        rounds = np.asarray(data["rounds"], dtype=float)[:num_rounds]

        local_steps = read_scalar_from_npz(data, "local_steps", default=None)

        if local_steps is not None:
            num_clients = infer_num_clients_from_npz(
                data=data,
                metric_array=metric_array,
            )
            return rounds * float(local_steps) * float(num_clients)

        return rounds

    return np.arange(num_rounds, dtype=float)


def normalize_eval_round_freq(eval_round_freq):
    """사용자가 지정한 evaluation round 주기를 양의 정수로 정규화한다."""

    if eval_round_freq is None:
        return None

    value = float(eval_round_freq)
    rounded = int(round(value))

    if not np.isfinite(value) or value <= 0 or not np.isclose(value, rounded):
        raise ValueError(
            f"eval_round_freq must be a positive integer, but got {eval_round_freq}"
        )

    return rounded


def get_logged_eval_round_freq(data, rounds):
    """
    한 evaluations.npz가 실제로 저장한 evaluation round 주기를 가져온다.

    우선순위:
        1. eval_round_freq.npy
        2. 없으면 rounds의 positive diff median으로 추정

    마지막 학습 round를 강제로 평가해서
    rounds가 [..., 3880, 3907]처럼 끝나더라도 median을 사용하므로
    정규 evaluation 주기(예: 40)를 안정적으로 추정할 수 있다.
    """

    logged_freq = read_scalar_from_npz(data, "eval_round_freq", default=None)

    if logged_freq is not None:
        value = float(logged_freq)
        rounded = int(round(value))
        if not np.isfinite(value) or value <= 0 or not np.isclose(value, rounded):
            raise ValueError(
                f"Invalid eval_round_freq in evaluations.npz: {logged_freq}"
            )
        return rounded

    rounds = np.asarray(rounds, dtype=float).reshape(-1)
    diffs = np.diff(rounds)
    positive_diffs = diffs[np.isfinite(diffs) & (diffs > 0)]

    if len(positive_diffs) == 0:
        raise ValueError(
            "eval_round_freq is missing and it cannot be inferred from rounds."
        )

    value = float(np.median(positive_diffs))
    rounded = int(round(value))

    if not np.isclose(value, rounded):
        raise ValueError(
            "eval_round_freq is missing and the inferred round spacing is not an integer: "
            f"{value}"
        )

    return rounded


def select_exact_eval_rounds(data, x, curve, requested_eval_round_freq):
    """
    requested_eval_round_freq에 정확히 해당하는 실제 저장값만 선택한다.

    예:
        seed A logged freq = 10
        seed B logged freq = 40
        requested freq     = 80

    -> 두 seed 모두 round 80, 160, 240, ... 의 실제 저장값만 사용한다.

    중요:
    - 보간(interpolation)은 절대 하지 않는다.
    - requested freq가 logged freq의 정수배가 아니면 ValueError.
      예: logged=30, requested=80 -> error
    - 배수 관계여도 실제로 필요한 round가 파일에 빠져 있으면 ValueError.
    """

    requested_eval_round_freq = normalize_eval_round_freq(requested_eval_round_freq)

    x = np.asarray(x, dtype=float).reshape(-1)
    curve = np.asarray(curve, dtype=float).reshape(-1)

    if "rounds" not in data:
        raise ValueError(
            "Strict eval_round_freq selection requires 'rounds' in evaluations.npz."
        )

    rounds = np.asarray(data["rounds"], dtype=float).reshape(-1)

    usable_len = min(len(rounds), len(x), len(curve))
    rounds = rounds[:usable_len]
    x = x[:usable_len]
    curve = curve[:usable_len]

    if usable_len == 0:
        raise ValueError("No evaluation data exists in evaluations.npz.")

    if not np.all(np.isfinite(rounds)):
        raise ValueError("rounds contains NaN or inf values.")

    rounded_rounds = np.rint(rounds).astype(np.int64)
    if not np.allclose(rounds, rounded_rounds):
        raise ValueError("rounds must contain integer-valued round numbers.")

    logged_eval_round_freq = get_logged_eval_round_freq(data, rounded_rounds)

    if requested_eval_round_freq % logged_eval_round_freq != 0:
        raise ValueError(
            "Requested eval_round_freq is incompatible with this seed: "
            f"requested={requested_eval_round_freq}, "
            f"logged={logged_eval_round_freq}. "
            "The requested frequency must be an integer multiple of the logged frequency "
            "because interpolation is disabled."
        )

    # 사용자가 요청한 정확한 round들만 선택한다.
    exact_mask = (rounded_rounds % requested_eval_round_freq) == 0
    selected_rounds = rounded_rounds[exact_mask]
    selected_x = x[exact_mask]
    selected_curve = curve[exact_mask]

    if len(selected_rounds) == 0:
        raise ValueError(
            f"No rounds matching eval_round_freq={requested_eval_round_freq} exist "
            f"in the logged range [{rounded_rounds.min()}, {rounded_rounds.max()}]."
        )

    # 파일이 커버하는 범위 안에서 존재해야 하는 requested round가 실제로 모두 있는지 검사한다.
    # 마지막 round가 eval 주기와 무관한 강제 final evaluation이어도 문제없도록
    # requested freq의 배수들만 expected로 만든다.
    first_expected = (
        (int(rounded_rounds.min()) + requested_eval_round_freq - 1)
        // requested_eval_round_freq
    ) * requested_eval_round_freq
    last_expected = (
        int(rounded_rounds.max()) // requested_eval_round_freq
    ) * requested_eval_round_freq

    if first_expected <= last_expected:
        expected_rounds = np.arange(
            first_expected,
            last_expected + 1,
            requested_eval_round_freq,
            dtype=np.int64,
        )
        missing_rounds = np.setdiff1d(expected_rounds, selected_rounds)

        if len(missing_rounds) > 0:
            preview = missing_rounds[:10].tolist()
            more = " ..." if len(missing_rounds) > 10 else ""
            raise ValueError(
                f"Missing exact evaluation rounds for requested eval_round_freq="
                f"{requested_eval_round_freq}: {preview}{more}. "
                "Interpolation is disabled, so the curve cannot be constructed."
            )

    return selected_x, selected_curve, selected_rounds, logged_eval_round_freq


def load_single_curve(npz_path, metric="nominal", eval_round_freq=None):
    """
    한 seed의 evaluations.npz에서 round별 metric mean curve를 가져온다.

    metric:
        - nominal
        - local_mean
        - local_min

    eval_round_freq:
        - None: 파일에 저장된 모든 evaluation point 사용
        - 정수: 해당 round 주기의 실제 저장값만 사용
          예: 80 -> rounds 80, 160, 240, ... 만 사용

    return:
        x: timesteps 기준 x축
        mean_curve: 선택된 metric curve
        rounds: 선택된 실제 round 번호
        logged_eval_round_freq: 해당 seed의 원래 evaluation 저장 주기
    """

    with np.load(npz_path, allow_pickle=True) as data:
        if metric == "nominal":
            mean_all = np.array(data["nominal_mean"], dtype=float)

            if mean_all.ndim == 2:
                mean_curve = np.mean(mean_all, axis=1)
            else:
                mean_curve = mean_all

        elif metric == "local_mean":
            mean_all = np.array(data["local_mean"], dtype=float)

            if mean_all.ndim == 2:
                mean_curve = np.mean(mean_all, axis=1)
            else:
                mean_curve = mean_all

        elif metric == "local_min":
            mean_all = np.array(data["local_mean"], dtype=float)

            if mean_all.ndim == 2:
                mean_curve = np.min(mean_all, axis=1)
            else:
                mean_curve = mean_all

        else:
            raise ValueError(f"Unknown metric: {metric}")

        num_rounds = len(mean_curve)

        x = make_timesteps_from_npz(
            data=data,
            num_rounds=num_rounds,
            metric_array=mean_all,
        )

        if "rounds" in data:
            rounds = np.asarray(data["rounds"], dtype=float)[:num_rounds]
        else:
            rounds = np.arange(num_rounds, dtype=float)

        if eval_round_freq is None:
            logged_eval_round_freq = (
                get_logged_eval_round_freq(data, rounds)
                if "rounds" in data
                else None
            )
            return x, mean_curve, rounds, logged_eval_round_freq

        return select_exact_eval_rounds(
            data=data,
            x=x,
            curve=mean_curve,
            requested_eval_round_freq=eval_round_freq,
        )


def align_seed_curves_by_round_exact(x_list, curve_list, rounds_list):
    """
    여러 seed를 실제 round 번호 기준으로 정확히 정렬한다.

    - 보간하지 않는다.
    - 모든 seed에 실제로 존재하는 공통 round만 사용한다.
    - 같은 round가 seed마다 다른 timestep에 대응하면 ValueError를 발생시킨다.
    """

    if not (len(x_list) == len(curve_list) == len(rounds_list)):
        raise ValueError("x_list, curve_list, and rounds_list must have the same length.")

    if len(rounds_list) == 0:
        raise ValueError("No seed curves to align.")

    normalized = []
    for x, curve, rounds in zip(x_list, curve_list, rounds_list):
        x = np.asarray(x, dtype=float).reshape(-1)
        curve = np.asarray(curve, dtype=float).reshape(-1)
        rounds = np.asarray(rounds, dtype=np.int64).reshape(-1)

        usable_len = min(len(x), len(curve), len(rounds))
        x = x[:usable_len]
        curve = curve[:usable_len]
        rounds = rounds[:usable_len]

        if len(np.unique(rounds)) != len(rounds):
            raise ValueError("Duplicate round values exist in a seed after filtering.")

        order = np.argsort(rounds, kind="stable")
        normalized.append((x[order], curve[order], rounds[order]))

    common_rounds = normalized[0][2]
    for _, _, rounds in normalized[1:]:
        common_rounds = np.intersect1d(common_rounds, rounds, assume_unique=True)

    if len(common_rounds) == 0:
        raise ValueError("No exact common evaluation rounds exist across seeds.")

    aligned_x = []
    aligned_curves = []

    for x, curve, rounds in normalized:
        index_by_round = {int(r): i for i, r in enumerate(rounds)}
        indices = np.array([index_by_round[int(r)] for r in common_rounds], dtype=int)
        aligned_x.append(x[indices])
        aligned_curves.append(curve[indices])

    reference_x = aligned_x[0]
    for seed_idx, seed_x in enumerate(aligned_x[1:], start=2):
        if not np.allclose(seed_x, reference_x, rtol=1e-9, atol=1e-9, equal_nan=False):
            raise ValueError(
                "The same evaluation rounds map to different timestep x-values across seeds. "
                f"Mismatch detected at seed index {seed_idx}. "
                "Check local_steps, num_clients, or saved timesteps."
            )

    return reference_x, np.vstack(aligned_curves), common_rounds


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

    eval_round_freq를 지정하면 모든 seed에서 그 주기의 실제 round만 사용한다.
    예: eval_round_freq=80 -> 80, 160, 240, ...

    seed의 원래 eval_round_freq가 요청값을 정확히 만들 수 없거나,
    필요한 round가 실제 파일에 없으면 ValueError를 발생시킨다.
    보간은 사용하지 않는다.

    expected path:
        {result_root_path}/{algo_id}/{env_id}_{seed}/evaluations.npz
    """

    eval_round_freq = normalize_eval_round_freq(eval_round_freq)

    x_list = []
    curve_list = []
    rounds_list = []
    valid_seeds = []
    logged_freqs = []

    for seed in range(1, num_trials + 1):
        npz_path = os.path.join(
            result_root_path,
            algo_id,
            f"{env_id}_{seed}",
            "evaluations.npz",
        )

        if not os.path.exists(npz_path):
            print(f"[Missing] seed {seed}: {npz_path}")
            continue

        try:
            x, mean_curve, rounds, logged_freq = load_single_curve(
                npz_path=npz_path,
                metric=metric,
                eval_round_freq=eval_round_freq,
            )
        except ValueError as e:
            # eval_round_freq 불일치/누락은 조용히 seed를 건너뛰면 안 된다.
            # 사용자가 잘못된 평균 plot을 보지 않도록 즉시 중단한다.
            raise ValueError(
                f"Failed to load seed {seed} strictly: {npz_path}\n{e}"
            ) from e
        except Exception as e:
            print(f"[Error] seed {seed}: {npz_path}")
            print(f"        {e}")
            continue

        x_list.append(x)
        curve_list.append(mean_curve)
        rounds_list.append(np.asarray(rounds, dtype=np.int64))
        valid_seeds.append(seed)
        logged_freqs.append(logged_freq)

    if len(curve_list) == 0:
        return None, None, []

    x, seed_curves, common_rounds = align_seed_curves_by_round_exact(
        x_list=x_list,
        curve_list=curve_list,
        rounds_list=rounds_list,
    )

    x = common_rounds.astype(float)

    freq_info = ", ".join(
        f"seed {seed}: {freq}" for seed, freq in zip(valid_seeds, logged_freqs)
    )
    requested_text = "all" if eval_round_freq is None else str(eval_round_freq)

    print(
        f"[Exact Eval Round] requested={requested_text}; "
        f"logged=({freq_info}); common points={len(common_rounds)}; "
        f"round range={common_rounds[0]}..{common_rounds[-1]}"
    )

    return x, seed_curves, valid_seeds


def normalize_extra_args(extra_args=None):
    """
    추가 하위 폴더 인자를 항상 tuple 형태로 정규화한다.

    예:
        None        -> ()
        []          -> ()
        1024        -> (1024,)
        [1024, 64]  -> (1024, 64)
        (1024, 64)  -> (1024, 64)
    """

    if extra_args is None:
        return ()

    if isinstance(extra_args, (list, tuple)):
        return tuple(extra_args)

    return (extra_args,)


def normalize_extra_arg_sets(extra_arg_sets=None):
    """
    여러 실험 설정을 항상 tuple의 list 형태로 정규화한다.

    예:
        None 또는 []
            -> [()]
            # 추가 폴더 없이 실행

        [1024, 2048]
            -> [(1024,), (2048,)]
            # 기존 first_arg_list처럼 1개 인자씩 여러 번 실행

        [(1024, 64), (2048, 128)]
            -> [(1024, 64), (2048, 128)]
            # first_arg, second_arg처럼 여러 인자를 묶어서 실행

        [(), (1024,), (1024, 64)]
            -> [(), (1024,), (1024, 64)]
            # 인자 없음/1개/여러 개를 같이 실행
    """

    if extra_arg_sets is None or len(extra_arg_sets) == 0:
        return [()]

    return [normalize_extra_args(extra_args) for extra_args in extra_arg_sets]


def append_extra_args_to_path(root_path, extra_args=None):
    """extra_args가 있으면 root_path 아래 subdir로 붙이고, 없으면 root_path 그대로 반환한다."""

    extra_args = normalize_extra_args(extra_args)

    if len(extra_args) == 0:
        return root_path

    return os.path.join(root_path, *[str(arg) for arg in extra_args])


def make_extra_args_suffix(extra_args=None):
    """파일명에 붙일 suffix를 만든다. extra_args가 없으면 빈 문자열을 반환한다."""

    extra_args = normalize_extra_args(extra_args)

    if len(extra_args) == 0:
        return ""

    return "_" + "_".join(str(arg) for arg in extra_args)


def count_algo_ids(algo_config_list):
    """algo_config_list 안에서 algo_id가 몇 번 등장하는지 센다."""

    algo_id_counts = {}

    for algo_id, _ in algo_config_list:
        algo_id_counts[algo_id] = algo_id_counts.get(algo_id, 0) + 1

    return algo_id_counts


def make_unique_plot_label(
    algo_id,
    result_root_path,
    algo_id_counts,
    used_plot_labels,
):
    """논문 그림용으로 알고리즘 legend label을 간결하게 만든다."""

    normalized_path = os.path.normpath(result_root_path).replace("\\", "/").lower()

    if algo_id == "ppo_avg":
        return "PPOAvg"

    if algo_id == "fed_ampo_ppo":
        if "/uniform" in normalized_path:
            return "AMPO-PPO(U)"
        if "/adaptive" in normalized_path:
            return "AMPO-PPO(A)"
        return "AMPO-PPO"

    if algo_id == "fed_svrpg_m":
        return "FedSVRPG-M-PPO"

    # 위 알고리즘 외에는 기존 algo_id를 그대로 사용한다.
    return algo_id


def normalize_window_size(window_size=None):
    """
    plot smoothing에 사용할 window 크기를 정규화한다.

    window_size가 None 또는 1 이하이면 smoothing을 적용하지 않는다.
    """

    if window_size is None:
        return 1

    window_size = int(window_size)

    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, but got {window_size}")

    return window_size


def make_window_suffix(window_size=None):
    """window smoothing을 적용한 경우 파일명에 붙일 suffix를 만든다."""

    window_size = normalize_window_size(window_size)

    if window_size <= 1:
        return ""

    return f"_window{window_size}"


def smooth_curve_with_window(curve, window_size=None):
    """
    1D curve에 centered moving-average window를 적용한다.

    특징:
    - window_size <= 1이면 원본 curve를 그대로 반환한다.
    - 출력 길이는 입력 길이와 동일하게 유지한다.
    - 양 끝 구간에서는 가능한 범위 안의 값만 사용한다.
    - NaN이 섞여 있으면 해당 window 안의 NaN을 제외하고 평균을 낸다.
    """

    curve = np.asarray(curve, dtype=float)
    window_size = normalize_window_size(window_size)

    if window_size <= 1 or len(curve) == 0:
        return curve.copy()

    window_size = min(window_size, len(curve))
    left = window_size // 2
    right = window_size - left - 1

    smoothed_curve = np.empty_like(curve, dtype=float)

    for idx in range(len(curve)):
        start = max(0, idx - left)
        end = min(len(curve), idx + right + 1)
        window_values = curve[start:end]

        if np.all(np.isnan(window_values)):
            smoothed_curve[idx] = np.nan
        else:
            smoothed_curve[idx] = np.nanmean(window_values)

    return smoothed_curve


def smooth_seed_curves_with_window(seed_curves, window_size=None):
    """
    seed_curves 전체에 window smoothing을 적용한다.

    seed_curves shape:
        (num_seeds, num_rounds)
    """

    window_size = normalize_window_size(window_size)
    seed_curves = np.asarray(seed_curves, dtype=float)

    if window_size <= 1:
        return seed_curves.copy()

    return np.array(
        [smooth_curve_with_window(curve, window_size) for curve in seed_curves],
        dtype=float,
    )


def plot_seed_average_curve(
    x,
    seed_curves,
    algo_id,
    env_id,
    metric,
    save_dir,
    filename_prefix,
    window_size=None,
):
    """
    시드 평균 curve와 시드 std band를 함께 그린다.

    주의:
    - 여기서 std는 nominal_std/local_std를 쓰는 것이 아니다.
    - 각 seed에서 얻은 'mean curve'를 기준으로
      seed 방향으로 std를 계산한다.
    """

    os.makedirs(save_dir, exist_ok=True)

    window_size = normalize_window_size(window_size)
    window_suffix = make_window_suffix(window_size)

    # window smoothing은 seed별 curve에 먼저 적용한다.
    # 이후 smoothed seed curves를 기준으로 seed 평균 / seed std를 계산한다.
    plot_seed_curves = smooth_seed_curves_with_window(seed_curves, window_size)
    avg_curve = np.nanmean(plot_seed_curves, axis=0)
    std_curve = np.nanstd(plot_seed_curves, axis=0)

    plt.figure(figsize=(10, 6))

    if window_size > 1:
        line_label = f"seed average (window={window_size})"
    else:
        line_label = "seed average"

    # seed average line
    plt.plot(
        x,
        avg_curve,
        linewidth=2.5,
        label=line_label,
    )

    # seed std band
    plt.fill_between(
        x,
        avg_curve - std_curve,
        avg_curve + std_curve,
        alpha=0.25,
        label="± seed std",
    )

    # plt.xlabel("Timesteps")
    plt.xlabel("Global Communication Rounds")
    plt.ylabel("Return")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    # Reduce outer whitespace more aggressively for paper figures.
    plt.tight_layout(pad=0.15, h_pad=0.15, w_pad=0.15)

    save_file = os.path.join(
        save_dir,
        f"{filename_prefix}_{metric}_learning_curve.png",
    )

    plt.savefig(
        save_file,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"[Saved] {save_file}")


def plot_multiple_algos(
    algo_data_list,
    env_id,
    metric,
    save_dir,
    filename_prefix,
    window_size=None,
):
    """
    여러 알고리즘의 seed average curve를 한 plot에 그린다.

    algo_data_list:
        [(plot_label, x, seed_curves), ...]

    주의:
    - dict를 쓰면 같은 algo_id가 여러 번 있을 때 key가 중복되어 덮어써진다.
    - 그래서 list를 사용해 같은 알고리즘의 여러 설정도 모두 plot한다.
    """

    os.makedirs(save_dir, exist_ok=True)

    window_size = normalize_window_size(window_size)
    window_suffix = make_window_suffix(window_size)

    plt.figure(figsize=(10, 6))

    for plot_label, x, seed_curves in algo_data_list:
        # window smoothing은 seed별 curve에 먼저 적용한다.
        # 이후 smoothed seed curves를 기준으로 seed 평균 / seed std를 계산한다.
        plot_seed_curves = smooth_seed_curves_with_window(seed_curves, window_size)
        avg_curve = np.nanmean(plot_seed_curves, axis=0)
        std_curve = np.nanstd(plot_seed_curves, axis=0)

        # seed average line
        plt.plot(
            x,
            avg_curve,
            linewidth=2.5,
            label=plot_label,
        )

        # seed std band
        plt.fill_between(
            x,
            avg_curve - std_curve,
            avg_curve + std_curve,
            alpha=0.15,
        )

    # plt.xlabel("Timesteps")
    plt.xlabel("Global Communication Rounds")
    plt.ylabel("Return")

    plt.grid(True, alpha=0.3)

    # 동일 알고리즘의 여러 하이퍼파라미터 곡선은 모두 유지하되,
    # legend에는 같은 이름을 한 번만 표시한다.
    handles, labels = plt.gca().get_legend_handles_labels()
    unique_legend = {}
    for handle, label in zip(handles, labels):
        if label not in unique_legend:
            unique_legend[label] = handle

    plt.legend(
        unique_legend.values(),
        unique_legend.keys(),
        frameon=False,
    )
    # Reduce outer whitespace more aggressively for paper figures.
    plt.tight_layout(pad=0.15, h_pad=0.15, w_pad=0.15)

    save_file = os.path.join(
        save_dir,
        f"{filename_prefix}_{metric}_learning_curve.png",
    )

    plt.savefig(
        save_file,
        dpi=200,
        bbox_inches="tight",
        pad_inches=0.02,
    )
    plt.close()

    print(f"[Saved] {save_file}")


def generate_learning_plots(
    algo_id,
    result_root_path,
    plot_root_path,
    env_list,
    num_trials,
    metric_list,
    extra_args=None,
    window_size=None,
    eval_round_freq=None,
):
    """
    env별, metric별 learning curve를 저장한다.

    result_root_path:
        evaluations.npz가 저장된 root

    plot_root_path:
        plot을 저장할 root

    extra_args:
        예: [1024]
        result_root_path 아래 추가 subdir로 붙고,
        파일 이름에도 붙는다.
    """

    extra_args = normalize_extra_args(extra_args)

    # 예:
    #   extra_args=[]          -> result_root_path
    #   extra_args=[1024]      -> result_root_path/1024
    #   extra_args=[1024, 64]  -> result_root_path/1024/64
    result_save_root = append_extra_args_to_path(result_root_path, extra_args)

    # 파일 이름 suffix
    suffix = make_extra_args_suffix(extra_args)
    filename_prefix = f"{algo_id}{suffix}"

    for env_id in env_list:
        save_dir = os.path.join(plot_root_path, env_id)

        for metric in metric_list:
            x, seed_curves, valid_seeds = collect_seed_curves(
                algo_id=algo_id,
                env_id=env_id,
                result_root_path=result_save_root,
                metric=metric,
                num_trials=num_trials,
                eval_round_freq=eval_round_freq,
            )

            if seed_curves is None:
                print(f"[Skip] No valid data: {algo_id}, {env_id}, {metric}")
                continue

            print(
                f"[Info] {algo_id}, {env_id}, {metric}: "
                f"{len(valid_seeds)} seeds loaded"
            )

            plot_seed_average_curve(
                x=x,
                seed_curves=seed_curves,
                algo_id=algo_id,
                env_id=env_id,
                metric=metric,
                save_dir=save_dir,
                filename_prefix=filename_prefix,
                window_size=window_size,
            )



def summarize_last_n_evals(seed_curves, last_n=10):
    """
    각 seed별로 마지막 last_n evaluation point의 평균을 계산한다.

    Parameters
    ----------
    seed_curves : np.ndarray
        shape = (num_seeds, num_eval_points)
    last_n : int
        마지막 몇 개 evaluation point를 평균낼지.

    Returns
    -------
    seed_final_scores : np.ndarray
        shape = (num_seeds,)
        각 seed의 마지막 last_n evaluation 평균.
    mean_score : float
        seed_final_scores의 평균.
    std_score : float
        seed_final_scores의 표준편차 (seed 방향).
    """
    seed_curves = np.asarray(seed_curves, dtype=float)

    if seed_curves.ndim != 2:
        raise ValueError(
            f"seed_curves must be 2D (num_seeds, num_eval_points), "
            f"but got shape={seed_curves.shape}"
        )

    if seed_curves.shape[1] == 0:
        raise ValueError("No evaluation points are available.")

    if last_n < 1:
        raise ValueError(f"last_n must be >= 1, but got {last_n}")

    n = min(last_n, seed_curves.shape[1])

    # 먼저 seed마다 마지막 n개 evaluation point를 평균낸다.
    seed_final_scores = np.nanmean(seed_curves[:, -n:], axis=1)

    # 그 scalar들을 seed 방향으로 평균 / 표준편차 계산.
    mean_score = float(np.nanmean(seed_final_scores))
    std_score = float(np.nanstd(seed_final_scores))

    return seed_final_scores, mean_score, std_score


def collect_dual_lr_sensitivity(
    env_id,
    perturbation_type,
    algo_config_list,
    eta_pi,
    num_trials,
    metric="local_min",
    eval_round_freq=40,
    last_n=10,
):
    """
    하나의 perturbation type에 대해 dual learning rate sensitivity 결과를 수집한다.

    algo_config_list 형식:
        [
            (eta_lambda, algo_id, result_root_path),
            ...
        ]

    기존 plotting 코드처럼 로그 경로를 main()에서 명시적으로 지정하고,
    여기서는 전달받은 경로를 그대로 사용한다.
    """
    ratios = []
    means = []
    stds = []
    details = []

    for eta_lambda, algo_id, result_root_path in algo_config_list:
        _, seed_curves, valid_seeds = collect_seed_curves(
            algo_id=algo_id,
            env_id=env_id,
            result_root_path=result_root_path,
            metric=metric,
            num_trials=num_trials,
            eval_round_freq=eval_round_freq,
        )

        if seed_curves is None:
            print(
                f"[Skip] No valid data: perturbation={perturbation_type}, "
                f"eta_lambda={eta_lambda}, path={result_root_path}"
            )
            continue

        # seed마다 마지막 last_n evaluation point를 먼저 평균
        seed_scores, mean_score, std_score = summarize_last_n_evals(
            seed_curves=seed_curves,
            last_n=last_n,
        )

        ratio = float(eta_lambda) / float(eta_pi)

        ratios.append(ratio)
        means.append(mean_score)
        stds.append(std_score)
        details.append(
            {
                "perturbation": perturbation_type,
                "eta_lambda": float(eta_lambda),
                "eta_pi": float(eta_pi),
                "ratio": ratio,
                "result_root_path": result_root_path,
                "valid_seeds": valid_seeds,
                "seed_last_n_means": seed_scores,
                "mean": mean_score,
                "std": std_score,
            }
        )

        print(
            f"[Dual LR] {perturbation_type:8s} | "
            f"eta_lambda={eta_lambda:g} | "
            f"eta_lambda/eta_pi={ratio:.6g} | "
            f"mean={mean_score:.4f} | std={std_score:.4f} | "
            f"seeds={valid_seeds}"
        )

    if len(ratios) == 0:
        return None

    order = np.argsort(np.asarray(ratios, dtype=float))

    return {
        "ratios": np.asarray(ratios, dtype=float)[order],
        "means": np.asarray(means, dtype=float)[order],
        "stds": np.asarray(stds, dtype=float)[order],
        "details": [details[i] for i in order],
    }

def collect_uniform_baseline(
    env_id,
    perturbation_type,
    algo_id,
    result_root_path,
    num_trials,
    metric="local_min",
    eval_round_freq=40,
    last_n=10,
):
    """
    lambda를 업데이트하지 않는 uniform AMPO-PPO baseline 성능을 계산한다.

    계산 방식은 adaptive sensitivity point와 동일하다.

    1. 각 seed의 evaluation curve를 불러온다.
    2. 각 seed에서 마지막 last_n evaluation point의 평균을 계산한다.
    3. seed별 scalar score들의 평균과 표준편차를 계산한다.

    return:
        {
            "perturbation": str,
            "mean": float,
            "std": float,
            "valid_seeds": list,
            "seed_last_n_means": np.ndarray,
            "result_root_path": str,
        }
    """

    _, seed_curves, valid_seeds = collect_seed_curves(
        algo_id=algo_id,
        env_id=env_id,
        result_root_path=result_root_path,
        metric=metric,
        num_trials=num_trials,
        eval_round_freq=eval_round_freq,
    )

    if seed_curves is None:
        print(
            f"[Skip Uniform] No valid data: "
            f"perturbation={perturbation_type}, "
            f"path={result_root_path}"
        )
        return None

    seed_scores, mean_score, std_score = summarize_last_n_evals(
        seed_curves=seed_curves,
        last_n=last_n,
    )

    print(
        f"[Uniform Baseline] {perturbation_type:8s} | "
        f"mean={mean_score:.4f} | std={std_score:.4f} | "
        f"seeds={valid_seeds}"
    )

    return {
        "perturbation": perturbation_type,
        "mean": mean_score,
        "std": std_score,
        "valid_seeds": valid_seeds,
        "seed_last_n_means": seed_scores,
        "result_root_path": result_root_path,
    }



def plot_dual_lr_ratio_friction_gravity(
    results,
    uniform_baselines,
    eta_pi,
    plot_root_path,
    env_id,
    metric="local_min",
    last_n=10,
    use_log_x=False,
):
    """
    Friction과 Gravity의 dual learning-rate sensitivity를 하나의 figure에 표시한다.

    Adaptive curves:
        x = eta_lambda / eta_pi
        각 x tick은 실제 ratio 크기와 관계없이 동일한 간격으로 표시한다.
        point = seed별 마지막 last_n evaluation 평균들의 seed 평균
        shaded region = seed 표준편차

    Uniform baselines:
        lambda를 업데이트하지 않는 uniform AMPO-PPO의 성능을
        perturbation별 dashed horizontal line으로 표시한다.

    Legend:
        - Friction
        - Gravity
        - AMPO-PPO(U) (Friction)
        - AMPO-PPO(U) (Gravity)
    """
    perturbation_styles = {
        "friction": {"label": "Friction", "marker": "o"},
        "gravity": {"label": "Gravity", "marker": "s"},
    }

    if len(results) == 0:
        raise RuntimeError("No valid friction/gravity data was found.")

    os.makedirs(plot_root_path, exist_ok=True)
    plt.figure(figsize=(7.2, 5.2))

    # -------------------------------------------------------
    # 모든 perturbation에서 사용된 ratio를 먼저 모은다.
    # -------------------------------------------------------
    all_ratios = []

    for perturbation_type in ["friction", "gravity"]:
        if perturbation_type in results:
            all_ratios.extend(results[perturbation_type]["ratios"].tolist())

    if len(all_ratios) == 0:
        raise RuntimeError("No eta_lambda / eta_pi ratios are available for plotting.")

    ratio_ticks = np.unique(np.asarray(all_ratios, dtype=float))
    ratio_ticks.sort()

    # 실제 ratio -> 동일 간격의 categorical position
    # 예:
    # 0.01   -> 0
    # 0.0333 -> 1
    # 0.1    -> 2
    # ...
    ratio_to_pos = {
        ratio: idx
        for idx, ratio in enumerate(ratio_ticks)
    }

    # adaptive curve에서 실제 사용된 색을 저장해
    # 같은 perturbation의 uniform baseline에 재사용한다.
    perturbation_colors = {}

    # -------------------------------------------------------
    # Adaptive lambda sensitivity curves
    # -------------------------------------------------------
    for perturbation_type in ["friction", "gravity"]:
        if perturbation_type not in results:
            continue

        result = results[perturbation_type]
        style = perturbation_styles[perturbation_type]

        ratios = result["ratios"]
        y = result["means"]
        y_std = result["stds"]

        # 실제 ratio 대신 동일 간격 index 사용
        x_pos = np.array(
            [ratio_to_pos[r] for r in ratios],
            dtype=float,
        )

        line, = plt.plot(
            x_pos,
            y,
            marker=style["marker"],
            markersize=8,
            linewidth=2.5,
            label=style["label"],
        )

        perturbation_colors[perturbation_type] = line.get_color()

        plt.fill_between(
            x_pos,
            y - y_std,
            y + y_std,
            alpha=0.18,
            color=line.get_color(),
        )

    # -------------------------------------------------------
    # Uniform AMPO-PPO baselines
    # -------------------------------------------------------
    for perturbation_type in ["friction", "gravity"]:
        if perturbation_type not in uniform_baselines:
            continue

        baseline = uniform_baselines[perturbation_type]
        style = perturbation_styles[perturbation_type]
        baseline_mean = baseline["mean"]

        plt.axhline(
            y=baseline_mean,
            color=perturbation_colors.get(perturbation_type, None),
            linestyle="--",
            linewidth=2.0,
            label=f'AMPO-PPO(U) ({style["label"]})',
        )

    # -------------------------------------------------------
    # x축은 동일한 간격, tick label만 실제 ratio
    # -------------------------------------------------------
    tick_positions = np.arange(len(ratio_ticks))

    plt.xticks(
        tick_positions,
        [f"{x:.3g}" for x in ratio_ticks],
    )

    # categorical x-position을 사용하므로 log scale은 적용하지 않는다.
    if use_log_x:
        print(
            "[Warning] use_log_x=True was requested, but the x-axis uses "
            "equal-spaced categorical positions. Log scaling is ignored."
        )

    plt.xlabel(r"$\eta_\lambda / \eta_\pi$")
    plt.ylabel(
        "Worst-case Local Return"
        if metric == "local_min"
        else "Return"
    )

    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False, loc="upper left")
    plt.tight_layout(pad=0.2)

    save_file = os.path.join(
        plot_root_path,
        f"{env_id}_dual_lr_ratio_{metric}_last{last_n}.png",
    )

    plt.savefig(
        save_file,
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.02,
    )

    plt.close()

    print(f"[Saved] {save_file}")


def main():
    env_id = "PerturbHopper-v4"
    metric = "local_min"
    num_trials = 5

    # policy learning rate는 모든 실험에서 고정
    eta_pi = 0.003

    # 각 point / baseline은 seed별 마지막 10 evaluation point 평균을 사용
    last_n = 10

    # 기존 코드와 동일하게 exact evaluation round만 사용
    plot_eval_round_freq = 40

    # friction / gravity 모두 계산
    perturbation_types = ["friction", "gravity"]

    # adaptive lambda sensitivity 결과
    results = {}

    # lambda를 업데이트하지 않는 uniform AMPO-PPO baseline 결과
    uniform_baselines = {}

    for perturbation_type in perturbation_types:
        # ---------------------------------------------------
        # Adaptive AMPO-PPO: dual learning-rate sensitivity
        # ---------------------------------------------------
        algo_config_list = [
            (
                0.00003,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.00003",
            ),
            (
                0.0001,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/lambda_cap/0.0001",
            ),
            (
                0.0003,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/lambda_cap/0.0003",
            ),
            (
                0.0005,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.0005",
            ),
            (
                0.0007,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.0007",
            ),
            (
                0.001,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.001",
            ),
            (
                0.003,
                "fed_ampo_ppo",
                f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.003",
            ),
            # (
            #     0.01,
            #     "fed_ampo_ppo",
            #     f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.01",
            # ),
            # (
            #     0.03,
            #     "fed_ampo_ppo",
            #     f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.03",
            # ),
            # (
            #     0.1,
            #     "fed_ampo_ppo",
            #     f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/0.1",
            # ),
        ]

        result = collect_dual_lr_sensitivity(
            env_id=env_id,
            perturbation_type=perturbation_type,
            algo_config_list=algo_config_list,
            eta_pi=eta_pi,
            num_trials=num_trials,
            metric=metric,
            eval_round_freq=plot_eval_round_freq,
            last_n=last_n,
        )

        if result is not None:
            results[perturbation_type] = result

        # ---------------------------------------------------
        # Uniform AMPO-PPO baseline
        # lambda를 업데이트하지 않는 알고리즘
        # ---------------------------------------------------
        uniform_algo_config = (
            "fed_ampo_ppo",
            f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/uniform",
        )

        uniform_algo_id, uniform_result_root_path = uniform_algo_config

        uniform_result = collect_uniform_baseline(
            env_id=env_id,
            perturbation_type=perturbation_type,
            algo_id=uniform_algo_id,
            result_root_path=uniform_result_root_path,
            num_trials=num_trials,
            metric=metric,
            eval_round_freq=plot_eval_round_freq,
            last_n=last_n,
        )

        if uniform_result is not None:
            uniform_baselines[perturbation_type] = uniform_result

    # -------------------------------------------------------
    # Friction / Gravity adaptive curves + uniform baselines
    # -------------------------------------------------------
    plot_root_path = f"plots/iclr2027_ampo/dual_lr_sensitivity/{env_id}"

    plot_dual_lr_ratio_friction_gravity(
        results=results,
        uniform_baselines=uniform_baselines,
        eta_pi=eta_pi,
        plot_root_path=plot_root_path,
        env_id=env_id,
        metric=metric,
        last_n=last_n,
        use_log_x=False,
    )


if __name__ == "__main__":
    main()
