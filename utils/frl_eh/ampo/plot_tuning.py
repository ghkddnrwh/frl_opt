import os
import numpy as np
import matplotlib.pyplot as plt


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
    """
    plot legend에 사용할 고유 label을 만든다.

    같은 algo_id가 여러 번 등장하면 result_root_path의 마지막 폴더명을 붙인다.
    예:
        fedsp_pg_ppo_paper_aligned/0.0003
        fedsp_pg_ppo_paper_aligned/0.001
        fedsp_pg_ppo_paper_aligned/0.003

    그래도 label이 중복되면 path suffix를 점점 길게 붙인다.
    """

    normalized_path = os.path.normpath(result_root_path)
    path_parts = normalized_path.split(os.sep)

    if algo_id_counts.get(algo_id, 0) > 1:
        suffix_len = 1
        path_suffix = "/".join(path_parts[-suffix_len:])
        base_label = f"{algo_id}/{path_suffix}"
    else:
        suffix_len = 0
        base_label = algo_id

    plot_label = base_label

    while plot_label in used_plot_labels:
        suffix_len += 1

        if suffix_len <= len(path_parts):
            path_suffix = "/".join(path_parts[-suffix_len:])
            plot_label = f"{algo_id}/{path_suffix}"
        else:
            plot_label = f"{base_label} #{len(used_plot_labels) + 1}"
            break

    return plot_label


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

    plt.figure(figsize=(8, 5))

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

    plt.xlabel("Timesteps")
    plt.ylabel("Return")
    plt.title(f"{algo_id} - {env_id} - {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_file = os.path.join(
        save_dir,
        f"{filename_prefix}_{metric}_learning_curve.png",
    )

    plt.savefig(save_file, dpi=300)
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

    plt.figure(figsize=(12, 6))

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

    plt.xlabel("Timesteps")
    plt.ylabel("Return")

    if window_size > 1:
        plt.title(f"Algorithm Comparison - {env_id} - {metric} - window={window_size}")
    else:
        plt.title(f"Algorithm Comparison - {env_id} - {metric}")

    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_file = os.path.join(
        save_dir,
        f"{filename_prefix}_{metric}_learning_curve.png",
    )

    plt.savefig(save_file, dpi=300)
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


def main():
    # env_id_list = ["PerturbPendulum-v1"]
    env_id = "PerturbHalfCheetah-v4"
    metric_list = ["nominal", "local_mean", "local_min"]
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
    perturbation_types = ["friction", "gravity"]

    for perturbation_type in perturbation_types:
        # 여러 알고리즘의 경로를 리스트로 정의
        algo_config_list = [
            # ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/ppo_avg"),

            # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform"),
            # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform/0.0003"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform/0.001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform/0.003"),
            
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.001"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.0003"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.0001"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.00001"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.000001"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.0000001"),
            
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0001"),
            # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0002"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0003"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0005"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.001"),
            
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.00001"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.000001"),
            # # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0000001"),


            # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.7/0.0001"),
            # # # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.7/0.0003"),

            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_svrpg_m/0.85/0.5"),
            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_svrpg_m/0.9/0.5"),
            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_svrpg_m/0.95/0.5"),

            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/uniform/0.5"),
            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/uniform/0.7"),
            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/uniform/1.0"),

            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/0.0001"),
            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/0.0002"),
            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/0.0003"),

            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/momentum/0.0002/0.85"),
            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/momentum/0.0002/0.9"),
            # # # ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/momentum/0.0002/0.95"),

            # ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/ppo_avg"),
            ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/ppo_avg/normalize"),

            ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/uniform"),
            
            ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/lambda_cap/0.0001"),
            ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_ampo_ppo/adaptive/undiscounted/lambda_cap/0.0003"),

            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/1.0/0.5"),
            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/1.0/0.7"),
            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/1.0/1.0"),

            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/fedavg/1.0/0.5"),
            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/fedavg/1.0/0.7"),
            # ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/fedavg/1.0/1.0"),

            ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/fedavg/0.85/0.5"),
            ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/fedavg/0.9/0.5"),
            ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/0.3/fed_svrpg_m/fedavg/0.95/0.5"),
        ]

        # plot 저장 root
        plot_root_path = f"plots/frl_eh/tuning_mujoco/fixed/noise_assignment/{env_id}/{perturbation_type}"

        # 추가 하위 폴더 인자 묶음
        #
        # 사용 예시:
        #   extra_arg_sets = []
        #       -> 추가 폴더 없이 실행
        #
        #   extra_arg_sets = [1024, 2048]
        #       -> result_root_path/1024, result_root_path/2048 각각 실행
        #
        #   extra_arg_sets = [(1024, 64), (2048, 128)]
        #       -> result_root_path/1024/64, result_root_path/2048/128 각각 실행
        #
        #   extra_arg_sets = [(), (1024,), (1024, 64)]
        #       -> 인자 없음/1개/2개 설정을 모두 실행
        extra_arg_sets = []

        for extra_args in normalize_extra_arg_sets(extra_arg_sets):
            # 파일 이름 suffix
            suffix = make_extra_args_suffix(extra_args)
            filename_prefix = f"comparison{suffix}"

            # 각 환경과 metric에 대해 모든 알고리즘을 비교 플롯
            for metric in metric_list:
                algo_data_list = []
                algo_id_counts = count_algo_ids(algo_config_list)
                used_plot_labels = set()

                # 각 알고리즘의 데이터를 수집
                for algo_id, result_root_path in algo_config_list:
                    result_root_with_args = append_extra_args_to_path(
                        result_root_path,
                        extra_args,
                    )

                    plot_label = make_unique_plot_label(
                        algo_id=algo_id,
                        result_root_path=result_root_with_args,
                        algo_id_counts=algo_id_counts,
                        used_plot_labels=used_plot_labels,
                    )
                    used_plot_labels.add(plot_label)

                    x, seed_curves, valid_seeds = collect_seed_curves(
                        algo_id=algo_id,
                        env_id=env_id,
                        result_root_path=result_root_with_args,
                        metric=metric,
                        num_trials=num_trials,
                        eval_round_freq=plot_eval_round_freq,
                    )

                    if seed_curves is None:
                        print(f"[Skip] No valid data: {plot_label}, {env_id}, {metric}")
                        continue

                    print(
                        f"[Info] {plot_label}, {env_id}, {metric}: "
                        f"{len(valid_seeds)} seeds loaded"
                    )
                    algo_data_list.append((plot_label, x, seed_curves))

                # 수집한 모든 알고리즘을 하나의 플롯에 표시
                if len(algo_data_list) > 0:
                    save_dir = os.path.join(plot_root_path, env_id)
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




# ("ppo_avg", f"logs/tuned/{env_id}/3/{perturbation_type}/128"),
# ("ppo_avg", f"logs/tuned/{env_id}/5/{perturbation_type}/64"),
# ("ppo_avg", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/ppo_avg/64/8"),
# ("fedsp_pg_ppo", f"logs/tuned/{env_id}/20/{perturbation_type}/64"),
# ("fedsp_pg_ppo_paper_aligned", f"logs/tuned/{env_id}/20/{perturbation_type}/64/0.0003"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.0001"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.0003"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.001"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.003"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.01"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.03"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/64/0.00001"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/64/0.0001"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/64/0.001"),
# ("fed_ampo_ppo", f"logs/tuning_classic_control/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/64/0.1"),



# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/0.2/0.5"),
# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/0.5/0.5"),
# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/0.8/0.5"),
# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/0.85/0.5"),
# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/0.9/0.5"),
# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/0.95/0.5"),
# ("fed_svrpg_m", f"logs/tuning_mujoco_long/revised4/{env_id}/{perturbation_type}/fed_svrpg_m/1.0/0.5"),
