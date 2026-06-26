import os
import numpy as np
import matplotlib.pyplot as plt


VALID_X_AXES = {"rounds", "total_timesteps"}


def normalize_x_axis(x_axis="total_timesteps"):
    """plot x축 종류를 정규화한다."""

    if x_axis is None:
        return "total_timesteps"

    x_axis = str(x_axis)

    if x_axis not in VALID_X_AXES:
        raise ValueError(f"x_axis must be one of {sorted(VALID_X_AXES)}, but got {x_axis}")

    return x_axis


def get_x_axis_label(x_axis="total_timesteps"):
    """plot에 표시할 x축 label을 반환한다."""

    x_axis = normalize_x_axis(x_axis)

    if x_axis == "total_timesteps":
        return "Total Timesteps"

    return "Global Round"


class EvalDataStore:
    """
    한 seed directory 안의 evaluation 결과를 읽는 wrapper.

    지원하는 저장 방식:
        1. seed_dir/evaluations.npz 안에 arrays가 들어있는 경우
        2. seed_dir/rounds.npy, seed_dir/local_steps.npy 처럼 개별 npy 파일로 저장된 경우

    같은 key가 npz와 npy 양쪽에 모두 있으면 npz 값을 먼저 사용한다.
    """

    def __init__(self, seed_dir):
        self.seed_dir = seed_dir
        self.npz_path = os.path.join(seed_dir, "evaluations.npz")
        self.npz_data = None

        if os.path.exists(self.npz_path):
            self.npz_data = np.load(self.npz_path, allow_pickle=True)

    def has_key(self, key):
        if self.npz_data is not None and key in self.npz_data:
            return True

        npy_path = os.path.join(self.seed_dir, f"{key}.npy")
        return os.path.exists(npy_path)

    def get(self, key):
        if self.npz_data is not None and key in self.npz_data:
            return self.npz_data[key]

        npy_path = os.path.join(self.seed_dir, f"{key}.npy")
        if os.path.exists(npy_path):
            return np.load(npy_path, allow_pickle=True)

        raise KeyError(key)

    def get_first(self, key_candidates):
        for key in key_candidates:
            if self.has_key(key):
                return self.get(key), key

        return None, None

    def has_any_data(self):
        if self.npz_data is not None:
            return True

        if not os.path.isdir(self.seed_dir):
            return False

        return any(filename.endswith(".npy") for filename in os.listdir(self.seed_dir))


def as_scalar(value):
    """numpy scalar 또는 1개짜리 array를 python scalar로 변환한다."""

    if value is None:
        return None

    value = np.asarray(value)

    if value.shape == ():
        return value.item()

    if value.size == 1:
        return value.reshape(-1)[0].item()

    return value


def as_float_scalar(value):
    """metadata scalar를 float으로 변환한다. 변환할 수 없으면 None을 반환한다."""

    value = as_scalar(value)

    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int_scalar(value):
    """metadata scalar를 int로 변환한다. 변환할 수 없으면 None을 반환한다."""

    value = as_float_scalar(value)

    if value is None:
        return None

    return int(value)


def truncate_array(array, num_points):
    """array를 1D float array로 바꾸고 필요한 길이만큼 자른다."""

    array = np.asarray(array, dtype=float).reshape(-1)
    return array[:num_points]


def infer_num_clients_from_metric_array(metric_array):
    """
    metric array에서 client 수를 추정한다.

    nominal_mean/local_mean이 shape (num_rounds, num_clients)이면 num_clients를 추정할 수 있다.
    shape (num_rounds,)이면 추정할 수 없으므로 None을 반환한다.
    """

    metric_array = np.asarray(metric_array)

    if metric_array.ndim == 2:
        return int(metric_array.shape[1])

    return None


def resolve_num_clients(store, metric_array, explicit_num_clients=None):
    """
    num_clients를 자동으로 결정한다.

    우선순위:
        1. 함수 인자로 명시한 explicit_num_clients
        2. seed_dir/num_clients.npy 또는 evaluations.npz의 num_clients
        3. metric array shape에서 추정한 client 수
        4. fallback 1
    """

    if explicit_num_clients is not None:
        return int(explicit_num_clients)

    loaded_num_clients, used_key = store.get_first(["num_clients", "n_clients"])
    if loaded_num_clients is not None:
        loaded_num_clients = as_int_scalar(loaded_num_clients)
        if loaded_num_clients is not None and loaded_num_clients >= 1:
            return loaded_num_clients

    inferred_num_clients = infer_num_clients_from_metric_array(metric_array)
    if inferred_num_clients is not None:
        return int(inferred_num_clients)

    return 1


def load_metric_curve(store, metric="nominal"):
    """
    metric별 mean curve를 읽는다.

    metric:
        - nominal
        - local_mean
        - local_min

    return:
        mean_curve: shape (num_rounds,)
        raw_metric_array: num_clients 추정에 사용할 원본 metric array
        used_metric_key: 실제로 사용한 key 설명
    """

    if metric == "nominal":
        mean_all, used_key = store.get_first(
            ["nominal_mean", "nominal_mean_across_clients"]
        )
        if mean_all is None:
            raise KeyError("nominal_mean or nominal_mean_across_clients")

        mean_all = np.asarray(mean_all, dtype=float)
        if mean_all.ndim == 2:
            mean_curve = np.mean(mean_all, axis=1)
        else:
            mean_curve = mean_all.reshape(-1)

        return mean_curve, mean_all, used_key

    if metric == "local_mean":
        mean_all, used_key = store.get_first(
            ["local_mean", "local_mean_across_clients"]
        )
        if mean_all is None:
            raise KeyError("local_mean or local_mean_across_clients")

        mean_all = np.asarray(mean_all, dtype=float)
        if mean_all.ndim == 2:
            mean_curve = np.mean(mean_all, axis=1)
        else:
            mean_curve = mean_all.reshape(-1)

        return mean_curve, mean_all, used_key

    if metric == "local_min":
        local_min_curve, used_key = store.get_first(["local_min_across_clients"])
        if local_min_curve is not None:
            local_min_curve = np.asarray(local_min_curve, dtype=float).reshape(-1)
            return local_min_curve, local_min_curve, used_key

        mean_all, used_key = store.get_first(["local_mean", "local_mean_across_clients"])
        if mean_all is None:
            raise KeyError("local_min_across_clients or local_mean")

        mean_all = np.asarray(mean_all, dtype=float)
        if mean_all.ndim == 2:
            mean_curve = np.min(mean_all, axis=1)
        else:
            mean_curve = mean_all.reshape(-1)

        return mean_curve, mean_all, f"min({used_key})"

    raise ValueError(f"Unknown metric: {metric}")


def make_rounds_fallback(store, num_rounds):
    """
    rounds.npy가 없을 때 eval_round_freq.npy를 이용해 round index를 복원한다.

    eval_round_freq=100, num_rounds=15이면
        [100, 200, ..., 1500]
    을 만든다.
    """

    eval_round_freq, used_key = store.get_first(["eval_round_freq"])
    eval_round_freq = as_float_scalar(eval_round_freq)

    if eval_round_freq is not None and eval_round_freq > 0:
        rounds = np.arange(1, num_rounds + 1, dtype=float) * eval_round_freq
        return rounds, f"arange * {used_key}({eval_round_freq:g})"

    rounds = np.arange(1, num_rounds + 1, dtype=float)
    return rounds, "arange_1_to_num_rounds"


def load_local_steps(store, explicit_local_steps=None):
    """
    local_steps를 자동으로 읽는다.

    우선순위:
        1. 함수 인자로 명시한 explicit_local_steps
        2. local_steps.npy 또는 evaluations.npz의 local_steps
        3. local_steps_per_round / train_freq / gradient_steps 같은 후보 key
    """

    if explicit_local_steps is not None:
        return float(explicit_local_steps), "explicit_local_steps"

    local_steps, used_key = store.get_first(
        [
            "local_steps",
            "local_steps_per_round",
            "num_local_steps",
            "local_timesteps",
            "train_freq",
            "gradient_steps",
        ]
    )
    local_steps = as_float_scalar(local_steps)

    if local_steps is None or local_steps <= 0:
        return None, None

    return local_steps, used_key


def make_total_timesteps_x(
    store,
    num_rounds,
    metric_array,
    explicit_local_steps=None,
    explicit_num_clients=None,
    count_all_clients_as_timesteps=True,
    prefer_saved_timesteps=True,
):
    """
    x축을 total timesteps로 만든다.

    우선순위:
        1. total_timesteps.npy / timesteps.npy / num_timesteps.npy가 있으면 사용
        2. rounds.npy와 local_steps.npy를 읽어서 자동 계산
        3. rounds.npy가 없으면 eval_round_freq.npy로 round를 복원한 뒤 자동 계산
        4. local_steps도 없으면 rounds만 사용하고 warning 성격의 used_x_key를 반환

    기본 계산식:
        total_timesteps = rounds * local_steps * num_clients

    count_all_clients_as_timesteps=False이면:
        total_timesteps = rounds * local_steps
    """

    if prefer_saved_timesteps:
        saved_x, used_key = store.get_first(
            ["total_timesteps", "timesteps", "num_timesteps"]
        )
        if saved_x is not None:
            saved_x = truncate_array(saved_x, num_rounds)
            if len(saved_x) == num_rounds:
                return saved_x, used_key

    rounds, used_round_key = store.get_first(["rounds", "global_rounds"])
    if rounds is not None:
        rounds = truncate_array(rounds, num_rounds)
    else:
        rounds, used_round_key = make_rounds_fallback(store, num_rounds)

    local_steps, used_local_steps_key = load_local_steps(
        store,
        explicit_local_steps=explicit_local_steps,
    )

    if local_steps is None:
        return rounds, f"{used_round_key} (not converted: local_steps not found)"

    multiplier = float(local_steps)

    if count_all_clients_as_timesteps:
        num_clients = resolve_num_clients(
            store=store,
            metric_array=metric_array,
            explicit_num_clients=explicit_num_clients,
        )
        multiplier *= float(num_clients)
        used_x_key = (
            f"{used_round_key} * {used_local_steps_key}({local_steps:g}) "
            f"* num_clients({num_clients})"
        )
    else:
        used_x_key = f"{used_round_key} * {used_local_steps_key}({local_steps:g})"

    x = rounds * multiplier
    return x, used_x_key


def load_single_curve(
    seed_dir,
    metric="nominal",
    x_axis="total_timesteps",
    local_steps_per_round=None,
    num_clients=None,
    count_all_clients_as_timesteps=True,
    prefer_saved_timesteps=True,
):
    """
    한 seed directory에서 round별 metric mean curve와 x축을 가져온다.

    seed_dir 안에 다음 둘 중 하나가 있으면 된다.
        - evaluations.npz
        - rounds.npy, local_mean.npy, local_steps.npy 등 개별 npy 파일
    """

    x_axis = normalize_x_axis(x_axis)
    store = EvalDataStore(seed_dir)

    if not store.has_any_data():
        raise FileNotFoundError(f"No evaluations.npz or npy files found in {seed_dir}")

    mean_curve, raw_metric_array, used_metric_key = load_metric_curve(
        store=store,
        metric=metric,
    )
    mean_curve = np.asarray(mean_curve, dtype=float).reshape(-1)
    num_rounds = len(mean_curve)

    if x_axis == "rounds":
        x, used_x_key = store.get_first(["rounds", "global_rounds"])
        if x is not None:
            x = truncate_array(x, num_rounds)
        else:
            x, used_x_key = make_rounds_fallback(store, num_rounds)
    else:
        x, used_x_key = make_total_timesteps_x(
            store=store,
            num_rounds=num_rounds,
            metric_array=raw_metric_array,
            explicit_local_steps=local_steps_per_round,
            explicit_num_clients=num_clients,
            count_all_clients_as_timesteps=count_all_clients_as_timesteps,
            prefer_saved_timesteps=prefer_saved_timesteps,
        )

    x = np.asarray(x, dtype=float).reshape(-1)[:num_rounds]
    mean_curve = mean_curve[: len(x)]

    return x, mean_curve, used_x_key, used_metric_key


def collect_seed_curves(
    algo_id,
    env_id,
    result_root_path,
    metric,
    num_trials,
    x_axis="total_timesteps",
    local_steps_per_round=None,
    num_clients=None,
    count_all_clients_as_timesteps=True,
    prefer_saved_timesteps=True,
):
    """
    여러 seed의 mean curve를 모은다.

    expected path:
        {result_root_path}/{algo_id}/{env_id}_{seed}/

    이 directory 안에서 evaluations.npz 또는 개별 npy 파일들을 자동으로 읽는다.
    """

    x_list = []
    curve_list = []
    valid_seeds = []
    used_x_key_list = []
    used_metric_key_list = []

    for seed in range(1, num_trials + 1):
        seed_dir = os.path.join(
            result_root_path,
            algo_id,
            f"{env_id}_{seed}",
        )

        if not os.path.isdir(seed_dir):
            print(f"[Missing] seed {seed}: {seed_dir}")
            continue

        try:
            x, mean_curve, used_x_key, used_metric_key = load_single_curve(
                seed_dir=seed_dir,
                metric=metric,
                x_axis=x_axis,
                local_steps_per_round=local_steps_per_round,
                num_clients=num_clients,
                count_all_clients_as_timesteps=count_all_clients_as_timesteps,
                prefer_saved_timesteps=prefer_saved_timesteps,
            )
            x_list.append(x)
            curve_list.append(mean_curve)
            valid_seeds.append(seed)
            used_x_key_list.append(used_x_key)
            used_metric_key_list.append(used_metric_key)
        except Exception as e:
            print(f"[Error] seed {seed}: {seed_dir}")
            print(f"        {e}")

    if len(curve_list) == 0:
        return None, None, [], None, None

    # seed마다 길이가 다를 수 있으므로 가장 짧은 길이에 맞춤
    min_len = min(len(curve) for curve in curve_list)
    min_len = min(min_len, min(len(x) for x in x_list))

    x = x_list[0][:min_len]
    seed_curves = np.array([curve[:min_len] for curve in curve_list], dtype=float)

    # 같은 알고리즘/설정의 seed들은 x축이 같아야 한다.
    # 다르면 일단 첫 번째 seed의 x축을 사용하되 warning을 출력한다.
    for seed, other_x in zip(valid_seeds[1:], x_list[1:]):
        other_x = other_x[:min_len]
        if len(other_x) != len(x) or not np.allclose(x, other_x, equal_nan=True):
            print(
                f"[Warning] x-axis mismatch inside {algo_id}, {env_id}, {metric}. "
                f"Using seed {valid_seeds[0]}'s x-axis; mismatched seed={seed}."
            )
            break

    used_x_key_summary = summarize_used_keys(used_x_key_list)
    used_metric_key_summary = summarize_used_keys(used_metric_key_list)

    return x, seed_curves, valid_seeds, used_x_key_summary, used_metric_key_summary


def summarize_used_keys(key_list):
    """여러 seed에서 사용한 key 설명을 중복 제거해 요약한다."""

    unique_keys = []
    for key in key_list:
        if key not in unique_keys:
            unique_keys.append(key)

    if len(unique_keys) == 1:
        return unique_keys[0]

    return unique_keys


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


def unpack_algo_config(algo_config):
    """
    algo_config를 표준 형태로 풀어낸다.

    지원 형태:
        (algo_id, result_root_path)
        (algo_id, result_root_path, config_dict)

    일반적으로 이제는 config_dict가 필요 없다.
    local_steps.npy, num_clients.npy를 seed directory에서 자동으로 읽는다.

    그래도 강제로 override하고 싶으면 config_dict를 쓸 수 있다.
        {
            "local_steps_per_round": 200,
            "num_clients": 5,
            "count_all_clients_as_timesteps": True,
        }
    """

    if len(algo_config) == 2:
        algo_id, result_root_path = algo_config
        config = {}
    elif len(algo_config) == 3:
        algo_id, result_root_path, config = algo_config
        if not isinstance(config, dict):
            config = {"local_steps_per_round": config}
    else:
        raise ValueError(
            "algo_config must be (algo_id, result_root_path) or "
            "(algo_id, result_root_path, config_dict)"
        )

    return algo_id, result_root_path, config


def count_algo_ids(algo_config_list):
    """algo_config_list 안에서 algo_id가 몇 번 등장하는지 센다."""

    algo_id_counts = {}

    for algo_config in algo_config_list:
        algo_id = algo_config[0]
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
    """plot smoothing에 사용할 window 크기를 정규화한다."""

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
    x_axis="total_timesteps",
):
    """
    시드 평균 curve와 시드 std band를 함께 그린다.

    주의:
    - 여기서 std는 nominal_std/local_std를 쓰는 것이 아니다.
    - 각 seed에서 얻은 'mean curve'를 기준으로 seed 방향으로 std를 계산한다.
    """

    os.makedirs(save_dir, exist_ok=True)

    window_size = normalize_window_size(window_size)
    window_suffix = make_window_suffix(window_size)

    plot_seed_curves = smooth_seed_curves_with_window(seed_curves, window_size)
    avg_curve = np.nanmean(plot_seed_curves, axis=0)
    std_curve = np.nanstd(plot_seed_curves, axis=0)

    plt.figure(figsize=(8, 5))

    if window_size > 1:
        line_label = f"seed average (window={window_size})"
    else:
        line_label = "seed average"

    plt.plot(
        x,
        avg_curve,
        linewidth=2.5,
        label=line_label,
    )

    plt.fill_between(
        x,
        avg_curve - std_curve,
        avg_curve + std_curve,
        alpha=0.25,
        label="± seed std",
    )

    plt.xlabel(get_x_axis_label(x_axis))
    plt.ylabel("Return")
    plt.title(f"{algo_id} - {env_id} - {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_file = os.path.join(
        save_dir,
        f"{filename_prefix}_{metric}{window_suffix}_learning_curve.png",
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
    x_axis="total_timesteps",
):
    """
    여러 알고리즘의 seed average curve를 한 plot에 그린다.

    algo_data_list:
        [(plot_label, x, seed_curves), ...]
    """

    os.makedirs(save_dir, exist_ok=True)

    window_size = normalize_window_size(window_size)
    window_suffix = make_window_suffix(window_size)

    plt.figure(figsize=(12, 6))

    for plot_label, x, seed_curves in algo_data_list:
        plot_seed_curves = smooth_seed_curves_with_window(seed_curves, window_size)
        avg_curve = np.nanmean(plot_seed_curves, axis=0)
        std_curve = np.nanstd(plot_seed_curves, axis=0)

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

    plt.xlabel(get_x_axis_label(x_axis))
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
        f"{filename_prefix}_comparison_{metric}{window_suffix}_learning_curve.png",
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
    x_axis="total_timesteps",
    local_steps_per_round=None,
    num_clients=None,
    count_all_clients_as_timesteps=True,
    prefer_saved_timesteps=True,
):
    """
    env별, metric별 learning curve를 저장한다.

    local_steps_per_round와 num_clients를 None으로 두면
    seed directory 안의 local_steps.npy, num_clients.npy를 자동으로 읽는다.
    """

    extra_args = normalize_extra_args(extra_args)
    result_save_root = append_extra_args_to_path(result_root_path, extra_args)
    suffix = make_extra_args_suffix(extra_args)
    filename_prefix = f"{algo_id}{suffix}"

    for env_id in env_list:
        save_dir = os.path.join(plot_root_path, env_id)

        for metric in metric_list:
            x, seed_curves, valid_seeds, used_x_key, used_metric_key = collect_seed_curves(
                algo_id=algo_id,
                env_id=env_id,
                result_root_path=result_save_root,
                metric=metric,
                num_trials=num_trials,
                x_axis=x_axis,
                local_steps_per_round=local_steps_per_round,
                num_clients=num_clients,
                count_all_clients_as_timesteps=count_all_clients_as_timesteps,
                prefer_saved_timesteps=prefer_saved_timesteps,
            )

            if seed_curves is None:
                print(f"[Skip] No valid data: {algo_id}, {env_id}, {metric}")
                continue

            print(
                f"[Info] {algo_id}, {env_id}, {metric}: "
                f"{len(valid_seeds)} seeds loaded, metric={used_metric_key}, x={used_x_key}"
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
                x_axis=x_axis,
            )


def main():
    env_id_list = ["PerturbAnt-v4", "PerturbHalfCheetah-v4", "PerturbHopper-v4", "PerturbWalker2d-v4"]
    # env_id_list = ["PerturbHopper-v4", "PerturbWalker2d-v4"]
    # env_id_list = ["PerturbHalfCheetah-v4"]
    metric_list = ["nominal", "local_mean", "local_min"]
    num_trials = 3

    # plot smoothing window
    # - 1 또는 None이면 기존처럼 smoothing 없이 plot
    # - 예: 5, 10, 20 등으로 설정하면 centered moving average 적용
    plot_window_size = 1

    # x축 설정
    # - "total_timesteps": rounds.npy, local_steps.npy, num_clients.npy를 자동으로 읽어서 x축 변환
    # - "rounds": 기존 방식처럼 global round 기준으로 비교
    x_axis = "total_timesteps"

    # True이면 total_timesteps = rounds * local_steps * num_clients
    # False이면 total_timesteps = rounds * local_steps
    # 보통 federated setting에서 전체 client가 사용한 총 환경 interaction을 비교하려면 True가 맞다.
    count_all_clients_as_timesteps = True

    # True이면 total_timesteps.npy / timesteps.npy / num_timesteps.npy가 있을 때 그것을 우선 사용한다.
    # 없으면 rounds.npy, local_steps.npy, num_clients.npy를 이용해 자동 계산한다.
    prefer_saved_timesteps = True

    # perturbation_types = ["none", "gravity", "mass", "length"]
    perturbation_types = ["gravity"]

    for env_id in env_id_list:
        for perturbation_type in perturbation_types:
            # 여러 알고리즘의 경로를 리스트로 정의
            # 이제 local_steps, num_clients는 각 seed directory 안의 npy/npz metadata에서 자동으로 읽는다.
            algo_config_list = [
                # ("td3_avg", f"logs/federate_logs/tuning/td3_avg/{perturbation_type}/200"),
                ("sac_avg", f"logs/federate_logs/tuning/sac_avg/{perturbation_type}/200"),
                # ("ar_td3_avg", f"logs/federate_logs/tuning/ar_td3_avg/{perturbation_type}/0.05/200"),
                # ("ar_td3_avg", f"logs/federate_logs/tuning/ar_td3_avg/{perturbation_type}/0.1/200"),
                # ("ar_td3_avg", f"logs/federate_logs/tuning/ar_td3_avg/{perturbation_type}/0.15/200"),
                ("ar_sac_avg", f"logs/federate_logs/tuning/ar_sac_avg/{perturbation_type}/0.05/200"),
                ("ar_sac_avg", f"logs/federate_logs/tuning/ar_sac_avg/{perturbation_type}/0.1/200"),
                ("ar_sac_avg", f"logs/federate_logs/tuning/ar_sac_avg/{perturbation_type}/0.15/200"),
            ]

            plot_root_path = f"plots/federate/avg/{env_id}/{perturbation_type}"
            extra_arg_sets = []

            for extra_args in normalize_extra_arg_sets(extra_arg_sets):
                suffix = make_extra_args_suffix(extra_args)
                filename_prefix = f"comparison{suffix}"

                for metric in metric_list:
                    algo_data_list = []
                    algo_id_counts = count_algo_ids(algo_config_list)
                    used_plot_labels = set()

                    for algo_config in algo_config_list:
                        algo_id, result_root_path, config = unpack_algo_config(algo_config)

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

                        x, seed_curves, valid_seeds, used_x_key, used_metric_key = collect_seed_curves(
                            algo_id=algo_id,
                            env_id=env_id,
                            result_root_path=result_root_with_args,
                            metric=metric,
                            num_trials=num_trials,
                            x_axis=x_axis,
                            local_steps_per_round=config.get("local_steps_per_round"),
                            num_clients=config.get("num_clients"),
                            count_all_clients_as_timesteps=config.get(
                                "count_all_clients_as_timesteps",
                                count_all_clients_as_timesteps,
                            ),
                            prefer_saved_timesteps=config.get(
                                "prefer_saved_timesteps",
                                prefer_saved_timesteps,
                            ),
                        )

                        if seed_curves is None:
                            print(f"[Skip] No valid data: {plot_label}, {env_id}, {metric}")
                            continue

                        print(
                            f"[Info] {plot_label}, {env_id}, {metric}: "
                            f"{len(valid_seeds)} seeds loaded, metric={used_metric_key}, x={used_x_key}"
                        )
                        algo_data_list.append((plot_label, x, seed_curves))

                    if len(algo_data_list) > 0:
                        save_dir = os.path.join(plot_root_path, env_id)
                        plot_multiple_algos(
                            algo_data_list=algo_data_list,
                            env_id=env_id,
                            metric=metric,
                            save_dir=save_dir,
                            filename_prefix=filename_prefix,
                            window_size=plot_window_size,
                            x_axis=x_axis,
                        )


if __name__ == "__main__":
    main()
