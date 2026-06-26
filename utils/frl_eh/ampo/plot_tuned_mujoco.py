import os
import numpy as np
import matplotlib.pyplot as plt


def load_single_curve(npz_path, metric="nominal"):
    """
    한 seed의 evaluations.npz에서 round별 metric mean curve를 가져온다.

    metric:
        - nominal
        - local_mean
        - local_min

    return:
        x: shape (num_rounds,)
        mean_curve: shape (num_rounds,)
    """

    data = np.load(npz_path)

    if metric == "nominal":
        # nominal_mean: shape (num_rounds, num_clients) 또는 (num_rounds,)
        mean_all = np.array(data["nominal_mean"], dtype=float)

        if mean_all.ndim == 2:
            # 각 round에서 client 평균
            mean_curve = np.mean(mean_all, axis=1)
        else:
            mean_curve = mean_all

    elif metric == "local_mean":
        # local_mean: shape (num_rounds, num_clients) 또는 (num_rounds,)
        mean_all = np.array(data["local_mean"], dtype=float)

        if mean_all.ndim == 2:
            # 각 round에서 client 평균
            mean_curve = np.mean(mean_all, axis=1)
        else:
            mean_curve = mean_all

    elif metric == "local_min":
        # local_mean에서 각 round별 최소 client 값 사용
        mean_all = np.array(data["local_mean"], dtype=float)

        if mean_all.ndim == 2:
            mean_curve = np.min(mean_all, axis=1)
        else:
            mean_curve = mean_all

    else:
        raise ValueError(f"Unknown metric: {metric}")

    num_rounds = len(mean_curve)

    # x축
    if "timesteps" in data:
        x = np.array(data["timesteps"], dtype=float)[:num_rounds]
    elif "rounds" in data:
        x = np.array(data["rounds"], dtype=float)[:num_rounds]
    else:
        x = np.arange(num_rounds)

    return x, mean_curve


def collect_seed_curves(
    algo_id,
    env_id,
    result_root_path,
    metric,
    num_trials,
):
    """
    여러 seed의 mean curve를 모은다.

    expected path:
        {result_root_path}/{algo_id}/{env_id}_{seed}/evaluations.npz
    """

    x_list = []
    curve_list = []
    valid_seeds = []

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
            x, mean_curve = load_single_curve(npz_path=npz_path, metric=metric)
            x_list.append(x)
            curve_list.append(mean_curve)
            valid_seeds.append(seed)
        except Exception as e:
            print(f"[Error] seed {seed}: {npz_path}")
            print(f"        {e}")

    if len(curve_list) == 0:
        return None, None, []

    # seed마다 길이가 다를 수 있으므로 가장 짧은 길이에 맞춤
    min_len = min(len(curve) for curve in curve_list)

    x = x_list[0][:min_len]
    seed_curves = np.array([curve[:min_len] for curve in curve_list], dtype=float)

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
    """plot smoothing에 사용할 window 크기를 정규화한다.

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

    plt.xlabel("Round / Timesteps")
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

    plt.xlabel("Round / Timesteps")
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
    env_id = "PerturbWalker2d-v4"
    metric_list = ["nominal", "local_mean", "local_min"]
    num_trials = 5

    # plot smoothing window
    # - 1 또는 None이면 기존처럼 smoothing 없이 plot
    # - 예: 5, 10, 20 등으로 설정하면 centered moving average 적용
    plot_window_size = 10

    # perturbation_types = ["none", "gravity", "mass", "length"]
    perturbation_types = ["friction", "gravity"]

    for perturbation_type in perturbation_types:
        # 여러 알고리즘의 경로를 리스트로 정의
        algo_config_list = [
            ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/{env_id}/{perturbation_type}/ppo_avg"),
            # ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/{env_id}/{perturbation_type}/ppo_avg/local"),
            # ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/{env_id}/{perturbation_type}/ppo_avg/none"),
            # ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/{env_id}/{perturbation_type}/ppo_avg/none/local"),

            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/0.0001"),
            ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/0.0003"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/0.001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/0.003"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/uniform/0.01"),
            
            ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/0.0000001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/0.000001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/0.00001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/0.0001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/revised/{env_id}/{perturbation_type}/fed_ampo_ppo/adaptive/0.0003"),

        ]

        # plot 저장 root
        plot_root_path = f"plots/frl_eh/tuning_mujoco_long/{env_id}/{perturbation_type}"

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
            # for env_id in env_id_list:
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
