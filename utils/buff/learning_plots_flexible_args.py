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




def plot_seed_average_curve(
    x,
    seed_curves,
    algo_id,
    env_id,
    metric,
    save_dir,
    filename_prefix,
):
    """
    시드 평균 curve와 시드 std band를 함께 그린다.

    주의:
    - 여기서 std는 nominal_std/local_std를 쓰는 것이 아니다.
    - 각 seed에서 얻은 'mean curve'를 기준으로
      seed 방향으로 std를 계산한다.
    """

    os.makedirs(save_dir, exist_ok=True)

    # 각 round에서 seed 평균 / seed std
    avg_curve = np.mean(seed_curves, axis=0)
    std_curve = np.std(seed_curves, axis=0)

    plt.figure(figsize=(8, 5))

    # seed average line
    plt.plot(
        x,
        avg_curve,
        linewidth=2.5,
        label="seed average",
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
        f"{filename_prefix}_{metric}_learning_curve.png",
    )

    plt.savefig(save_file, dpi=300)
    plt.close()

    print(f"[Saved] {save_file}")


def plot_multiple_algos(
    algo_data_dict,
    env_id,
    metric,
    save_dir,
    filename_prefix,
):
    """
    여러 알고리즘의 seed average curve를 한 plot에 그린다.

    algo_data_dict:
        {algo_id: (x, seed_curves), ...}
    """

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12, 6))

    for algo_id, (x, seed_curves) in algo_data_dict.items():
        # 각 round에서 seed 평균 / seed std
        avg_curve = np.mean(seed_curves, axis=0)
        std_curve = np.std(seed_curves, axis=0)

        # seed average line
        plt.plot(
            x,
            avg_curve,
            linewidth=2.5,
            label=algo_id,
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
    plt.title(f"Algorithm Comparison - {env_id} - {metric}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    save_file = os.path.join(
        save_dir,
        f"{filename_prefix}_comparison_{metric}_learning_curve.png",
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
            )


def main():
    # env_id_list = ["PerturbPendulum-v1"]
    env_id_list = ["PerturbAcrobot-v1"]
    metric_list = ["nominal", "local_mean", "local_min"]
    num_trials = 3

    # perturbation_types = ["none", "gravity", "mass", "length"]
    perturbation_types = ["none"]

    for perturbation_type in perturbation_types:
        # 여러 알고리즘의 경로를 리스트로 정의
        algo_config_list = [
            ("ppo_avg", "logs/parameter_tuning/ppo_avg/"),
            # ("fedsp_pg_ppo", "logs/developing/frl_eh/fedsp_pg_ppo/" + perturbation_type),
            # ("fedsvrpg_m", "logs/developing/frl_eh/fedsvrpg_m/" + perturbation_type),
        ]

        # plot 저장 root
        plot_root_path = f"plots/frl_eh/{perturbation_type}"

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
            for env_id in env_id_list:
                for metric in metric_list:
                    algo_data_dict = {}

                    # 각 알고리즘의 데이터를 수집
                    for algo_id, result_root_path in algo_config_list:
                        result_root_with_args = append_extra_args_to_path(
                            result_root_path,
                            extra_args,
                        )

                        x, seed_curves, valid_seeds = collect_seed_curves(
                            algo_id=algo_id,
                            env_id=env_id,
                            result_root_path=result_root_with_args,
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
                        algo_data_dict[algo_id] = (x, seed_curves)

                    # 수집한 모든 알고리즘을 하나의 플롯에 표시
                    if len(algo_data_dict) > 0:
                        save_dir = os.path.join(plot_root_path, env_id)
                        plot_multiple_algos(
                            algo_data_dict=algo_data_dict,
                            env_id=env_id,
                            metric=metric,
                            save_dir=save_dir,
                            filename_prefix=filename_prefix,
                        )


if __name__ == "__main__":
    main()