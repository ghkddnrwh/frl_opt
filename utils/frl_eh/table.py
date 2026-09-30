import numpy as np
import os


def load_single_score(npz_path, metric="nominal"):
    """한 run의 evaluations.npz에서 마지막 round의 mean/std를 가져옴"""

    data = np.load(npz_path)

    if metric == "nominal":
        # shape: (num_rounds, num_clients)
        mean_all = np.array(data["nominal_mean"], dtype=float)
        std_all = np.array(data["nominal_std"], dtype=float)

        last_mean = mean_all[-1]   # (num_clients,)
        last_std = std_all[-1]     # (num_clients,)

        final_mean = float(np.mean(last_mean))
        final_std = float(np.mean(last_std))

    elif metric == "local_mean":
        # shape: (num_rounds, num_clients)
        mean_all = np.array(data["local_mean"], dtype=float)
        std_all = np.array(data["local_std"], dtype=float)

        last_mean = mean_all[-1]   # (num_clients,)
        last_std = std_all[-1]     # (num_clients,)

        final_mean = float(np.mean(last_mean))
        final_std = float(np.mean(last_std))

    elif metric == "local_min":
        # shape: (num_rounds, num_clients)
        mean_all = np.array(data["local_mean"], dtype=float)
        std_all = np.array(data["local_std"], dtype=float)

        last_mean = mean_all[-1]   # (num_clients,)
        last_std = std_all[-1]     # (num_clients,)

        min_idx = int(np.argmin(last_mean))
        final_mean = float(last_mean[min_idx])
        final_std = float(last_std[min_idx])

    else:
        raise ValueError(f"Unknown metric: {metric}")

    return final_mean, final_std


def training_reward(algo_id, env_id, save_path, metric="nominal", num_trials=5):
    """여러 seed 결과를 불러와 seed 평균과 std 계산"""
    mean_list = []
    std_list = []

    for i in range(1, num_trials + 1):
        data_save_path = os.path.join(save_path, algo_id, f"{env_id}_{i}", "evaluations.npz")

        try:
            mean_score, std_score = load_single_score(data_save_path, metric=metric)
            mean_list.append(mean_score)
            std_list.append(std_score)
        except FileNotFoundError:
            print(f"File not found for trial {i}: {data_save_path}")
        except Exception as e:
            print(f"Error loading trial {i}: {data_save_path}, error: {e}")

    if len(mean_list) == 0:
        return np.nan, np.nan

    mean_scores = np.array(mean_list, dtype=float)
    std_scores = np.array(std_list, dtype=float)

    return np.round(np.mean(mean_scores), 1), np.round(np.mean(std_scores), 1)


def generate_table(algo_id, root_path, env_list, num_trials, metric, *extra_args):
    """하나의 알고리즘 설정에 대한 표 한 줄 생성"""
    algo_display_name = algo_id + "".join(", " + str(extra_arg) for extra_arg in extra_args)
    algo_display_name += f", {metric}"

    return_string = f"{algo_display_name} & "

    reward_list, std_list = [], []

    subdir = "".join(str(extra_arg) + "/" for extra_arg in extra_args)
    save_root = os.path.join(root_path, subdir)

    for env_id in env_list:
        reward, std = training_reward(
            algo_id=algo_id,
            env_id=env_id,
            save_path=save_root,
            metric=metric,
            num_trials=num_trials,
        )
        return_string += f"{reward} ± {std} & "
        reward_list.append(reward)
        std_list.append(std)

    return_string += f"{np.round(np.nanmean(reward_list), 2)} ± {np.round(np.nanmean(std_list), 2)} \\\\"
    return return_string


def main():
    env_id_list = ["PerturbPendulum-v1"]

    # first_arg_list = [10, 50, 100, 500, 1000]
    first_arg_list = [256, 512, 1024]
    num_trials = 4

    # "nominal", "local_mean", "local_min"
    metric_list = ["nominal", "local_mean", "local_min"]

    algo_configs = [
        # ("td3_avg", "logs/frl_eh/td3_avg/mass"),
        ("ppo_avg", "logs/frl_eh/ppo_avg2/none"),
    ]

    for algo_id, root_path in algo_configs:
        for metric in metric_list:
            print(f"\n===== metric: {metric} =====")
            for first_arg in first_arg_list:
                print(
                    generate_table(
                        algo_id,
                        root_path,
                        env_id_list,
                        num_trials,
                        metric,
                        first_arg,
                    )
                )


if __name__ == "__main__":
    main()