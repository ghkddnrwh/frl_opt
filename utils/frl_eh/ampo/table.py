import os
import sys
import numpy as np


def load_final_reward(npz_path, metric="nominal"):
    """
    한 seed의 evaluations.npz에서 마지막 evaluation reward를 읽는다.

    metric 정의는 기존 plot 코드와 동일:
        - nominal   : nominal_mean의 client 평균
        - local_mean: local_mean의 client 평균
        - local_min : local_mean의 client 최솟값

    반환값:
        final_reward: 해당 seed의 마지막 evaluation reward
        final_round : rounds가 저장되어 있으면 마지막 round, 아니면 None
    """
    with np.load(npz_path, allow_pickle=True) as data:
        if metric == "nominal":
            reward_all = np.asarray(data["nominal_mean"], dtype=float)
            if reward_all.ndim == 2:
                reward_curve = np.mean(reward_all, axis=1)
            else:
                reward_curve = reward_all

        elif metric == "local_mean":
            reward_all = np.asarray(data["local_mean"], dtype=float)
            if reward_all.ndim == 2:
                reward_curve = np.mean(reward_all, axis=1)
            else:
                reward_curve = reward_all

        elif metric == "local_min":
            reward_all = np.asarray(data["local_mean"], dtype=float)
            if reward_all.ndim == 2:
                reward_curve = np.min(reward_all, axis=1)
            else:
                reward_curve = reward_all

        else:
            raise ValueError(f"Unknown metric: {metric}")

        reward_curve = np.asarray(reward_curve, dtype=float).reshape(-1)

        if len(reward_curve) == 0:
            raise ValueError("No reward data exists in evaluations.npz")

        final_reward = float(reward_curve[-1])

        final_round = None
        if "rounds" in data:
            rounds = np.asarray(data["rounds"]).reshape(-1)
            if len(rounds) > 0:
                final_round = rounds[-1].item()

    return final_reward, final_round


def resolve_npz_path(result_root_path, algo_id, env_id, seed):
    """
    실제 evaluations.npz 경로를 찾는다.

    우선순위:
        1. {result_root_path}/{algo_id}/{env_id}_{seed}/evaluations.npz
        2. {result_root_path}/{env_id}_{seed}/evaluations.npz

    현재 실험 저장 구조처럼 hyperparameter 폴더 뒤에 algo_id가 다시 붙는 경우와,
    algo_id 없이 바로 seed 폴더가 오는 경우를 둘 다 지원한다.
    """
    candidates = [
        os.path.join(
            result_root_path,
            algo_id,
            f"{env_id}_{seed}",
            "evaluations.npz",
        ),
        os.path.join(
            result_root_path,
            f"{env_id}_{seed}",
            "evaluations.npz",
        ),
    ]

    for path in candidates:
        if os.path.exists(path):
            return path

    return None


def final_reward_stats(
    algo_id,
    env_id,
    result_root_path,
    metric,
    num_trials=5,
):
    """
    여러 seed의 '마지막 reward'를 모아서 seed 평균과 seed 표준편차를 계산한다.

    evaluations.npz는 아래 두 저장 구조를 모두 지원한다:
        1. {result_root_path}/{algo_id}/{env_id}_{seed}/evaluations.npz
        2. {result_root_path}/{env_id}_{seed}/evaluations.npz

    예:
        result_root_path = .../momentum/0.0002/0.95
        algo_id          = fed_ampo_local_ppo
        seed 1 path      = .../0.95/fed_ampo_local_ppo/PerturbWalker2d-v4_1/evaluations.npz
    """
    final_rewards = []
    final_rounds = []
    valid_seeds = []

    for seed in range(1, num_trials + 1):
        npz_path = resolve_npz_path(
            result_root_path=result_root_path,
            algo_id=algo_id,
            env_id=env_id,
            seed=seed,
        )

        if npz_path is None:
            primary_path = os.path.join(
                result_root_path,
                algo_id,
                f"{env_id}_{seed}",
                "evaluations.npz",
            )
            fallback_path = os.path.join(
                result_root_path,
                f"{env_id}_{seed}",
                "evaluations.npz",
            )
            print(
                f"[Missing] seed {seed}: tried\n"
                f"          {primary_path}\n"
                f"          {fallback_path}",
                file=sys.stderr,
            )
            continue

        try:
            final_reward, final_round = load_final_reward(
                npz_path=npz_path,
                metric=metric,
            )
        except Exception as e:
            print(f"[Error] seed {seed}: {npz_path}", file=sys.stderr)
            print(f"        {e}", file=sys.stderr)
            continue

        final_rewards.append(final_reward)
        valid_seeds.append(seed)

        if final_round is not None:
            final_rounds.append(final_round)

    if len(final_rewards) == 0:
        return np.nan, np.nan, valid_seeds

    final_rewards = np.asarray(final_rewards, dtype=float)

    # 표에 보이지 않는 warning은 stderr로 출력해서 stdout의 LaTeX table을 깨지 않게 한다.
    if len(final_rounds) > 1 and len(set(final_rounds)) > 1:
        print(
            f"[Warning] {env_id}, {metric}: seeds have different final rounds: "
            f"{sorted(set(final_rounds))}",
            file=sys.stderr,
        )

    reward_mean = np.nanmean(final_rewards)
    reward_std = np.nanstd(final_rewards)  # seed 방향 std, ddof=0

    return reward_mean, reward_std, valid_seeds


def normalize_extra_args(extra_args=None):
    """추가 하위 폴더 인자를 항상 tuple로 만든다."""
    if extra_args is None:
        return ()

    if isinstance(extra_args, (list, tuple)):
        return tuple(extra_args)

    return (extra_args,)


def normalize_extra_arg_sets(extra_arg_sets=None):
    """
    예:
        []                         -> [()]
        [1024, 2048]               -> [(1024,), (2048,)]
        [(1024, 64), (2048, 128)]  -> [(1024, 64), (2048, 128)]
    """
    if extra_arg_sets is None or len(extra_arg_sets) == 0:
        return [()]

    return [normalize_extra_args(extra_args) for extra_args in extra_arg_sets]


def append_extra_args_to_path(root_path, extra_args=None):
    """extra_args가 있으면 root_path 아래 subdir로 붙인다."""
    extra_args = normalize_extra_args(extra_args)

    if len(extra_args) == 0:
        return root_path

    return os.path.join(root_path, *[str(arg) for arg in extra_args])


def make_algo_display_name(algo_id, result_root_path, extra_args=None):
    """
    예제 table 형식처럼 알고리즘 이름 뒤에 경로상의 hyperparameter를 붙인다.

    예:
        .../fed_svrpg_m/0.85/0.5
        -> fed_svrpg_m, 0.85, 0.5

        .../fed_ampo_local_ppo/uniform/0.7
        -> fed_ampo_local_ppo, uniform, 0.7
    """
    path_parts = os.path.normpath(result_root_path).split(os.sep)

    # path 안에서 마지막 algo_id 위치를 찾는다.
    algo_positions = [i for i, part in enumerate(path_parts) if part == algo_id]

    if len(algo_positions) > 0:
        algo_idx = algo_positions[-1]
        path_args = path_parts[algo_idx + 1 :]
    else:
        path_args = []

    all_args = list(path_args) + [str(arg) for arg in normalize_extra_args(extra_args)]

    if len(all_args) == 0:
        return algo_id

    return algo_id + "".join(f", {arg}" for arg in all_args)


def generate_table_row(
    algo_id,
    result_root_path,
    env_id,
    metric_list,
    num_trials,
    extra_args=None,
    decimals=1,
):
    """
    한 알고리즘 설정에 대한 LaTeX table row 생성.

    출력 예:
        fed_svrpg_m, 0.85, 0.5 & 1234.5 ± 120.3 & 1100.2 ± 95.4 & 870.1 ± 80.0 \\
    """
    result_root_with_args = append_extra_args_to_path(
        result_root_path,
        extra_args,
    )

    display_name = make_algo_display_name(
        algo_id=algo_id,
        result_root_path=result_root_path,
        extra_args=extra_args,
    )

    table_values = []

    for metric in metric_list:
        reward_mean, reward_std, valid_seeds = final_reward_stats(
            algo_id=algo_id,
            env_id=env_id,
            result_root_path=result_root_with_args,
            metric=metric,
            num_trials=num_trials,
        )

        if len(valid_seeds) == 0 or np.isnan(reward_mean):
            table_values.append("N/A")
            continue

        if len(valid_seeds) != num_trials:
            print(
                f"[Warning] {display_name}, {env_id}, {metric}: "
                f"{len(valid_seeds)}/{num_trials} seeds loaded: {valid_seeds}",
                file=sys.stderr,
            )

        table_values.append(
            f"{reward_mean:.{decimals}f} ± {reward_std:.{decimals}f}"
        )

    return f"{display_name} & " + " & ".join(table_values) + r" \\"


def main():
    """각 perturbation type에 대해 마지막 reward의 seed mean ± std를 출력한다."""

    env_id = "PerturbAnt-v4"
    metric_list = ["nominal", "local_mean", "local_min"]
    num_trials = 5

    perturbation_types = ["friction", "gravity"]

    for perturbation_type in perturbation_types:
        # 각 tuple의 두 번째 값은 hyperparameter 설정까지의 root path이다.
        # 실제 파일 탐색 시 아래 두 형태를 순서대로 확인한다:
        #   1) {result_root_path}/{algo_id}/{env_id}_{seed}/evaluations.npz
        #   2) {result_root_path}/{env_id}_{seed}/evaluations.npz
        # 따라서 momentum/0.0002/0.95/fed_ampo_local_ppo/PerturbWalker2d-v4_1
        # 같은 저장 구조도 그대로 읽을 수 있다.
        algo_config_list = [
            ("ppo_avg", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/ppo_avg"),

            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform/0.0003"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform/0.001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/uniform/0.003"),

            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.0003"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.0001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.00001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.000001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/0.0000001"),

            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0002"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0003"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0005"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.00001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.000001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.0000001"),

            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.7/0.0001"),
            # ("fed_ampo_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_ppo/adaptive/undiscounted/0.7/0.0003"),

            ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_svrpg_m/0.85/0.5"),
            ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_svrpg_m/0.9/0.5"),
            ("fed_svrpg_m", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_svrpg_m/0.95/0.5"),

            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/uniform/0.5"),
            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/uniform/0.7"),
            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/uniform/1.0"),

            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/0.0001"),
            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/0.0002"),
            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/0.0003"),

            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/momentum/0.0002/0.85"),
            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/momentum/0.0002/0.9"),
            ("fed_ampo_local_ppo", f"logs/fed_ampo/tuned_mujoco/fixed/noise_assignment/{perturbation_type}/fed_ampo_local_ppo/adaptive/undiscounted/momentum/0.0002/0.95"),
        ]

        print(f"% {perturbation_type}")
        print("Algorithm & " + " & ".join(metric_list) + r" \\")

        for algo_id, result_root_path in algo_config_list:
            print(
                generate_table_row(
                    algo_id=algo_id,
                    result_root_path=result_root_path,
                    env_id=env_id,
                    metric_list=metric_list,
                    num_trials=num_trials,
                    extra_args=None,
                    decimals=1,
                )
            )

        print()


if __name__ == "__main__":
    main()
