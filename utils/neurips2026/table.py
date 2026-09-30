import numpy as np
import os


def training_reward(env_id, save_path, num_trials=5):
    """실험 결과를 불러와 평균과 표준편차 계산"""
    reward_list = []
    
    for i in range(1, num_trials + 1):
        data_save_path = os.path.join(save_path, f"{env_id}_{i}")
        try:
            data = np.load(os.path.join(data_save_path, 'evaluations.npz'))
            reward_list.append(data['results'])
        except FileNotFoundError:
            # Handle missing file
            print(f"File not found for trial {i} at {data_save_path}")
        except Exception as e:
            # Handle any other errors
            print(f"Error loading trial {i} at {data_save_path}: {e}")
        # try:
        #     data = np.load(os.path.join(save_path, 'evaluations.npz'))
        # catch:

        # reward_list.append(data['results'])

        # reward_list.append(data['results'][:33])

    rewards = np.array(reward_list)
    rewards = np.mean(rewards, axis = 2)
    return np.round(np.mean(rewards, axis = 0), 1), np.round(np.std(rewards, axis=0), 1)


# def training_reward(env_id: str, algo_id: str, root_name='logs', robustness_level=0, implicit_tau=0, num_trials=5):
#     """실험 결과를 불러와 평균과 표준편차 계산"""
#     reward_list = []
    
#     for i in range(1, num_trials + 1):
#         save_path = os.path.join(root_name, str(robustness_level), str(implicit_tau), algo_id, f"{env_id}_{i}")
#         data = np.load(os.path.join(save_path, 'evaluations.npz'))
#         reward_list.append(data['results'])

#     rewards = np.array(reward_list)
#     return np.round(np.mean(rewards, axis=(0, 2)), 1), np.round(np.mean(np.std(rewards, axis = 2), axis = 0), 1)


def generate_table(algo_id, root_path, env_list, num_trails, *extra_args):
    """하나의 알고리즘에 대한 표를 생성"""
    # env_id_list = ["Ant-v4", "Walker2d-v4", "Hopper-v4", "HalfCheetah-v4", "Humanoid-v4"]
    env_id_list = env_list
    # env_id_list = ["FetchPush-v1"]
    # env_id_list = ["Ant-v4", "Walker2d-v4", "Hopper-v4", "HalfCheetah-v4"]
    # env_id_list = ["HalfCheetah-v4"]

    
    # algo_display_name = algo_id + ((", ")+ str(extra_arg) for _, extra_arg in enumerate(extra_args))
    algo_display_name = algo_id + "".join((", " + str(extra_arg)) for extra_arg in extra_args)
    return_string = f"{algo_display_name} & "
    
    reward_list, std_list = [], []
    
    for env_id in env_id_list:
        reward, std = training_reward(env_id, os.path.join(root_path, "".join((str(extra_arg) + "/") for extra_arg in extra_args), algo_id), num_trails)
        reward, std = np.round(reward[-1], 1), np.round(std[-1], 1)
        return_string += f"{reward} ± {std} & "
        reward_list.append(reward)
        std_list.append(std)

    # 평균 및 표준편차 추가
    return_string += f"{np.round(np.mean(reward_list), 2)} ± {np.round(np.mean(std_list), 2)} \\\\"
    
    return return_string


def main():
    """실험 결과를 출력"""

    env_id_list = [
        "Ant-v4",
        # "HalfCheetah-v4",
        # "Hopper-v4",
        # "Walker2d-v4",
        # "Humanoid-v4",
    ]

    # ################################################################################################################################ 
    # ### Block
    # ################################################################################################################################ 
    # num_trials = 10

    # algo_configs = [
    #     # ("td3", "logs/basic/min_q_loss"),
    #     # ("td3", "logs/basic/q0_loss"),
    #     ("td3", "logs/basic/min_q_loss/low_lr"),
    #     ("td3", "logs/basic/q0_loss/low_lr"),
    #     # ("sac", "logs/basic/"),
    #     # ("td3", "logs/basic/dm_control/tuned/min_q_loss"),
    #     # ("td3", "logs/basic/dm_control/tuned/q0_loss"),
    #     # ("sac", "logs/basic/dm_control/tuned"),
    # ]

    # for algo_id, root_path in algo_configs:
    #     print(generate_table(algo_id, root_path, env_id_list, num_trials))

    ################################################################################################################################ 
    ### Block
    ################################################################################################################################ 

    first_arg_list = [0.05, 0.1, 0.15, 0.2]  
    second_arg_list = [0.005]  

    num_trials = 5

    algo_configs = [
        ("icml_d2mc", "logs/neurips2026/rebuttal"),
    ]

    for algo_id, root_path in algo_configs:
        for first_arg in first_arg_list:
            for second_arg in second_arg_list:
                print(generate_table(algo_id, root_path, env_id_list, num_trials, first_arg, second_arg))



if __name__ == "__main__":
    main()


# sa_td3, 0.1, 1.5, 5 & 4804.4 ± 802.6 & 4804.4 ± 802.6 \\
# sa_td3, 0.2, 1.0, 5 & 5066.1 ± 170.9 & 5066.1 ± 170.9 \\
# sa_td3, 0.4, 1.0, 5 & 4839.3 ± 139.0 & 4839.3 ± 139.0 \\
# sa_td3, 0.4, 1.0, 10 & 4977.2 ± 82.6 & 4977.2 ± 82.6 \\