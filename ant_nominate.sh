#!/bin/bash

# first_args=("gravity") # Acrobot-v1
# first_args=("gravity" "mass_cart" "length") # CartPole-v1
# first_args=("gravity" "mass" "length") # Pendulum-v1
# first_args=("power" "gravity" "none") # Pendulum-v1
# first_args=("length" "mass_cart") # Pendulum-v1
# first_args=("all_geom_density") # Pendulum-v1
first_args=("friction" "gravity") # Pendulum-v1

# second_args=(8 32 128 512)
# second_args=(256 1024)
second_args=(512)

# third_args=(0.0003 0.001 0.003)
third_args=(0.001)
# third_args=(8)
# fourth_args=(0.0001 0.001 0.01 0.1)
fourth_args=(0.0000001)

fifth_args=("PerturbHalfCheetah-v4")
# fourth_args=(0.0001 0.0003 0.001)


# conda activate frl_opt && ./ant_nominate.sh

# Outer loop for second_arg
for first_arg in "${first_args[@]}"
do
    # Inner loop for implicit_tau
    for second_arg in "${second_args[@]}"
    do
        for third_arg in "${third_args[@]}"
        do
            for fourth_arg in "${fourth_args[@]}"
            do
                for fifth_arg in "${fifth_args[@]}"
                do
                    for i in {1..5}
                    # for i in 2
                    do
                        python rl_zoo3/train.py --algo ppo_avg --env ${fifth_arg} \
                            --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/ppo_avg/" \
                            --hyperparams n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                                n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                            --seed $i --frl --log-wandb False --device "cpu" &
                        sleep 3

                        python rl_zoo3/train.py --algo fed_ampo_ppo --env ${fifth_arg} \
                            --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/fed_ampo_ppo/uniform" \
                            --hyperparams 'dual_update_mode:"uniform"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                                n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                            --seed $i --frl --log-wandb False --device "cpu" &
                        sleep 3

                        # python rl_zoo3/train.py --algo fed_ampo_ppo --env ${fifth_arg} \
                        #     --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/fed_ampo_ppo/adaptive/${fourth_arg}" \
                        #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                        #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                        #         dual_lr:${fourth_arg} \
                        #     --seed $i --frl --log-wandb False --device "cpu" &
                        # sleep 3

                        # python rl_zoo3/train.py --algo fed_ampo_local_ppo --env ${fifth_arg} \
                        #     --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/fed_ampo_local_ppo/uniform/${third_arg}" \
                        #     --hyperparams 'dual_update_mode:"uniform"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                        #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                        #         server_actor_delta_scale:${third_arg} \
                        #     --seed $i --frl --log-wandb False --device "cpu" &
                        # sleep 3

                        # python rl_zoo3/train.py --algo fed_ampo_local_ppo --env ${fifth_arg} \
                        #     --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/fed_ampo_local_ppo/adaptive/${fourth_arg}" \
                        #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                        #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                        #         dual_lr:${fourth_arg} \
                        #     --seed $i --frl --log-wandb False --device "cpu" &
                        # sleep 3

                        # python rl_zoo3/train.py --algo fed_ampo_local_ppo --env ${fifth_arg} \
                        #     --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/fed_ampo_local_ppo/adaptive/momentum/${fourth_arg}/${third_arg}" \
                        #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                        #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                        #         'local_actor_update_mode:"momentum"' dual_lr:${fourth_arg} momentum_beta:${third_arg} \
                        #     --seed $i --frl --log-wandb False --device "cpu" &
                        # sleep 3

                        # python rl_zoo3/train.py --algo fed_svrpg_m --env ${fifth_arg} \
                        #     --log-folder "logs/fed_ampo/tuned_mujoco/noise_assignment/${first_arg}/fed_svrpg_m/${third_arg}/${fourth_arg}" \
                        #     --hyperparams n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" 'client_noise_values:[-0.5,-0.25,0.0,0.25,0.5]' local_steps:${second_arg} \
                        #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                        #         momentum_beta:${third_arg} server_update_weight:${fourth_arg} \
                        #     --seed $i --frl --log-wandb False --device "cpu" &
                        # sleep 3
                    done
                    wait
                done
            done
        done
    done
done
