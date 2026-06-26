#!/bin/bash

# first_args=("gravity") # Acrobot-v1
# first_args=("gravity" "mass_cart" "length") # CartPole-v1
# first_args=("gravity" "mass" "length") # Pendulum-v1
# first_args=("power" "gravity" "none") # Pendulum-v1
# first_args=("length" "mass_cart") # Pendulum-v1
# first_args=("all_geom_density") # Pendulum-v1
first_args=("gravity") # Pendulum-v1

# second_args=(8 32 128 512)
# second_args=(256 1024)
second_args=(100)

third_args=(0.1)
# fourth_args=(0.0001 0.001 0.01 0.1)
fourth_args=(0.0000001)
# fourth_args=(0.0001 0.0003 0.001)


# conda activate frl_opt && ./fed_avg.sh

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
                for i in {1..3}
                # for i in 3
                do

                    # python rl_zoo3/train.py --algo td3_avg --env PerturbAnt-v4 \
                    #     --log-folder "logs/federate_logs/tuning/td3_avg/reset_optimizer/${first_arg}/${second_arg}" \
                    #     --hyperparams "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 n_timesteps:3e6 local_steps:${second_arg} \
                    #         reset_optimizer_on_broadcast:True \
                    #         eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu"
                    # sleep 1

                    # python rl_zoo3/train.py --algo sac_avg --env PerturbAnt-v4 \
                    #     --log-folder "logs/federate_logs/tuning/sac_avg/reset_optimizer/${first_arg}/${second_arg}" \
                    #     --hyperparams "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 n_timesteps:3e6 local_steps:${second_arg} \
                    #         eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                    #         reset_optimizer_on_broadcast:True \
                    #     --seed $i --frl --log-wandb True --device "cpu"
                    # sleep 1

                    # python rl_zoo3/train.py --algo ar_sac_avg --env PerturbAnt-v4 \
                    #     --log-folder "logs/federate_logs/tuning/ar_sac_avg/reset_optimizer/${first_arg}/${third_arg}/${second_arg}" \
                    #     --hyperparams "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 n_timesteps:3e6 local_steps:${second_arg} \
                    #         cautious_weight:${third_arg}  reset_optimizer_on_broadcast:True \
                    #         eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu"
                    # sleep 1

                    python rl_zoo3/train.py --algo ar_td3_avg --env PerturbAnt-v4 \
                        --log-folder "logs/federate_logs/tuning/ar_td3_avg/reset_optimizer/${first_arg}/${third_arg}/${second_arg}" \
                        --hyperparams "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 n_timesteps:3e6 local_steps:${second_arg} \
                            cautious_weight:${third_arg} reset_optimizer_on_broadcast:True \
                            eval_round_freq:$(python -c "import math; print(math.ceil(100000 / 5 / $second_arg))") \
                        --seed $i --frl --log-wandb True --device "cpu"
                    sleep 1

                done
            done
        done
    done
done
