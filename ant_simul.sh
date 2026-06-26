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
second_args=(64)

third_args=(0.0001 0.0003 0.001 0.003 0.01 0.03)
# third_args=(8)
# fourth_args=(0.0001 0.001 0.01 0.1)
fourth_args=(0.001)
# fourth_args=(0.0001 0.0003 0.001)


# conda activate frl_opt && ./ant_simul.sh

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
                # # for i in {1..5}
                # for i in 1
                # do
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/tests" --hyperparams "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 eval_round_freq:40 --frl --device "cpu" --seed $i 
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i & 
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbLunarLanderContinuous-v3 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i & 
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbLunarLander-v3 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i & 

                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbAcrobot-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i & 
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 --log-folder "logs/self_tuning" --hyperparams n_timesteps:1e6 local_steps:256 n_envs:${first_arg} n_steps:${second_arg} --frl --device "cpu" --seed $i & 


                #     # python rl_zoo3/train.py --algo fedsp_pg_ppo --env PerturbAcrobot-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i & 
                #     # python rl_zoo3/train.py --algo fedsp_pg_ppo_paper_aligned --env PerturbAcrobot-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i & 




                #     # python rl_zoo3/train.py --algo fedsp_pg_ppo --env PerturbMountainCarContinuous-v0 --log-folder "logs/tuned/PerturbMountainCarContinuous-v0/${first_arg}/${second_arg}" --hyperparams n_timesteps:1e5 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:$((second_arg / 4)) eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $second_arg))") --frl --device "cpu" --seed $i & 
                #     # python rl_zoo3/train.py --algo fedsp_pg_ppo --env PerturbMountainCar-v0 --log-folder "logs/tuned/PerturbMountainCar-v0/${first_arg}/${second_arg}" --hyperparams n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:$((second_arg / 4)) eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --frl --device "cpu" --seed $i & 



                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbAnt-v4 --log-folder "logs/tuning/PerturbAnt-v4/original10/${first_arg}/${second_arg}/${third_arg}/8" --hyperparams n_timesteps:2e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:${third_arg} batch_size:8 eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --seed $i --frl --device "cpu" &
                #     # sleep 1
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbHalfCheetah-v4 --log-folder "logs/tuning/PerturbHalfCheetah-v4/original10/${first_arg}/${second_arg}/${third_arg}/16" --hyperparams n_timesteps:2e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:${third_arg} batch_size:16 eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --seed $i --frl --device "cpu" &
                #     # sleep 1
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbHopper-v4 --log-folder "logs/tuning/PerturbHopper-v4/original10/${first_arg}/${second_arg}/${third_arg}/8" --hyperparams n_timesteps:2e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:${third_arg} batch_size:8 eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --seed $i --frl --device "cpu" &
                #     # sleep 1
                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbWalker2d-v4 --log-folder "logs/tuning/PerturbWalker2d-v4/original10/${first_arg}/${second_arg}/${third_arg}/8" --hyperparams n_timesteps:2e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:${third_arg} batch_size:8 eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --seed $i --frl --device "cpu" &
                #     # sleep 1



                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbHalfCheetah-v4 --log-folder "logs/tuned/PerturbHalfCheetah-v4/${first_arg}/${second_arg}" --hyperparams n_timesteps:2e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --seed $i --frl --device "cpu" &

                #     # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 --log-folder "logs/tuned/PerturbCartPole-v1/1/${first_arg}/${second_arg}" --hyperparams n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} n_steps:$((second_arg / 4)) eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") --seed $i --frl --device "cpu" & 

                #     sleep 1
                #     wait
                # done

                for i in {1..3}
                # for i in 1
                do
                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 \
                    #     --log-folder "logs/tuning_classic_control/PerturbPendulum-v1/${first_arg}/ppo_avg/${second_arg}" \
                    #     --hyperparams n_timesteps:3e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbPendulum-v1 \
                    #     --log-folder "logs/tuning_classic_control/PerturbPendulum-v1/${first_arg}/fed_ampo_ppo/uniform/${second_arg}/${third_arg}" \
                    #     --hyperparams 'dual_update_mode:"uniform"' n_timesteps:3e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         server_actor_lr:${third_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbPendulum-v1 \
                    #     --log-folder "logs/tuning_classic_control/PerturbPendulum-v1/${first_arg}/fed_ampo_ppo/adaptive/${second_arg}/${fourth_arg}" \
                    #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:3e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         dual_lr:${fourth_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbMountainCar-v0 \
                    #     --log-folder "logs/tuning_classic_control/PerturbMountainCar-v0/${first_arg}/ppo_avg/${second_arg}" \
                    #     --hyperparams n_timesteps:5e4 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(500 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbMountainCar-v0 \
                    #     --log-folder "logs/tuning_classic_control/PerturbMountainCar-v0/${first_arg}/fed_ampo_ppo/uniform/${second_arg}/${third_arg}" \
                    #     --hyperparams 'dual_update_mode:"uniform"' n_timesteps:5e4 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(500 / 5 / $second_arg))") \
                    #         server_actor_lr:${third_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbMountainCar-v0 \
                    #     --log-folder "logs/tuning_classic_control/PerturbMountainCar-v0/${first_arg}/fed_ampo_ppo/adaptive/${second_arg}/${fourth_arg}" \
                    #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:5e4 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(500 / 5 / $second_arg))") \
                    #         dual_lr:${fourth_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 \
                    #     --log-folder "logs/tuning_classic_control/PerturbCartPole-v1/${first_arg}/ppo_avg/${second_arg}" \
                    #     --hyperparams n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbCartPole-v1 \
                    #     --log-folder "logs/tuning_classic_control/PerturbCartPole-v1/${first_arg}/fed_ampo_ppo/uniform/${second_arg}/${third_arg}" \
                    #     --hyperparams 'dual_update_mode:"uniform"' n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         server_actor_lr:${third_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbCartPole-v1 \
                    #     --log-folder "logs/tuning_classic_control/PerturbCartPole-v1/${first_arg}/fed_ampo_ppo/adaptive/${second_arg}/${fourth_arg}" \
                    #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         dual_lr:${fourth_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbLunarLander-v3 \
                    #     --log-folder "logs/tuning_classic_control/PerturbLunarLander-v3/${first_arg}/ppo_avg/${second_arg}" \
                    #     --hyperparams n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbLunarLander-v3 \
                        --log-folder "logs/tuning_classic_control/PerturbLunarLander-v3/${first_arg}/fed_ampo_ppo/uniform/${second_arg}/${third_arg}" \
                        --hyperparams 'dual_update_mode:"uniform"' n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                            n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                            server_actor_lr:${third_arg} \
                        --seed $i --frl --log-wandb True --device "cpu" &
                    sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbLunarLander-v3 \
                    #     --log-folder "logs/tuning_classic_control/PerturbLunarLander-v3/${first_arg}/fed_ampo_ppo/adaptive/${second_arg}/${fourth_arg}" \
                    #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:1e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:$(python -c "import math; print(math.ceil($second_arg / 4))") eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         dual_lr:${fourth_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1
                done

                wait


                # for i in {1..3}
                # # for i in 1
                # do
                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbHopper-v4 \
                    #     --log-folder "logs/tuning_mujoco_long/PerturbHopper-v4/${first_arg}/ppo_avg/" \
                    #     --hyperparams n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbAnt-v4 \
                    #     --log-folder "logs/tuning_mujoco_long/PerturbAnt-v4/${first_arg}/ppo_avg/local/" \
                    #     --hyperparams 'critic_sync_mode:"local"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbHalfCheetah-v4 \
                    #     --log-folder "logs/tuning_mujoco_long/PerturbHalfCheetah-v4/${first_arg}/ppo_avg/none/" \
                    #     --hyperparams 'vecnormalize_sync_mode:"none"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo ppo_avg --env PerturbHalfCheetah-v4 \
                    #     --log-folder "logs/tuning_mujoco_long/PerturbHalfCheetah-v4/${first_arg}/ppo_avg/none/local/" \
                    #     --hyperparams 'vecnormalize_sync_mode:"none"' 'critic_sync_mode:"local"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbHopper-v4 \
                    #     --log-folder "logs/tuning_mujoco_long/revised/PerturbHopper-v4/${first_arg}/fed_ampo_ppo/uniform/${third_arg}" \
                    #     --hyperparams 'dual_update_mode:"uniform"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         server_actor_lr:${third_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1

                    # python rl_zoo3/train.py --algo fed_ampo_ppo --env PerturbHopper-v4 \
                    #     --log-folder "logs/tuning_mujoco_long/revised/PerturbHopper-v4/${first_arg}/fed_ampo_ppo/adaptive/${fourth_arg}" \
                    #     --hyperparams 'dual_update_mode:"adaptive"' n_timesteps:10e6 "perturb_noise_type:\"${first_arg}\"" perturb_noise_range:0.5 local_steps:${second_arg} \
                    #         n_steps:${second_arg} eval_round_freq:$(python -c "import math; print(math.ceil(25000 / 5 / $second_arg))") \
                    #         dual_lr:${fourth_arg} \
                    #     --seed $i --frl --log-wandb True --device "cpu" &
                    # sleep 1
                # done
            done
        done
    done
done
