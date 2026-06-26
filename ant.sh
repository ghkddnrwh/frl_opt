#!/bin/bash

first_args=(32 64 128 256 512)

second_args=("none")

third_args=(0.5)


# conda activate frl_opt && ./ant.sh

# Outer loop for second_arg
for first_arg in "${first_args[@]}"
do
    # Inner loop for implicit_tau
    for second_arg in "${second_args[@]}"
    do
        for third_arg in "${third_args[@]}"
        do
            # Loop for num_id
            for i in 1
            # for i in {1..5}
            do
                # python rl_zoo3/train.py --algo td3_avg --env Pendulum-v1 --log-folder "logs/tests/td3_avg" --device "cpu" --seed $i

                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbMountainCarContinuous-v0 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbLunarLanderContinuous-v3 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i

                # python rl_zoo3/train.py --algo ppo_avg --env PerturbAcrobot-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbLunarLander-v3 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbMountainCar-v0 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i
                
                # python rl_zoo3/train.py --algo ppo --env PerturbHopper-v4 --log-folder "logs/basic/ppo/${first_arg}" --hyperparams n_steps:${first_arg} --device "cpu" --seed $i
                python rl_zoo3/train.py --algo ppo --env Hopper-v4 --log-folder "logs/basic/ppo/${first_arg}" --hyperparams n_steps:${first_arg} --device "cpu" --seed $i &

                # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 --log-folder "logs/parameter_tuning" --frl --device "cpu" --seed $i



                ### local_steps, eval_round_freq, perturb_noise_type, perturb_noise_range

                # python rl_zoo3/train.py --algo td3_avg --env PerturbHalfCheetah-v4 --log-folder "logs/tests/frl_eh/td3_avg/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$((25000 / 5 / first_arg)) 'perturb_noise_type:"radius_torso"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i

                ### PerturbPendulum-v1 ### gravity, length, mass
                # python rl_zoo3/train.py --algo td3_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/td3_avg/mass/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"mass"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo td3_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/td3_avg/gravity/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"gravity"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo td3_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/td3_avg/length/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"length"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/mass/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"mass"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/gravity/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"gravity"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/length/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"length"' perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/none/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") --frl --device "cpu" --seed $i
                
                # python rl_zoo3/train.py --algo ppo --env PerturbCartPole-v1 --log-folder "logs/basic/ppo" --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo --env PerturbPendulum-v1 --log-folder "logs/basic/ppo" --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo --env PerturbMountainCarContinuous-v0 --log-folder "logs/basic/ppo" --device "cpu" --eval-freq 500 --seed $i
                # python rl_zoo3/train.py --algo ppo --env PerturbMountainCarContinuous-v0 --log-folder "logs/basic/ppo/unnormalize" --hyperparams normalize:False --device "cpu" --eval-freq 500 --seed $i

                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/tests/frl_eh/ppo_avg/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbMountainCarContinuous-v0 --log-folder "logs/frl_eh/ppo_avg/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbAcrobot-v1 --log-folder "logs/tuning/frl_eh/ppo_avg/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 --log-folder "logs/tuning/frl_eh/ppo_avg/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbCartPole-v1 --log-folder "logs/tuning/frl_eh/ppo_avg/${second_arg}/${first_arg}" --hyperparams num_clients:1 local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i

                # python rl_zoo3/train.py --algo fedsp_pg_ppo --env PerturbPendulum-v1 --log-folder "logs/developing/frl_eh/fedsp_pg_ppo/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                # python rl_zoo3/train.py --algo fedsp_pg_ppo --env PerturbMountainCarContinuous-v0 --log-folder "logs/developing/frl_eh/fedsp_pg_ppo/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "cpu" --seed $i
                
                # FedSVRPG-M converts the per-round interaction budget local_steps
                # into the paper's local iteration count K using the env horizon.
                # python train.py --algo fedsvrpg_m --env PerturbPendulum-v1 --log-folder "logs/developing/frl_eh/fedsvrpg_m/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "auto" --seed $i
                # python train.py --algo fedsvrpg_m --env PerturbMountainCarContinuous-v0 --log-folder "logs/developing/frl_eh/fedsvrpg_m/${second_arg}/${first_arg}" --hyperparams "perturb_noise_type:\"${second_arg}\"" local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") perturb_noise_range:0.5 --frl --device "auto" --seed $i
            done
        done
    done
done
