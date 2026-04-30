#!/bin/bash

first_args=(1024)

second_args=(0.15)

third_args=(0.5)


# conda activate frl_eh && ./ant.sh

# Outer loop for second_arg
for first_arg in "${first_args[@]}"
do
    # Inner loop for implicit_tau
    for second_arg in "${second_args[@]}"
    do
        for third_arg in "${third_args[@]}"
        do
            # Loop for num_id
            for i in 4
            # for i in {1..5}
            do
                # python rl_zoo3/train.py --algo td3_avg --env Pendulum-v1 --log-folder "logs/tests/td3_avg" --device "cuda:0" --seed $i


                ### local_steps, eval_round_freq, perturb_noise_type, perturb_noise_range

                # python rl_zoo3/train.py --algo td3_avg --env PerturbHalfCheetah-v4 --log-folder "logs/tests/frl_eh/td3_avg/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$((25000 / 5 / first_arg)) 'perturb_noise_type:"radius_torso"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i

                ### PerturbPendulum-v1 ### gravity, length, mass
                # python rl_zoo3/train.py --algo td3_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/td3_avg/mass/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"mass"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i
                # python rl_zoo3/train.py --algo td3_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/td3_avg/gravity/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"gravity"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i
                # python rl_zoo3/train.py --algo td3_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/td3_avg/length/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"length"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i
                
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/mass/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"mass"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/gravity/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"gravity"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/length/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") 'perturb_noise_type:"length"' perturb_noise_range:0.5 --frl --device "cuda:0" --seed $i
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg2/none/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") --frl --device "cuda:0" --seed $i
                
                # python rl_zoo3/train.py --algo ppo_avg --env PerturbPendulum-v1 --log-folder "logs/frl_eh/ppo_avg4/none/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") --frl --device "cuda:0" --seed $i

                python rl_zoo3/train.py --algo fedsp_pg_ppo --env PerturbPendulum-v1 --log-folder "logs/tests/frl_eh/fedsp_pg_ppo/none/${first_arg}" --hyperparams local_steps:${first_arg} eval_round_freq:$(python -c "import math; print(math.ceil(2500 / 5 / $first_arg))") --frl --device "cuda:0" --seed $i
                
            done
        done
    done
done