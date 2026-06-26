#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_NAME="${1:-compare_200k}"
TIMESTEPS="${TIMESTEPS:-200000}"
EVAL_ROUND_FREQ="${EVAL_ROUND_FREQ:-40}"
EVAL_EPISODES="${EVAL_EPISODES:-5}"
LOG_ROOT="logs/codex/fedsp_pg_ppo_codex/${RUN_NAME}"
STDOUT_DIR="${LOG_ROOT}/stdout"
mkdir -p "$STDOUT_DIR"

COMMON_ARGS=(
  --env PerturbPendulum-v1
  --hyperparams
  "n_timesteps:${TIMESTEPS}"
  'perturb_noise_type:"gravity"'
  "perturb_noise_range:0.5"
  "eval_round_freq:${EVAL_ROUND_FREQ}"
  "eval_local_episodes:${EVAL_EPISODES}"
  "eval_nominal_episodes:${EVAL_EPISODES}"
  --frl
  --device cpu
  --verbose 0
)

for seed in 1 2 3; do
  python codex/train_codex.py \
    --algo ppo_avg \
    --conf-file hyperparams/ppo_avg.yml \
    --log-folder "${LOG_ROOT}/train_logs/ppo_avg_seed_${seed}" \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    > "${STDOUT_DIR}/ppo_avg_seed_${seed}.log" 2>&1 &

  python codex/train_codex.py \
    --algo fedsp_pg_ppo_codex \
    --conf-file codex/fedsp_pg_ppo_codex.yml \
    --log-folder "${LOG_ROOT}/train_logs/fedsp_pg_ppo_codex_seed_${seed}" \
    "${COMMON_ARGS[@]}" \
    --seed "$seed" \
    > "${STDOUT_DIR}/fedsp_pg_ppo_codex_seed_${seed}.log" 2>&1 &
done

wait

python codex/summarize_frl_evals.py "${LOG_ROOT}/train_logs"
