# Codex FedSP-PG PPO Notes

## Diagnosis

- The existing `fedsp_pg_ppo_paper_aligned` server actor update happens once
  per federated round. With 5 clients, `local_steps=1024` gives only about 196
  server actor updates per 1e6 aggregated timesteps, and `local_steps=4096`
  gives only about 49.
- Inside each client, multiple rollout gradients are averaged before upload, so
  longer local collection does not compensate with a larger actor signal.
- The paper-aligned raw-gradient path can eventually learn with many rounds
  (for example 1e7 timesteps in the existing logs), but it is weak at shorter
  budgets.
- `ppo_avg` does not directly optimize local-min performance; it averages full
  policies. It can still become a strong baseline because every client performs
  full local PPO actor/critic updates before averaging.

## New Implementation

Files added under `codex/` only:

- `codex/fedsp_pg_ppo_codex.py`
- `codex/train_codex.py`
- `codex/fedsp_pg_ppo_codex.yml`
- `codex/run_pendulum_gravity_compare.sh`
- `codex/summarize_frl_evals.py`

The default `fedsp_pg_ppo_codex` implementation uses:

- `fedsp_rollout_steps=32` and `max_local_steps_per_round=128`, so large
  command-line `local_steps` values do not collapse the server update count.
- A soft worst-client dual update using EMA-smoothed rollout returns.
- A hybrid actor update: clients estimate a temporary actor-only PPO delta,
  restore their actor, upload the delta, and the server applies a lambda-weighted
  delta with `server_delta_scale=0.5`.

## PerturbPendulum-v1 Gravity Results

All runs used gravity perturbation range `0.5`, 5 clients, seeds 1-3, 400k
aggregated timesteps, 5 evaluation episodes.

| Run | final local_min mean | best local_min mean | final local_mean mean | final nominal mean |
| --- | ---: | ---: | ---: | ---: |
| ppo_avg | -994.51 | -861.90 | -638.30 | -480.92 |
| codex raw-gradient | -1208.77 | -1048.86 | -1070.14 | -1004.51 |
| codex delta, scale 1.0 | -902.42 | -548.21 | -561.64 | -457.06 |
| codex delta, scale 0.5 | -536.69 | -382.04 | -395.22 | -304.92 |
| codex delta, smooth 0.75 | -778.98 | -385.52 | -533.37 | -365.96 |

The selected default is `codex delta, scale 0.5`.

## Usage

```bash
python codex/train_codex.py \
  --algo fedsp_pg_ppo_codex \
  --env PerturbPendulum-v1 \
  --conf-file codex/fedsp_pg_ppo_codex.yml \
  --log-folder logs/codex/manual \
  --hyperparams 'perturb_noise_type:"gravity"' perturb_noise_range:0.5 \
  --seed 1 --frl --device cpu
```

The wrapper registers the new algorithm at runtime, so `rl_zoo3/utils.py` and
the original FedSP source files do not need to be edited.

