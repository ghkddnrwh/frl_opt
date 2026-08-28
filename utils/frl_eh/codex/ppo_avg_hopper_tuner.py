from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "logs" / "codex" / "ppo_avg_hopper_tuning"
ENV_ID = "PerturbHopper-v4"
CLIENT_NOISE_VALUES = "[-0.5,-0.25,0.0,0.25,0.5]"
RELU_256 = "dict(log_std_init={log_std}, ortho_init=False, activation_fn=nn.ReLU, net_arch=dict(pi=[256, 256], vf=[256, 256]))"
TANH_256 = "dict(log_std_init={log_std}, ortho_init=False, activation_fn=nn.Tanh, net_arch=dict(pi=[256, 256], vf=[256, 256]))"


@dataclass(frozen=True)
class Candidate:
    label: str
    learning_rate: str
    batch_size: int
    clip_range: str
    n_epochs: int
    gae_lambda: float
    gamma: float
    ent_coef: float
    max_grad_norm: float
    vf_coef: float
    log_std_init: float
    target_kl: str = "None"
    server_update_weight: float = 1.0
    critic_sync_mode: str = "fedavg"
    vecnormalize_sync_mode: str = "obs_reward"
    activation: str = "relu"
    n_steps: int = 512
    local_steps: int = 512
    n_envs: int = 1

    @property
    def policy_kwargs(self) -> str:
        template = TANH_256 if self.activation == "tanh" else RELU_256
        return template.format(log_std=fmt_float(self.log_std_init))


@dataclass
class TrialResult:
    label: str
    stage: str
    noise_type: str
    seed: int
    status: str
    returncode: int
    duration_sec: float
    n_timesteps: int
    final_local_mean: float | None
    final_local_min: float | None
    final_nominal_mean: float | None
    best_local_mean: float | None
    best_local_min: float | None
    score: float | None
    eval_count: int
    run_dir: str | None
    log_path: str
    error: str | None


def fmt_float(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.8g}"


def eval_arg(value: str) -> str:
    try:
        float(value)
    except ValueError:
        if value in {"None", "True", "False"}:
            return value
        return repr(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PPOAvg Hopper hyperparameter candidates.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--stage", choices=("screen", "validate"), default="screen")
    parser.add_argument("--timesteps", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument("--noise-types", nargs="+", default=["gravity", "friction"])
    parser.add_argument("--candidates", nargs="+", default=None, help="Candidate labels to run.")
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--num-threads", type=int, default=1)
    parser.add_argument("--eval-local-episodes", type=int, default=3)
    parser.add_argument("--eval-nominal-episodes", type=int, default=3)
    parser.add_argument("--eval-round-freq", type=int, default=10**9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--train-verbose", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_candidates() -> list[Candidate]:
    original = Candidate(
        label="orig_relu_lr9p8e5_bs64_clip20",
        learning_rate="9.80828e-05",
        batch_size=64,
        clip_range="0.2",
        n_epochs=5,
        gae_lambda=0.99,
        gamma=0.99,
        ent_coef=0.00229519,
        max_grad_norm=0.7,
        vf_coef=0.835671,
        log_std_init=-2,
    )
    return [
        original,
        Candidate(
            label="orig_relu_targetkl015",
            learning_rate=original.learning_rate,
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.00229519,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            target_kl="0.015",
        ),
        Candidate(
            label="orig_relu_targetkl03",
            learning_rate=original.learning_rate,
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.00229519,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            target_kl="0.03",
        ),
        Candidate(
            label="env1_rlzoo_lin3e4_bs512_cliplin10_kl015_std0",
            learning_rate="lin_3e-4",
            batch_size=512,
            clip_range="lin_0.1",
            n_epochs=5,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_rlzoo_lin3e4_bs128_cliplin10_kl015_std0",
            learning_rate="lin_3e-4",
            batch_size=128,
            clip_range="lin_0.1",
            n_epochs=5,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_rlzoo_const3e4_bs512_clip10_kl015_std0",
            learning_rate="3e-4",
            batch_size=512,
            clip_range="0.1",
            n_epochs=5,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_std0_lr3e4_bs256_ep10_clip20",
            learning_rate="3e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
        ),
        Candidate(
            label="env1_std0_lr2e4_bs128_ep10_clip20",
            learning_rate="2e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
        ),
        Candidate(
            label="env1_std0_lin2e4_bs128_ep10_clip20",
            learning_rate="lin_2e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
        ),
        Candidate(
            label="env1_std0_lr2e4_bs128_ep10_clip20_kl015",
            learning_rate="2e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_std0_lr2e4_bs128_ep10_clip15_kl015",
            learning_rate="2e-4",
            batch_size=128,
            clip_range="0.15",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_std0_lr2e4_bs128_ep10_clip10_kl015",
            learning_rate="2e-4",
            batch_size=128,
            clip_range="0.1",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_std0_lr15e5_bs128_ep10_clip20",
            learning_rate="1.5e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
        ),
        Candidate(
            label="env1_std0_lr1e4_bs128_ep10_clip20",
            learning_rate="1e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
        ),
        Candidate(
            label="env1_std0_lr2e4_bs64_ep10_clip20_kl015",
            learning_rate="2e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_std0_lr2e4_bs256_ep5_clip20_kl015",
            learning_rate="2e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
        ),
        Candidate(
            label="env1_orig_gamma999_bs32",
            learning_rate="9.80828e-05",
            batch_size=32,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.999,
            ent_coef=0.00229519,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
        ),
        Candidate(
            label="env1_orig_lr15e5_bs64_std2",
            learning_rate="1.5e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.00229519,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
        ),
        Candidate(
            label="orig_relu_sw075",
            learning_rate=original.learning_rate,
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.00229519,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            server_update_weight=0.75,
        ),
        Candidate(
            label="orig_relu_sw05",
            learning_rate=original.learning_rate,
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.00229519,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            server_update_weight=0.5,
        ),
        Candidate(
            label="relu_lr5e5_bs64_clip20",
            learning_rate="5e-05",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
        ),
        Candidate(
            label="relu_lin1e4_bs64_clip20",
            learning_rate="lin_1e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
        ),
        Candidate(
            label="relu_lr1e4_bs128_clip20",
            learning_rate="1e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
        ),
        Candidate(
            label="relu_lr1e4_bs64_clip15_gae97",
            learning_rate="1e-4",
            batch_size=64,
            clip_range="0.15",
            n_epochs=5,
            gae_lambda=0.97,
            gamma=0.99,
            ent_coef=0.001,
            max_grad_norm=0.7,
            vf_coef=0.7,
            log_std_init=-2,
        ),
        Candidate(
            label="relu_lr2e4_bs128_clip10_std1",
            learning_rate="2e-4",
            batch_size=128,
            clip_range="0.1",
            n_epochs=3,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=-1,
            target_kl="0.03",
        ),
        Candidate(
            label="relu_env8_steps512_local4096_lr3e4_bs256_ep10_clip20_std0",
            learning_rate="3e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            n_envs=8,
            n_steps=512,
            local_steps=4096,
        ),
        Candidate(
            label="relu_env8_steps512_local4096_lin3e4_bs512_ep5_cliplin10_std0",
            learning_rate="lin_3e-4",
            batch_size=512,
            clip_range="lin_0.1",
            n_epochs=5,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            target_kl="0.015",
            n_envs=8,
            n_steps=512,
            local_steps=4096,
        ),
        Candidate(
            label="relu_env8_steps512_local4096_lr1e4_bs256_clip15_gae97",
            learning_rate="1e-4",
            batch_size=256,
            clip_range="0.15",
            n_epochs=5,
            gae_lambda=0.97,
            gamma=0.99,
            ent_coef=0.001,
            max_grad_norm=0.7,
            vf_coef=0.7,
            log_std_init=-2,
            n_envs=8,
            n_steps=512,
            local_steps=4096,
        ),
        Candidate(
            label="relu_env8_steps512_local4096_lr2e4_bs256_clip10_std1",
            learning_rate="2e-4",
            batch_size=256,
            clip_range="0.1",
            n_epochs=3,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=-1,
            target_kl="0.03",
            n_envs=8,
            n_steps=512,
            local_steps=4096,
        ),
        Candidate(
            label="relu_env4_steps512_local2048_lr3e4_bs256_ep10_clip20_std0",
            learning_rate="3e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            n_envs=4,
            n_steps=512,
            local_steps=2048,
        ),
        Candidate(
            label="relu_env8_steps256_local2048_lr3e4_bs256_ep10_clip20_std0",
            learning_rate="3e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            n_envs=8,
            n_steps=256,
            local_steps=2048,
        ),
        Candidate(
            label="relu_env4_steps128_local512_lr3e4_bs128_ep10_clip20_std0",
            learning_rate="3e-4",
            batch_size=128,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            n_envs=4,
            n_steps=128,
            local_steps=512,
        ),
        Candidate(
            label="relu_env8_steps64_local512_lr3e4_bs256_ep10_clip20_std0",
            learning_rate="3e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            n_envs=8,
            n_steps=64,
            local_steps=512,
        ),
        Candidate(
            label="tanh_env8_steps512_local4096_lr3e4_bs256_ep10_clip20_std0",
            learning_rate="3e-4",
            batch_size=256,
            clip_range="0.2",
            n_epochs=10,
            gae_lambda=0.95,
            gamma=0.99,
            ent_coef=0.0,
            max_grad_norm=0.5,
            vf_coef=0.5,
            log_std_init=0,
            activation="tanh",
            n_envs=8,
            n_steps=512,
            local_steps=4096,
        ),
        Candidate(
            label="relu_lr1e4_obsnorm_only",
            learning_rate="1e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            vecnormalize_sync_mode="obs",
        ),
        Candidate(
            label="relu_lr1e4_localcritic",
            learning_rate="1e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            critic_sync_mode="local",
        ),
        Candidate(
            label="relu_lr1e4_obs_localcritic",
            learning_rate="1e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            critic_sync_mode="local",
            vecnormalize_sync_mode="obs",
        ),
        Candidate(
            label="tanh_lr1e4_bs64_clip20",
            learning_rate="1e-4",
            batch_size=64,
            clip_range="0.2",
            n_epochs=5,
            gae_lambda=0.99,
            gamma=0.99,
            ent_coef=0.002,
            max_grad_norm=0.7,
            vf_coef=0.835671,
            log_std_init=-2,
            activation="tanh",
        ),
    ]


def candidate_by_label() -> dict[str, Candidate]:
    return {candidate.label: candidate for candidate in build_candidates()}


def trial_dir(output_root: Path, stage: str, candidate: Candidate, noise_type: str, seed: int, n_timesteps: int) -> Path:
    return output_root / stage / f"steps_{n_timesteps}" / candidate.label / noise_type / f"seed_{seed}"


def find_run_dir(log_root: Path) -> Path | None:
    algo_dir = log_root / "ppo_avg"
    if not algo_dir.exists():
        return None
    run_dirs = sorted(path for path in algo_dir.iterdir() if path.is_dir())
    return run_dirs[-1] if run_dirs else None


def optional_float(value: np.ndarray) -> float | None:
    if value.size == 0:
        return None
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return None
    return float(finite[-1])


def max_optional_float(value: np.ndarray) -> float | None:
    if value.size == 0:
        return None
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return None
    return float(np.max(finite))


def load_metrics(run_dir: Path) -> tuple[float | None, float | None, float | None, float | None, float | None, int]:
    eval_path = run_dir / "evaluations.npz"
    if not eval_path.exists():
        return None, None, None, None, None, 0
    with np.load(eval_path, allow_pickle=True) as data:
        local = np.asarray(data.get("local_mean_across_clients", []), dtype=np.float64)
        local_min = np.asarray(data.get("local_min_across_clients", []), dtype=np.float64)
        nominal = np.asarray(data.get("nominal_mean_across_clients", []), dtype=np.float64)
    return (
        optional_float(local),
        optional_float(local_min),
        optional_float(nominal),
        max_optional_float(local),
        max_optional_float(local_min),
        int(local.size),
    )


def compute_score(
    final_local_mean: float | None,
    final_local_min: float | None,
    final_nominal_mean: float | None,
) -> float | None:
    values = [value for value in (final_local_mean, final_local_min, final_nominal_mean) if value is not None]
    if not values:
        return None
    # Robustness first: the worst client gets half the score budget.
    local_mean = final_local_mean if final_local_mean is not None else 0.0
    local_min = final_local_min if final_local_min is not None else 0.0
    nominal = final_nominal_mean if final_nominal_mean is not None else local_mean
    return float(0.35 * local_mean + 0.50 * local_min + 0.15 * nominal)


def tail_text(path: Path, max_lines: int = 80) -> str:
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(lines[-max_lines:])


def build_command(
    candidate: Candidate,
    args: argparse.Namespace,
    log_root: Path,
    noise_type: str,
    seed: int,
    n_timesteps: int,
) -> list[str]:
    return [
        args.python,
        "-m",
        "rl_zoo3.train",
        "--algo",
        "ppo_avg",
        "--env",
        ENV_ID,
        "--frl",
        "--conf-file",
        str(REPO_ROOT / "hyperparams" / "ppo_avg.yml"),
        "--log-folder",
        str(log_root),
        "--seed",
        str(seed),
        "--num-threads",
        str(args.num_threads),
        "--device",
        args.device,
        "--vec-env",
        "dummy",
        "--eval-freq",
        "-1",
        "--save-freq",
        "-1",
        "--log-interval",
        "-1",
        "--verbose",
        str(args.train_verbose),
        "--log-wandb",
        "False",
        "-n",
        str(n_timesteps),
        "--hyperparams",
        f"n_timesteps:{n_timesteps}",
        "num_clients:5",
        f"local_steps:{candidate.local_steps}",
        f"server_update_weight:{candidate.server_update_weight}",
        f'n_envs:{candidate.n_envs}',
        f"n_steps:{candidate.n_steps}",
        f"batch_size:{candidate.batch_size}",
        "normalize:True",
        f"learning_rate:{eval_arg(candidate.learning_rate)}",
        f"ent_coef:{candidate.ent_coef}",
        f"clip_range:{eval_arg(candidate.clip_range)}",
        f"n_epochs:{candidate.n_epochs}",
        f"gae_lambda:{candidate.gae_lambda}",
        f"gamma:{candidate.gamma}",
        f"max_grad_norm:{candidate.max_grad_norm}",
        f"vf_coef:{candidate.vf_coef}",
        f"target_kl:{candidate.target_kl}",
        f'policy_kwargs:"{candidate.policy_kwargs}"',
        f'perturb_noise_type:"{noise_type}"',
        f"client_noise_values:{CLIENT_NOISE_VALUES}",
        f"critic_sync_mode:\"{candidate.critic_sync_mode}\"",
        f"vecnormalize_sync_mode:\"{candidate.vecnormalize_sync_mode}\"",
        f"eval_round_freq:{args.eval_round_freq}",
        f"eval_local_episodes:{args.eval_local_episodes}",
        f"eval_nominal_episodes:{args.eval_nominal_episodes}",
    ]


def run_trial(candidate: Candidate, args: argparse.Namespace, noise_type: str, seed: int, n_timesteps: int) -> TrialResult:
    current_trial_dir = trial_dir(args.output_root, args.stage, candidate, noise_type, seed, n_timesteps)
    log_root = current_trial_dir / "train_logs"
    log_path = current_trial_dir / "train.log"
    current_trial_dir.mkdir(parents=True, exist_ok=True)
    log_root.mkdir(parents=True, exist_ok=True)

    existing_run_dir = find_run_dir(log_root)
    if existing_run_dir is not None and (existing_run_dir / "evaluations.npz").exists():
        final_local, final_min, final_nominal, best_local, best_min, eval_count = load_metrics(existing_run_dir)
        score = compute_score(final_local, final_min, final_nominal)
        return TrialResult(
            label=candidate.label,
            stage=args.stage,
            noise_type=noise_type,
            seed=seed,
            status="cached",
            returncode=0,
            duration_sec=0.0,
            n_timesteps=n_timesteps,
            final_local_mean=final_local,
            final_local_min=final_min,
            final_nominal_mean=final_nominal,
            best_local_mean=best_local,
            best_local_min=best_min,
            score=score,
            eval_count=eval_count,
            run_dir=str(existing_run_dir),
            log_path=str(log_path),
            error=None,
        )

    command = build_command(candidate, args, log_root, noise_type, seed, n_timesteps)
    if args.dry_run:
        log_path.write_text(" ".join(command) + "\n", encoding="utf-8")
        return TrialResult(
            label=candidate.label,
            stage=args.stage,
            noise_type=noise_type,
            seed=seed,
            status="dry-run",
            returncode=0,
            duration_sec=0.0,
            n_timesteps=n_timesteps,
            final_local_mean=None,
            final_local_min=None,
            final_nominal_mean=None,
            best_local_mean=None,
            best_local_min=None,
            score=None,
            eval_count=0,
            run_dir=None,
            log_path=str(log_path),
            error=None,
        )

    env = os.environ.copy()
    env["WANDB_MODE"] = "disabled"
    env["MPLCONFIGDIR"] = str((args.output_root / ".matplotlib").resolve())
    env["XDG_CACHE_HOME"] = str((args.output_root / ".cache").resolve())
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    start = time.time()
    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write("COMMAND:\n")
        log_file.write(" ".join(command))
        log_file.write("\n\nOUTPUT:\n")
        process = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    duration = time.time() - start

    run_dir = find_run_dir(log_root)
    final_local, final_min, final_nominal, best_local, best_min, eval_count = (
        load_metrics(run_dir) if run_dir is not None else (None, None, None, None, None, 0)
    )
    score = compute_score(final_local, final_min, final_nominal)
    status = "ok" if process.returncode == 0 and score is not None else "failed"
    return TrialResult(
        label=candidate.label,
        stage=args.stage,
        noise_type=noise_type,
        seed=seed,
        status=status,
        returncode=process.returncode,
        duration_sec=duration,
        n_timesteps=n_timesteps,
        final_local_mean=final_local,
        final_local_min=final_min,
        final_nominal_mean=final_nominal,
        best_local_mean=best_local,
        best_local_min=best_min,
        score=score,
        eval_count=eval_count,
        run_dir=str(run_dir) if run_dir is not None else None,
        log_path=str(log_path),
        error=None if status == "ok" else tail_text(log_path),
    )


def write_rows(path: Path, rows: list[TrialResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0]).keys()) if rows else [])
        if rows:
            writer.writeheader()
            for row in rows:
                writer.writerow(asdict(row))


def mean_or_nan(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else float("nan")


def std_or_nan(values: list[float | None]) -> float:
    finite = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    return float(np.std(finite)) if finite else float("nan")


def summarize(rows: list[TrialResult]) -> list[dict[str, Any]]:
    groups: dict[str, list[TrialResult]] = {}
    for row in rows:
        if row.status not in {"ok", "cached"}:
            continue
        groups.setdefault(row.label, []).append(row)
    summary: list[dict[str, Any]] = []
    for label, items in sorted(groups.items()):
        scores = [item.score for item in items]
        locals_ = [item.final_local_mean for item in items]
        local_mins = [item.final_local_min for item in items]
        nominals = [item.final_nominal_mean for item in items]
        by_noise: dict[str, list[TrialResult]] = {}
        for item in items:
            by_noise.setdefault(item.noise_type, []).append(item)
        summary.append(
            {
                "label": label,
                "n": len(items),
                "noises": ",".join(sorted(by_noise)),
                "seeds": ",".join(map(str, sorted({item.seed for item in items}))),
                "score_mean": mean_or_nan(scores),
                "score_std": std_or_nan(scores),
                "score_min": min((score for score in scores if score is not None), default=float("nan")),
                "final_local_mean": mean_or_nan(locals_),
                "final_local_std": std_or_nan(locals_),
                "final_local_min_over_trials": min(
                    (value for value in locals_ if value is not None),
                    default=float("nan"),
                ),
                "final_worst_client_mean": mean_or_nan(local_mins),
                "final_nominal_mean": mean_or_nan(nominals),
                "noise_score_means": {
                    noise: mean_or_nan([item.score for item in noise_items])
                    for noise, noise_items in sorted(by_noise.items())
                },
            }
        )
    summary.sort(
        key=lambda item: (
            -float(item["score_min"]) if math.isfinite(float(item["score_min"])) else float("inf"),
            -float(item["score_mean"]) if math.isfinite(float(item["score_mean"])) else float("inf"),
        )
    )
    return summary


def write_summary(output_root: Path, stage: str, n_timesteps: int, rows: list[TrialResult]) -> None:
    summary = summarize(rows)
    summary_path = output_root / stage / f"steps_{n_timesteps}" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    md_lines = [f"# PPOAvg Hopper {stage} summary", ""]
    md_lines.append("| rank | label | n | score mean | score min | local mean | worst-client mean | nominal mean | noise score means |")
    md_lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")
    for idx, item in enumerate(summary, start=1):
        noise_scores = ", ".join(f"{k}:{v:.1f}" for k, v in item["noise_score_means"].items())
        md_lines.append(
            f"| {idx} | `{item['label']}` | {item['n']} | {item['score_mean']:.1f} | "
            f"{item['score_min']:.1f} | {item['final_local_mean']:.1f} | "
            f"{item['final_worst_client_mean']:.1f} | {item['final_nominal_mean']:.1f} | {noise_scores} |"
        )
    (summary_path.parent / "summary.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")


def print_result(row: TrialResult) -> None:
    score = "n/a" if row.score is None else f"{row.score:.1f}"
    local = "n/a" if row.final_local_mean is None else f"{row.final_local_mean:.1f}"
    worst = "n/a" if row.final_local_min is None else f"{row.final_local_min:.1f}"
    print(
        f"[{row.status}] {row.stage} {row.label} {row.noise_type} seed={row.seed} "
        f"score={score} local={local} worst={worst} time={row.duration_sec:.1f}s",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    candidates = build_candidates()
    if args.candidates:
        labels = candidate_by_label()
        missing = sorted(set(args.candidates) - set(labels))
        if missing:
            raise ValueError(f"Unknown candidate labels: {missing}")
        candidates = [labels[label] for label in args.candidates]

    if args.timesteps is None:
        n_timesteps = 500_000 if args.stage == "screen" else 2_000_000
    else:
        n_timesteps = args.timesteps
    seeds = args.seeds if args.seeds is not None else ([1] if args.stage == "screen" else [1, 2, 3])

    print(f"Output root: {args.output_root}")
    print(f"Stage: {args.stage}, timesteps={n_timesteps}, seeds={seeds}, noises={args.noise_types}")
    print(f"Candidates: {[candidate.label for candidate in candidates]}")

    trial_specs = [
        (candidate, noise_type, seed)
        for candidate in candidates
        for noise_type in args.noise_types
        for seed in seeds
    ]

    rows: list[TrialResult] = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [
            executor.submit(run_trial, candidate, args, noise_type, seed, n_timesteps)
            for candidate, noise_type, seed in trial_specs
        ]
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            print_result(row)

    rows.sort(key=lambda row: (row.label, row.noise_type, row.seed))
    result_path = args.output_root / args.stage / f"steps_{n_timesteps}" / "results.csv"
    if rows:
        write_rows(result_path, rows)
        write_summary(args.output_root, args.stage, n_timesteps, rows)
        print(f"Results: {result_path}")
        print(f"Summary: {result_path.parent / 'summary.md'}")


if __name__ == "__main__":
    main()
