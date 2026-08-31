from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.ppo import PPO

from rl_zoo3.algorithms.federate.common.federated_algorithm import (
    FederatedAlgorithmMixin,
    FederatedModules,
    FederatedPayload,
)

VecNormalizeState = dict[str, Any]
RunningMeanStdState = dict[str, Any]


class FedSVRPGM(FederatedAlgorithmMixin, PPO):
    """PPO-based practical FedSVRPG-M implemented directly on top of SB3 PPO.

    This class does not inherit from FedSPPGPPO.  It uses SB3's PPO as the RL
    backbone and implements only the federated FedSVRPG-M-style communication
    logic locally in this file.

    This version uses an actual local PPO update by default.  With
    ``fedsvrpg_local_update_mode="ppo_update"`` and ``momentum_beta=1.0``, a
    client performs the same local PPO optimizer update used by SB3 PPO and
    uploads the resulting actor displacement.  The server still follows the
    FedSVRPG-M paper-style update: it reconstructs ``u_{r+1}`` by dividing the
    averaged displacement by ``local_lr * num_local_updates`` and then applies
    ``server_update_weight`` as lambda.  Therefore
    ``server_update_weight = local_lr * K`` recovers PPOAvg-style actor
    parameter averaging for beta=1.  Critic synchronization is configurable
    through ``critic_sync_mode``: ``"local"`` preserves the original
    FedSVRPG-M behavior, while ``"fedavg"`` averages the critic/non-actor
    policy state across clients and clears the full PPO optimizer state after
    each global synchronization, matching PPOAvg's optimizer-reset convention.

    For ``momentum_beta < 1.0``, the local PPO actor displacement is blended
    with a FedSVRPG-M-style correction displacement.  The correction uses the
    selected gradient estimator family
    ``fedsvrpg_gradient_type in {"ppo_clip", "score"}``:

        delta_local = beta * delta_ppo
                      + (1 - beta) * local_lr * [u_r
                          + g(theta_{r,k}; B_{r,k})
                          - w(B_{r,k}) g(theta_{r-1}; B_{r,k})].

    Here w(B_{r,k}) is a trajectory-level importance-sampling weight computed
    from log pi_{theta_{r-1}} - log pi_{theta_{r,k}} on the rollout.  Setting
    ``importance_ratio_clip=None`` leaves this IS weight unclipped; a positive
    value clips it to [1 / clip, clip], which is a practical biased variant.

    Set ``fedsvrpg_local_update_mode="gradient_update"`` to recover the older
    gradient-level local update path that directly applies the FedSVRPG-M
    direction instead of running a true PPO optimizer update.

    ``actor_gradient_mode`` controls how the minibatch actor-gradient deltas
    inside one rollout are aggregated.  ``mean`` averages over all
    epoch/minibatch gradients and is the recommended, scale-stable default;
    ``cumulative`` reproduces the previous behavior that sums every
    epoch/minibatch delta.

    The implementation uses the following scale convention.  The server always
    computes ``u_{r+1}`` by dividing averaged client actor displacements by
    ``local_lr * num_local_updates``.  In ``gradient_update`` mode this is the
    paper's ``eta * K`` exactly.  In ``ppo_update`` mode the local displacement
    is produced by SB3 PPO's optimizer, so ``local_lr`` is mainly the scale used
    to interpret the displacement as a FedSVRPG-M direction and to choose the
    matching lambda; setting ``server_update_weight = local_lr * K`` makes the
    beta=1 server update equal to parameter averaging.

    The auxiliary FedSVRPG-M correction still uses an actor-gradient estimator
    selected by ``fedsvrpg_gradient_type``.  ``actor_n_epochs`` and
    ``actor_batch_size`` control that correction-gradient estimator only; the
    real local PPO update uses SB3's usual ``n_epochs`` and ``batch_size``.

    The initial server momentum/anchor u_0 is estimated once in
    ``prepare_federated_training`` by averaging ``init_grad_episodes`` initial
    rollout gradient directions from each client.  The actor parameters are not
    mutated while estimating the current and previous directions; the real
    local actor update is applied once through u_{r,k}.  The critic/value branch
    remains client-local when ``critic_sync_mode="local"`` and is federated by
    parameter averaging when ``critic_sync_mode="fedavg"``.  During each local
    rollout update, the actor gradient uses the rollout advantages computed at
    collection time, matching PPO's
    fixed-advantage convention; the critic is updated afterward on the same
    rollout buffer without refreshing actor advantages.
    """

    federated_actor_module_name = "policy"
    federated_critic_module_name = "policy"

    federated_manager_keys: tuple[str, ...] = (
        "num_clients",
        "local_steps",
        "server_update_weight",
        "perturb_noise_type",
        "perturb_noise_range",
        "eval_local_episodes",
        "eval_nominal_episodes",
        "eval_round_freq",
        "eval_deterministic",
        "log_wandb",
        "local_lr",
        "momentum_beta",
        "init_grad_episodes",
        "max_update_norm",
        "importance_ratio_clip",
        "actor_gradient_mode",
        "actor_n_epochs",
        "actor_batch_size",
        "fedsvrpg_gradient_type",
        "fedsvrpg_local_update_mode",
        "vecnormalize_sync_mode",
        "critic_sync_mode",
    )
    valid_critic_sync_modes: tuple[str, ...] = ("local", "fedavg")
    valid_vecnormalize_sync_modes: tuple[str, ...] = ("none", "obs", "reward", "obs_reward")
    valid_actor_gradient_modes: tuple[str, ...] = ("mean", "cumulative")
    valid_fedsvrpg_gradient_types: tuple[str, ...] = ("ppo_clip", "score")
    valid_fedsvrpg_local_update_modes: tuple[str, ...] = ("ppo_update", "gradient_update")

    # Last global VecNormalize state produced by the server.  This mirrors
    # PPOAvg and prevents double-counting shared RunningMeanStd history when
    # aggregating client normalizer updates over multiple communication rounds.
    _last_global_vecnormalize_state: VecNormalizeState | None = None

    def __init__(self, *args, **kwargs):
        self.critic_sync_mode = self._normalize_critic_sync_mode(
            kwargs.pop("critic_sync_mode", "local")
        )
        self.vecnormalize_sync_mode = self._normalize_vecnormalize_sync_mode(
            kwargs.pop("vecnormalize_sync_mode", "obs_reward")
        )
        self.local_lr = float(kwargs.pop("local_lr", 1.0))
        self.momentum_beta = float(kwargs.pop("momentum_beta", 0.9))
        self.init_grad_episodes = int(kwargs.pop("init_grad_episodes", 1))
        self.max_update_norm = kwargs.pop("max_update_norm", None)
        self.actor_gradient_mode = self._normalize_actor_gradient_mode(
            kwargs.pop("actor_gradient_mode", "cumulative")
        )
        self.actor_n_epochs = int(kwargs.pop("actor_n_epochs", 1))
        actor_batch_size = kwargs.pop("actor_batch_size", None)
        if actor_batch_size is None:
            self.actor_batch_size = None
        elif isinstance(actor_batch_size, str) and actor_batch_size.strip().lower() in {
            "none",
            "full",
            "full_rollout",
            "rollout",
            "all",
        }:
            self.actor_batch_size = None
        else:
            self.actor_batch_size = int(actor_batch_size)
        self.fedsvrpg_gradient_type = self._normalize_fedsvrpg_gradient_type(
            kwargs.pop("fedsvrpg_gradient_type", "ppo_clip")
        )
        self.fedsvrpg_local_update_mode = self._normalize_fedsvrpg_local_update_mode(
            kwargs.pop("fedsvrpg_local_update_mode", "ppo_update")
        )
        self._federated_progress_lock: float | None = None

        # Kept for compatibility with previous config files.
        kwargs.pop("local_iteration_horizon", None)
        self.importance_ratio_clip = kwargs.pop("importance_ratio_clip", 20.0)

        # Remove federated-manager-only keys before calling SB3 PPO.__init__().
        # This is necessary because we inherit directly from PPO now.
        for key in self.federated_manager_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        if self.local_lr <= 0.0:
            raise ValueError("local_lr must be positive")
        if not (0.0 <= self.momentum_beta <= 1.0):
            raise ValueError("momentum_beta must be in [0, 1]")
        if self.init_grad_episodes < 1:
            raise ValueError("init_grad_episodes must be >= 1")
        if self.actor_n_epochs < 1:
            raise ValueError("actor_n_epochs must be >= 1")
        if self.actor_batch_size is not None and self.actor_batch_size < 1:
            raise ValueError("actor_batch_size must be positive, None, or 'full_rollout'")
        if self.max_update_norm is not None and float(self.max_update_norm) <= 0.0:
            raise ValueError("max_update_norm must be positive when provided")
        if self.importance_ratio_clip is not None and float(self.importance_ratio_clip) < 1.0:
            raise ValueError("importance_ratio_clip must be >= 1.0 or None")

        initial_actor = self._get_actor_state()
        self._fedsvrpg_round_start_actor_state = self._clone_modules(initial_actor)
        self._fedsvrpg_prev_global_actor_state = self._clone_modules(initial_actor)
        self._fedsvrpg_server_direction = self._zero_like_actor_state()
        self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
        self._fedsvrpg_last_return = 0.0
        self._fedsvrpg_last_num_local_updates = 0
        self._last_federated_metrics: dict[str, float] = {}

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        return True

    @classmethod
    def reset_federated_state(cls) -> None:
        cls._last_global_vecnormalize_state = None

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        if self._federated_progress_lock is None:
            super()._update_current_progress_remaining(num_timesteps, total_timesteps)
            return
        self._current_progress_remaining = self._federated_progress_lock

    @classmethod
    def _normalize_critic_sync_mode(cls, mode: str) -> str:
        """Normalize critic synchronization mode aliases.

        Modes:
          - local: keep critic/value state client-local (original FedSVRPG-M behavior).
          - fedavg: average critic/non-actor policy state across clients every round.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "none": "local",
            "no_sync": "local",
            "local_only": "local",
            "client_local": "local",
            "private": "local",
            "avg": "fedavg",
            "average": "fedavg",
            "global": "fedavg",
            "server": "fedavg",
            "sync": "fedavg",
            "synchronized": "fedavg",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_critic_sync_modes:
            raise ValueError(
                f"Unsupported critic_sync_mode={mode!r}. "
                f"Choose one of {cls.valid_critic_sync_modes}."
            )
        return normalized

    @classmethod
    def _normalize_vecnormalize_sync_mode(cls, mode: str) -> str:
        """Normalize VecNormalize synchronization mode aliases.

        Modes:
          - none: do not upload, aggregate, or apply VecNormalize statistics.
          - obs: synchronize only observation RunningMeanStd statistics.
          - reward: synchronize only reward/return RunningMeanStd statistics.
          - obs_reward: synchronize both observation and reward statistics.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "false": "none",
            "off": "none",
            "no": "none",
            "0": "none",
            "disable": "none",
            "disabled": "none",
            "true": "obs_reward",
            "on": "obs_reward",
            "yes": "obs_reward",
            "1": "obs_reward",
            "all": "obs_reward",
            "full": "obs_reward",
            "both": "obs_reward",
            "obs_only": "obs",
            "observation": "obs",
            "observations": "obs",
            "state": "obs",
            "states": "obs",
            "ret": "reward",
            "return": "reward",
            "returns": "reward",
            "reward_only": "reward",
            "rewards": "reward",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_vecnormalize_sync_modes:
            raise ValueError(
                f"Unsupported vecnormalize_sync_mode={mode!r}. "
                f"Choose one of {cls.valid_vecnormalize_sync_modes}."
            )
        return normalized

    @classmethod
    def _normalize_actor_gradient_mode(cls, mode: str) -> str:
        """Normalize actor-gradient aggregation mode aliases.

        Modes:
          - mean: average over all epoch/minibatch actor-gradient deltas.
          - cumulative: sum all epoch/minibatch actor-gradient deltas.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "avg": "mean",
            "average": "mean",
            "averaged": "mean",
            "normalized": "mean",
            "scale_invariant": "mean",
            "sum": "cumulative",
            "summed": "cumulative",
            "accumulate": "cumulative",
            "accumulated": "cumulative",
            "old": "cumulative",
            "legacy": "cumulative",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_actor_gradient_modes:
            raise ValueError(
                f"Unsupported actor_gradient_mode={mode!r}. "
                f"Choose one of {cls.valid_actor_gradient_modes}."
            )
        return normalized



    @classmethod
    def _normalize_fedsvrpg_gradient_type(cls, mode: str) -> str:
        """Normalize FedSVRPG-M actor-gradient estimator type.

        Modes:
          - ppo_clip: use PPO's clipped actor surrogate gradient.
          - score: use the raw score-function surrogate -A log pi.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "ppo": "ppo_clip",
            "clipped": "ppo_clip",
            "clip": "ppo_clip",
            "ppo_clipped": "ppo_clip",
            "ppo_surrogate": "ppo_clip",
            "score_function": "score",
            "score_func": "score",
            "reinforce": "score",
            "gpomdp": "score",
            "raw": "score",
            "vanilla_pg": "score",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_fedsvrpg_gradient_types:
            raise ValueError(
                f"Unsupported fedsvrpg_gradient_type={mode!r}. "
                f"Choose one of {cls.valid_fedsvrpg_gradient_types}."
            )
        return normalized

    @classmethod
    def _normalize_fedsvrpg_local_update_mode(cls, mode: str) -> str:
        """Normalize the practical local-update backend.

        Modes:
          - ppo_update: run a real SB3 PPO local optimizer update first.
          - gradient_update: apply the FedSVRPG-M gradient direction directly.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "ppo": "ppo_update",
            "ppoavg": "ppo_update",
            "ppo_avg": "ppo_update",
            "local_ppo": "ppo_update",
            "local_ppo_update": "ppo_update",
            "true_ppo": "ppo_update",
            "gradient": "gradient_update",
            "grad": "gradient_update",
            "manual": "gradient_update",
            "manual_gradient": "gradient_update",
            "direction": "gradient_update",
            "old": "gradient_update",
            "legacy": "gradient_update",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_fedsvrpg_local_update_modes:
            raise ValueError(
                f"Unsupported fedsvrpg_local_update_mode={mode!r}. "
                f"Choose one of {cls.valid_fedsvrpg_local_update_modes}."
            )
        return normalized

    def _use_ppo_clip_gradient(self) -> bool:
        return self.fedsvrpg_gradient_type == "ppo_clip"

    @staticmethod
    def _get_rms_state(rms: Any) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
        """Serialize SB3 RunningMeanStd, including dict-observation variants."""
        if isinstance(rms, Mapping):
            return {key: FedSVRPGM._get_rms_state(value) for key, value in rms.items()}

        return {
            "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
            "var": np.asarray(rms.var, dtype=np.float64).copy(),
            "count": float(rms.count),
        }

    @staticmethod
    def _set_rms_state(rms: Any, state: RunningMeanStdState | dict[str, RunningMeanStdState]) -> None:
        """Restore SB3 RunningMeanStd, including dict-observation variants."""
        if isinstance(rms, Mapping):
            for key, value in state.items():
                if key in rms:
                    FedSVRPGM._set_rms_state(rms[key], value)
            return

        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])

        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or not np.isfinite(count):
            raise ValueError(f"Invalid VecNormalize RMS state: count={count}, mean={mean}, var={var}")

        rms.mean = mean.copy()
        rms.var = np.maximum(var, 1e-12).copy()
        rms.count = max(count, 1e-4)

    def _get_vecnormalize_state(self) -> VecNormalizeState | None:
        """Serialize VecNormalize statistics from this model's env, if present."""
        vecnormalize = self.get_vec_normalize_env()
        if vecnormalize is None:
            return None

        state: VecNormalizeState = {
            "norm_obs": bool(vecnormalize.norm_obs),
            "norm_reward": bool(vecnormalize.norm_reward),
            "clip_obs": float(vecnormalize.clip_obs),
            "clip_reward": float(vecnormalize.clip_reward),
            "gamma": float(vecnormalize.gamma),
            "epsilon": float(vecnormalize.epsilon),
            "training": bool(vecnormalize.training),
            "obs_rms": None,
            "ret_rms": None,
        }

        if getattr(vecnormalize, "obs_rms", None) is not None:
            state["obs_rms"] = self._get_rms_state(vecnormalize.obs_rms)
        if getattr(vecnormalize, "ret_rms", None) is not None:
            state["ret_rms"] = self._get_rms_state(vecnormalize.ret_rms)

        return state

    def _get_filtered_vecnormalize_state(self) -> VecNormalizeState | None:
        """Serialize only the VecNormalize statistics selected by sync mode."""
        if self.vecnormalize_sync_mode == "none":
            return None

        state = self._get_vecnormalize_state()
        if state is None:
            return None

        if self.vecnormalize_sync_mode == "obs":
            state["ret_rms"] = None
        elif self.vecnormalize_sync_mode == "reward":
            state["obs_rms"] = None

        return state

    def _set_vecnormalize_state(self, state: VecNormalizeState | None) -> None:
        """Apply VecNormalize statistics while preserving the current environment trajectory."""
        if state is None:
            return

        vecnormalize = self.get_vec_normalize_env()
        if vecnormalize is None:
            return

        current_original_obs = None
        if self._last_obs is not None:
            current_original_obs = vecnormalize.get_original_obs()

        for attr in ("norm_obs", "norm_reward", "clip_obs", "clip_reward", "gamma", "epsilon", "training"):
            if attr in state:
                setattr(vecnormalize, attr, state[attr])

        if state.get("obs_rms") is not None and getattr(vecnormalize, "obs_rms", None) is not None:
            self._set_rms_state(vecnormalize.obs_rms, state["obs_rms"])
        if state.get("ret_rms") is not None and getattr(vecnormalize, "ret_rms", None) is not None:
            self._set_rms_state(vecnormalize.ret_rms, state["ret_rms"])

        if current_original_obs is not None:
            self._last_obs = vecnormalize.normalize_obs(current_original_obs)
            self._last_original_obs = self._clone_state(current_original_obs)

    def _reset_after_vecnormalize_sync(self) -> None:
        return

    @staticmethod
    def _clone_state(state: Any) -> Any:
        if state is None:
            return None
        if isinstance(state, np.ndarray):
            return state.copy()
        if isinstance(state, Mapping):
            return {key: FedSVRPGM._clone_state(value) for key, value in state.items()}
        return state

    @staticmethod
    def _is_single_rms_state(state: Any) -> bool:
        return isinstance(state, Mapping) and {"mean", "var", "count"}.issubset(state.keys())

    @staticmethod
    def _rms_is_finite(state: RunningMeanStdState) -> bool:
        return (
            np.isfinite(float(state["count"]))
            and np.all(np.isfinite(np.asarray(state["mean"], dtype=np.float64)))
            and np.all(np.isfinite(np.asarray(state["var"], dtype=np.float64)))
        )

    @classmethod
    def _merge_single_rms_states(cls, states: Sequence[RunningMeanStdState]) -> RunningMeanStdState:
        """Merge independent RunningMeanStd states exactly via M2 statistics."""
        valid_states = [state for state in states if cls._rms_is_finite(state) and float(state["count"]) > 0.0]
        if len(valid_states) == 0:
            reference = states[0]
            return {
                "mean": np.asarray(reference["mean"], dtype=np.float64).copy(),
                "var": np.maximum(np.asarray(reference["var"], dtype=np.float64), 1e-12).copy(),
                "count": max(float(reference["count"]), 1e-4),
            }

        mean = np.asarray(valid_states[0]["mean"], dtype=np.float64).copy()
        var = np.maximum(np.asarray(valid_states[0]["var"], dtype=np.float64), 0.0).copy()
        count = float(valid_states[0]["count"])
        m2 = var * count

        for state in valid_states[1:]:
            other_count = float(state["count"])
            other_mean = np.asarray(state["mean"], dtype=np.float64)
            other_var = np.maximum(np.asarray(state["var"], dtype=np.float64), 0.0)
            other_m2 = other_var * other_count

            total = count + other_count
            if total <= 0.0:
                continue
            delta = other_mean - mean
            mean = mean + delta * other_count / total
            m2 = m2 + other_m2 + np.square(delta) * count * other_count / total
            count = total

        var = np.maximum(m2 / max(count, 1e-12), 1e-12)
        return {"mean": mean, "var": var, "count": max(count, 1e-4)}

    @classmethod
    def _subtract_single_rms_state(
        cls,
        final_state: RunningMeanStdState,
        base_state: RunningMeanStdState,
    ) -> RunningMeanStdState | None:
        """Recover the incremental samples D from final_state = merge(base_state, D)."""
        if not cls._rms_is_finite(final_state) or not cls._rms_is_finite(base_state):
            return None

        final_count = float(final_state["count"])
        base_count = float(base_state["count"])
        inc_count = final_count - base_count
        if inc_count <= 1e-8:
            return None

        final_mean = np.asarray(final_state["mean"], dtype=np.float64)
        base_mean = np.asarray(base_state["mean"], dtype=np.float64)
        final_var = np.maximum(np.asarray(final_state["var"], dtype=np.float64), 0.0)
        base_var = np.maximum(np.asarray(base_state["var"], dtype=np.float64), 0.0)

        if final_mean.shape != base_mean.shape or final_var.shape != base_var.shape:
            return None

        inc_mean = (final_count * final_mean - base_count * base_mean) / inc_count

        m2_final = final_var * final_count
        m2_base = base_var * base_count
        correction = np.square(inc_mean - base_mean) * base_count * inc_count / final_count
        m2_inc = m2_final - m2_base - correction
        m2_inc = np.maximum(m2_inc, 0.0)
        inc_var = np.maximum(m2_inc / inc_count, 1e-12)

        if not np.all(np.isfinite(inc_mean)) or not np.all(np.isfinite(inc_var)):
            return None

        return {"mean": inc_mean, "var": inc_var, "count": inc_count}

    @classmethod
    def _aggregate_rms_states(
        cls,
        states: Sequence[RunningMeanStdState | dict[str, RunningMeanStdState]],
        base_state: RunningMeanStdState | dict[str, RunningMeanStdState] | None = None,
    ) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
        """Aggregate RunningMeanStd states without double-counting shared history."""
        if len(states) == 0:
            raise ValueError("At least one RunningMeanStd state is required.")

        reference = states[0]
        if not cls._is_single_rms_state(reference):
            base_mapping = base_state if isinstance(base_state, Mapping) else None
            return {
                key: cls._aggregate_rms_states(
                    [state[key] for state in states],
                    base_mapping.get(key) if base_mapping is not None and key in base_mapping else None,
                )
                for key in reference.keys()
            }

        if base_state is not None and cls._is_single_rms_state(base_state):
            increments = [cls._subtract_single_rms_state(state, base_state) for state in states]
            if all(increment is not None for increment in increments):
                merged_increments = cls._merge_single_rms_states(increments)  # type: ignore[arg-type]
                return cls._merge_single_rms_states([base_state, merged_increments])  # type: ignore[list-item]

        return cls._merge_single_rms_states(states)  # type: ignore[arg-type]

    @classmethod
    def average_vecnormalize_states(cls, states: Sequence[VecNormalizeState | None]) -> VecNormalizeState | None:
        """Aggregate VecNormalize obs_rms/ret_rms states from clients."""
        valid_states = [state for state in states if state is not None]
        if len(valid_states) == 0:
            cls._last_global_vecnormalize_state = None
            return None

        reference = valid_states[0]
        previous_global = cls._last_global_vecnormalize_state

        averaged: VecNormalizeState = {
            "norm_obs": bool(reference["norm_obs"]),
            "norm_reward": bool(reference["norm_reward"]),
            "clip_obs": float(reference["clip_obs"]),
            "clip_reward": float(reference["clip_reward"]),
            "gamma": float(reference["gamma"]),
            "epsilon": float(reference["epsilon"]),
            "training": bool(reference["training"]),
            "obs_rms": None,
            "ret_rms": None,
        }

        if reference.get("obs_rms") is not None:
            averaged["obs_rms"] = cls._aggregate_rms_states(
                [state["obs_rms"] for state in valid_states if state.get("obs_rms") is not None],
                previous_global.get("obs_rms") if previous_global is not None else None,
            )
        if reference.get("ret_rms") is not None:
            averaged["ret_rms"] = cls._aggregate_rms_states(
                [state["ret_rms"] for state in valid_states if state.get("ret_rms") is not None],
                previous_global.get("ret_rms") if previous_global is not None else None,
            )

        cls._last_global_vecnormalize_state = cls._clone_state(averaged)
        return averaged

    def resolve_federated_local_steps(
        self,
        configured_local_steps: int,
        remaining_timesteps: int,
        num_clients: int,
    ) -> int:
        per_client_budget = int(np.ceil(max(int(remaining_timesteps), 1) / float(max(int(num_clients), 1))))
        return max(1, min(int(configured_local_steps), per_client_budget))

    # ------------------------------------------------------------------
    # Actor/critic key filtering utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _is_critic_key(key: str) -> bool:
        return (
            key.startswith("value_net.")
            or key.startswith("mlp_extractor.value_net.")
            or key.startswith("vf_features_extractor.")
        )

    @staticmethod
    def _is_explicit_actor_key(key: str) -> bool:
        return (
            key == "log_std"
            or key.startswith("action_net.")
            or key.startswith("mlp_extractor.policy_net.")
            or key.startswith("pi_features_extractor.")
            or key.startswith("features_extractor.")
        )

    def _actor_state_keys(self) -> tuple[str, ...]:
        state = self.policy.state_dict()
        explicit_actor_keys = [key for key in state.keys() if self._is_explicit_actor_key(key)]
        if explicit_actor_keys:
            return tuple(explicit_actor_keys)

        # Fallback for unusual/custom SB3 policies: synchronize all non-critic
        # floating entries rather than accidentally including the value branch.
        return tuple(
            key for key, value in state.items() if th.is_floating_point(value) and not self._is_critic_key(key)
        )

    def _actor_named_parameters(self) -> dict[str, th.nn.Parameter]:
        actor_keys = set(self._actor_state_keys())
        return {
            name: parameter
            for name, parameter in self.policy.named_parameters()
            if name in actor_keys and parameter.requires_grad
        }

    def _critic_named_parameters(self) -> dict[str, th.nn.Parameter]:
        return {
            name: parameter
            for name, parameter in self.policy.named_parameters()
            if self._is_critic_key(name) and parameter.requires_grad
        }

    # ------------------------------------------------------------------
    # Federated module helpers
    # ------------------------------------------------------------------
    def _get_actor_state(self) -> FederatedModules:
        policy_state = self.policy.state_dict()
        actor_keys = self._actor_state_keys()
        return {
            self.federated_actor_module_name: OrderedDict(
                (key, policy_state[key].detach().cpu().clone()) for key in actor_keys
            )
        }

    def _set_actor_state(self, modules: FederatedModules) -> None:
        module_name = self.federated_actor_module_name
        if module_name not in modules:
            raise KeyError(f"Missing {module_name!r} in federated actor payload.")

        current_state = self.policy.state_dict()
        incoming_state = modules[module_name]
        expected_keys = set(self._actor_state_keys())
        incoming_keys = set(incoming_state.keys())
        missing = expected_keys - incoming_keys
        if missing:
            raise KeyError(f"Actor payload is missing keys: {sorted(missing)}")

        for key in expected_keys:
            current_state[key] = incoming_state[key].to(self.device)
        self.policy.load_state_dict(current_state, strict=True)

    def _critic_state_keys(self) -> tuple[str, ...]:
        """Return policy state entries not already carried by the actor payload.

        For standard SB3 ActorCriticPolicy this corresponds to the value/critic
        branch.  Defining it as the complement of actor keys makes
        ``actor_state + critic_state`` cover the full policy state_dict, which is
        what PPOAvg synchronizes in ``critic_sync_mode="fedavg"``.
        """
        state = self.policy.state_dict()
        actor_keys = set(self._actor_state_keys())
        return tuple(key for key in state.keys() if key not in actor_keys)

    def _get_critic_state(self) -> FederatedModules:
        policy_state = self.policy.state_dict()
        critic_keys = self._critic_state_keys()
        return {
            self.federated_critic_module_name: OrderedDict(
                (key, policy_state[key].detach().cpu().clone()) for key in critic_keys
            )
        }

    def _set_critic_state(self, modules: FederatedModules) -> None:
        module_name = self.federated_critic_module_name
        if module_name not in modules:
            raise KeyError(f"Missing {module_name!r} in federated critic payload.")

        current_state = self.policy.state_dict()
        incoming_state = modules[module_name]
        expected_keys = set(self._critic_state_keys())
        incoming_keys = set(incoming_state.keys())
        missing = expected_keys - incoming_keys
        if missing:
            raise KeyError(f"Critic payload is missing keys: {sorted(missing)}")
        unexpected = incoming_keys - expected_keys
        if unexpected:
            raise KeyError(f"Critic payload contains unexpected keys: {sorted(unexpected)}")

        for key in expected_keys:
            current_state[key] = incoming_state[key].to(self.device)
        self.policy.load_state_dict(current_state, strict=True)

    def _zero_like_actor_state(self) -> FederatedModules:
        current = self._get_actor_state()
        return {
            module_name: OrderedDict(
                (key, th.zeros_like(value, dtype=value.dtype)) for key, value in module_state.items()
            )
            for module_name, module_state in current.items()
        }

    @staticmethod
    def _clone_static_modules(modules: FederatedModules) -> FederatedModules:
        return {
            module_name: OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module_state.items()
            )
            for module_name, module_state in modules.items()
        }

    def _clone_modules(self, modules: FederatedModules) -> FederatedModules:
        return self._clone_static_modules(modules)

    @staticmethod
    def _subtract_static_modules(after: FederatedModules, before: FederatedModules) -> FederatedModules:
        delta: FederatedModules = {}
        for module_name, after_state in after.items():
            if module_name not in before:
                raise KeyError(f"Missing module {module_name!r} in reference modules.")
            delta[module_name] = OrderedDict()
            for key, after_value in after_state.items():
                before_value = before[module_name][key]
                if th.is_floating_point(after_value):
                    delta[module_name][key] = after_value - before_value.to(after_value.dtype)
                else:
                    delta[module_name][key] = th.zeros_like(after_value)
        return delta

    @staticmethod
    def _add_static_modules(base: FederatedModules, delta: FederatedModules) -> FederatedModules:
        result: FederatedModules = {}
        for module_name, base_state in base.items():
            if module_name not in delta:
                raise KeyError(f"Missing module {module_name!r} in delta modules.")
            result[module_name] = OrderedDict()
            for key, base_value in base_state.items():
                if th.is_floating_point(base_value):
                    result[module_name][key] = base_value + delta[module_name][key].to(base_value.dtype)
                else:
                    result[module_name][key] = base_value.clone()
        return result

    @classmethod
    def _mix_static_modules(
        cls,
        old_modules: FederatedModules,
        new_modules: FederatedModules,
        mix_weight: float,
    ) -> FederatedModules:
        if mix_weight >= 1.0:
            return cls._clone_static_modules(new_modules)

        mixed: FederatedModules = {}
        for module_name, new_state in new_modules.items():
            if module_name not in old_modules:
                raise KeyError(f"Missing module {module_name!r} in old modules.")
            mixed[module_name] = OrderedDict()
            for key, new_value in new_state.items():
                old_value = old_modules[module_name][key]
                if th.is_floating_point(new_value):
                    mixed[module_name][key] = mix_weight * new_value + (1.0 - mix_weight) * old_value.to(new_value.dtype)
                else:
                    mixed[module_name][key] = new_value.clone()
        return mixed

    @staticmethod
    def _scale_static_modules(modules: FederatedModules, scale: float) -> FederatedModules:
        return {
            module_name: OrderedDict(
                (key, value.detach().cpu().clone() * float(scale))
                if th.is_floating_point(value)
                else (key, th.zeros_like(value))
                for key, value in module_state.items()
            )
            for module_name, module_state in modules.items()
        }

    def _scale_actor_delta(self, modules: FederatedModules, scale: float) -> FederatedModules:
        return self._scale_static_modules(modules, scale)

    def _add_actor_deltas(
        self,
        lhs: FederatedModules,
        rhs: FederatedModules,
        *,
        rhs_scale: float = 1.0,
    ) -> FederatedModules:
        out: FederatedModules = {}
        for module_name in lhs.keys():
            if module_name not in rhs:
                raise KeyError(f"Missing module {module_name!r} in rhs delta.")
            out[module_name] = OrderedDict()
            for key in lhs[module_name].keys():
                lhs_value = lhs[module_name][key]
                rhs_value = rhs[module_name][key]
                if th.is_floating_point(lhs_value):
                    out[module_name][key] = lhs_value + rhs_value.to(lhs_value.dtype) * float(rhs_scale)
                else:
                    out[module_name][key] = th.zeros_like(lhs_value)
        return out

    def _actor_delta_norm(self, modules: FederatedModules) -> float:
        total_sq_norm = 0.0
        for module_state in modules.values():
            for value in module_state.values():
                if th.is_floating_point(value):
                    total_sq_norm += float(th.sum(value.float() ** 2).item())
        return total_sq_norm**0.5

    def _clip_actor_delta(self, modules: FederatedModules) -> FederatedModules:
        if self.max_update_norm is None:
            return modules
        delta_norm = self._actor_delta_norm(modules)
        max_norm = float(self.max_update_norm)
        if delta_norm <= max_norm or delta_norm == 0.0:
            return modules
        return self._scale_actor_delta(modules, max_norm / delta_norm)

    def _reset_actor_optimizer_state(self) -> None:
        actor_params = set(self._actor_named_parameters().values())
        for parameter in actor_params:
            self.policy.optimizer.state.pop(parameter, None)
        self.policy.optimizer.zero_grad(set_to_none=True)

    def _reset_optimizer_state_after_global_sync(self) -> None:
        """Reset optimizer state consistently with the selected sync mode.

        ``local`` preserves the original FedSVRPG-M behavior and drops only
        actor Adam moments.  ``fedavg`` replaces critic parameters too, so the
        full optimizer state is stale; PPOAvg clears ``policy.optimizer.state``
        after global synchronization, and we mirror that behavior here.
        """
        if self.critic_sync_mode == "fedavg":
            self.policy.optimizer.state.clear()
            self.policy.optimizer.zero_grad(set_to_none=True)
            return
        self._reset_actor_optimizer_state()

    # ------------------------------------------------------------------
    # Rollout / PPO utility helpers
    # ------------------------------------------------------------------
    def _init_fedsvrpg_training_state(self) -> None:
        if self.ep_info_buffer is None or self.ep_success_buffer is None:
            total_timesteps = max(int(getattr(self, "_total_timesteps", 0)), 1)
            self._setup_learn(
                total_timesteps=total_timesteps,
                callback=None,
                reset_num_timesteps=False,
                tb_log_name="fedsvrpg_m_ppo",
                progress_bar=False,
            )

        if self._last_obs is None:
            if self.env is None:
                raise RuntimeError("FedSVRPGM requires an environment for local rollouts.")
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        if self.rollout_buffer is None:
            self.rollout_buffer = RolloutBuffer(
                self.n_steps,
                self.observation_space,
                self.action_space,
                device=self.device,
                gamma=self.gamma,
                gae_lambda=self.gae_lambda,
                n_envs=self.n_envs,
            )

    def _estimate_discounted_mc_return(self) -> float:
        rewards = np.asarray(self.rollout_buffer.rewards.copy(), dtype=np.float64)
        episode_starts = np.asarray(self.rollout_buffer.episode_starts.copy(), dtype=bool)
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if episode_starts.ndim == 1:
            episode_starts = episode_starts[:, None]

        returns: list[float] = []
        fallback_partial_returns: list[float] = []
        n_steps, n_envs = rewards.shape
        for env_idx in range(n_envs):
            running_return = 0.0
            discount = 1.0
            has_steps = False
            for step_idx in range(n_steps):
                if bool(episode_starts[step_idx, env_idx]) and has_steps:
                    returns.append(running_return)
                    running_return = 0.0
                    discount = 1.0
                    has_steps = False

                running_return += discount * float(rewards[step_idx, env_idx])
                discount *= float(self.gamma)
                has_steps = True

            if has_steps:
                fallback_partial_returns.append(running_return)

        if not returns:
            if not fallback_partial_returns:
                return 0.0
            return float(np.mean(fallback_partial_returns))
        return float(np.mean(returns))

    def _collect_one_rollout(self) -> float:
        self._init_fedsvrpg_training_state()

        callback = CallbackList([])
        callback.init_callback(self)

        success = self.collect_rollouts(
            self.env,
            callback,
            self.rollout_buffer,
            n_rollout_steps=self.n_steps,
        )
        if not success:
            raise RuntimeError("collect_rollouts() returned False.")

        total_timesteps = int(getattr(self, "_total_timesteps", 0))
        if total_timesteps > 0:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        return self._estimate_discounted_mc_return()

    def _prepare_rollout_actions(self, actions: th.Tensor) -> th.Tensor:
        if isinstance(self.action_space, spaces.Discrete):
            return actions.long().flatten()
        return actions

    def _current_clip_range(self) -> float:
        return float(self.clip_range(self._current_progress_remaining))

    def _current_learning_rate(self) -> float:
        lr = self.lr_schedule(self._current_progress_remaining) if callable(self.lr_schedule) else self.learning_rate
        return float(lr) if lr is not None else 1e-3

    @staticmethod
    def _set_optimizer_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
        for param_group in optimizer.param_groups:
            param_group["lr"] = learning_rate

    def _current_clip_range_vf(self) -> float | None:
        if self.clip_range_vf is None:
            return None
        return float(self.clip_range_vf(self._current_progress_remaining))

    @staticmethod
    def _ppo_value_loss(values: th.Tensor, rollout_data: Any, clip_range_vf: float | None) -> th.Tensor:
        values = values.flatten()
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        return F.mse_loss(rollout_data.returns, values_pred)

    def _iter_rollout_minibatches(self) -> Sequence[RolloutBufferSamples]:
        total_size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        batch_size = self.batch_size or total_size
        indices = np.random.permutation(total_size)

        observations = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.observations)
        actions = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.actions).astype(np.float32, copy=False)
        old_values = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.values).flatten()
        old_log_prob = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.log_probs).flatten()
        advantages = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.advantages).flatten()
        returns = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.returns).flatten()

        minibatches: list[RolloutBufferSamples] = []
        start_idx = 0
        while start_idx < total_size:
            batch_inds = indices[start_idx : start_idx + batch_size]
            minibatches.append(
                RolloutBufferSamples(
                    observations=self.rollout_buffer.to_torch(observations[batch_inds]),
                    actions=self.rollout_buffer.to_torch(actions[batch_inds]),
                    old_values=self.rollout_buffer.to_torch(old_values[batch_inds]),
                    old_log_prob=self.rollout_buffer.to_torch(old_log_prob[batch_inds]),
                    advantages=self.rollout_buffer.to_torch(advantages[batch_inds]),
                    returns=self.rollout_buffer.to_torch(returns[batch_inds]),
                )
            )
            start_idx += batch_size
        return minibatches

    def _refresh_rollout_advantages(self) -> None:
        obs = self.rollout_buffer.observations
        flat_obs = obs.reshape((-1, *obs.shape[2:]))
        with th.no_grad():
            value_tensor = self.policy.predict_values(obs_as_tensor(flat_obs, self.device))
            last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))

        values = value_tensor.detach().cpu().numpy().reshape(self.rollout_buffer.buffer_size, self.rollout_buffer.n_envs)
        self.rollout_buffer.values = values
        self.rollout_buffer.generator_ready = False
        self.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values,
            dones=np.asarray(self._last_episode_starts, dtype=bool),
        )

    def _update_local_critic(self) -> None:
        critic_params = self._critic_named_parameters()
        if not critic_params:
            return

        self.policy.set_training_mode(True)
        self._set_optimizer_learning_rate(self.policy.optimizer, self._current_learning_rate())
        clip_range_vf = self._current_clip_range_vf()

        for _ in range(self.n_epochs):
            for rollout_data in self._iter_rollout_minibatches():
                actions = self._prepare_rollout_actions(rollout_data.actions)
                values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                value_loss = self._ppo_value_loss(values, rollout_data, clip_range_vf)
                loss = self.vf_coef * value_loss

                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                for name, parameter in self.policy.named_parameters():
                    if name not in critic_params:
                        parameter.grad = None

                th.nn.utils.clip_grad_norm_(list(critic_params.values()), self.max_grad_norm)
                self.policy.optimizer.step()

        self.policy.optimizer.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # PPO actor-gradient proxy
    # ------------------------------------------------------------------
    def _snapshot_rng_state(self) -> dict[str, Any]:
        return {
            "numpy": np.random.get_state(),
            "torch_cpu": th.random.get_rng_state(),
            "torch_cuda": th.cuda.get_rng_state_all() if th.cuda.is_available() else None,
        }

    @staticmethod
    def _restore_rng_state(snapshot: dict[str, Any]) -> None:
        np.random.set_state(snapshot["numpy"])
        th.random.set_rng_state(snapshot["torch_cpu"])
        if snapshot.get("torch_cuda") is not None and th.cuda.is_available():
            th.cuda.set_rng_state_all(snapshot["torch_cuda"])

    def _actor_surrogate_loss(
        self,
        rollout_data: RolloutBufferSamples,
        clip_range: float,
        sample_weights: th.Tensor | None = None,
        *,
        use_ppo_clip: bool = True,
    ) -> th.Tensor:
        """Return an actor-only PPO/score-function loss for one minibatch.

        FedSVRPG-M's variance-reduction correction should subtract two
        estimators of the same form.  Therefore the current local actor and the
        previous global actor are evaluated with the same ``use_ppo_clip``
        value.  With ``use_ppo_clip=True`` this is PPO's clipped actor
        surrogate; with ``False`` this is the raw score/advantage surrogate
        ``-A log pi_theta(a|s)``.  The previous-global term may additionally be
        multiplied by trajectory-level IS weights.
        """
        actions = self._prepare_rollout_actions(rollout_data.actions)
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        _, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        if sample_weights is None:
            weights = th.ones_like(advantages)
        else:
            weights = sample_weights.to(device=advantages.device, dtype=advantages.dtype).flatten()

        if use_ppo_clip:
            ratio = th.exp(log_prob - rollout_data.old_log_prob)
            policy_loss_1 = advantages * ratio
            policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
            policy_loss_per_sample = -th.min(policy_loss_1, policy_loss_2)
        else:
            # Score-function surrogate: grad[-A log pi_theta(a|s)] = -A grad log pi.
            # The trajectory IS weights are applied outside this term.
            policy_loss_per_sample = -(advantages * log_prob)

        policy_loss = th.mean(weights * policy_loss_per_sample)

        if entropy is None:
            entropy_per_sample = log_prob
        else:
            entropy_per_sample = entropy
        entropy_loss = -th.mean(weights * entropy_per_sample)
        return policy_loss + self.ent_coef * entropy_loss

    def _iter_rollout_minibatches_with_weights(
        self,
        sample_weights: np.ndarray | None = None,
    ) -> Sequence[tuple[RolloutBufferSamples, th.Tensor | None]]:
        total_size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
        if sample_weights is not None and int(sample_weights.shape[0]) != total_size:
            raise ValueError(
                f"sample_weights must contain {total_size} entries, got {int(sample_weights.shape[0])}."
            )

        # Actor-gradient estimation is intentionally decoupled from the
        # PPO critic's minibatch schedule.  By default, use one full-rollout
        # batch so one rollout produces one stochastic policy-gradient
        # estimate, as in the FedSVRPG-M paper.
        batch_size = total_size if self.actor_batch_size is None else min(int(self.actor_batch_size), total_size)
        indices = np.random.permutation(total_size)

        observations = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.observations)
        actions = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.actions).astype(np.float32, copy=False)
        old_values = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.values).flatten()
        old_log_prob = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.log_probs).flatten()
        advantages = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.advantages).flatten()
        returns = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.returns).flatten()

        minibatches: list[tuple[RolloutBufferSamples, th.Tensor | None]] = []
        start_idx = 0
        while start_idx < total_size:
            batch_inds = indices[start_idx : start_idx + batch_size]
            rollout_samples = RolloutBufferSamples(
                observations=self.rollout_buffer.to_torch(observations[batch_inds]),
                actions=self.rollout_buffer.to_torch(actions[batch_inds]),
                old_values=self.rollout_buffer.to_torch(old_values[batch_inds]),
                old_log_prob=self.rollout_buffer.to_torch(old_log_prob[batch_inds]),
                advantages=self.rollout_buffer.to_torch(advantages[batch_inds]),
                returns=self.rollout_buffer.to_torch(returns[batch_inds]),
            )
            if sample_weights is None:
                batch_weights = None
            else:
                batch_weights = self.rollout_buffer.to_torch(sample_weights[batch_inds].astype(np.float32, copy=False))
            minibatches.append((rollout_samples, batch_weights))
            start_idx += batch_size
        return minibatches

    def _trajectory_importance_weights(self, previous_global_actor_state: FederatedModules) -> np.ndarray:
        """Compute per-transition trajectory-level IS weights.

        The rollout buffer was sampled by the current local actor theta_{r,k}.
        For each environment segment in the buffer, this computes

            w(tau) = exp(sum_t log pi_{theta_{r-1}}(a_t|s_t)
                         - sum_t log pi_{theta_{r,k}}(a_t|s_t)).

        The resulting segment weight is assigned to all transitions in that
        segment and flattened in the same order as RolloutBuffer.get().
        """
        original_actor_state = self._get_actor_state()
        try:
            self._set_actor_state(previous_global_actor_state)
            self.policy.set_training_mode(False)

            observations = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.observations)
            actions_np = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.actions).astype(np.float32, copy=False)
            total_size = int(observations.shape[0])
            batch_size = self.batch_size or total_size
            previous_log_probs: list[np.ndarray] = []

            with th.no_grad():
                for start_idx in range(0, total_size, batch_size):
                    end_idx = min(start_idx + batch_size, total_size)
                    obs_tensor = self.rollout_buffer.to_torch(observations[start_idx:end_idx])
                    action_tensor = self.rollout_buffer.to_torch(actions_np[start_idx:end_idx])
                    action_tensor = self._prepare_rollout_actions(action_tensor)
                    _, log_prob, _ = self.policy.evaluate_actions(obs_tensor, action_tensor)
                    previous_log_probs.append(log_prob.detach().cpu().numpy().reshape(-1))

            prev_log_prob_flat = np.concatenate(previous_log_probs, axis=0)
        finally:
            self._set_actor_state(original_actor_state)
            self.policy.set_training_mode(True)

        n_steps = int(self.rollout_buffer.buffer_size)
        n_envs = int(self.rollout_buffer.n_envs)
        prev_log_prob = prev_log_prob_flat.reshape(n_envs, n_steps).T
        behavior_log_prob = np.asarray(self.rollout_buffer.log_probs, dtype=np.float64).reshape(n_steps, n_envs)
        episode_starts = np.asarray(self.rollout_buffer.episode_starts, dtype=bool).reshape(n_steps, n_envs)
        step_log_ratio = prev_log_prob.astype(np.float64) - behavior_log_prob

        log_weight_per_step = np.zeros_like(step_log_ratio, dtype=np.float64)
        clip_value = None if self.importance_ratio_clip is None else float(self.importance_ratio_clip)
        log_clip_value = None if clip_value is None else float(np.log(clip_value))

        for env_idx in range(n_envs):
            segment_start = 0
            for step_idx in range(1, n_steps):
                if bool(episode_starts[step_idx, env_idx]):
                    segment_log_weight = float(np.sum(step_log_ratio[segment_start:step_idx, env_idx]))
                    if log_clip_value is not None:
                        # Practical biased variant: clip the actual IS ratio to
                        # [1 / importance_ratio_clip, importance_ratio_clip].
                        # With importance_ratio_clip=None, no clipping is used.
                        segment_log_weight = float(np.clip(segment_log_weight, -log_clip_value, log_clip_value))
                    log_weight_per_step[segment_start:step_idx, env_idx] = segment_log_weight
                    segment_start = step_idx

            segment_log_weight = float(np.sum(step_log_ratio[segment_start:n_steps, env_idx]))
            if log_clip_value is not None:
                segment_log_weight = float(np.clip(segment_log_weight, -log_clip_value, log_clip_value))
            log_weight_per_step[segment_start:n_steps, env_idx] = segment_log_weight

        weights_step_env = np.exp(log_weight_per_step)
        return np.swapaxes(weights_step_env, 0, 1).reshape(-1).astype(np.float32, copy=False)

    def _ppo_actor_gradient_delta_on_current_buffer(
        self,
        actor_state: FederatedModules,
        *,
        sample_weights: np.ndarray | None = None,
        use_ppo_clip: bool = True,
    ) -> FederatedModules:
        """Compute a raw actor-gradient direction without mutating actor parameters.

        This returns a sum/average of ``-grad(actor_loss)`` terms.  By default
        actor_loss is PPO's clipped actor surrogate; use
        ``fedsvrpg_gradient_type='score'`` to switch back to the raw
        score-function surrogate.  This helper does not multiply SB3's PPO
        learning rate and it does not run Adam.  ``local_lr`` is the only local
        actor step-size eta applied to this direction.  ``actor_n_epochs`` and
        ``actor_batch_size`` control the actor-gradient estimator schedule
        independently from the critic's PPO ``n_epochs`` and ``batch_size``.
        """
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSVRPG-M.")

        original_actor_state = self._get_actor_state()
        module_name = self.federated_actor_module_name
        accumulated_delta = self._zero_like_actor_state()
        num_actor_minibatches = 0

        try:
            self._set_actor_state(actor_state)
            self.policy.set_training_mode(True)
            clip_range = self._current_clip_range()

            for _ in range(self.actor_n_epochs):
                for rollout_data, batch_weights in self._iter_rollout_minibatches_with_weights(sample_weights):
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    loss = self._actor_surrogate_loss(
                        rollout_data,
                        clip_range,
                        sample_weights=batch_weights,
                        use_ppo_clip=use_ppo_clip,
                    )
                    loss.backward()

                    for name, parameter in self.policy.named_parameters():
                        if name not in actor_params:
                            parameter.grad = None

                    th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)

                    for name, parameter in actor_params.items():
                        if parameter.grad is None:
                            continue
                        if name not in accumulated_delta[module_name]:
                            continue
                        accumulated_delta[module_name][name] += (
                            -parameter.grad.detach().cpu().to(accumulated_delta[module_name][name].dtype)
                        )
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    num_actor_minibatches += 1

            if self.actor_gradient_mode == "mean":
                if num_actor_minibatches <= 0:
                    raise RuntimeError("No actor minibatches were generated for FedSVRPG-M actor gradient.")
                accumulated_delta = self._scale_actor_delta(accumulated_delta, 1.0 / float(num_actor_minibatches))

            return accumulated_delta
        finally:
            self._set_actor_state(original_actor_state)
            self.policy.optimizer.zero_grad(set_to_none=True)

    def _paired_ppo_actor_gradient_deltas(
        self,
        current_actor_state: FederatedModules,
        previous_global_actor_state: FederatedModules,
    ) -> tuple[FederatedModules, FederatedModules, dict[str, float]]:
        rng_snapshot = self._snapshot_rng_state()
        # Use the same estimator family for both terms in
        # g(theta_{r,k}) - w g(theta_{r-1}).  By default this is PPO's clipped
        # actor surrogate gradient.  Setting fedsvrpg_gradient_type='score'
        # recovers the previous raw score-function-gradient variant.
        use_ppo_clip = self._use_ppo_clip_gradient()
        current_delta = self._ppo_actor_gradient_delta_on_current_buffer(
            current_actor_state,
            sample_weights=None,
            use_ppo_clip=use_ppo_clip,
        )
        self._restore_rng_state(rng_snapshot)

        trajectory_weights = self._trajectory_importance_weights(previous_global_actor_state)
        previous_global_delta = self._ppo_actor_gradient_delta_on_current_buffer(
            previous_global_actor_state,
            sample_weights=trajectory_weights,
            use_ppo_clip=use_ppo_clip,
        )
        weight_metrics = {
            "frl/fedsvrpg_is_weight_mean": float(np.mean(trajectory_weights)),
            "frl/fedsvrpg_is_weight_std": float(np.std(trajectory_weights)),
            "frl/fedsvrpg_is_weight_min": float(np.min(trajectory_weights)),
            "frl/fedsvrpg_is_weight_max": float(np.max(trajectory_weights)),
        }
        return current_delta, previous_global_delta, weight_metrics

    def _apply_actor_delta(self, actor_delta: FederatedModules) -> None:
        scaled_delta = self._scale_actor_delta(actor_delta, self.local_lr)
        scaled_delta = self._clip_actor_delta(scaled_delta)

        current_actor = self._get_actor_state()
        updated_actor = self._add_static_modules(current_actor, scaled_delta)
        self._set_actor_state(updated_actor)
        self._reset_actor_optimizer_state()

    def _run_local_ppo_update(self) -> FederatedModules:
        """Run the real SB3 PPO local update and return the actor displacement.

        Unlike ``_ppo_actor_gradient_delta_on_current_buffer()``, this mutates
        the policy with PPO's normal training operator: clipped actor loss,
        value loss, entropy term, Adam optimizer, minibatch replay, and PPO's
        configured ``n_epochs``/``batch_size`` schedule.  The returned object is
        a parameter displacement ``actor_after - actor_before``.
        """
        actor_before = self._get_actor_state()
        self.train()
        actor_after = self._get_actor_state()
        return self._subtract_static_modules(actor_after, actor_before)


    # ------------------------------------------------------------------
    # Initial u_0 estimation
    # ------------------------------------------------------------------
    def _estimate_initial_actor_direction(self) -> tuple[FederatedModules, dict[str, float]]:
        """Estimate the initial FedSVRPG-M anchor u_0 on one client.

        The paper initializes u_0 with an average of policy-gradient estimates
        collected at theta_0.  In this PPO-based implementation, one estimate
        is the same actor-only score/advantage delta used in the SVRPG
        correction.  ``init_grad_episodes`` controls how many initial rollout
        buffers are averaged on each client.
        """
        initial_actor = self._get_actor_state()
        accumulated_direction = self._zero_like_actor_state()
        returns: list[float] = []
        direction_norms: list[float] = []

        for _ in range(self.init_grad_episodes):
            rollout_return = self._collect_one_rollout()
            # Keep theta fixed at theta_0 while estimating u_0.  We do not run
            # a critic update here: this is only the initial policy-gradient
            # anchor, not a local PPO optimization step.
            direction = self._ppo_actor_gradient_delta_on_current_buffer(
                initial_actor,
                sample_weights=None,
                use_ppo_clip=self._use_ppo_clip_gradient(),
            )
            accumulated_direction = self._add_actor_deltas(accumulated_direction, direction)
            returns.append(float(rollout_return))
            direction_norms.append(self._actor_delta_norm(direction))

        averaged_direction = self._scale_actor_delta(accumulated_direction, 1.0 / float(self.init_grad_episodes))
        self._set_actor_state(initial_actor)
        self._reset_actor_optimizer_state()
        self._fedsvrpg_server_direction = self._clone_modules(averaged_direction)

        return averaged_direction, {
            "frl/fedsvrpg_u0_client_episodes": float(self.init_grad_episodes),
            "frl/fedsvrpg_u0_client_return": float(np.mean(returns)) if returns else 0.0,
            "frl/fedsvrpg_u0_client_direction_norm": float(np.mean(direction_norms)) if direction_norms else 0.0,
        }

    def prepare_federated_training(self, clients: Sequence[FederatedAlgorithmMixin]) -> None:
        """Estimate and broadcast the initial server direction u_0.

        This hook is called once by the federated experiment manager after all
        clients have been initialized from the server's theta_0.  It implements

            u_0 ~= (1 / N) sum_i (1 / B) sum_b g_i(tau_b^{(i)} | theta_0)

        with B = ``init_grad_episodes`` rollout buffers per client.
        """
        if len(clients) == 0:
            return

        client_directions: list[FederatedModules] = []
        client_weights: list[float] = []
        client_metrics: list[dict[str, float]] = []
        theta0_actor = self._get_actor_state()

        if self.vecnormalize_sync_mode == "none":
            type(self)._last_global_vecnormalize_state = None
        else:
            type(self)._last_global_vecnormalize_state = type(self)._clone_state(
                self._get_filtered_vecnormalize_state()
            )

        # In the default practical PPO backend with beta=1, the SVRPG anchor is
        # unused.  Skipping the extra u0 rollout makes the beta=1 path much
        # closer to PPOAvg: no pre-training rollout is collected solely for
        # momentum initialization.
        if self.fedsvrpg_local_update_mode == "ppo_update" and self.momentum_beta >= 1.0:
            zero_direction = self._zero_like_actor_state()
            self._fedsvrpg_prev_global_actor_state = self._clone_modules(theta0_actor)
            self._fedsvrpg_round_start_actor_state = self._clone_modules(theta0_actor)
            self._fedsvrpg_server_direction = self._clone_modules(zero_direction)
            self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
            self._fedsvrpg_last_num_local_updates = 0
            for client in clients:
                if not isinstance(client, FedSVRPGM):
                    raise TypeError("FedSVRPGM.prepare_federated_training expects FedSVRPGM clients.")
                client._set_actor_state(theta0_actor)
                client._fedsvrpg_prev_global_actor_state = client._clone_modules(theta0_actor)
                client._fedsvrpg_round_start_actor_state = client._clone_modules(theta0_actor)
                client._fedsvrpg_server_direction = client._clone_modules(zero_direction)
                client._fedsvrpg_last_actor_delta = client._zero_like_actor_state()
                client._fedsvrpg_last_num_local_updates = 0
            self._last_federated_metrics = {
                "frl/fedsvrpg_u0_initialized": 0.0,
                "frl/fedsvrpg_u0_skipped_for_ppo_beta1": 1.0,
                "frl/fedsvrpg_gradient_type_ppo_clip": float(self.fedsvrpg_gradient_type == "ppo_clip"),
                "frl/fedsvrpg_gradient_type_score": float(self.fedsvrpg_gradient_type == "score"),
                "frl/fedsvrpg_local_update_mode_ppo": 1.0,
                "frl/fedsvrpg_local_update_mode_gradient": 0.0,
            }
            return

        for client in clients:
            if not isinstance(client, FedSVRPGM):
                raise TypeError("FedSVRPGM.prepare_federated_training expects FedSVRPGM clients.")
            client._set_actor_state(theta0_actor)
            client._fedsvrpg_prev_global_actor_state = client._clone_modules(theta0_actor)
            client._fedsvrpg_round_start_actor_state = client._clone_modules(theta0_actor)
            previous_progress_lock = client._federated_progress_lock
            client._federated_progress_lock = float(client._current_progress_remaining)
            try:
                direction, metrics = client._estimate_initial_actor_direction()
            finally:
                client._federated_progress_lock = previous_progress_lock
            client_directions.append(direction)
            client_weights.append(float(client.get_client_weight()))
            client_metrics.append(metrics)

        u0 = self.average_module_states(client_directions, weights=client_weights)
        self._fedsvrpg_prev_global_actor_state = self._clone_modules(theta0_actor)
        self._fedsvrpg_round_start_actor_state = self._clone_modules(theta0_actor)
        self._fedsvrpg_server_direction = self._clone_modules(u0)
        self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
        self._fedsvrpg_last_num_local_updates = 0

        for client in clients:
            assert isinstance(client, FedSVRPGM)
            client._fedsvrpg_prev_global_actor_state = client._clone_modules(theta0_actor)
            client._fedsvrpg_round_start_actor_state = client._clone_modules(theta0_actor)
            client._fedsvrpg_server_direction = client._clone_modules(u0)
            client._fedsvrpg_last_actor_delta = client._zero_like_actor_state()
            client._fedsvrpg_last_num_local_updates = 0

        # ``_estimate_initial_actor_direction()`` collects real rollouts, which
        # updates VecNormalize statistics on each client before the first
        # communication round.  PPOAvg does not have this extra pre-training
        # rollout phase, so copying its round-end synchronization alone leaves
        # FedSVRPG-M clients starting round 1 with different normalizers.  Bring
        # all clients and the server back to a common VecNormalize state here,
        # and store it as the global baseline used by the next aggregation to
        # avoid double-counting shared RunningMeanStd history.
        if self.vecnormalize_sync_mode != "none":
            initial_vecnormalize = self.average_vecnormalize_states(
                [client._get_filtered_vecnormalize_state() for client in clients if isinstance(client, FedSVRPGM)]
            )
            self._set_vecnormalize_state(initial_vecnormalize)
            self._reset_after_vecnormalize_sync()
            for client in clients:
                assert isinstance(client, FedSVRPGM)
                client._set_vecnormalize_state(initial_vecnormalize)
                client._reset_after_vecnormalize_sync()

        u0_client_norms = [m["frl/fedsvrpg_u0_client_direction_norm"] for m in client_metrics]
        u0_client_returns = [m["frl/fedsvrpg_u0_client_return"] for m in client_metrics]
        self._last_federated_metrics = {
            "frl/fedsvrpg_u0_initialized": 1.0,
            "frl/fedsvrpg_u0_episodes_per_client": float(self.init_grad_episodes),
            "frl/fedsvrpg_u0_direction_norm": self._actor_delta_norm(u0),
            "frl/fedsvrpg_u0_client_direction_norm_mean": float(np.mean(u0_client_norms)) if u0_client_norms else 0.0,
            "frl/fedsvrpg_u0_client_return_mean": float(np.mean(u0_client_returns)) if u0_client_returns else 0.0,
            "frl/fedsvrpg_gradient_type_ppo_clip": float(self.fedsvrpg_gradient_type == "ppo_clip"),
            "frl/fedsvrpg_gradient_type_score": float(self.fedsvrpg_gradient_type == "score"),
            "frl/fedsvrpg_local_update_mode_ppo": float(self.fedsvrpg_local_update_mode == "ppo_update"),
            "frl/fedsvrpg_local_update_mode_gradient": float(self.fedsvrpg_local_update_mode == "gradient_update"),
        }

    # ------------------------------------------------------------------
    # Federated client side
    # ------------------------------------------------------------------
    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        self._federated_progress_lock = float(self._current_progress_remaining)
        try:
            del kwargs

            target_steps = int(local_steps)
            if target_steps <= 0:
                raise ValueError(f"local_steps must be positive, got {local_steps}")

            actor_before_round = self._get_actor_state()
            self._fedsvrpg_round_start_actor_state = self._clone_modules(actor_before_round)

            previous_global_actor = self._fedsvrpg_prev_global_actor_state
            if previous_global_actor is None:
                previous_global_actor = self._clone_modules(actor_before_round)

            server_direction = self._fedsvrpg_server_direction
            if server_direction is None:
                server_direction = self._zero_like_actor_state()

            collected_steps = 0
            num_local_updates = 0
            local_returns: list[float] = []
            update_norms: list[float] = []
            ppo_delta_norms: list[float] = []
            current_delta_norms: list[float] = []
            correction_norms: list[float] = []
            is_weight_means: list[float] = []
            is_weight_maxs: list[float] = []

            while collected_steps < target_steps:
                rollout_return = self._collect_one_rollout()
                current_actor = self._get_actor_state()

                if self.fedsvrpg_local_update_mode == "ppo_update":
                    # Default practical PPO backend.
                    # First, optionally compute the FedSVRPG-M correction at the
                    # pre-PPO local actor.  For beta=1 this branch is skipped, so
                    # the client update is exactly the local SB3 PPO update.
                    if self.momentum_beta < 1.0:
                        current_delta, previous_delta, is_metrics = self._paired_ppo_actor_gradient_deltas(
                            current_actor,
                            previous_global_actor,
                        )
                        correction = self._add_actor_deltas(current_delta, previous_delta, rhs_scale=-1.0)
                        svrp_anchor = self._add_actor_deltas(server_direction, correction)
                        svrp_delta = self._scale_actor_delta(svrp_anchor, self.local_lr)
                        svrp_delta = self._clip_actor_delta(svrp_delta)
                    else:
                        current_delta = self._zero_like_actor_state()
                        correction = self._zero_like_actor_state()
                        svrp_delta = self._zero_like_actor_state()
                        is_metrics = {
                            "frl/fedsvrpg_is_weight_mean": 1.0,
                            "frl/fedsvrpg_is_weight_max": 1.0,
                        }

                    # Run the actual local PPO optimizer update.  This updates both
                    # actor and critic exactly as SB3 PPO would for this rollout.
                    ppo_delta = self._run_local_ppo_update()

                    if self.momentum_beta < 1.0:
                        # Replace the PPO actor displacement by the beta-blended
                        # practical FedSVRPG-M displacement.  The critic update made
                        # by PPO is kept local.
                        blended_delta = self._add_actor_deltas(
                            self._scale_actor_delta(ppo_delta, self.momentum_beta),
                            self._scale_actor_delta(svrp_delta, 1.0 - self.momentum_beta),
                        )
                        updated_actor = self._add_static_modules(current_actor, blended_delta)
                        self._set_actor_state(updated_actor)
                        self._reset_actor_optimizer_state()
                        effective_delta = blended_delta
                    else:
                        # beta=1: this is the local PPO update displacement.
                        # The server will still divide the uploaded round delta by
                        # local_lr * K and then multiply by lambda
                        # (server_update_weight).  Setting lambda = local_lr * K
                        # makes the shared actor update equal to actor-only PPOAvg
                        # parameter averaging.
                        effective_delta = ppo_delta

                    update_norms.append(self._actor_delta_norm(effective_delta))
                    ppo_delta_norms.append(self._actor_delta_norm(ppo_delta))
                    current_delta_norms.append(self._actor_delta_norm(current_delta))
                    correction_norms.append(self._actor_delta_norm(correction))
                    is_weight_means.append(float(is_metrics["frl/fedsvrpg_is_weight_mean"]))
                    is_weight_maxs.append(float(is_metrics["frl/fedsvrpg_is_weight_max"]))

                else:
                    # Legacy/theory-closer gradient-level backend.  This does not
                    # run PPO's optimizer for the actor; it applies the FedSVRPG-M
                    # direction manually and then fits the critic locally.
                    current_delta, previous_delta, is_metrics = self._paired_ppo_actor_gradient_deltas(
                        current_actor,
                        previous_global_actor,
                    )

                    correction = self._add_actor_deltas(current_delta, previous_delta, rhs_scale=-1.0)
                    svrp_anchor = self._add_actor_deltas(server_direction, correction)
                    momentum_direction = self._add_actor_deltas(
                        self._scale_actor_delta(current_delta, self.momentum_beta),
                        self._scale_actor_delta(svrp_anchor, 1.0 - self.momentum_beta),
                    )
                    momentum_direction = self._clip_actor_delta(momentum_direction)

                    self._apply_actor_delta(momentum_direction)
                    self._update_local_critic()

                    update_norms.append(self._actor_delta_norm(momentum_direction))
                    ppo_delta_norms.append(0.0)
                    current_delta_norms.append(self._actor_delta_norm(current_delta))
                    correction_norms.append(self._actor_delta_norm(correction))
                    is_weight_means.append(float(is_metrics["frl/fedsvrpg_is_weight_mean"]))
                    is_weight_maxs.append(float(is_metrics["frl/fedsvrpg_is_weight_max"]))

                local_returns.append(float(rollout_return))
                collected_steps += self.n_steps * self.n_envs
                num_local_updates += 1

            actor_after_round = self._get_actor_state()
            self._fedsvrpg_last_actor_delta = self._subtract_static_modules(actor_after_round, actor_before_round)
            self._fedsvrpg_last_return = float(np.mean(local_returns)) if local_returns else 0.0
            self._fedsvrpg_last_num_local_updates = int(num_local_updates)
            self._last_federated_metrics = {
                "frl/fedsvrpg_local_updates": float(num_local_updates),
                "frl/fedsvrpg_mean_update_norm": float(np.mean(update_norms)) if update_norms else 0.0,
                "frl/fedsvrpg_mean_ppo_delta_norm": float(np.mean(ppo_delta_norms)) if ppo_delta_norms else 0.0,
                "frl/fedsvrpg_mean_current_delta_norm": float(np.mean(current_delta_norms)) if current_delta_norms else 0.0,
                "frl/fedsvrpg_mean_correction_norm": float(np.mean(correction_norms)) if correction_norms else 0.0,
                "frl/fedsvrpg_mean_is_weight": float(np.mean(is_weight_means)) if is_weight_means else 1.0,
                "frl/fedsvrpg_max_is_weight": float(np.max(is_weight_maxs)) if is_weight_maxs else 1.0,
                "frl/fedsvrpg_gradient_type_ppo_clip": float(self.fedsvrpg_gradient_type == "ppo_clip"),
                "frl/fedsvrpg_gradient_type_score": float(self.fedsvrpg_gradient_type == "score"),
                "frl/fedsvrpg_local_update_mode_ppo": float(self.fedsvrpg_local_update_mode == "ppo_update"),
                "frl/fedsvrpg_local_update_mode_gradient": float(self.fedsvrpg_local_update_mode == "gradient_update"),
                "frl/fedsvrpg_actor_n_epochs": float(self.actor_n_epochs),
                "frl/fedsvrpg_actor_batch_size": float(
                    self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
                    if self.actor_batch_size is None
                    else self.actor_batch_size
                ),
            }

        finally:
            self._federated_progress_lock = None

    def get_upload_payload(self) -> FederatedPayload:
        critic_state = self._get_critic_state() if self.critic_sync_mode == "fedavg" else None
        return {
            "round_start_actor_state": self._clone_modules(self._fedsvrpg_round_start_actor_state),
            "actor_delta": self._clone_modules(self._fedsvrpg_last_actor_delta),
            "critic_state": self._clone_modules(critic_state) if critic_state is not None else None,
            "critic_sync_mode": self.critic_sync_mode,
            "vecnormalize": self._get_filtered_vecnormalize_state(),
            "vecnormalize_sync_mode": self.vecnormalize_sync_mode,
            "return": float(self._fedsvrpg_last_return),
            "num_local_updates": int(self._fedsvrpg_last_num_local_updates),
            "local_lr": float(self.local_lr),
            "fedsvrpg_local_update_mode": self.fedsvrpg_local_update_mode,
        }

    # ------------------------------------------------------------------
    # Federated server side
    # ------------------------------------------------------------------
    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        if len(uploads) == 0:
            raise ValueError("At least one upload is required for federated aggregation.")

        critic_modes = {
            cls._normalize_critic_sync_mode(str(upload.get("critic_sync_mode", "local")))
            for upload in uploads
        }
        if len(critic_modes) != 1:
            raise ValueError(
                f"Mixed critic_sync_mode values are not supported in one aggregation: {critic_modes}"
            )
        critic_sync_mode = next(iter(critic_modes))

        vecnormalize_modes = {
            cls._normalize_vecnormalize_sync_mode(str(upload.get("vecnormalize_sync_mode", "obs_reward")))
            for upload in uploads
        }
        if len(vecnormalize_modes) != 1:
            raise ValueError(
                f"Mixed vecnormalize_sync_mode values are not supported in one aggregation: {vecnormalize_modes}"
            )
        vecnormalize_sync_mode = next(iter(vecnormalize_modes))
        local_update_modes = {
            cls._normalize_fedsvrpg_local_update_mode(str(upload.get("fedsvrpg_local_update_mode", "gradient_update")))
            for upload in uploads
        }
        if len(local_update_modes) != 1:
            raise ValueError(
                f"Mixed fedsvrpg_local_update_mode values are not supported in one aggregation: {local_update_modes}"
            )
        fedsvrpg_local_update_mode = next(iter(local_update_modes))

        actor_deltas = [upload["actor_delta"] for upload in uploads]
        aggregated_actor_delta = cls.average_module_states(actor_deltas, weights=weights)

        aggregated_critic_state = None
        if critic_sync_mode == "fedavg":
            critic_states = [upload.get("critic_state") for upload in uploads]
            if any(state is None for state in critic_states):
                raise ValueError("critic_sync_mode='fedavg' requires critic_state from every client upload.")
            aggregated_critic_state = cls.average_module_states(
                critic_states,  # type: ignore[arg-type]
                weights=weights,
            )

        normalized_weights = cls.normalize_weights(len(uploads), weights)
        if vecnormalize_sync_mode == "none":
            aggregated_vecnormalize = None
            cls._last_global_vecnormalize_state = None
        else:
            aggregated_vecnormalize = cls.average_vecnormalize_states(
                [upload.get("vecnormalize") for upload in uploads]
            )

        update_scales = []
        for upload in uploads:
            num_local_updates = max(int(upload.get("num_local_updates", 1)), 1)
            local_lr = float(upload.get("local_lr", 1.0))
            update_scales.append(max(local_lr * float(num_local_updates), 1e-12))
        denominator = float(sum(weight * scale for weight, scale in zip(normalized_weights, update_scales, strict=True)))
        server_direction = cls._scale_static_modules(aggregated_actor_delta, 1.0 / denominator)

        mean_return = float(np.mean([float(upload["return"]) for upload in uploads]))
        reference_actor_state = cls._clone_static_modules(uploads[0]["round_start_actor_state"])

        return {
            "aggregation_type": "fedsvrpg_m_update",
            "reference_actor_state": reference_actor_state,
            "aggregated_actor_delta": aggregated_actor_delta,
            "server_direction": server_direction,
            "critic_state": aggregated_critic_state,
            "critic_sync_mode": critic_sync_mode,
            "fedsvrpg_local_update_mode": fedsvrpg_local_update_mode,
            "vecnormalize": aggregated_vecnormalize,
            "vecnormalize_sync_mode": vecnormalize_sync_mode,
            "return": mean_return,
            "num_clients": len(uploads),
            "mean_local_update_scale": denominator,
        }

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (float(mix_weight) > 0.0):
            raise ValueError("mix_weight must be positive.")

        payload_critic_mode = self._normalize_critic_sync_mode(
            str(payload.get("critic_sync_mode", self.critic_sync_mode))
        )
        if payload_critic_mode != self.critic_sync_mode:
            raise ValueError(
                f"Client critic_sync_mode={self.critic_sync_mode!r} does not match "
                f"payload critic_sync_mode={payload_critic_mode!r}."
            )

        payload_vecnormalize_mode = self._normalize_vecnormalize_sync_mode(
            str(payload.get("vecnormalize_sync_mode", self.vecnormalize_sync_mode))
        )
        if payload_vecnormalize_mode != self.vecnormalize_sync_mode:
            raise ValueError(
                f"Client vecnormalize_sync_mode={self.vecnormalize_sync_mode!r} does not match "
                f"payload vecnormalize_sync_mode={payload_vecnormalize_mode!r}."
            )

        # Server update path: keep the FedSVRPG-M paper-style server update
        # for both local-update backends.  Clients upload actor displacements
        # Delta_i.  The server reconstructs
        #     u_{r+1} = average_i(Delta_i) / (eta * K)
        # and then applies
        #     theta_{r+1} = theta_r + lambda * u_{r+1}.
        # The federated manager passes lambda as server_update_weight through
        # mix_weight.  Therefore, in ppo_update mode with beta=1, choosing
        # lambda = eta * K makes this equivalent to actor-only PPOAvg-style
        # parameter averaging.
        if payload.get("aggregation_type") == "fedsvrpg_m_update":
            reference_actor = payload["reference_actor_state"]
            server_direction = payload["server_direction"]
            payload_update_mode = self._normalize_fedsvrpg_local_update_mode(
                str(payload.get("fedsvrpg_local_update_mode", "gradient_update"))
            )
            server_step = self._scale_actor_delta(server_direction, mix_weight)
            updated_actor = self._add_static_modules(reference_actor, server_step)

            self._set_actor_state(updated_actor)
            if self.critic_sync_mode == "fedavg":
                aggregated_critic_state = payload.get("critic_state")
                if aggregated_critic_state is None:
                    raise KeyError("Missing critic_state in FedSVRPG-M aggregation payload for critic_sync_mode='fedavg'.")
                # Critic uses ordinary FedAvg parameter averaging.  It is not
                # scaled by the FedSVRPG-M actor lambda/server_update_weight.
                self._set_critic_state(aggregated_critic_state)

            self._fedsvrpg_prev_global_actor_state = self._clone_modules(reference_actor)
            self._fedsvrpg_server_direction = self._clone_modules(server_direction)
            self._fedsvrpg_round_start_actor_state = self._clone_modules(updated_actor)
            self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
            self._fedsvrpg_last_return = float(payload.get("return", 0.0))
            self._fedsvrpg_last_num_local_updates = 0
            if self.vecnormalize_sync_mode != "none":
                self._set_vecnormalize_state(payload.get("vecnormalize"))
                self._reset_after_vecnormalize_sync()
            self._reset_optimizer_state_after_global_sync()
            self._last_federated_metrics = {
                "frl/fedsvrpg_server_direction_norm": self._actor_delta_norm(server_direction),
                "frl/fedsvrpg_server_step_norm": self._actor_delta_norm(server_step),
                "frl/fedsvrpg_local_update_mode_ppo": float(payload_update_mode == "ppo_update"),
                "frl/fedsvrpg_local_update_mode_gradient": float(payload_update_mode == "gradient_update"),
                "frl/fedsvrpg_critic_local": float(self.critic_sync_mode == "local"),
                "frl/fedsvrpg_critic_fedavg": float(self.critic_sync_mode == "fedavg"),
                "frl/fedsvrpg_lambda_over_eta_k": float(mix_weight) / max(float(payload.get("mean_local_update_scale", 1.0)), 1e-12),
                "frl/fedsvrpg_mean_return": float(payload.get("return", 0.0)),
                "frl/fedsvrpg_mean_local_update_scale": float(payload.get("mean_local_update_scale", 1.0)),
                "frl/fedsvrpg_vecnormalize_none": float(self.vecnormalize_sync_mode == "none"),
                "frl/fedsvrpg_vecnormalize_obs": float(self.vecnormalize_sync_mode == "obs"),
                "frl/fedsvrpg_vecnormalize_reward": float(self.vecnormalize_sync_mode == "reward"),
                "frl/fedsvrpg_vecnormalize_obs_reward": float(self.vecnormalize_sync_mode == "obs_reward"),
            }
            return

        # Broadcast path: clients receive theta_r, theta_{r-1}, and u_r.
        incoming_actor = payload["actor_state"]
        if mix_weight < 1.0:
            incoming_actor = self._mix_static_modules(self._get_actor_state(), incoming_actor, mix_weight)
        self._set_actor_state(incoming_actor)

        if self.critic_sync_mode == "fedavg":
            incoming_critic = payload.get("critic_state")
            if incoming_critic is None:
                raise KeyError("Missing critic_state in FedSVRPG-M broadcast payload for critic_sync_mode='fedavg'.")
            # The server has already formed the global critic by ordinary
            # parameter averaging; broadcast that state directly to clients.
            self._set_critic_state(incoming_critic)

        previous_actor = payload.get("prev_actor_state", incoming_actor)
        server_direction = payload.get("server_direction", self._zero_like_actor_state())
        self._fedsvrpg_prev_global_actor_state = self._clone_modules(previous_actor)
        self._fedsvrpg_server_direction = self._clone_modules(server_direction)
        self._fedsvrpg_round_start_actor_state = self._clone_modules(incoming_actor)
        self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
        self._fedsvrpg_last_return = float(payload.get("return", 0.0))
        self._fedsvrpg_last_num_local_updates = 0
        if self.vecnormalize_sync_mode != "none":
            self._set_vecnormalize_state(payload.get("vecnormalize"))
            self._reset_after_vecnormalize_sync()
        self._reset_optimizer_state_after_global_sync()

    def get_broadcast_payload(self) -> FederatedPayload:
        actor_state = self._get_actor_state()
        previous_actor = self._fedsvrpg_prev_global_actor_state
        if previous_actor is None:
            previous_actor = actor_state
        server_direction = self._fedsvrpg_server_direction
        if server_direction is None:
            server_direction = self._zero_like_actor_state()
        critic_state = self._get_critic_state() if self.critic_sync_mode == "fedavg" else None
        return {
            "aggregation_type": "fedsvrpg_m_broadcast",
            "actor_state": self._clone_modules(actor_state),
            "prev_actor_state": self._clone_modules(previous_actor),
            "server_direction": self._clone_modules(server_direction),
            "critic_state": self._clone_modules(critic_state) if critic_state is not None else None,
            "critic_sync_mode": self.critic_sync_mode,
            "fedsvrpg_local_update_mode": self.fedsvrpg_local_update_mode,
            "vecnormalize": self._get_filtered_vecnormalize_state(),
            "vecnormalize_sync_mode": self.vecnormalize_sync_mode,
            "return": float(self._fedsvrpg_last_return),
        }

    def get_client_weight(self) -> float:
        return 1.0
