from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.ppo import PPO

from rl_zoo3.algorithms.federate.common.federated_algorithm import (
    FederatedAlgorithmMixin,
    FederatedModules,
    FederatedPayload,
)

VecNormalizeState = dict[str, Any]
RunningMeanStdState = dict[str, Any]


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"Expected a 1D vector, got shape={values.shape}.")
    if values.size == 0:
        raise ValueError("Cannot project an empty vector onto the simplex.")
    if values.size == 1:
        return np.ones(1, dtype=np.float64)

    sorted_values = np.sort(values)[::-1]
    cssv = np.cumsum(sorted_values)
    support = np.nonzero(sorted_values * np.arange(1, values.size + 1) > cssv - 1.0)[0]
    if support.size == 0:
        return np.ones_like(values) / float(values.size)

    rho = int(support[-1])
    theta = (cssv[rho] - 1.0) / float(rho + 1)
    projected = np.maximum(values - theta, 0.0)
    projected_sum = float(np.sum(projected))
    if projected_sum <= 0.0:
        return np.ones_like(values) / float(values.size)
    return projected / projected_sum


class _RawRewardCaptureCallback(BaseCallback):
    """Capture raw/normalized training rewards and episode completions for the dual estimator."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_rewards: list[np.ndarray] = []
        self.normalized_rewards: list[np.ndarray] = []
        self.dones: list[np.ndarray] = []

    def _on_step(self) -> bool:
        normalized_rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        raw_rewards = normalized_rewards
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None and getattr(vecnormalize, "norm_reward", False):
            raw_rewards = np.asarray(vecnormalize.get_original_reward(), dtype=np.float64)

        dones = self.locals.get("dones")
        if dones is None:
            dones = np.zeros_like(normalized_rewards, dtype=bool)

        self.raw_rewards.append(raw_rewards.copy())
        self.normalized_rewards.append(normalized_rewards.copy())
        self.dones.append(np.asarray(dones, dtype=bool).copy())
        return True


class FedAMPOLocalPPO(FederatedAlgorithmMixin, PPO):
    """Fed-AMPO-LocalPPO: adaptive adversarial weighting with local PPO updates.

    Paper-level round:
      1. the server broadcasts the shared actor theta,
      2. each client collects exactly one rollout with that global actor,
      3. completed training-stream episode returns update the client dual-return EMA,
      4. each client performs local PPO actor/critic updates on that rollout,
      5. each client uploads the actor delta and current dual-return EMA,
      6. the server applies theta <- theta + rho * sum_k lambda_k Delta_k,
      7. the server applies lambda <- Proj_Delta(lambda - eta_lambda * J).

    ``local_steps`` must equal ``n_steps * n_envs`` exactly so the local PPO
    update is based on one pre-update global-policy rollout per round. The
    dual estimator is a training-stream episodic estimator whose partial
    episodes may span communication rounds.

    ``dual_update_mode="uniform"`` freezes lambda to 1/K for the ablation.
    ``critic_sync_mode="local"`` is the paper default; ``"fedavg"`` and
    ``"actor_like"`` are server-critic ablations.

    ``local_actor_update_mode="standard"`` (default) preserves the original
    local PPO update exactly.  ``"momentum"`` replaces each raw local actor
    gradient by the Remark-4-style variance-reduced direction

        d <- beta * g(theta_local)
             + (1 - beta) * [d + g(theta_local) - g(theta_global)],

    where the local and global-reference gradients are evaluated on the same
    rollout minibatch.  Each client keeps its own momentum state across local
    minibatches and communication rounds; the state is initialized from the
    first global-reference gradient when momentum mode is first activated.

    ``dual_aware_drift_mode="none"`` is the default and preserves the original
    algorithm. ``"diagnostic"`` logs local-policy KL without changing training.
    ``"kl_gate"`` optionally suppresses only high-lambda client deltas when their
    local PPO KL exceeds ``dual_aware_kl_target``; the dual update itself is not
    changed.
    """

    federated_actor_module_name = "policy"
    federated_critic_module_name = "policy"
    valid_dual_update_modes: tuple[str, ...] = ("adaptive", "uniform")
    valid_critic_sync_modes: tuple[str, ...] = ("local", "fedavg", "actor_like")
    valid_vecnormalize_sync_modes: tuple[str, ...] = ("none", "obs", "reward", "obs_reward")
    valid_dual_return_sources: tuple[str, ...] = ("raw", "normalized")
    valid_dual_return_modes: tuple[str, ...] = ("discounted", "undiscounted")
    valid_dual_scale_modes: tuple[str, ...] = ("none", "std_ema")
    valid_local_actor_update_modes: tuple[str, ...] = ("standard", "momentum")
    valid_dual_aware_drift_modes: tuple[str, ...] = ("none", "diagnostic", "kl_gate")

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
        "dual_lr",
        "initial_lambda",
        "dual_update_mode",
        "fixed_uniform_lambda",
        "dual_fixed_uniform",
        "critic_sync_mode",
        "server_actor_lr",
        "server_actor_delta_scale",
        "actor_gradient_mode",
        "server_actor_optimizer",
        "vecnormalize_sync_mode",
        "dual_return_source",
        "dual_return_mode",
        "dual_return_ema_beta",
        "dual_update_interval",
        "dual_warmup_min_episodes",
        "dual_scale_mode",
        "dual_scale_ema_beta",
        "dual_scale_epsilon",
        "local_actor_update_mode",
        "actor_update_mode",
        "momentum_beta",
        "local_momentum_beta",
        "dual_aware_drift_mode",
        "dual_aware_kl_target",
        "dual_aware_correction_strength",
        "dual_aware_min_delta_scale",
        "log_wandb",
    )

    _last_global_vecnormalize_state: VecNormalizeState | None = None

    def __init__(self, *args, **kwargs):
        self.dual_lr = float(kwargs.pop("dual_lr", 0.05))
        self.initial_lambda = kwargs.pop("initial_lambda", None)

        # ``server_actor_lr`` is retained as a backward-compatible alias.
        # In the local-update variant it scales an already-computed actor delta,
        # so the natural default is 1.0 rather than the PPO learning rate.
        server_actor_delta_scale = kwargs.pop(
            "server_actor_delta_scale",
            kwargs.pop("server_actor_lr", None),
        )
        self.server_actor_delta_scale = (
            1.0 if server_actor_delta_scale is None else float(server_actor_delta_scale)
        )
        if self.server_actor_delta_scale <= 0.0:
            raise ValueError(
                "server_actor_delta_scale must be positive, "
                f"got {self.server_actor_delta_scale}."
            )

        # Accepted only for backward compatibility with existing configs.
        # Actor gradients are no longer uploaded in this implementation.
        kwargs.pop("actor_gradient_mode", None)
        self.critic_sync_mode = self._normalize_critic_sync_mode(kwargs.pop("critic_sync_mode", "local"))
        self.vecnormalize_sync_mode = self._normalize_vecnormalize_sync_mode(
            kwargs.pop("vecnormalize_sync_mode", "obs_reward")
        )
        self.dual_return_source = self._normalize_dual_return_source(kwargs.pop("dual_return_source", "raw"))
        self.dual_return_mode = self._normalize_dual_return_mode(kwargs.pop("dual_return_mode", "discounted"))
        self.dual_return_ema_beta = float(kwargs.pop("dual_return_ema_beta", 0.9))
        if not (0.0 <= self.dual_return_ema_beta < 1.0):
            raise ValueError("dual_return_ema_beta must be in [0, 1).")

        self.dual_update_interval = int(kwargs.pop("dual_update_interval", 5))
        if self.dual_update_interval <= 0:
            raise ValueError("dual_update_interval must be positive.")

        self.dual_warmup_min_episodes = int(kwargs.pop("dual_warmup_min_episodes", 1))
        if self.dual_warmup_min_episodes <= 0:
            raise ValueError("dual_warmup_min_episodes must be positive.")

        self.dual_scale_mode = self._normalize_dual_scale_mode(kwargs.pop("dual_scale_mode", "std_ema"))
        self.dual_scale_ema_beta = float(kwargs.pop("dual_scale_ema_beta", 0.95))
        if not (0.0 <= self.dual_scale_ema_beta < 1.0):
            raise ValueError("dual_scale_ema_beta must be in [0, 1).")
        self.dual_scale_epsilon = float(kwargs.pop("dual_scale_epsilon", 1e-8))
        if self.dual_scale_epsilon <= 0.0:
            raise ValueError("dual_scale_epsilon must be positive.")

        actor_update_alias = kwargs.pop("actor_update_mode", None)
        raw_actor_update_mode = kwargs.pop("local_actor_update_mode", actor_update_alias)
        if raw_actor_update_mode is None:
            raw_actor_update_mode = "standard"
        self.local_actor_update_mode = self._normalize_local_actor_update_mode(raw_actor_update_mode)

        momentum_beta_alias = kwargs.pop("local_momentum_beta", None)
        raw_momentum_beta = kwargs.pop("momentum_beta", momentum_beta_alias)
        self.momentum_beta = 0.9 if raw_momentum_beta is None else float(raw_momentum_beta)
        if not (0.0 <= self.momentum_beta <= 1.0):
            raise ValueError(f"momentum_beta must be in [0, 1], got {self.momentum_beta}.")

        # Optional dual-aware drift control.  The default ``none`` leaves the
        # original FedAMPO-LocalPPO optimization path unchanged.  ``diagnostic``
        # measures local-policy KL without changing the update, while ``kl_gate``
        # attenuates only high-dual-weight client deltas whose local PPO drift is
        # above the configured KL target.
        self.dual_aware_drift_mode = self._normalize_dual_aware_drift_mode(
            kwargs.pop("dual_aware_drift_mode", "diagnostic")
        )
        raw_dual_aware_kl_target = kwargs.pop("dual_aware_kl_target", None)
        self.dual_aware_correction_strength = float(kwargs.pop("dual_aware_correction_strength", 1.0))
        if not (0.0 <= self.dual_aware_correction_strength <= 1.0):
            raise ValueError("dual_aware_correction_strength must be in [0, 1].")
        self.dual_aware_min_delta_scale = float(kwargs.pop("dual_aware_min_delta_scale", 0.5))
        if not (0.0 < self.dual_aware_min_delta_scale <= 1.0):
            raise ValueError("dual_aware_min_delta_scale must be in (0, 1].")

        raw_dual_mode = kwargs.pop("dual_update_mode", None)
        fixed_uniform = self._as_bool(kwargs.pop("fixed_uniform_lambda", False)) or self._as_bool(
            kwargs.pop("dual_fixed_uniform", False)
        )
        if raw_dual_mode is None:
            raw_dual_mode = "uniform" if fixed_uniform else "adaptive"
        self.dual_update_mode = self._normalize_dual_update_mode(raw_dual_mode)
        self._federated_progress_lock: float | None = None

        for key in self.federated_manager_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        if raw_dual_aware_kl_target is None:
            # Reuse PPO's target_kl when available; otherwise use a modest
            # diagnostic/gating reference.  This value is inert in mode="none".
            self.dual_aware_kl_target = (
                float(self.target_kl) if self.target_kl is not None else 0.02
            )
        else:
            self.dual_aware_kl_target = float(raw_dual_aware_kl_target)
        if self.dual_aware_kl_target <= 0.0:
            raise ValueError("dual_aware_kl_target must be positive.")

        self.lambda_weights: np.ndarray | None = None
        self._ampo_last_actor_delta: FederatedModules | None = None
        self._ampo_last_return: float | None = None
        self._ampo_last_critic_state: FederatedModules | None = None
        self._ampo_last_critic_delta: FederatedModules | None = None
        self._ampo_last_num_actor_batches: int = 0
        self._ampo_last_current_actor_grad_norm: float = 0.0
        self._ampo_last_reference_actor_grad_norm: float = 0.0
        self._ampo_last_gradient_difference_norm: float = 0.0
        self._ampo_last_corrected_actor_grad_norm: float = 0.0
        self._ampo_last_local_actor_kl_mean: float = float("nan")
        self._ampo_last_local_actor_kl_max: float = float("nan")
        self._ampo_last_gradient_cosine_mean: float = float("nan")
        self._ampo_last_gradient_cosine_min: float = float("nan")
        self._ampo_actor_momentum_state: dict[str, th.Tensor] | None = None

        # Training-stream episodic dual estimator. Partial episodes survive
        # communication rounds and are finalized only on actual environment done.
        self._ampo_dual_episode_returns: np.ndarray | None = None
        self._ampo_dual_episode_discounts: np.ndarray | None = None
        self._ampo_dual_return_ema: float | None = None
        self._ampo_dual_completed_episodes: int = 0
        self._ampo_last_completed_episode_mean: float | None = None
        self._ampo_last_completed_episodes: int = 0

        # Server-side slow-timescale/scaling state.
        self._ampo_server_round: int = 0
        self._ampo_dual_scale_ema: float | None = None

        self._last_federated_metrics: dict[str, float] = {}

    @classmethod
    def reset_federated_state(cls) -> None:
        cls._last_global_vecnormalize_state = None

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        return True

    def prepare_federated_training(self, clients: Sequence[FederatedAlgorithmMixin]) -> None:
        del clients
        if self.vecnormalize_sync_mode == "none":
            type(self)._last_global_vecnormalize_state = None
            return
        type(self)._last_global_vecnormalize_state = type(self)._clone_state(
            self._get_filtered_vecnormalize_state()
        )

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        if self._federated_progress_lock is None:
            super()._update_current_progress_remaining(num_timesteps, total_timesteps)
            return
        self._current_progress_remaining = self._federated_progress_lock

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "none", "null", ""}:
            return False
        raise ValueError(f"Cannot interpret {value!r} as a boolean.")

    @classmethod
    def _normalize_dual_update_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_").replace("/", "_over_")
        aliases = {
            "ampo": "adaptive",
            "learn": "adaptive",
            "learned": "adaptive",
            "update": "adaptive",
            "fixed": "uniform",
            "fixed_uniform": "uniform",
            "uniform_fixed": "uniform",
            "uniform_lambda": "uniform",
            "fixed_lambda": "uniform",
            "1_k": "uniform",
            "1_over_k": "uniform",
            "one_over_k": "uniform",
            "mean": "uniform",
            "average": "uniform",
            "fedavg": "uniform",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_update_modes:
            raise ValueError(
                f"Unsupported dual_update_mode={mode!r}. Choose one of {cls.valid_dual_update_modes}."
            )
        return normalized

    @classmethod
    def _normalize_local_actor_update_mode(cls, mode: str) -> str:
        """Normalize the local actor optimizer backend.

        Modes:
          - standard: unchanged local PPO actor/critic update (default).
          - momentum: Remark-4-style gradient-difference momentum correction.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "none": "standard",
            "off": "standard",
            "false": "standard",
            "no": "standard",
            "baseline": "standard",
            "ppo": "standard",
            "local_ppo": "standard",
            "plain": "standard",
            "vanilla": "standard",
            "m": "momentum",
            "vr": "momentum",
            "vr_momentum": "momentum",
            "variance_reduced": "momentum",
            "variance_reduction": "momentum",
            "momentum_corrected": "momentum",
            "fedsvrpg": "momentum",
            "fedsvrpg_m": "momentum",
            "svrpg": "momentum",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_local_actor_update_modes:
            raise ValueError(
                f"Unsupported local_actor_update_mode={mode!r}. "
                f"Choose one of {cls.valid_local_actor_update_modes}."
            )
        return normalized

    @classmethod
    def _normalize_dual_aware_drift_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "off": "none",
            "false": "none",
            "disabled": "none",
            "log": "diagnostic",
            "diagnostics": "diagnostic",
            "monitor": "diagnostic",
            "kl": "kl_gate",
            "gate": "kl_gate",
            "dual_aware": "kl_gate",
            "dual_aware_kl": "kl_gate",
            "kl_gated": "kl_gate",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_aware_drift_modes:
            raise ValueError(
                f"Unsupported dual_aware_drift_mode={mode!r}. "
                f"Choose one of {cls.valid_dual_aware_drift_modes}."
            )
        return normalized

    @classmethod
    def _normalize_critic_sync_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "none": "local",
            "local_only": "local",
            "private": "local",
            "global": "fedavg",
            "server_avg": "fedavg",
            "avg": "fedavg",
            "average": "fedavg",
            "sync": "fedavg",
            "server": "actor_like",
            "server_update": "actor_like",
            "same_as_actor": "actor_like",
            "lambda": "actor_like",
            "lambda_delta": "actor_like",
            "weighted_delta": "actor_like",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_critic_sync_modes:
            raise ValueError(
                f"Unsupported critic_sync_mode={mode!r}. Choose one of {cls.valid_critic_sync_modes}."
            )
        return normalized

    @classmethod
    def _normalize_vecnormalize_sync_mode(cls, mode: str) -> str:
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
    def _normalize_dual_return_source(cls, source: str) -> str:
        normalized = str(source).strip().lower().replace("-", "_")
        aliases = {
            "env": "raw",
            "environment": "raw",
            "original": "raw",
            "unnormalized": "raw",
            "normalised": "normalized",
            "norm": "normalized",
            "vecnorm": "normalized",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_return_sources:
            raise ValueError(
                f"Unsupported dual_return_source={source!r}. Choose one of {cls.valid_dual_return_sources}."
            )
        return normalized

    @classmethod
    def _normalize_dual_return_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "discount": "discounted",
            "disc": "discounted",
            "gamma": "discounted",
            "episodic": "undiscounted",
            "episode": "undiscounted",
            "raw_episode": "undiscounted",
            "sum": "undiscounted",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_return_modes:
            raise ValueError(
                f"Unsupported dual_return_mode={mode!r}. Choose one of {cls.valid_dual_return_modes}."
            )
        return normalized

    @classmethod
    def _normalize_dual_scale_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "off": "none",
            "false": "none",
            "raw": "none",
            "std": "std_ema",
            "standardize": "std_ema",
            "standardized": "std_ema",
            "zscore": "std_ema",
            "z_score": "std_ema",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_scale_modes:
            raise ValueError(
                f"Unsupported dual_scale_mode={mode!r}. Choose one of {cls.valid_dual_scale_modes}."
            )
        return normalized


    def _uses_global_critic(self) -> bool:
        return self.critic_sync_mode in {"fedavg", "actor_like"}

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
        explicit_keys = [key for key in state.keys() if self._is_explicit_actor_key(key)]
        if explicit_keys:
            return tuple(explicit_keys)
        return tuple(key for key, value in state.items() if th.is_floating_point(value) and not self._is_critic_key(key))

    def _critic_state_keys(self) -> tuple[str, ...]:
        state = self.policy.state_dict()
        return tuple(key for key, value in state.items() if th.is_floating_point(value) and self._is_critic_key(key))

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

    def _get_actor_state(self) -> FederatedModules:
        policy_state = self.policy.state_dict()
        return {
            self.federated_actor_module_name: OrderedDict(
                (key, policy_state[key].detach().cpu().clone()) for key in self._actor_state_keys()
            )
        }

    def _set_actor_state(self, modules: FederatedModules) -> None:
        self._set_filtered_policy_state(modules, self._actor_state_keys(), "actor")
        self._clear_optimizer_state_for_names(tuple(self._actor_named_parameters().keys()))

    def _zero_like_actor_state(self) -> FederatedModules:
        return self._zero_like_modules(self._get_actor_state())

    def _get_critic_state(self) -> FederatedModules:
        policy_state = self.policy.state_dict()
        return {
            self.federated_critic_module_name: OrderedDict(
                (key, policy_state[key].detach().cpu().clone()) for key in self._critic_state_keys()
            )
        }

    def _set_critic_state(self, modules: FederatedModules) -> None:
        self._set_filtered_policy_state(modules, self._critic_state_keys(), "critic")
        self._clear_optimizer_state_for_names(tuple(self._critic_named_parameters().keys()))

    def _zero_like_critic_state(self) -> FederatedModules:
        return self._zero_like_modules(self._get_critic_state())

    def _set_filtered_policy_state(self, modules: FederatedModules, keys: Sequence[str], label: str) -> None:
        module_name = self.federated_actor_module_name
        if module_name not in modules:
            raise KeyError(f"Missing {module_name!r} in {label} payload.")

        current_state = self.policy.state_dict()
        incoming = modules[module_name]
        expected = set(keys)
        incoming_keys = set(incoming.keys())
        missing = expected - incoming_keys
        if missing:
            raise KeyError(f"{label} payload is missing keys: {sorted(missing)}")
        unexpected = incoming_keys - set(current_state.keys())
        if unexpected:
            raise KeyError(f"{label} payload contains unknown keys: {sorted(unexpected)}")

        for key in expected:
            current_state[key] = incoming[key].to(self.device)
        self.policy.load_state_dict(current_state, strict=True)

    def _clear_optimizer_state_for_names(self, names: Sequence[str]) -> None:
        if not hasattr(self.policy, "optimizer"):
            return
        params = dict(self.policy.named_parameters())
        for name in names:
            parameter = params.get(name)
            if parameter is not None:
                self.policy.optimizer.state.pop(parameter, None)

    @staticmethod
    def _clone_modules(modules: FederatedModules) -> FederatedModules:
        return {
            module_name: OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module_state.items()
            )
            for module_name, module_state in modules.items()
        }

    @staticmethod
    def _zero_like_modules(modules: FederatedModules) -> FederatedModules:
        return {
            module_name: OrderedDict(
                (key, th.zeros_like(value)) for key, value in module_state.items()
            )
            for module_name, module_state in modules.items()
        }

    @staticmethod
    def _subtract_modules(after: FederatedModules, before: FederatedModules) -> FederatedModules:
        delta: FederatedModules = {}
        for module_name, after_state in after.items():
            if module_name not in before:
                raise KeyError(f"Missing module {module_name!r} in reference state.")
            delta[module_name] = OrderedDict()
            for key, after_value in after_state.items():
                before_value = before[module_name][key]
                if th.is_floating_point(after_value):
                    delta[module_name][key] = after_value - before_value.to(after_value.dtype)
                else:
                    delta[module_name][key] = th.zeros_like(after_value)
        return delta

    @staticmethod
    def _add_scaled_modules(base: FederatedModules, delta: FederatedModules, scale: float) -> FederatedModules:
        result: FederatedModules = {}
        for module_name, base_state in base.items():
            if module_name not in delta:
                raise KeyError(f"Missing module {module_name!r} in delta state.")
            result[module_name] = OrderedDict()
            for key, base_value in base_state.items():
                if th.is_floating_point(base_value):
                    result[module_name][key] = base_value + float(scale) * delta[module_name][key].to(base_value.dtype)
                else:
                    result[module_name][key] = base_value.clone()
        return result

    @staticmethod
    def _mix_modules(old: FederatedModules, new: FederatedModules, mix_weight: float) -> FederatedModules:
        if mix_weight >= 1.0:
            return FedAMPOLocalPPO._clone_modules(new)
        mixed: FederatedModules = {}
        for module_name, new_state in new.items():
            mixed[module_name] = OrderedDict()
            for key, new_value in new_state.items():
                old_value = old[module_name][key]
                if th.is_floating_point(new_value):
                    mixed[module_name][key] = mix_weight * new_value + (1.0 - mix_weight) * old_value.to(new_value.dtype)
                else:
                    mixed[module_name][key] = new_value.clone()
        return mixed

    @classmethod
    def _average_modules(
        cls,
        modules_list: Sequence[FederatedModules],
        weights: Sequence[float] | None = None,
    ) -> FederatedModules:
        if len(modules_list) == 0:
            raise ValueError("At least one module state is required.")
        if weights is None:
            normalized_weights = np.ones(len(modules_list), dtype=np.float64) / float(len(modules_list))
        else:
            normalized_weights = np.asarray(weights, dtype=np.float64)
            if normalized_weights.shape != (len(modules_list),):
                raise ValueError(
                    f"weights shape mismatch: expected {(len(modules_list),)}, got {normalized_weights.shape}."
                )
            weight_sum = float(np.sum(normalized_weights))
            if weight_sum <= 0.0:
                raise ValueError("weights must sum to a positive value.")
            normalized_weights = normalized_weights / weight_sum

        reference = modules_list[0]
        averaged: FederatedModules = {}
        for module_name, reference_state in reference.items():
            averaged[module_name] = OrderedDict()
            for key, reference_value in reference_state.items():
                if th.is_floating_point(reference_value):
                    value_sum = th.zeros_like(reference_value)
                    for weight, modules in zip(normalized_weights, modules_list, strict=True):
                        value_sum += modules[module_name][key].to(reference_value.dtype) * float(weight)
                    averaged[module_name][key] = value_sum
                else:
                    averaged[module_name][key] = reference_value.clone()
        return averaged

    @staticmethod
    def _module_l2_norm(modules: FederatedModules) -> float:
        total = 0.0
        for module_state in modules.values():
            for value in module_state.values():
                if th.is_floating_point(value):
                    tensor = value.detach().to(dtype=th.float64)
                    total += float(th.sum(tensor * tensor).cpu().item())
        return float(np.sqrt(total))

    @staticmethod
    def _simplex_entropy(weights: np.ndarray) -> float:
        safe = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, 1.0)
        return float(-np.sum(safe * np.log(safe)))

    @staticmethod
    def _effective_num_clients(weights: np.ndarray) -> float:
        denom = float(np.sum(np.square(np.asarray(weights, dtype=np.float64))))
        if denom <= 0.0:
            return 0.0
        return 1.0 / denom

    def _compute_dual_aware_delta_scales(
        self,
        lambda_weights: np.ndarray,
        client_kl_maxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return per-client delta scales and diagnostics for dual-aware KL gating.

        The gate is deliberately one-sided: a client is attenuated only when
        (i) its current AMPO weight is above uniform and (ii) its local PPO KL
        exceeds the reference target.  Thus uniform/dual-off training is exactly
        unchanged even when ``kl_gate`` is enabled.

        Let a_k = max(K lambda_k - 1, 0)/(K-1), and for KL_k > tau let
        d_k = (KL_k/tau - 1)/(KL_k/tau).  The risk q_k=a_k d_k lies in [0,1],
        and the server uses s_k=max(s_min, 1-alpha q_k).
        """
        weights = np.asarray(lambda_weights, dtype=np.float64)
        kl_values = np.asarray(client_kl_maxes, dtype=np.float64)
        if weights.ndim != 1 or kl_values.shape != weights.shape:
            raise ValueError(
                f"dual-aware shape mismatch: lambda={weights.shape}, kl={kl_values.shape}."
            )
        num_clients = int(weights.size)
        scales = np.ones(num_clients, dtype=np.float64)
        importance = np.zeros(num_clients, dtype=np.float64)
        drift = np.zeros(num_clients, dtype=np.float64)
        risk = np.zeros(num_clients, dtype=np.float64)
        if num_clients <= 1 or self.dual_aware_drift_mode == "none":
            return scales, importance, drift, risk

        importance = np.maximum(num_clients * weights - 1.0, 0.0) / float(num_clients - 1)
        finite = np.isfinite(kl_values)
        ratio = np.zeros(num_clients, dtype=np.float64)
        ratio[finite] = kl_values[finite] / float(self.dual_aware_kl_target)
        excess = np.maximum(ratio - 1.0, 0.0)
        # Smoothly maps 1x target -> 0, 2x -> 0.5, and large excess -> 1.
        drift = np.divide(excess, 1.0 + excess, out=np.zeros_like(excess), where=np.isfinite(excess))
        risk = np.clip(importance * drift, 0.0, 1.0)

        if self.dual_aware_drift_mode == "kl_gate":
            scales = 1.0 - self.dual_aware_correction_strength * risk
            scales = np.clip(scales, self.dual_aware_min_delta_scale, 1.0)
        return scales, importance, drift, risk

    @staticmethod
    def _uniform_lambda(num_clients: int) -> np.ndarray:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}.")
        return np.ones(num_clients, dtype=np.float64) / float(num_clients)

    def _ensure_lambda(self, num_clients: int) -> None:
        if self.dual_update_mode == "uniform":
            self.lambda_weights = self._uniform_lambda(num_clients)
            return
        if self.lambda_weights is not None and self.lambda_weights.shape == (num_clients,):
            return
        if self.initial_lambda is None:
            self.lambda_weights = self._uniform_lambda(num_clients)
            return
        initial = np.asarray(self.initial_lambda, dtype=np.float64)
        if initial.shape != (num_clients,):
            raise ValueError(f"initial_lambda shape mismatch: expected {(num_clients,)}, got {initial.shape}.")
        self.lambda_weights = project_to_simplex(initial)

    def _current_server_actor_delta_scale(self) -> float:
        return float(self.server_actor_delta_scale)

    @staticmethod
    def _set_optimizer_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def _current_clip_range(self) -> float:
        return float(self.clip_range(self._current_progress_remaining))

    def _current_clip_range_vf(self) -> float | None:
        if self.clip_range_vf is None:
            return None
        return float(self.clip_range_vf(self._current_progress_remaining))

    def _prepare_rollout_actions(self, actions: th.Tensor) -> th.Tensor:
        if isinstance(self.action_space, spaces.Discrete):
            return actions.long().flatten()
        return actions

    def _init_ampo_training_state(self) -> None:
        if self.ep_info_buffer is None or self.ep_success_buffer is None:
            self._setup_learn(
                total_timesteps=max(int(getattr(self, "_total_timesteps", 0)), 1),
                callback=None,
                reset_num_timesteps=False,
                tb_log_name="fed_ampo_local_ppo",
                progress_bar=False,
            )

        if self._last_obs is None:
            assert self.env is not None
            self._last_obs = self.env.reset()
            self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)
            self._reset_dual_episode_accumulators()

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

    def _ensure_dual_episode_accumulators(self) -> None:
        n_envs = int(self.n_envs)
        if self._ampo_dual_episode_returns is None or self._ampo_dual_episode_returns.shape != (n_envs,):
            self._ampo_dual_episode_returns = np.zeros(n_envs, dtype=np.float64)
            self._ampo_dual_episode_discounts = np.ones(n_envs, dtype=np.float64)

    def _reset_dual_episode_accumulators(self) -> None:
        if self._ampo_dual_episode_returns is not None:
            self._ampo_dual_episode_returns.fill(0.0)
        if self._ampo_dual_episode_discounts is not None:
            self._ampo_dual_episode_discounts.fill(1.0)

    def _consume_dual_training_steps(
        self,
        rewards: np.ndarray,
        dones: np.ndarray,
    ) -> list[float]:
        rewards = np.asarray(rewards, dtype=np.float64)
        dones = np.asarray(dones, dtype=bool)
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if dones.ndim == 1:
            dones = dones[:, None]
        if rewards.shape != dones.shape:
            raise ValueError(f"dual reward/done shape mismatch: {rewards.shape} vs {dones.shape}.")

        self._ensure_dual_episode_accumulators()
        assert self._ampo_dual_episode_returns is not None
        assert self._ampo_dual_episode_discounts is not None
        if rewards.shape[1] != self._ampo_dual_episode_returns.shape[0]:
            raise ValueError(
                f"Expected {self._ampo_dual_episode_returns.shape[0]} env reward streams, got {rewards.shape[1]}."
            )

        completed: list[float] = []
        for step_idx in range(rewards.shape[0]):
            for env_idx in range(rewards.shape[1]):
                reward = float(rewards[step_idx, env_idx])
                if self.dual_return_mode == "discounted":
                    self._ampo_dual_episode_returns[env_idx] += (
                        self._ampo_dual_episode_discounts[env_idx] * reward
                    )
                    self._ampo_dual_episode_discounts[env_idx] *= float(self.gamma)
                else:
                    self._ampo_dual_episode_returns[env_idx] += reward

                if bool(dones[step_idx, env_idx]):
                    completed.append(float(self._ampo_dual_episode_returns[env_idx]))
                    self._ampo_dual_episode_returns[env_idx] = 0.0
                    self._ampo_dual_episode_discounts[env_idx] = 1.0

        return completed

    def _update_dual_return_ema(self, completed_returns: Sequence[float]) -> None:
        self._ampo_last_completed_episodes = len(completed_returns)
        if not completed_returns:
            self._ampo_last_completed_episode_mean = None
            return

        episode_mean = float(np.mean(np.asarray(completed_returns, dtype=np.float64)))
        self._ampo_last_completed_episode_mean = episode_mean
        self._ampo_dual_completed_episodes += len(completed_returns)
        if self._ampo_dual_return_ema is None:
            self._ampo_dual_return_ema = episode_mean
        else:
            beta = self.dual_return_ema_beta
            self._ampo_dual_return_ema = beta * self._ampo_dual_return_ema + (1.0 - beta) * episode_mean

    def _collect_one_rollout(self) -> list[float]:
        self._init_ampo_training_state()

        reward_capture = _RawRewardCaptureCallback()
        callback = CallbackList([reward_capture])
        callback.init_callback(self)

        success = self.collect_rollouts(
            self.env,
            callback,
            self.rollout_buffer,
            n_rollout_steps=self.n_steps,
        )
        if not success:
            raise RuntimeError("collect_rollouts() returned False.")

        if self.dual_return_source == "raw":
            rewards = np.asarray(reward_capture.raw_rewards, dtype=np.float64)
        else:
            rewards = np.asarray(reward_capture.normalized_rewards, dtype=np.float64)
        dones = np.asarray(reward_capture.dones, dtype=bool)
        completed_returns = self._consume_dual_training_steps(rewards, dones)

        total_timesteps = int(getattr(self, "_total_timesteps", 0))
        if total_timesteps > 0:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        return completed_returns




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

    def _ppo_value_loss(self, values: th.Tensor, rollout_data: RolloutBufferSamples) -> th.Tensor:
        values = values.flatten()
        clip_range_vf = self._current_clip_range_vf()
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = rollout_data.old_values + th.clamp(
                values - rollout_data.old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        return F.mse_loss(rollout_data.returns, values_pred)

    def _ppo_actor_loss(
        self,
        log_prob: th.Tensor,
        entropy: th.Tensor | None,
        rollout_data: RolloutBufferSamples,
    ) -> th.Tensor:
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = th.exp(log_prob - rollout_data.old_log_prob)
        clip_range = self._current_clip_range()
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
        return policy_loss + self.ent_coef * entropy_loss

    @staticmethod
    def _named_gradient_norm(gradients: Mapping[str, th.Tensor]) -> float:
        total = 0.0
        for gradient in gradients.values():
            tensor = gradient.detach().to(dtype=th.float64)
            total += float(th.sum(tensor * tensor).cpu().item())
        return float(np.sqrt(total))

    @staticmethod
    def _named_gradient_cosine(
        first: Mapping[str, th.Tensor],
        second: Mapping[str, th.Tensor],
        epsilon: float = 1e-12,
    ) -> float:
        dot = 0.0
        first_sq = 0.0
        second_sq = 0.0
        shared_names = set(first.keys()) & set(second.keys())
        for name in shared_names:
            a = first[name].detach().to(dtype=th.float64)
            b = second[name].detach().to(device=a.device, dtype=th.float64)
            dot += float(th.sum(a * b).cpu().item())
            first_sq += float(th.sum(a * a).cpu().item())
            second_sq += float(th.sum(b * b).cpu().item())
        denom = float(np.sqrt(first_sq * second_sq))
        if denom <= epsilon:
            return float("nan")
        return float(np.clip(dot / denom, -1.0, 1.0))

    @staticmethod
    def _approx_kl_from_log_prob(log_prob: th.Tensor, old_log_prob: th.Tensor) -> float:
        with th.no_grad():
            log_ratio = log_prob - old_log_prob
            approx_kl = th.mean((th.exp(log_ratio) - 1.0) - log_ratio)
        return float(approx_kl.detach().cpu().item())

    def _capture_named_gradients(
        self,
        parameter_names: Sequence[str] | None = None,
    ) -> dict[str, th.Tensor]:
        named_parameters = dict(self.policy.named_parameters())
        names = tuple(named_parameters.keys()) if parameter_names is None else tuple(parameter_names)
        gradients: dict[str, th.Tensor] = {}
        for name in names:
            parameter = named_parameters.get(name)
            if parameter is None:
                raise KeyError(f"Unknown policy parameter {name!r}.")
            if parameter.grad is None:
                # Explicit parameter lists (used for actor gradients) need a
                # shape-compatible zero.  When capturing all gradients (used
                # for the critic), omit missing entries so Adam does not apply
                # stale momentum to parameters that had no gradient.
                if parameter_names is not None:
                    gradients[name] = th.zeros_like(parameter, memory_format=th.preserve_format)
            else:
                gradients[name] = parameter.grad.detach().clone()
        return gradients

    def _copy_actor_parameters_without_optimizer_reset(self, actor_state: FederatedModules) -> None:
        """Copy actor parameters without touching Adam state.

        This helper is used only for the temporary global-reference gradient
        evaluation inside one minibatch.  Calling ``_set_actor_state`` here
        would incorrectly clear the local PPO optimizer state.
        """
        module_name = self.federated_actor_module_name
        if module_name not in actor_state:
            raise KeyError(f"Missing {module_name!r} in actor reference state.")

        incoming = actor_state[module_name]
        actor_parameters = self._actor_named_parameters()
        missing = set(actor_parameters.keys()) - set(incoming.keys())
        if missing:
            raise KeyError(f"Actor reference state is missing parameters: {sorted(missing)}")

        with th.no_grad():
            for name, parameter in actor_parameters.items():
                parameter.copy_(incoming[name].to(device=parameter.device, dtype=parameter.dtype))

    def _set_optimizer_gradients(
        self,
        actor_gradients: Mapping[str, th.Tensor],
        critic_gradients: Mapping[str, th.Tensor],
    ) -> None:
        """Install corrected actor gradients plus ordinary critic gradients.

        Shared feature-extractor parameters may receive both actor and critic
        derivatives.  Adding both terms preserves the joint PPO update while
        replacing only the actor component by the corrected direction.
        """
        actor_names = set(self._actor_named_parameters().keys())
        for name, parameter in self.policy.named_parameters():
            gradient: th.Tensor | None = None
            critic_gradient = critic_gradients.get(name)
            if critic_gradient is not None:
                gradient = critic_gradient.to(device=parameter.device, dtype=parameter.dtype).clone()
            if name in actor_names:
                actor_gradient = actor_gradients[name].to(device=parameter.device, dtype=parameter.dtype)
                gradient = actor_gradient.clone() if gradient is None else gradient + actor_gradient
            parameter.grad = gradient

    def _update_local_actor_and_critic_standard(self) -> None:
        """Run standard local PPO optimization on the collected global-policy rollout."""
        self.policy.set_training_mode(True)
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        self._set_optimizer_learning_rate(self.policy.optimizer, float(learning_rate))

        approx_kls: list[float] = []
        num_batches = 0
        continue_training = True
        for _ in range(self.n_epochs):
            for rollout_data in self._iter_rollout_minibatches():
                actions = self._prepare_rollout_actions(rollout_data.actions)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )
                actor_minimization_loss = self._ppo_actor_loss(
                    log_prob,
                    entropy,
                    rollout_data,
                )
                critic_minimization_loss = self.vf_coef * self._ppo_value_loss(
                    values,
                    rollout_data,
                )
                loss = actor_minimization_loss + critic_minimization_loss

                # Keep the default path identical: KL is evaluated only when PPO
                # already needs target_kl or the optional diagnostic/gate is enabled.
                approx_kl: float | None = None
                if self.target_kl is not None or self.dual_aware_drift_mode != "none":
                    approx_kl = self._approx_kl_from_log_prob(log_prob, rollout_data.old_log_prob)
                    if self.dual_aware_drift_mode != "none":
                        approx_kls.append(approx_kl)

                self.policy.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                num_batches += 1

                if self.target_kl is not None:
                    assert approx_kl is not None
                    if approx_kl > 1.5 * self.target_kl:
                        continue_training = False
                        break

            if not continue_training:
                break

        if num_batches == 0:
            raise RuntimeError("No rollout minibatches were available for local PPO optimization.")
        self._ampo_last_num_actor_batches = num_batches
        if approx_kls:
            self._ampo_last_local_actor_kl_mean = float(np.mean(approx_kls))
            self._ampo_last_local_actor_kl_max = float(np.max(approx_kls))


    def _update_local_actor_and_critic_momentum(
        self,
        global_actor_state: FederatedModules,
    ) -> None:
        """Run local PPO with the Remark-4 gradient-difference correction.

        For every PPO minibatch, this computes the actor minimization gradients
        at the current local actor and at the round-start global actor on the
        same data.  In minimization-gradient notation, the update is

            d <- beta * g_local
                 + (1 - beta) * (d + g_local - g_global).

        The critic keeps its ordinary PPO gradient.  Each client preserves
        the correction state across communication rounds.  On its first use,
        the state is initialized with the global-reference gradient, making the
        first actor step identical to the standard PPO actor step when local
        and global actors coincide.
        """
        self.policy.set_training_mode(True)
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        self._set_optimizer_learning_rate(self.policy.optimizer, float(learning_rate))

        actor_parameter_names = tuple(self._actor_named_parameters().keys())
        if not actor_parameter_names:
            raise RuntimeError("Could not identify actor parameters for momentum-corrected PPO.")

        momentum_state = self._ampo_actor_momentum_state
        if momentum_state is not None:
            expected_names = set(actor_parameter_names)
            if set(momentum_state.keys()) != expected_names:
                # This can happen only after changing the policy architecture
                # while reusing an old model object/checkpoint.
                momentum_state = None
        current_norms: list[float] = []
        reference_norms: list[float] = []
        difference_norms: list[float] = []
        corrected_norms: list[float] = []
        gradient_cosines: list[float] = []
        approx_kls: list[float] = []

        num_batches = 0
        continue_training = True
        for _ in range(self.n_epochs):
            for rollout_data in self._iter_rollout_minibatches():
                actions = self._prepare_rollout_actions(rollout_data.actions)
                if self.use_sde:
                    # The same sampled exploration matrix is retained for the
                    # local and reference evaluations of this minibatch.
                    self.policy.reset_noise(self.batch_size)

                # Current local actor and critic losses.
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )
                actor_minimization_loss = self._ppo_actor_loss(
                    log_prob,
                    entropy,
                    rollout_data,
                )
                critic_minimization_loss = self.vf_coef * self._ppo_value_loss(
                    values,
                    rollout_data,
                )

                approx_kl: float | None = None
                if self.target_kl is not None or self.dual_aware_drift_mode != "none":
                    approx_kl = self._approx_kl_from_log_prob(log_prob, rollout_data.old_log_prob)
                    if self.dual_aware_drift_mode != "none":
                        approx_kls.append(approx_kl)

                # Actor component at the current local point.
                self.policy.optimizer.zero_grad(set_to_none=True)
                actor_minimization_loss.backward(retain_graph=True)
                current_actor_gradients = self._capture_named_gradients(actor_parameter_names)

                # Ordinary critic component, including a possible contribution
                # to a shared feature extractor.
                self.policy.optimizer.zero_grad(set_to_none=True)
                critic_minimization_loss.backward()
                critic_gradients = self._capture_named_gradients()
                self.policy.optimizer.zero_grad(set_to_none=True)

                # Evaluate the actor gradient at the round-start global actor
                # on exactly the same minibatch.  Restore the local actor before
                # applying the optimizer step.
                local_actor_state = self._get_actor_state()
                try:
                    self._copy_actor_parameters_without_optimizer_reset(global_actor_state)
                    _, reference_log_prob, reference_entropy = self.policy.evaluate_actions(
                        rollout_data.observations,
                        actions,
                    )
                    reference_actor_loss = self._ppo_actor_loss(
                        reference_log_prob,
                        reference_entropy,
                        rollout_data,
                    )
                    self.policy.optimizer.zero_grad(set_to_none=True)
                    reference_actor_loss.backward()
                    reference_actor_gradients = self._capture_named_gradients(actor_parameter_names)
                finally:
                    self._copy_actor_parameters_without_optimizer_reset(local_actor_state)
                    self.policy.optimizer.zero_grad(set_to_none=True)

                if momentum_state is None:
                    momentum_state = {
                        name: gradient.detach().clone()
                        for name, gradient in reference_actor_gradients.items()
                    }

                beta = float(self.momentum_beta)
                corrected_actor_gradients: dict[str, th.Tensor] = {}
                gradient_differences: dict[str, th.Tensor] = {}
                for name in actor_parameter_names:
                    local_gradient = current_actor_gradients[name]
                    reference_gradient = reference_actor_gradients[name]
                    gradient_difference = local_gradient - reference_gradient
                    corrected = beta * local_gradient + (1.0 - beta) * (
                        momentum_state[name] + gradient_difference
                    )
                    gradient_differences[name] = gradient_difference
                    corrected_actor_gradients[name] = corrected

                if self.dual_aware_drift_mode != "none":
                    cosine = self._named_gradient_cosine(current_actor_gradients, reference_actor_gradients)
                    if np.isfinite(cosine):
                        gradient_cosines.append(cosine)

                momentum_state = {
                    name: gradient.detach().clone()
                    for name, gradient in corrected_actor_gradients.items()
                }

                current_norms.append(self._named_gradient_norm(current_actor_gradients))
                reference_norms.append(self._named_gradient_norm(reference_actor_gradients))
                difference_norms.append(self._named_gradient_norm(gradient_differences))
                corrected_norms.append(self._named_gradient_norm(corrected_actor_gradients))

                self._set_optimizer_gradients(corrected_actor_gradients, critic_gradients)
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad(set_to_none=True)
                num_batches += 1

                if self.target_kl is not None:
                    assert approx_kl is not None
                    if approx_kl > 1.5 * self.target_kl:
                        continue_training = False
                        break

            if not continue_training:
                break

        if num_batches == 0:
            raise RuntimeError("No rollout minibatches were available for local PPO optimization.")

        self._ampo_actor_momentum_state = {
            name: gradient.detach().clone()
            for name, gradient in (momentum_state or {}).items()
        }
        self._ampo_last_num_actor_batches = num_batches
        self._ampo_last_current_actor_grad_norm = float(np.mean(current_norms)) if current_norms else 0.0
        self._ampo_last_reference_actor_grad_norm = float(np.mean(reference_norms)) if reference_norms else 0.0
        self._ampo_last_gradient_difference_norm = float(np.mean(difference_norms)) if difference_norms else 0.0
        self._ampo_last_corrected_actor_grad_norm = float(np.mean(corrected_norms)) if corrected_norms else 0.0
        if approx_kls:
            self._ampo_last_local_actor_kl_mean = float(np.mean(approx_kls))
            self._ampo_last_local_actor_kl_max = float(np.max(approx_kls))
        if gradient_cosines:
            self._ampo_last_gradient_cosine_mean = float(np.mean(gradient_cosines))
            self._ampo_last_gradient_cosine_min = float(np.min(gradient_cosines))

    def _update_local_actor_and_critic(
        self,
        global_actor_state: FederatedModules,
    ) -> None:
        """Dispatch to the unchanged PPO update or the optional momentum mode."""
        self._ampo_last_current_actor_grad_norm = 0.0
        self._ampo_last_reference_actor_grad_norm = 0.0
        self._ampo_last_gradient_difference_norm = 0.0
        self._ampo_last_corrected_actor_grad_norm = 0.0
        self._ampo_last_local_actor_kl_mean = float("nan")
        self._ampo_last_local_actor_kl_max = float("nan")
        self._ampo_last_gradient_cosine_mean = float("nan")
        self._ampo_last_gradient_cosine_min = float("nan")

        if self.local_actor_update_mode == "momentum":
            self._update_local_actor_and_critic_momentum(global_actor_state)
        else:
            self._update_local_actor_and_critic_standard()

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        del kwargs
        target_steps = int(local_steps)
        expected_steps = int(self.n_steps * self.n_envs)
        if target_steps != expected_steps:
            raise ValueError(
                "Fed-AMPO-LocalPPO requires exactly one global-policy rollout per "
                "communication round. Set local_steps == n_steps * n_envs. "
                f"Received local_steps={target_steps}, n_steps={self.n_steps}, "
                f"n_envs={self.n_envs}, expected={expected_steps}."
            )

        self._federated_progress_lock = float(self._current_progress_remaining)
        try:
            actor_before = self._get_actor_state()
            critic_before = self._get_critic_state() if self.critic_sync_mode == "actor_like" else None

            completed_episode_returns = self._collect_one_rollout()

            # Preserve the local-PPO variant: update actor/critic locally on the same rollout.
            self._update_local_actor_and_critic(actor_before)

            actor_after = self._get_actor_state()
            self._ampo_last_actor_delta = self._subtract_modules(actor_after, actor_before)
            self._update_dual_return_ema(completed_episode_returns)
            self._ampo_last_return = (
                float(self._ampo_dual_return_ema) if self._ampo_dual_return_ema is not None else 0.0
            )

            if self.critic_sync_mode == "fedavg":
                self._ampo_last_critic_state = self._get_critic_state()
                self._ampo_last_critic_delta = None
            elif self.critic_sync_mode == "actor_like":
                assert critic_before is not None
                self._ampo_last_critic_state = None
                self._ampo_last_critic_delta = self._subtract_modules(
                    self._get_critic_state(),
                    critic_before,
                )
            else:
                self._ampo_last_critic_state = None
                self._ampo_last_critic_delta = None
        finally:
            self._federated_progress_lock = None

    def get_upload_payload(self) -> FederatedPayload:
        if self._ampo_last_actor_delta is None:
            self._ampo_last_actor_delta = self._zero_like_actor_state()
        if self._ampo_last_return is None:
            self._ampo_last_return = 0.0

        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "actor_delta": self._clone_modules(self._ampo_last_actor_delta),
            "return": float(self._ampo_last_return),
            "num_actor_batches": int(self._ampo_last_num_actor_batches),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "critic_sync_mode": self.critic_sync_mode,
            "dual_update_mode": self.dual_update_mode,
            "vecnormalize_sync_mode": self.vecnormalize_sync_mode,
            "dual_return_source": self.dual_return_source,
            "dual_return_mode": self.dual_return_mode,
            "dual_return_ema_beta": self.dual_return_ema_beta,
            "dual_update_interval": self.dual_update_interval,
            "dual_warmup_min_episodes": self.dual_warmup_min_episodes,
            "dual_scale_mode": self.dual_scale_mode,
            "dual_scale_ema_beta": self.dual_scale_ema_beta,
            "dual_scale_epsilon": self.dual_scale_epsilon,
            "dual_return_ready": self._ampo_dual_return_ema is not None,
            "dual_completed_episodes": int(self._ampo_dual_completed_episodes),
            "dual_completed_episodes_last_update": int(self._ampo_last_completed_episodes),
            "dual_completed_episode_mean": self._ampo_last_completed_episode_mean,
            "local_actor_update_mode": self.local_actor_update_mode,
            "momentum_beta": float(self.momentum_beta),
            "dual_aware_drift_mode": self.dual_aware_drift_mode,
            "dual_aware_kl_target": float(self.dual_aware_kl_target),
            "dual_aware_correction_strength": float(self.dual_aware_correction_strength),
            "dual_aware_min_delta_scale": float(self.dual_aware_min_delta_scale),
            "local_actor_kl_mean": float(self._ampo_last_local_actor_kl_mean),
            "local_actor_kl_max": float(self._ampo_last_local_actor_kl_max),
            "gradient_cosine_mean": float(self._ampo_last_gradient_cosine_mean),
            "gradient_cosine_min": float(self._ampo_last_gradient_cosine_min),
            "current_actor_grad_norm": float(self._ampo_last_current_actor_grad_norm),
            "reference_actor_grad_norm": float(self._ampo_last_reference_actor_grad_norm),
            "gradient_difference_norm": float(self._ampo_last_gradient_difference_norm),
            "corrected_actor_grad_norm": float(self._ampo_last_corrected_actor_grad_norm),
            "vecnormalize": self._get_filtered_vecnormalize_state(),
        }

        if self.critic_sync_mode == "fedavg":
            if self._ampo_last_critic_state is None:
                self._ampo_last_critic_state = self._get_critic_state()
            payload["critic_state"] = self._clone_modules(self._ampo_last_critic_state)
        elif self.critic_sync_mode == "actor_like":
            if self._ampo_last_critic_delta is None:
                self._ampo_last_critic_delta = self._zero_like_critic_state()
            payload["critic_delta"] = self._clone_modules(self._ampo_last_critic_delta)
        return payload

    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        del weights
        if len(uploads) == 0:
            raise ValueError("At least one upload is required.")

        critic_modes = {cls._normalize_critic_sync_mode(str(upload.get("critic_sync_mode", "local"))) for upload in uploads}
        if len(critic_modes) != 1:
            raise ValueError(f"Mixed critic_sync_mode values are not supported: {critic_modes}.")
        critic_sync_mode = next(iter(critic_modes))

        dual_modes = {cls._normalize_dual_update_mode(str(upload.get("dual_update_mode", "adaptive"))) for upload in uploads}
        if len(dual_modes) != 1:
            raise ValueError(f"Mixed dual_update_mode values are not supported: {dual_modes}.")
        dual_update_mode = next(iter(dual_modes))

        vecnormalize_modes = {
            cls._normalize_vecnormalize_sync_mode(str(upload.get("vecnormalize_sync_mode", "obs_reward")))
            for upload in uploads
        }
        if len(vecnormalize_modes) != 1:
            raise ValueError(f"Mixed vecnormalize_sync_mode values are not supported: {vecnormalize_modes}.")
        vecnormalize_sync_mode = next(iter(vecnormalize_modes))

        return_sources = {
            cls._normalize_dual_return_source(str(upload.get("dual_return_source", "raw"))) for upload in uploads
        }
        if len(return_sources) != 1:
            raise ValueError(f"Mixed dual_return_source values are not supported: {return_sources}.")

        return_modes = {
            cls._normalize_dual_return_mode(str(upload.get("dual_return_mode", "discounted"))) for upload in uploads
        }
        if len(return_modes) != 1:
            raise ValueError(f"Mixed dual_return_mode values are not supported: {return_modes}.")

        scale_modes = {
            cls._normalize_dual_scale_mode(str(upload.get("dual_scale_mode", "std_ema"))) for upload in uploads
        }
        if len(scale_modes) != 1:
            raise ValueError(f"Mixed dual_scale_mode values are not supported: {scale_modes}.")

        actor_update_modes = {
            cls._normalize_local_actor_update_mode(str(upload.get("local_actor_update_mode", "standard")))
            for upload in uploads
        }
        if len(actor_update_modes) != 1:
            raise ValueError(f"Mixed local_actor_update_mode values are not supported: {actor_update_modes}.")
        local_actor_update_mode = next(iter(actor_update_modes))

        dual_aware_modes = {
            cls._normalize_dual_aware_drift_mode(str(upload.get("dual_aware_drift_mode", "none")))
            for upload in uploads
        }
        if len(dual_aware_modes) != 1:
            raise ValueError(f"Mixed dual_aware_drift_mode values are not supported: {dual_aware_modes}.")
        dual_aware_drift_mode = next(iter(dual_aware_modes))

        momentum_betas = np.asarray(
            [float(upload.get("momentum_beta", 0.9)) for upload in uploads],
            dtype=np.float64,
        )
        if not np.allclose(momentum_betas, momentum_betas[0]):
            raise ValueError(f"Mixed momentum_beta values are not supported: {momentum_betas.tolist()}.")

        def _shared_value(key: str, default: Any) -> Any:
            values = {upload.get(key, default) for upload in uploads}
            if len(values) != 1:
                raise ValueError(f"Mixed {key} values are not supported: {values}.")
            return next(iter(values))

        dual_return_ema_beta = float(_shared_value("dual_return_ema_beta", 0.9))
        dual_update_interval = int(_shared_value("dual_update_interval", 5))
        dual_warmup_min_episodes = int(_shared_value("dual_warmup_min_episodes", 1))
        dual_scale_ema_beta = float(_shared_value("dual_scale_ema_beta", 0.95))
        dual_scale_epsilon = float(_shared_value("dual_scale_epsilon", 1e-8))
        dual_aware_kl_target = float(_shared_value("dual_aware_kl_target", 0.02))
        dual_aware_correction_strength = float(_shared_value("dual_aware_correction_strength", 1.0))
        dual_aware_min_delta_scale = float(_shared_value("dual_aware_min_delta_scale", 0.5))

        payload: FederatedPayload = {
            "client_actor_deltas": [upload["actor_delta"] for upload in uploads],
            "client_returns": np.asarray([float(upload["return"]) for upload in uploads], dtype=np.float64),
            "client_num_actor_batches": np.asarray(
                [int(upload.get("num_actor_batches", 0)) for upload in uploads],
                dtype=np.int32,
            ),
            "critic_sync_mode": critic_sync_mode,
            "dual_update_mode": dual_update_mode,
            "vecnormalize_sync_mode": vecnormalize_sync_mode,
            "dual_return_source": next(iter(return_sources)),
            "dual_return_mode": next(iter(return_modes)),
            "dual_return_ema_beta": dual_return_ema_beta,
            "dual_update_interval": dual_update_interval,
            "dual_warmup_min_episodes": dual_warmup_min_episodes,
            "dual_scale_mode": next(iter(scale_modes)),
            "dual_scale_ema_beta": dual_scale_ema_beta,
            "dual_scale_epsilon": dual_scale_epsilon,
            "client_dual_return_ready": np.asarray(
                [bool(upload.get("dual_return_ready", False)) for upload in uploads], dtype=bool
            ),
            "client_completed_episodes": np.asarray(
                [int(upload.get("dual_completed_episodes", 0)) for upload in uploads], dtype=np.int64
            ),
            "client_completed_episodes_last_update": np.asarray(
                [int(upload.get("dual_completed_episodes_last_update", 0)) for upload in uploads], dtype=np.int64
            ),
            "client_completed_episode_means": np.asarray(
                [
                    np.nan if upload.get("dual_completed_episode_mean") is None
                    else float(upload["dual_completed_episode_mean"])
                    for upload in uploads
                ],
                dtype=np.float64,
            ),
            "local_actor_update_mode": local_actor_update_mode,
            "momentum_beta": float(momentum_betas[0]),
            "dual_aware_drift_mode": dual_aware_drift_mode,
            "dual_aware_kl_target": dual_aware_kl_target,
            "dual_aware_correction_strength": dual_aware_correction_strength,
            "dual_aware_min_delta_scale": dual_aware_min_delta_scale,
            "client_local_actor_kl_means": np.asarray(
                [float(upload.get("local_actor_kl_mean", np.nan)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_local_actor_kl_maxes": np.asarray(
                [float(upload.get("local_actor_kl_max", np.nan)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_gradient_cosine_means": np.asarray(
                [float(upload.get("gradient_cosine_mean", np.nan)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_gradient_cosine_mins": np.asarray(
                [float(upload.get("gradient_cosine_min", np.nan)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_current_actor_grad_norms": np.asarray(
                [float(upload.get("current_actor_grad_norm", 0.0)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_reference_actor_grad_norms": np.asarray(
                [float(upload.get("reference_actor_grad_norm", 0.0)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_gradient_difference_norms": np.asarray(
                [float(upload.get("gradient_difference_norm", 0.0)) for upload in uploads],
                dtype=np.float64,
            ),
            "client_corrected_actor_grad_norms": np.asarray(
                [float(upload.get("corrected_actor_grad_norm", 0.0)) for upload in uploads],
                dtype=np.float64,
            ),
            "num_clients": len(uploads),
        }

        if critic_sync_mode == "fedavg":
            payload["aggregated_critic_state"] = cls._average_modules(
                [upload["critic_state"] for upload in uploads],
                weights=None,
            )
        elif critic_sync_mode == "actor_like":
            payload["client_critic_deltas"] = [upload["critic_delta"] for upload in uploads]

        if vecnormalize_sync_mode == "none":
            payload["vecnormalize"] = None
            cls._last_global_vecnormalize_state = None
        else:
            payload["vecnormalize"] = cls.average_vecnormalize_states([upload.get("vecnormalize") for upload in uploads])
        return payload

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        if "actor_state" in payload and "client_actor_deltas" not in payload:
            incoming_actor = payload["actor_state"]
            if mix_weight < 1.0:
                incoming_actor = self._mix_modules(self._get_actor_state(), incoming_actor, mix_weight)
            self._set_actor_state(incoming_actor)

            incoming_lambda = payload.get("lambda_weights")
            if incoming_lambda is not None:
                self.lambda_weights = np.asarray(incoming_lambda, dtype=np.float64).copy()

            if self._uses_global_critic() and "critic_state" in payload:
                incoming_critic = payload["critic_state"]
                if mix_weight < 1.0:
                    incoming_critic = self._mix_modules(self._get_critic_state(), incoming_critic, mix_weight)
                self._set_critic_state(incoming_critic)

            self._apply_vecnormalize_from_payload(payload)
            return

        payload_critic_mode = self._normalize_critic_sync_mode(str(payload.get("critic_sync_mode", "local")))
        payload_dual_mode = self._normalize_dual_update_mode(str(payload.get("dual_update_mode", "adaptive")))
        payload_vecnormalize_mode = self._normalize_vecnormalize_sync_mode(
            str(payload.get("vecnormalize_sync_mode", "obs_reward"))
        )
        payload_return_mode = self._normalize_dual_return_mode(
            str(payload.get("dual_return_mode", "discounted"))
        )
        payload_scale_mode = self._normalize_dual_scale_mode(
            str(payload.get("dual_scale_mode", "std_ema"))
        )
        payload_actor_update_mode = self._normalize_local_actor_update_mode(
            str(payload.get("local_actor_update_mode", "standard"))
        )
        payload_dual_aware_mode = self._normalize_dual_aware_drift_mode(
            str(payload.get("dual_aware_drift_mode", "none"))
        )
        if payload_critic_mode != self.critic_sync_mode:
            raise ValueError(
                f"Server critic_sync_mode={self.critic_sync_mode!r} does not match payload {payload_critic_mode!r}."
            )
        if payload_dual_mode != self.dual_update_mode:
            raise ValueError(
                f"Server dual_update_mode={self.dual_update_mode!r} does not match payload {payload_dual_mode!r}."
            )
        if payload_vecnormalize_mode != self.vecnormalize_sync_mode:
            raise ValueError(
                f"Server vecnormalize_sync_mode={self.vecnormalize_sync_mode!r} does not match "
                f"payload {payload_vecnormalize_mode!r}."
            )
        if payload_return_mode != self.dual_return_mode:
            raise ValueError(
                f"Server dual_return_mode={self.dual_return_mode!r} does not match payload {payload_return_mode!r}."
            )
        if payload_scale_mode != self.dual_scale_mode:
            raise ValueError(
                f"Server dual_scale_mode={self.dual_scale_mode!r} does not match payload {payload_scale_mode!r}."
            )
        if payload_actor_update_mode != self.local_actor_update_mode:
            raise ValueError(
                f"Server local_actor_update_mode={self.local_actor_update_mode!r} does not match "
                f"payload {payload_actor_update_mode!r}."
            )
        if payload_dual_aware_mode != self.dual_aware_drift_mode:
            raise ValueError(
                f"Server dual_aware_drift_mode={self.dual_aware_drift_mode!r} does not match "
                f"payload {payload_dual_aware_mode!r}."
            )
        payload_momentum_beta = float(payload.get("momentum_beta", self.momentum_beta))
        if not np.isclose(payload_momentum_beta, self.momentum_beta):
            raise ValueError(
                f"Server momentum_beta={self.momentum_beta} does not match payload {payload_momentum_beta}."
            )
        for key, server_value in (
            ("dual_aware_kl_target", self.dual_aware_kl_target),
            ("dual_aware_correction_strength", self.dual_aware_correction_strength),
            ("dual_aware_min_delta_scale", self.dual_aware_min_delta_scale),
        ):
            payload_value = float(payload.get(key, server_value))
            if not np.isclose(payload_value, server_value):
                raise ValueError(f"Server {key}={server_value} does not match payload {payload_value}.")

        client_actor_deltas: list[FederatedModules] = payload["client_actor_deltas"]
        client_returns = np.asarray(payload["client_returns"], dtype=np.float64)
        num_clients = int(payload["num_clients"])
        if client_returns.shape != (num_clients,):
            raise ValueError(f"client_returns shape mismatch: expected {(num_clients,)}, got {client_returns.shape}.")

        self._ensure_lambda(num_clients)
        assert self.lambda_weights is not None
        lambda_before = self.lambda_weights.copy()

        client_local_actor_kl_means = np.asarray(
            payload.get("client_local_actor_kl_means", np.full(num_clients, np.nan)), dtype=np.float64
        )
        client_local_actor_kl_maxes = np.asarray(
            payload.get("client_local_actor_kl_maxes", np.full(num_clients, np.nan)), dtype=np.float64
        )
        if client_local_actor_kl_means.shape != (num_clients,) or client_local_actor_kl_maxes.shape != (num_clients,):
            raise ValueError(
                "dual-aware KL metric shape mismatch: "
                f"mean={client_local_actor_kl_means.shape}, max={client_local_actor_kl_maxes.shape}."
            )
        client_delta_scales, client_dual_importance, client_kl_drift_scores, client_dual_aware_risks = (
            self._compute_dual_aware_delta_scales(lambda_before, client_local_actor_kl_maxes)
        )

        actor_before = self._get_actor_state()
        module_name = self.federated_actor_module_name
        aggregated_delta: FederatedModules = {module_name: OrderedDict()}
        for key, value in actor_before[module_name].items():
            if th.is_floating_point(value):
                delta = th.zeros_like(value)
                if self.dual_aware_drift_mode == "kl_gate":
                    for lambda_weight, client_scale, client_delta in zip(
                        lambda_before, client_delta_scales, client_actor_deltas, strict=True
                    ):
                        delta += (
                            client_delta[module_name][key].to(value.dtype)
                            * float(lambda_weight)
                            * float(client_scale)
                        )
                else:
                    # Preserve the exact original aggregation when the optional
                    # correction is disabled or diagnostic-only.
                    for lambda_weight, client_delta in zip(lambda_before, client_actor_deltas, strict=True):
                        delta += client_delta[module_name][key].to(value.dtype) * float(lambda_weight)
                aggregated_delta[module_name][key] = delta
            else:
                aggregated_delta[module_name][key] = th.zeros_like(value)

        actor_delta_scale = self._current_server_actor_delta_scale()
        actor_updated = self._add_scaled_modules(actor_before, aggregated_delta, actor_delta_scale)
        if mix_weight < 1.0:
            actor_updated = self._mix_modules(actor_before, actor_updated, mix_weight)
        self._set_actor_state(actor_updated)
        actor_step = self._subtract_modules(actor_updated, actor_before)

        if self.critic_sync_mode == "fedavg":
            if "aggregated_critic_state" not in payload:
                raise KeyError("Missing 'aggregated_critic_state' for critic_sync_mode='fedavg'.")
            current_critic = self._get_critic_state()
            incoming_critic = payload["aggregated_critic_state"]
            if mix_weight < 1.0:
                incoming_critic = self._mix_modules(current_critic, incoming_critic, mix_weight)
            self._set_critic_state(incoming_critic)
        elif self.critic_sync_mode == "actor_like":
            current_critic = self._get_critic_state()
            client_critic_deltas: list[FederatedModules] = payload["client_critic_deltas"]
            critic_delta = self._weighted_module_sum(current_critic, client_critic_deltas, self.lambda_weights)
            critic_updated = self._add_scaled_modules(current_critic, critic_delta, 1.0)
            if mix_weight < 1.0:
                critic_updated = self._mix_modules(current_critic, critic_updated, mix_weight)
            self._set_critic_state(critic_updated)

        self._ampo_server_round += 1
        client_return_ready = np.asarray(
            payload.get("client_dual_return_ready", np.isfinite(client_returns)), dtype=bool
        )
        client_completed_episodes = np.asarray(
            payload.get("client_completed_episodes", np.zeros(num_clients, dtype=np.int64)), dtype=np.int64
        )
        client_completed_episodes_last_update = np.asarray(
            payload.get("client_completed_episodes_last_update", np.zeros(num_clients, dtype=np.int64)), dtype=np.int64
        )
        if client_return_ready.shape != (num_clients,):
            raise ValueError(
                f"client_dual_return_ready shape mismatch: expected {(num_clients,)}, got {client_return_ready.shape}."
            )
        if client_completed_episodes.shape != (num_clients,):
            raise ValueError(
                f"client_completed_episodes shape mismatch: expected {(num_clients,)}, got {client_completed_episodes.shape}."
            )
        if client_completed_episodes_last_update.shape != (num_clients,):
            raise ValueError(
                "client_completed_episodes_last_update shape mismatch: "
                f"expected {(num_clients,)}, got {client_completed_episodes_last_update.shape}."
            )

        warmup_ready = bool(
            np.all(client_return_ready)
            and np.all(client_completed_episodes >= self.dual_warmup_min_episodes)
            and np.all(np.isfinite(client_returns))
        )
        dual_update_due = (self._ampo_server_round % self.dual_update_interval) == 0
        dual_update_applied = False
        dual_update_skipped_small_scale = False
        dual_signal = client_returns.copy()
        current_return_spread = float(np.std(client_returns)) if warmup_ready else float("nan")

        any_new_dual_episode = bool(np.any(client_completed_episodes_last_update > 0))
        if self.dual_scale_mode == "std_ema" and warmup_ready and any_new_dual_episode:
            if (
                self._ampo_dual_scale_ema is None
                or not np.isfinite(self._ampo_dual_scale_ema)
                or self._ampo_dual_scale_ema <= self.dual_scale_epsilon
            ):
                self._ampo_dual_scale_ema = current_return_spread
            else:
                beta = self.dual_scale_ema_beta
                self._ampo_dual_scale_ema = (
                    beta * self._ampo_dual_scale_ema + (1.0 - beta) * current_return_spread
                )

        if self.dual_update_mode == "uniform":
            self.lambda_weights = self._uniform_lambda(num_clients)
        elif not warmup_ready:
            self.lambda_weights = self._uniform_lambda(num_clients)
        elif dual_update_due:
            if self.dual_scale_mode == "std_ema":
                scale = 0.0 if self._ampo_dual_scale_ema is None else float(self._ampo_dual_scale_ema)
                if not np.isfinite(scale) or scale <= self.dual_scale_epsilon:
                    dual_update_skipped_small_scale = True
                else:
                    dual_signal = (client_returns - float(np.mean(client_returns))) / scale
                    self.lambda_weights = project_to_simplex(
                        self.lambda_weights - self.dual_lr * dual_signal
                    )
                    dual_update_applied = True
            else:
                self.lambda_weights = project_to_simplex(
                    self.lambda_weights - self.dual_lr * client_returns
                )
                dual_update_applied = True

        lambda_after = self.lambda_weights.copy()
        client_delta_norms = [self._module_l2_norm(client_delta) for client_delta in client_actor_deltas]
        self._record_server_metrics(
            num_clients=num_clients,
            client_returns=client_returns,
            client_delta_norms=client_delta_norms,
            aggregated_delta_norm=self._module_l2_norm(aggregated_delta),
            actor_step_norm=self._module_l2_norm(actor_step),
            actor_delta_scale=actor_delta_scale,
            lambda_before=lambda_before,
            lambda_after=lambda_after,
            client_num_actor_batches=np.asarray(payload.get("client_num_actor_batches", []), dtype=np.int32),
            client_current_actor_grad_norms=np.asarray(
                payload.get("client_current_actor_grad_norms", []), dtype=np.float64
            ),
            client_reference_actor_grad_norms=np.asarray(
                payload.get("client_reference_actor_grad_norms", []), dtype=np.float64
            ),
            client_gradient_difference_norms=np.asarray(
                payload.get("client_gradient_difference_norms", []), dtype=np.float64
            ),
            client_corrected_actor_grad_norms=np.asarray(
                payload.get("client_corrected_actor_grad_norms", []), dtype=np.float64
            ),
            client_local_actor_kl_means=client_local_actor_kl_means,
            client_local_actor_kl_maxes=client_local_actor_kl_maxes,
            client_gradient_cosine_means=np.asarray(
                payload.get("client_gradient_cosine_means", np.full(num_clients, np.nan)), dtype=np.float64
            ),
            client_gradient_cosine_mins=np.asarray(
                payload.get("client_gradient_cosine_mins", np.full(num_clients, np.nan)), dtype=np.float64
            ),
            client_delta_scales=client_delta_scales,
            client_dual_importance=client_dual_importance,
            client_kl_drift_scores=client_kl_drift_scores,
            client_dual_aware_risks=client_dual_aware_risks,
            client_completed_episodes=client_completed_episodes,
            client_completed_episodes_last_update=client_completed_episodes_last_update,
            warmup_ready=warmup_ready,
            dual_update_due=dual_update_due,
            dual_update_applied=dual_update_applied,
            dual_update_skipped_small_scale=dual_update_skipped_small_scale,
            dual_signal=dual_signal,
            current_return_spread=current_return_spread,
        )
        self._apply_vecnormalize_from_payload(payload)

    def _weighted_module_sum(
        self,
        reference: FederatedModules,
        modules_list: Sequence[FederatedModules],
        weights: np.ndarray,
    ) -> FederatedModules:
        module_name = next(iter(reference.keys()))
        result: FederatedModules = {module_name: OrderedDict()}
        for key, value in reference[module_name].items():
            if th.is_floating_point(value):
                total = th.zeros_like(value)
                for weight, modules in zip(weights, modules_list, strict=True):
                    total += modules[module_name][key].to(value.dtype) * float(weight)
                result[module_name][key] = total
            else:
                result[module_name][key] = th.zeros_like(value)
        return result

    def _record_server_metrics(
        self,
        *,
        num_clients: int,
        client_returns: np.ndarray,
        client_delta_norms: list[float],
        aggregated_delta_norm: float,
        actor_step_norm: float,
        actor_delta_scale: float,
        lambda_before: np.ndarray,
        lambda_after: np.ndarray,
        client_num_actor_batches: np.ndarray,
        client_current_actor_grad_norms: np.ndarray,
        client_reference_actor_grad_norms: np.ndarray,
        client_gradient_difference_norms: np.ndarray,
        client_corrected_actor_grad_norms: np.ndarray,
        client_local_actor_kl_means: np.ndarray,
        client_local_actor_kl_maxes: np.ndarray,
        client_gradient_cosine_means: np.ndarray,
        client_gradient_cosine_mins: np.ndarray,
        client_delta_scales: np.ndarray,
        client_dual_importance: np.ndarray,
        client_kl_drift_scores: np.ndarray,
        client_dual_aware_risks: np.ndarray,
        client_completed_episodes: np.ndarray,
        client_completed_episodes_last_update: np.ndarray,
        warmup_ready: bool,
        dual_update_due: bool,
        dual_update_applied: bool,
        dual_update_skipped_small_scale: bool,
        dual_signal: np.ndarray,
        current_return_spread: float,
    ) -> None:
        metrics: dict[str, float] = {
            "server/num_clients": float(num_clients),
            "server/ampo/return_mean": float(np.mean(client_returns)),
            "server/ampo/return_std": float(np.std(client_returns)),
            "server/ampo/return_min": float(np.min(client_returns)),
            "server/ampo/return_max": float(np.max(client_returns)),
            "server/ampo/client_delta_norm_mean": float(np.mean(client_delta_norms)),
            "server/ampo/client_delta_norm_std": float(np.std(client_delta_norms)),
            "server/ampo/client_delta_norm_min": float(np.min(client_delta_norms)),
            "server/ampo/client_delta_norm_max": float(np.max(client_delta_norms)),
            "server/ampo/aggregated_delta_norm": float(aggregated_delta_norm),
            "server/ampo/actor_step_norm": float(actor_step_norm),
            "server/ampo/server_actor_delta_scale": float(actor_delta_scale),
            "server/ampo/dual_lr": float(self.dual_lr),
            "server/ampo/dual_round": float(self._ampo_server_round),
            "server/ampo/dual_update_interval": float(self.dual_update_interval),
            "server/ampo/dual_warmup_ready": float(warmup_ready),
            "server/ampo/dual_update_due": float(dual_update_due),
            "server/ampo/dual_update_applied": float(dual_update_applied),
            "server/ampo/dual_update_skipped_small_scale": float(dual_update_skipped_small_scale),
            "server/ampo/dual_return_discounted": float(self.dual_return_mode == "discounted"),
            "server/ampo/dual_return_undiscounted": float(self.dual_return_mode == "undiscounted"),
            "server/ampo/dual_return_ema_beta": float(self.dual_return_ema_beta),
            "server/ampo/dual_scale_none": float(self.dual_scale_mode == "none"),
            "server/ampo/dual_scale_std_ema": float(self.dual_scale_mode == "std_ema"),
            "server/ampo/dual_scale_ema_beta": float(self.dual_scale_ema_beta),
            "server/ampo/dual_scale_value": float(self._ampo_dual_scale_ema)
            if self._ampo_dual_scale_ema is not None else float("nan"),
            "server/ampo/dual_current_return_spread": float(current_return_spread),
            "server/ampo/dual_signal_mean": float(np.mean(dual_signal)),
            "server/ampo/dual_signal_std": float(np.std(dual_signal)),
            "server/ampo/dual_uniform": float(self.dual_update_mode == "uniform"),
            "server/ampo/critic_local": float(self.critic_sync_mode == "local"),
            "server/ampo/critic_fedavg": float(self.critic_sync_mode == "fedavg"),
            "server/ampo/critic_actor_like": float(self.critic_sync_mode == "actor_like"),
            "server/ampo/vecnormalize_none": float(self.vecnormalize_sync_mode == "none"),
            "server/ampo/vecnormalize_obs": float(self.vecnormalize_sync_mode == "obs"),
            "server/ampo/vecnormalize_reward": float(self.vecnormalize_sync_mode == "reward"),
            "server/ampo/vecnormalize_obs_reward": float(self.vecnormalize_sync_mode == "obs_reward"),
            "server/ampo/dual_return_raw": float(self.dual_return_source == "raw"),
            "server/ampo/dual_return_normalized": float(self.dual_return_source == "normalized"),
            "server/ampo/local_actor_update_standard": float(self.local_actor_update_mode == "standard"),
            "server/ampo/local_actor_update_momentum": float(self.local_actor_update_mode == "momentum"),
            "server/ampo/momentum_beta": float(self.momentum_beta),
            "server/ampo/dual_aware_none": float(self.dual_aware_drift_mode == "none"),
            "server/ampo/dual_aware_diagnostic": float(self.dual_aware_drift_mode == "diagnostic"),
            "server/ampo/dual_aware_kl_gate": float(self.dual_aware_drift_mode == "kl_gate"),
            "server/ampo/dual_aware_kl_target": float(self.dual_aware_kl_target),
            "server/ampo/dual_aware_correction_strength": float(self.dual_aware_correction_strength),
            "server/ampo/dual_aware_min_delta_scale": float(self.dual_aware_min_delta_scale),
            "server/ampo/dual_aware_scale_mean": float(np.mean(client_delta_scales)),
            "server/ampo/dual_aware_scale_min": float(np.min(client_delta_scales)),
            "server/ampo/dual_aware_risk_mean": float(np.mean(client_dual_aware_risks)),
            "server/ampo/dual_aware_risk_max": float(np.max(client_dual_aware_risks)),
            "server/ampo/lambda_entropy": self._simplex_entropy(lambda_after),
            "server/ampo/lambda_min": float(np.min(lambda_after)),
            "server/ampo/lambda_max": float(np.max(lambda_after)),
            "server/ampo/lambda_delta_norm": float(np.linalg.norm(lambda_after - lambda_before)),
            "server/ampo/effective_clients": self._effective_num_clients(lambda_after),
        }
        if client_num_actor_batches.size > 0:
            metrics["server/ampo/actor_batches_mean"] = float(np.mean(client_num_actor_batches))
        if client_current_actor_grad_norms.size > 0:
            metrics["server/ampo/current_actor_grad_norm_mean"] = float(np.mean(client_current_actor_grad_norms))
        if client_reference_actor_grad_norms.size > 0:
            metrics["server/ampo/reference_actor_grad_norm_mean"] = float(np.mean(client_reference_actor_grad_norms))
        if client_gradient_difference_norms.size > 0:
            metrics["server/ampo/gradient_difference_norm_mean"] = float(np.mean(client_gradient_difference_norms))
        if client_corrected_actor_grad_norms.size > 0:
            metrics["server/ampo/corrected_actor_grad_norm_mean"] = float(np.mean(client_corrected_actor_grad_norms))

        def _finite_mean(values: np.ndarray) -> float | None:
            finite_values = values[np.isfinite(values)]
            if finite_values.size == 0:
                return None
            return float(np.mean(finite_values))

        for metric_name, values in (
            ("server/ampo/local_actor_kl_mean", client_local_actor_kl_means),
            ("server/ampo/local_actor_kl_max_mean", client_local_actor_kl_maxes),
            ("server/ampo/gradient_cosine_mean", client_gradient_cosine_means),
            ("server/ampo/gradient_cosine_min_mean", client_gradient_cosine_mins),
        ):
            finite_value = _finite_mean(values)
            if finite_value is not None:
                metrics[metric_name] = finite_value

        for client_idx, (client_return, lambda_weight, delta_norm) in enumerate(
            zip(client_returns, lambda_after, client_delta_norms, strict=True)
        ):
            metrics[f"server/ampo/client_{client_idx}/return"] = float(client_return)
            metrics[f"server/ampo/client_{client_idx}/lambda"] = float(lambda_weight)
            metrics[f"server/ampo/client_{client_idx}/delta_norm"] = float(delta_norm)
            metrics[f"server/ampo/client_{client_idx}/lambda_actor"] = float(lambda_before[client_idx])
            metrics[f"server/ampo/client_{client_idx}/dual_aware_scale"] = float(client_delta_scales[client_idx])
            metrics[f"server/ampo/client_{client_idx}/dual_importance"] = float(client_dual_importance[client_idx])
            metrics[f"server/ampo/client_{client_idx}/kl_drift_score"] = float(client_kl_drift_scores[client_idx])
            metrics[f"server/ampo/client_{client_idx}/dual_aware_risk"] = float(client_dual_aware_risks[client_idx])
            if np.isfinite(client_local_actor_kl_means[client_idx]):
                metrics[f"server/ampo/client_{client_idx}/local_actor_kl_mean"] = float(
                    client_local_actor_kl_means[client_idx]
                )
            if np.isfinite(client_local_actor_kl_maxes[client_idx]):
                metrics[f"server/ampo/client_{client_idx}/local_actor_kl_max"] = float(
                    client_local_actor_kl_maxes[client_idx]
                )
            if np.isfinite(client_gradient_cosine_means[client_idx]):
                metrics[f"server/ampo/client_{client_idx}/gradient_cosine_mean"] = float(
                    client_gradient_cosine_means[client_idx]
                )
            metrics[f"server/ampo/client_{client_idx}/completed_episodes"] = float(
                client_completed_episodes[client_idx]
            )
            metrics[f"server/ampo/client_{client_idx}/completed_episodes_last_update"] = float(
                client_completed_episodes_last_update[client_idx]
            )
            metrics[f"server/ampo/client_{client_idx}/dual_signal"] = float(dual_signal[client_idx])
        self._last_federated_metrics = metrics

    def get_client_weight(self) -> float:
        return 1.0

    def get_broadcast_payload(self) -> FederatedPayload:
        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "critic_sync_mode": self.critic_sync_mode,
            "dual_update_mode": self.dual_update_mode,
            "vecnormalize_sync_mode": self.vecnormalize_sync_mode,
            "dual_return_source": self.dual_return_source,
            "dual_return_mode": self.dual_return_mode,
            "dual_scale_mode": self.dual_scale_mode,
            "local_actor_update_mode": self.local_actor_update_mode,
            "momentum_beta": float(self.momentum_beta),
            "dual_aware_drift_mode": self.dual_aware_drift_mode,
            "dual_aware_kl_target": float(self.dual_aware_kl_target),
            "dual_aware_correction_strength": float(self.dual_aware_correction_strength),
            "dual_aware_min_delta_scale": float(self.dual_aware_min_delta_scale),
            "vecnormalize": self._get_filtered_vecnormalize_state(),
        }
        if self._uses_global_critic():
            payload["critic_state"] = self._get_critic_state()
        return payload

    @staticmethod
    def _get_rms_state(rms: Any) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
        if isinstance(rms, Mapping):
            return {key: FedAMPOLocalPPO._get_rms_state(value) for key, value in rms.items()}
        return {
            "mean": np.asarray(rms.mean, dtype=np.float64).copy(),
            "var": np.asarray(rms.var, dtype=np.float64).copy(),
            "count": float(rms.count),
        }

    @staticmethod
    def _set_rms_state(rms: Any, state: RunningMeanStdState | dict[str, RunningMeanStdState]) -> None:
        if isinstance(rms, Mapping):
            for key, value in state.items():
                if key in rms:
                    FedAMPOLocalPPO._set_rms_state(rms[key], value)
            return

        mean = np.asarray(state["mean"], dtype=np.float64)
        var = np.asarray(state["var"], dtype=np.float64)
        count = float(state["count"])
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(var)) or not np.isfinite(count):
            raise ValueError(f"Invalid VecNormalize RMS state: count={count}, mean={mean}, var={var}.")
        rms.mean = mean.copy()
        rms.var = np.maximum(var, 1e-12).copy()
        rms.count = max(count, 1e-4)

    def _get_vecnormalize_state(self) -> VecNormalizeState | None:
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
            self._last_original_obs = current_original_obs

    def _apply_vecnormalize_from_payload(self, payload: FederatedPayload) -> None:
        payload_mode = self._normalize_vecnormalize_sync_mode(
            str(payload.get("vecnormalize_sync_mode", self.vecnormalize_sync_mode))
        )
        if payload_mode != self.vecnormalize_sync_mode:
            raise ValueError(
                f"vecnormalize_sync_mode={self.vecnormalize_sync_mode!r} does not match payload {payload_mode!r}."
            )
        if self.vecnormalize_sync_mode == "none":
            return
        self._set_vecnormalize_state(payload.get("vecnormalize"))

    @staticmethod
    def _clone_state(state: Any) -> Any:
        if state is None:
            return None
        if isinstance(state, np.ndarray):
            return state.copy()
        if isinstance(state, Mapping):
            return {key: FedAMPOLocalPPO._clone_state(value) for key, value in state.items()}
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
        valid_states = [state for state in states if cls._rms_is_finite(state) and float(state["count"]) > 0.0]
        if not valid_states:
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
            total = count + other_count
            if total <= 0.0:
                continue
            delta = other_mean - mean
            m2 = m2 + other_var * other_count + np.square(delta) * count * other_count / total
            mean = mean + delta * other_count / total
            count = total
        return {"mean": mean, "var": np.maximum(m2 / max(count, 1e-12), 1e-12), "count": max(count, 1e-4)}

    @classmethod
    def _subtract_single_rms_state(
        cls,
        final_state: RunningMeanStdState,
        base_state: RunningMeanStdState,
    ) -> RunningMeanStdState | None:
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
        inc_var = np.maximum(np.maximum(m2_final - m2_base - correction, 0.0) / inc_count, 1e-12)
        if not np.all(np.isfinite(inc_mean)) or not np.all(np.isfinite(inc_var)):
            return None
        return {"mean": inc_mean, "var": inc_var, "count": inc_count}

    @classmethod
    def _aggregate_rms_states(
        cls,
        states: Sequence[RunningMeanStdState | dict[str, RunningMeanStdState]],
        base_state: RunningMeanStdState | dict[str, RunningMeanStdState] | None = None,
    ) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
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
                merged_increment = cls._merge_single_rms_states(increments)  # type: ignore[arg-type]
                return cls._merge_single_rms_states([base_state, merged_increment])  # type: ignore[list-item]
        return cls._merge_single_rms_states(states)  # type: ignore[arg-type]

    @classmethod
    def average_vecnormalize_states(cls, states: Sequence[VecNormalizeState | None]) -> VecNormalizeState | None:
        valid_states = [state for state in states if state is not None]
        if not valid_states:
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


# Preferred concise alias for the local-PPO variant.
FedAMPOLPPO = FedAMPOLocalPPO

# Backward-compatible aliases for existing registries/configurations.
FedAMPOPPO = FedAMPOLocalPPO
FedAMPPO = FedAMPOLocalPPO
