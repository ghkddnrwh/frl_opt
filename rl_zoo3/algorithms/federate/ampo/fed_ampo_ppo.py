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
    """Record rollout rewards before VecNormalize reward scaling changes J_k."""

    def __init__(self) -> None:
        super().__init__()
        self.rewards: list[np.ndarray] = []
        self.episode_starts: list[np.ndarray] = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals["rewards"], dtype=np.float64)
        vecnormalize = self.model.get_vec_normalize_env()
        if vecnormalize is not None and getattr(vecnormalize, "norm_reward", False):
            rewards = np.asarray(vecnormalize.get_original_reward(), dtype=np.float64)

        episode_starts = getattr(self.model, "_last_episode_starts", None)
        if episode_starts is None:
            episode_starts = np.zeros_like(rewards, dtype=bool)

        self.rewards.append(rewards.copy())
        self.episode_starts.append(np.asarray(episode_starts, dtype=bool).copy())
        return True


class FedAMPOPPO(FederatedAlgorithmMixin, PPO):
    """Fed-AMPO-PPO implemented directly from the AMPO/PPO algorithm.

    Paper-level round:
      1. the server broadcasts the shared actor theta,
      2. each client collects local trajectories,
      3. each client computes actor/critic PPO gradients on the same local
         minibatches, applies only the critic update locally, and uploads the
         actor gradient with a Monte-Carlo local return J_k,
      4. the server applies theta <- theta + eta_pi * sum_k lambda_k g_k,
      5. the server applies lambda <- Proj_Delta(lambda - eta_lambda * J).

    ``dual_update_mode="uniform"`` freezes lambda to 1/K for the ablation.
    ``critic_sync_mode="local"`` is the paper default; ``"fedavg"`` and
    ``"actor_like"`` are server-critic ablations.
    """

    federated_actor_module_name = "policy"
    federated_critic_module_name = "policy"
    valid_dual_update_modes: tuple[str, ...] = ("adaptive", "uniform")
    valid_critic_sync_modes: tuple[str, ...] = ("local", "fedavg", "actor_like")
    valid_vecnormalize_sync_modes: tuple[str, ...] = ("none", "obs", "reward", "obs_reward")
    valid_dual_return_sources: tuple[str, ...] = ("raw", "normalized")
    valid_actor_gradient_modes: tuple[str, ...] = ("mean", "cumulative")

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
        "actor_gradient_mode",
        "server_actor_optimizer",
        "vecnormalize_sync_mode",
        "dual_return_source",
        "log_wandb",
    )

    _last_global_vecnormalize_state: VecNormalizeState | None = None

    def __init__(self, *args, **kwargs):
        self.dual_lr = float(kwargs.pop("dual_lr", 0.05))
        self.initial_lambda = kwargs.pop("initial_lambda", None)

        server_actor_lr = kwargs.pop("server_actor_lr", None)
        self.server_actor_lr = None if server_actor_lr is None else float(server_actor_lr)

        self.actor_gradient_mode = self._normalize_actor_gradient_mode(kwargs.pop("actor_gradient_mode", "mean"))
        self.critic_sync_mode = self._normalize_critic_sync_mode(kwargs.pop("critic_sync_mode", "local"))
        self.vecnormalize_sync_mode = self._normalize_vecnormalize_sync_mode(
            kwargs.pop("vecnormalize_sync_mode", "obs_reward")
        )
        self.dual_return_source = self._normalize_dual_return_source(kwargs.pop("dual_return_source", "raw"))

        raw_dual_mode = kwargs.pop("dual_update_mode", None)
        fixed_uniform = self._as_bool(kwargs.pop("fixed_uniform_lambda", False)) or self._as_bool(
            kwargs.pop("dual_fixed_uniform", False)
        )
        if raw_dual_mode is None:
            raw_dual_mode = "uniform" if fixed_uniform else "adaptive"
        self.dual_update_mode = self._normalize_dual_update_mode(raw_dual_mode)

        for key in self.federated_manager_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        self.lambda_weights: np.ndarray | None = None
        self._ampo_last_actor_gradient: FederatedModules | None = None
        self._ampo_last_return: float | None = None
        self._ampo_last_critic_state: FederatedModules | None = None
        self._ampo_last_critic_delta: FederatedModules | None = None
        self._ampo_last_num_actor_batches: int = 0
        self._ampo_last_raw_rewards: np.ndarray | None = None
        self._ampo_last_raw_episode_starts: np.ndarray | None = None
        self._last_federated_metrics: dict[str, float] = {}

    @classmethod
    def reset_federated_state(cls) -> None:
        cls._last_global_vecnormalize_state = None

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        return True

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
    def _normalize_actor_gradient_mode(cls, mode: str) -> str:
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {"sum": "cumulative", "accumulate": "cumulative", "average": "mean", "avg": "mean"}
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_actor_gradient_modes:
            raise ValueError(
                f"Unsupported actor_gradient_mode={mode!r}. Choose one of {cls.valid_actor_gradient_modes}."
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
            return FedAMPOPPO._clone_modules(new)
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

    def _current_server_actor_lr(self) -> float:
        if self.server_actor_lr is not None:
            return float(self.server_actor_lr)
        lr = self.lr_schedule(self._current_progress_remaining) if callable(self.lr_schedule) else self.learning_rate
        return float(lr) if lr is not None else 3e-4

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
                tb_log_name="fed_ampo_ppo",
                progress_bar=False,
            )

        if self._last_obs is None:
            assert self.env is not None
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

    def _collect_one_rollout(self) -> float:
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

        self._ampo_last_raw_rewards = np.asarray(reward_capture.rewards, dtype=np.float64)
        self._ampo_last_raw_episode_starts = np.asarray(reward_capture.episode_starts, dtype=bool)

        total_timesteps = int(getattr(self, "_total_timesteps", 0))
        if total_timesteps > 0:
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        return self._estimate_local_return()

    def _estimate_local_return(self) -> float:
        if (
            self.dual_return_source == "raw"
            and self._ampo_last_raw_rewards is not None
            and self._ampo_last_raw_episode_starts is not None
            and self._ampo_last_raw_rewards.size > 0
        ):
            return self._discounted_return_from_arrays(
                self._ampo_last_raw_rewards,
                self._ampo_last_raw_episode_starts,
            )
        return self._discounted_return_from_arrays(
            np.asarray(self.rollout_buffer.rewards.copy(), dtype=np.float64),
            np.asarray(self.rollout_buffer.episode_starts.copy(), dtype=bool),
        )

    def _discounted_return_from_arrays(self, rewards: np.ndarray, episode_starts: np.ndarray) -> float:
        rewards = np.asarray(rewards, dtype=np.float64)
        episode_starts = np.asarray(episode_starts, dtype=bool)
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        if episode_starts.ndim == 1:
            episode_starts = episode_starts[:, None]

        returns: list[float] = []
        partial_returns: list[float] = []
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
                partial_returns.append(running_return)

        if returns:
            return float(np.mean(returns))
        if partial_returns:
            return float(np.mean(partial_returns))
        return 0.0

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

    def _update_local_critic_and_compute_actor_gradient(self) -> FederatedModules:
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for Fed-AMPO-PPO.")

        critic_params = self._critic_named_parameters()
        gradient = self._zero_like_actor_state()
        module_name = self.federated_actor_module_name
        num_batches = 0

        self.policy.set_training_mode(True)
        learning_rate = self.lr_schedule(self._current_progress_remaining)
        self._set_optimizer_learning_rate(self.policy.optimizer, float(learning_rate))

        for _ in range(self.n_epochs):
            for rollout_data in self._iter_rollout_minibatches():
                actions = self._prepare_rollout_actions(rollout_data.actions)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                actor_minimization_loss = self._ppo_actor_loss(log_prob, entropy, rollout_data)
                critic_minimization_loss = self.vf_coef * self._ppo_value_loss(values, rollout_data)

                # Actor theta is server-owned: use the same minibatch as the critic,
                # but keep value-loss gradients out of the uploaded actor gradient.
                self.policy.optimizer.zero_grad(set_to_none=True)
                actor_minimization_loss.backward(retain_graph=bool(critic_params))
                for name, parameter in self.policy.named_parameters():
                    if name not in actor_params:
                        parameter.grad = None
                th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)

                for name, parameter in actor_params.items():
                    if parameter.grad is not None and name in gradient[module_name]:
                        gradient[module_name][name] += -parameter.grad.detach().cpu()

                self.policy.optimizer.zero_grad(set_to_none=True)
                if not critic_params:
                    num_batches += 1
                    continue

                critic_minimization_loss.backward()
                for name, parameter in self.policy.named_parameters():
                    if name not in critic_params:
                        parameter.grad = None
                th.nn.utils.clip_grad_norm_(list(critic_params.values()), self.max_grad_norm)
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad(set_to_none=True)
                num_batches += 1

        if num_batches == 0:
            raise RuntimeError("No rollout minibatches were available for actor-gradient estimation.")
        if self.actor_gradient_mode == "mean":
            for key in gradient[module_name].keys():
                gradient[module_name][key] /= float(num_batches)
        self._ampo_last_num_actor_batches = num_batches
        return gradient

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        del kwargs
        target_steps = int(local_steps)
        if target_steps <= 0:
            raise ValueError(f"local_steps must be positive, got {local_steps}.")

        critic_before = self._get_critic_state() if self.critic_sync_mode == "actor_like" else None
        collected_steps = 0
        returns: list[float] = []
        accumulated_gradient: FederatedModules | None = None
        num_rollouts = 0
        num_actor_batches = 0

        while collected_steps < target_steps:
            rollout_return = self._collect_one_rollout()
            actor_gradient = self._update_local_critic_and_compute_actor_gradient()

            if accumulated_gradient is None:
                accumulated_gradient = self._clone_modules(actor_gradient)
            else:
                for module_name, module_state in accumulated_gradient.items():
                    for key in module_state.keys():
                        module_state[key] += actor_gradient[module_name][key]

            returns.append(rollout_return)
            collected_steps += self.n_steps * self.n_envs
            num_rollouts += 1
            num_actor_batches += self._ampo_last_num_actor_batches

        assert accumulated_gradient is not None
        for module_state in accumulated_gradient.values():
            for key in module_state.keys():
                module_state[key] /= float(num_rollouts)

        self._ampo_last_actor_gradient = accumulated_gradient
        self._ampo_last_return = float(np.mean(returns)) if returns else 0.0
        self._ampo_last_num_actor_batches = num_actor_batches

        if self.critic_sync_mode == "fedavg":
            self._ampo_last_critic_state = self._get_critic_state()
            self._ampo_last_critic_delta = None
        elif self.critic_sync_mode == "actor_like":
            assert critic_before is not None
            self._ampo_last_critic_state = None
            self._ampo_last_critic_delta = self._subtract_modules(self._get_critic_state(), critic_before)
        else:
            self._ampo_last_critic_state = None
            self._ampo_last_critic_delta = None

    def get_upload_payload(self) -> FederatedPayload:
        if self._ampo_last_actor_gradient is None:
            self._ampo_last_actor_gradient = self._zero_like_actor_state()
        if self._ampo_last_return is None:
            self._ampo_last_return = 0.0

        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "actor_gradient": self._clone_modules(self._ampo_last_actor_gradient),
            "return": float(self._ampo_last_return),
            "num_actor_batches": int(self._ampo_last_num_actor_batches),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "critic_sync_mode": self.critic_sync_mode,
            "dual_update_mode": self.dual_update_mode,
            "vecnormalize_sync_mode": self.vecnormalize_sync_mode,
            "dual_return_source": self.dual_return_source,
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

        payload: FederatedPayload = {
            "client_actor_gradients": [upload["actor_gradient"] for upload in uploads],
            "client_returns": np.asarray([float(upload["return"]) for upload in uploads], dtype=np.float64),
            "client_num_actor_batches": np.asarray(
                [int(upload.get("num_actor_batches", 0)) for upload in uploads],
                dtype=np.int32,
            ),
            "critic_sync_mode": critic_sync_mode,
            "dual_update_mode": dual_update_mode,
            "vecnormalize_sync_mode": vecnormalize_sync_mode,
            "dual_return_source": next(iter(return_sources)),
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

        if "actor_state" in payload and "client_actor_gradients" not in payload:
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

        client_actor_gradients: list[FederatedModules] = payload["client_actor_gradients"]
        client_returns = np.asarray(payload["client_returns"], dtype=np.float64)
        num_clients = int(payload["num_clients"])
        if client_returns.shape != (num_clients,):
            raise ValueError(f"client_returns shape mismatch: expected {(num_clients,)}, got {client_returns.shape}.")

        self._ensure_lambda(num_clients)
        assert self.lambda_weights is not None
        lambda_before = self.lambda_weights.copy()

        actor_before = self._get_actor_state()
        module_name = self.federated_actor_module_name
        aggregated_gradient: FederatedModules = {module_name: OrderedDict()}
        for key, value in actor_before[module_name].items():
            if th.is_floating_point(value):
                grad = th.zeros_like(value)
                for lambda_weight, client_gradient in zip(self.lambda_weights, client_actor_gradients, strict=True):
                    grad += client_gradient[module_name][key].to(value.dtype) * float(lambda_weight)
                aggregated_gradient[module_name][key] = grad
            else:
                aggregated_gradient[module_name][key] = th.zeros_like(value)

        actor_lr = self._current_server_actor_lr()
        actor_updated = self._add_scaled_modules(actor_before, aggregated_gradient, actor_lr)
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

        if self.dual_update_mode == "uniform":
            self.lambda_weights = self._uniform_lambda(num_clients)
        else:
            self.lambda_weights = project_to_simplex(self.lambda_weights - self.dual_lr * client_returns)

        lambda_after = self.lambda_weights.copy()
        client_gradient_norms = [self._module_l2_norm(client_gradient) for client_gradient in client_actor_gradients]
        self._record_server_metrics(
            num_clients=num_clients,
            client_returns=client_returns,
            client_gradient_norms=client_gradient_norms,
            aggregated_gradient_norm=self._module_l2_norm(aggregated_gradient),
            actor_step_norm=self._module_l2_norm(actor_step),
            actor_lr=actor_lr,
            lambda_before=lambda_before,
            lambda_after=lambda_after,
            client_num_actor_batches=np.asarray(payload.get("client_num_actor_batches", []), dtype=np.int32),
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
        client_gradient_norms: list[float],
        aggregated_gradient_norm: float,
        actor_step_norm: float,
        actor_lr: float,
        lambda_before: np.ndarray,
        lambda_after: np.ndarray,
        client_num_actor_batches: np.ndarray,
    ) -> None:
        metrics: dict[str, float] = {
            "server/num_clients": float(num_clients),
            "server/ampo/return_mean": float(np.mean(client_returns)),
            "server/ampo/return_std": float(np.std(client_returns)),
            "server/ampo/return_min": float(np.min(client_returns)),
            "server/ampo/return_max": float(np.max(client_returns)),
            "server/ampo/client_grad_norm_mean": float(np.mean(client_gradient_norms)),
            "server/ampo/client_grad_norm_std": float(np.std(client_gradient_norms)),
            "server/ampo/client_grad_norm_min": float(np.min(client_gradient_norms)),
            "server/ampo/client_grad_norm_max": float(np.max(client_gradient_norms)),
            "server/ampo/aggregated_grad_norm": float(aggregated_gradient_norm),
            "server/ampo/actor_step_norm": float(actor_step_norm),
            "server/ampo/server_actor_lr": float(actor_lr),
            "server/ampo/dual_lr": float(self.dual_lr),
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
            "server/ampo/lambda_entropy": self._simplex_entropy(lambda_after),
            "server/ampo/lambda_min": float(np.min(lambda_after)),
            "server/ampo/lambda_max": float(np.max(lambda_after)),
            "server/ampo/lambda_delta_norm": float(np.linalg.norm(lambda_after - lambda_before)),
            "server/ampo/effective_clients": self._effective_num_clients(lambda_after),
        }
        if client_num_actor_batches.size > 0:
            metrics["server/ampo/actor_batches_mean"] = float(np.mean(client_num_actor_batches))
        for client_idx, (client_return, lambda_weight, grad_norm) in enumerate(
            zip(client_returns, lambda_after, client_gradient_norms, strict=True)
        ):
            metrics[f"server/ampo/client_{client_idx}/return"] = float(client_return)
            metrics[f"server/ampo/client_{client_idx}/lambda"] = float(lambda_weight)
            metrics[f"server/ampo/client_{client_idx}/grad_norm"] = float(grad_norm)
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
            "vecnormalize": self._get_filtered_vecnormalize_state(),
        }
        if self._uses_global_critic():
            payload["critic_state"] = self._get_critic_state()
        return payload

    @staticmethod
    def _get_rms_state(rms: Any) -> RunningMeanStdState | dict[str, RunningMeanStdState]:
        if isinstance(rms, Mapping):
            return {key: FedAMPOPPO._get_rms_state(value) for key, value in rms.items()}
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
                    FedAMPOPPO._set_rms_state(rms[key], value)
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
        for attr in ("norm_obs", "norm_reward", "clip_obs", "clip_reward", "gamma", "epsilon", "training"):
            if attr in state:
                setattr(vecnormalize, attr, state[attr])
        if state.get("obs_rms") is not None and getattr(vecnormalize, "obs_rms", None) is not None:
            self._set_rms_state(vecnormalize.obs_rms, state["obs_rms"])
        if state.get("ret_rms") is not None and getattr(vecnormalize, "ret_rms", None) is not None:
            self._set_rms_state(vecnormalize.ret_rms, state["ret_rms"])
            if getattr(vecnormalize, "returns", None) is not None:
                vecnormalize.returns = np.zeros_like(vecnormalize.returns)

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
        self._last_obs = None
        self._last_original_obs = None
        self._last_episode_starts = None

    @staticmethod
    def _clone_state(state: Any) -> Any:
        if state is None:
            return None
        if isinstance(state, np.ndarray):
            return state.copy()
        if isinstance(state, Mapping):
            return {key: FedAMPOPPO._clone_state(value) for key, value in state.items()}
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


FedAMPPO = FedAMPOPPO
