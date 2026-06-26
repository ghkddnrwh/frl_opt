from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
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


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """Euclidean projection onto the probability simplex."""
    if v.ndim != 1:
        raise ValueError(f"Expected 1D array, got shape={v.shape}")
    n = v.shape[0]
    if n == 1:
        return np.array([1.0], dtype=np.float64)

    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if len(rho) == 0:
        theta = 0.0
    else:
        rho_idx = rho[-1]
        theta = (cssv[rho_idx] - 1.0) / float(rho_idx + 1)
    w = np.maximum(v - theta, 0.0)
    w_sum = w.sum()
    if w_sum <= 0:
        return np.ones_like(w) / float(n)
    return w / w_sum


class FedSPPGPPO(FederatedAlgorithmMixin, PPO):
    """PPO backbone for Federated Saddle-Point Policy Gradient (FedSP-PG).

    This implementation follows the paper-level algorithm more closely than a
    FedAvg-style PPO implementation:

    Client side:
      1. receive only the global/shared actor parameters,
      2. collect trajectories in the local environment,
      3. update the local critic only,
      4. compute a PPO actor-gradient estimate without taking a local actor step,
      5. send the actor gradient and Monte-Carlo return estimate to the server.

    Server side:
      1. aggregate client actor gradients with the current dual weights lambda,
      2. perform one global actor gradient-ascent step,
      3. update lambda by projected gradient descent on the simplex, unless
         ``dual_update_mode="uniform"`` fixes lambda to 1/K.

    Important design choice:
      Stable-Baselines3 stores actor and critic inside one ActorCriticPolicy.
      Therefore, all synchronization/update helpers below explicitly filter the
      policy state_dict. The critic is client-local by default, but can also be
      synchronized by FedAvg or by lambda-weighted delta aggregation through
      ``critic_sync_mode``.
    """

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
    )

    federated_actor_module_name = "policy"
    federated_critic_module_name = "policy"
    valid_critic_sync_modes: tuple[str, ...] = ("local", "fedavg", "actor_like")
    valid_dual_update_modes: tuple[str, ...] = ("adaptive", "uniform")

    def __init__(self, *args, **kwargs):
        self.dual_lr = float(kwargs.pop("dual_lr", 0.05))
        self.initial_lambda = kwargs.pop("initial_lambda", None)

        # ``dual_update_mode="adaptive"`` is the AMPO/FedSP-PG default.
        # ``dual_update_mode="uniform"`` freezes lambda to 1/K, which is useful
        # as a FedAvg-style aggregation ablation while keeping the FedSP actor/critic flow.
        dual_update_mode = kwargs.pop("dual_update_mode", None)
        fixed_uniform_lambda = kwargs.pop("fixed_uniform_lambda", None)
        dual_fixed_uniform = kwargs.pop("dual_fixed_uniform", None)
        if dual_update_mode is None:
            fixed_uniform = self._optional_bool(fixed_uniform_lambda) or self._optional_bool(dual_fixed_uniform)
            dual_update_mode = "uniform" if fixed_uniform else "adaptive"
        self.dual_update_mode = self._normalize_dual_update_mode(dual_update_mode)

        self.critic_sync_mode = self._normalize_critic_sync_mode(kwargs.pop("critic_sync_mode", "local"))

        for key in self.federated_manager_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        self.lambda_weights: np.ndarray | None = None
        self._fedsp_last_gradient: FederatedModules | None = None
        self._fedsp_last_return: float | None = None
        self._fedsp_last_critic_state: FederatedModules | None = None
        self._fedsp_last_critic_delta: FederatedModules | None = None
        self._fedsp_num_clients_hint: int | None = None

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        return True

    @staticmethod
    def _optional_bool(value: Any) -> bool:
        """Parse optional bool-like config values used by YAML/CLI wrappers."""
        if value is None:
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, np.integer)):
            return bool(value)
        normalized = str(value).strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off", "none", ""}:
            return False
        raise ValueError(f"Cannot interpret {value!r} as a boolean option.")

    @classmethod
    def _normalize_dual_update_mode(cls, mode: str) -> str:
        """Normalize the dual/lambda update mode.

        Modes:
          - adaptive: AMPO/FedSP-PG dual update, lambda <- Proj(lambda - eta * J).
          - uniform: freeze lambda to the uniform distribution 1/K.
        """
        normalized = str(mode).strip().lower().replace("-", "_").replace("/", "_over_")
        aliases = {
            "ampo": "adaptive",
            "fedsp": "adaptive",
            "dynamic": "adaptive",
            "learned": "adaptive",
            "learnable": "adaptive",
            "dual": "adaptive",
            "fixed": "uniform",
            "fixed_uniform": "uniform",
            "uniform_fixed": "uniform",
            "uniform_lambda": "uniform",
            "fixed_lambda": "uniform",
            "1_k": "uniform",
            "1_over_k": "uniform",
            "one_over_k": "uniform",
            "avg": "uniform",
            "average": "uniform",
            "fedavg": "uniform",
            "mean": "uniform",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_update_modes:
            raise ValueError(
                f"Unsupported dual_update_mode={mode!r}. "
                f"Choose one of {cls.valid_dual_update_modes}."
            )
        return normalized

    def _uses_fixed_uniform_lambda(self) -> bool:
        return self.dual_update_mode == "uniform"

    @staticmethod
    def _uniform_lambda(num_clients: int) -> np.ndarray:
        if num_clients <= 0:
            raise ValueError(f"num_clients must be positive, got {num_clients}")
        return np.ones(num_clients, dtype=np.float64) / float(num_clients)

    @classmethod
    def _normalize_critic_sync_mode(cls, mode: str) -> str:
        """Normalize user-facing critic synchronization mode aliases.

        Modes:
          - local: keep critic fully client-local (default, paper-aligned).
          - fedavg: average final client critic parameters on the server.
          - actor_like: aggregate client critic deltas using the same lambda
            weights as actor deltas and broadcast the resulting global critic.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "none": "local",
            "local_only": "local",
            "server_avg": "fedavg",
            "avg": "fedavg",
            "average": "fedavg",
            "lambda": "actor_like",
            "lambda_delta": "actor_like",
            "weighted_delta": "actor_like",
            "same_as_actor": "actor_like",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_critic_sync_modes:
            raise ValueError(
                f"Unsupported critic_sync_mode={mode!r}. "
                f"Choose one of {cls.valid_critic_sync_modes}."
            )
        return normalized

    def _uses_global_critic(self) -> bool:
        return self.critic_sync_mode in {"fedavg", "actor_like"}

    # ------------------------------------------------------------------
    # Actor/critic key filtering utilities
    # ------------------------------------------------------------------
    @staticmethod
    def _is_critic_key(key: str) -> bool:
        """Return True for SB3 value/critic parameters."""
        return (
            key.startswith("value_net.")
            or key.startswith("mlp_extractor.value_net.")
            or key.startswith("vf_features_extractor.")
        )

    @staticmethod
    def _is_explicit_actor_key(key: str) -> bool:
        """Return True for SB3 actor/policy parameters.

        MlpPolicy commonly uses:
          - mlp_extractor.policy_net.*
          - action_net.*
          - log_std for Gaussian policies

        CNN/custom policies may also have feature-extractor parameters. When
        those features feed the actor, they must be part of the shared actor.
        """
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

        # Fallback for unusual/custom SB3 policies: keep all non-critic floating
        # entries. This is safer than synchronizing the value branch by default.
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
    # Federated actor-state helpers
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
            raise KeyError(f"Missing '{module_name}' in federated actor payload.")

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

    def _zero_like_actor_state(self) -> FederatedModules:
        current = self._get_actor_state()
        return {
            module_name: OrderedDict(
                (key, th.zeros_like(value, dtype=value.dtype)) for key, value in module_state.items()
            )
            for module_name, module_state in current.items()
        }

    def _critic_state_keys(self) -> tuple[str, ...]:
        state = self.policy.state_dict()
        return tuple(
            key for key, value in state.items() if th.is_floating_point(value) and self._is_critic_key(key)
        )

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
            raise KeyError(f"Missing '{module_name}' in federated critic payload.")

        current_state = self.policy.state_dict()
        incoming_state = modules[module_name]
        expected_keys = set(self._critic_state_keys())
        incoming_keys = set(incoming_state.keys())
        missing = expected_keys - incoming_keys
        if missing:
            raise KeyError(f"Critic payload is missing keys: {sorted(missing)}")

        for key in expected_keys:
            current_state[key] = incoming_state[key].to(self.device)
        self.policy.load_state_dict(current_state, strict=True)

    def _zero_like_critic_state(self) -> FederatedModules:
        current = self._get_critic_state()
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

    @staticmethod
    def _mix_static_modules(
        old_modules: FederatedModules,
        new_modules: FederatedModules,
        mix_weight: float,
    ) -> FederatedModules:
        if mix_weight >= 1.0:
            return FedSPPGPPO._clone_static_modules(new_modules)

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

    @classmethod
    def _average_static_modules(
        cls,
        modules_list: Sequence[FederatedModules],
        weights: Sequence[float] | None = None,
    ) -> FederatedModules:
        if len(modules_list) == 0:
            raise ValueError("At least one module state is required for averaging.")

        if weights is None:
            average_weights = np.ones(len(modules_list), dtype=np.float64) / float(len(modules_list))
        else:
            average_weights = np.asarray(weights, dtype=np.float64)
            if average_weights.shape != (len(modules_list),):
                raise ValueError(
                    f"weights shape mismatch: expected {(len(modules_list),)}, got {average_weights.shape}"
                )
            weight_sum = float(np.sum(average_weights))
            if weight_sum <= 0.0:
                raise ValueError("weights must sum to a positive value.")
            average_weights = average_weights / weight_sum

        reference = modules_list[0]
        averaged: FederatedModules = {}
        for module_name, reference_state in reference.items():
            averaged[module_name] = OrderedDict()
            for key, reference_value in reference_state.items():
                if th.is_floating_point(reference_value):
                    value_sum = th.zeros_like(reference_value)
                    for weight, modules in zip(average_weights, modules_list, strict=True):
                        value_sum += modules[module_name][key].to(reference_value.dtype) * float(weight)
                    averaged[module_name][key] = value_sum
                else:
                    averaged[module_name][key] = reference_value.clone()
        return averaged

    def _ensure_lambda(self, num_clients: int) -> None:
        if self._uses_fixed_uniform_lambda():
            self.lambda_weights = self._uniform_lambda(num_clients)
            return

        if self.lambda_weights is not None and len(self.lambda_weights) == num_clients:
            return

        if self.initial_lambda is None:
            self.lambda_weights = self._uniform_lambda(num_clients)
        else:
            init = np.asarray(self.initial_lambda, dtype=np.float64)
            if init.shape != (num_clients,):
                raise ValueError(f"initial_lambda shape mismatch: expected {(num_clients,)}, got {init.shape}")
            self.lambda_weights = project_to_simplex(init)

    @staticmethod
    def _normalized_dual_update_signal(client_returns: np.ndarray) -> np.ndarray:
        """Return a scale-stable dual signal based on relative client returns.

        Projection onto the simplex is invariant to adding the same constant to
        every coordinate, so only relative return gaps matter. Normalizing those
        gaps keeps ``dual_lr`` meaningful across environments with very
        different reward scales and avoids immediate one-hot lambda collapse.
        """
        centered_returns = client_returns - float(np.mean(client_returns))
        scale = max(float(np.std(centered_returns)), 1.0)
        return centered_returns / scale

    # ------------------------------------------------------------------
    # SB3 rollout/training helpers
    # ------------------------------------------------------------------
    def _init_fedsp_training_state(self) -> None:
        """Initialize SB3 on-policy state used by collect_rollouts()."""
        if self.ep_info_buffer is None or self.ep_success_buffer is None:
            total_timesteps = max(int(getattr(self, "_total_timesteps", 0)), 1)
            self._setup_learn(
                total_timesteps=total_timesteps,
                callback=None,
                reset_num_timesteps=False,
                tb_log_name="fedsp_pg_ppo",
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

    def _estimate_discounted_mc_return(self) -> float:
        """Estimate J_k(pi) from the most recent rollout buffer.

        The paper defines J_k as a discounted trajectory return. SB3's rollout
        buffer may contain several episodes plus a final truncated segment. To
        stay close to the Monte-Carlo intent in the paper, we average only
        completed episode returns when available and ignore the trailing partial
        segment. If no episode completes inside the rollout, we fall back to the
        discounted return of that partial segment.
        """
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
        self._init_fedsp_training_state()

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

    def _ppo_value_loss(self, values: th.Tensor, rollout_data: Any, clip_range_vf: float | None) -> th.Tensor:
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

    def _snapshot_rollout_buffer(self) -> dict[str, Any]:
        """Keep an unflattened copy because RolloutBuffer.get() mutates arrays."""
        return {
            "observations": np.array(self.rollout_buffer.observations, copy=True),
            "actions": np.array(self.rollout_buffer.actions, copy=True),
            "rewards": np.array(self.rollout_buffer.rewards, copy=True),
            "episode_starts": np.array(self.rollout_buffer.episode_starts, copy=True),
            "values": np.array(self.rollout_buffer.values, copy=True),
            "log_probs": np.array(self.rollout_buffer.log_probs, copy=True),
            "advantages": np.array(self.rollout_buffer.advantages, copy=True),
            "returns": np.array(self.rollout_buffer.returns, copy=True),
            "generator_ready": bool(self.rollout_buffer.generator_ready),
        }

    def _restore_rollout_buffer(self, snapshot: dict[str, Any]) -> None:
        self.rollout_buffer.observations = snapshot["observations"]
        self.rollout_buffer.actions = snapshot["actions"]
        self.rollout_buffer.rewards = snapshot["rewards"]
        self.rollout_buffer.episode_starts = snapshot["episode_starts"]
        self.rollout_buffer.values = snapshot["values"]
        self.rollout_buffer.log_probs = snapshot["log_probs"]
        self.rollout_buffer.advantages = snapshot["advantages"]
        self.rollout_buffer.returns = snapshot["returns"]
        self.rollout_buffer.generator_ready = snapshot["generator_ready"]

    def _snapshot_optimizer_state(self) -> dict[str, Any]:
        return deepcopy(self.policy.optimizer.state_dict())

    def _restore_optimizer_state(self, snapshot: dict[str, Any]) -> None:
        self.policy.optimizer.load_state_dict(snapshot)

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
        """Recompute values/GAE after critic-only training using the updated critic."""
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
        """Update only client-local critic/value parameters on the rollout buffer."""
        critic_params = self._critic_named_parameters()
        if not critic_params:
            # Custom policies might not expose value_net-style names. In that
            # case, skip critic-only training rather than accidentally updating
            # the shared actor.
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

    def _compute_actor_gradient(self) -> FederatedModules:
        """Compute a practical actor update direction from PPO actor-only steps.

        In theory FedSP-PG communicates a policy-gradient estimate. In practice,
        directly aggregating a temporary actor-only PPO update is substantially
        more stable with SB3 because it preserves the optimizer/preconditioning
        used by PPO. We apply actor-only PPO steps locally, measure the actor
        delta, then restore the original actor before uploading that delta.
        """
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSP-PG PPO.")

        actor_before = self._get_actor_state()
        optimizer_snapshot = self._snapshot_optimizer_state()
        self.policy.set_training_mode(True)
        self._set_optimizer_learning_rate(self.policy.optimizer, self._current_learning_rate())
        clip_range = self._current_clip_range()
        module_name = self.federated_actor_module_name

        for _ in range(self.n_epochs):
            for rollout_data in self._iter_rollout_minibatches():
                actions = self._prepare_rollout_actions(rollout_data.actions)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                _, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                actor_minimization_loss = policy_loss + self.ent_coef * entropy_loss

                self.policy.optimizer.zero_grad(set_to_none=True)
                actor_minimization_loss.backward()

                for name, parameter in self.policy.named_parameters():
                    if name not in actor_params:
                        parameter.grad = None

                th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)
                self.policy.optimizer.step()

        actor_after = self._get_actor_state()
        self._set_actor_state(actor_before)
        self._restore_optimizer_state(optimizer_snapshot)
        self.policy.optimizer.zero_grad(set_to_none=True)

        actor_delta: FederatedModules = {module_name: OrderedDict()}
        for key, before_value in actor_before[module_name].items():
            after_value = actor_after[module_name][key]
            if th.is_floating_point(after_value):
                actor_delta[module_name][key] = after_value - before_value
            else:
                actor_delta[module_name][key] = th.zeros_like(after_value)
        return actor_delta

    def _train_local_critic_and_compute_actor_gradient(self) -> FederatedModules:
        rollout_snapshot = self._snapshot_rollout_buffer()
        self._update_local_critic()
        self._restore_rollout_buffer(rollout_snapshot)
        self._refresh_rollout_advantages()
        return self._compute_actor_gradient()

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------
    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        """Collect local rollouts, update local critic, and estimate actor gradient.

        ``local_steps`` is interpreted as a target amount of environment
        interaction. PPO collects fixed-size rollouts, so the actual number of
        steps is quantized by ``n_steps * n_envs``.
        """
        del kwargs

        target_steps = int(local_steps)
        if target_steps <= 0:
            raise ValueError(f"local_steps must be positive, got {local_steps}")

        collected_steps = 0
        returns: list[float] = []
        gradient_accumulator: FederatedModules | None = None
        num_updates = 0
        critic_before_round = self._get_critic_state() if self.critic_sync_mode == "actor_like" else None

        while collected_steps < target_steps:
            rollout_return = self._collect_one_rollout()
            local_grad = self._train_local_critic_and_compute_actor_gradient()

            if gradient_accumulator is None:
                gradient_accumulator = self._clone_modules(local_grad)
            else:
                for module_name in gradient_accumulator.keys():
                    for key in gradient_accumulator[module_name].keys():
                        gradient_accumulator[module_name][key] += local_grad[module_name][key]

            returns.append(rollout_return)
            collected_steps += self.n_steps * self.n_envs
            num_updates += 1

        assert gradient_accumulator is not None
        for module_name in gradient_accumulator.keys():
            for key in gradient_accumulator[module_name].keys():
                gradient_accumulator[module_name][key] /= float(num_updates)

        self._fedsp_last_gradient = gradient_accumulator
        self._fedsp_last_return = float(np.mean(returns))

        if self.critic_sync_mode == "fedavg":
            self._fedsp_last_critic_state = self._get_critic_state()
            self._fedsp_last_critic_delta = None
        elif self.critic_sync_mode == "actor_like":
            assert critic_before_round is not None
            self._fedsp_last_critic_state = None
            self._fedsp_last_critic_delta = self._subtract_static_modules(
                self._get_critic_state(),
                critic_before_round,
            )
        else:
            self._fedsp_last_critic_state = None
            self._fedsp_last_critic_delta = None

        # Do NOT manually increment self.num_timesteps here. SB3's
        # collect_rollouts() already increments num_timesteps internally.

    def get_upload_payload(self) -> FederatedPayload:
        if self._fedsp_last_gradient is None:
            self._fedsp_last_gradient = self._zero_like_actor_state()
        if self._fedsp_last_return is None:
            self._fedsp_last_return = 0.0

        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "actor_gradient": self._clone_modules(self._fedsp_last_gradient),
            "return": float(self._fedsp_last_return),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "dual_update_mode": self.dual_update_mode,
            "critic_sync_mode": self.critic_sync_mode,
        }

        if self.critic_sync_mode == "fedavg":
            if self._fedsp_last_critic_state is None:
                self._fedsp_last_critic_state = self._get_critic_state()
            payload["critic_state"] = self._clone_modules(self._fedsp_last_critic_state)
        elif self.critic_sync_mode == "actor_like":
            if self._fedsp_last_critic_delta is None:
                self._fedsp_last_critic_delta = self._zero_like_critic_state()
            payload["critic_delta"] = self._clone_modules(self._fedsp_last_critic_delta)

        return payload

    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        """Package client gradients/returns for the FedSP-PG server update."""
        del weights

        if len(uploads) == 0:
            raise ValueError("At least one upload is required.")

        actor_gradients = [upload["actor_gradient"] for upload in uploads]
        returns = np.asarray([float(upload["return"]) for upload in uploads], dtype=np.float64)

        critic_modes = {cls._normalize_critic_sync_mode(str(upload.get("critic_sync_mode", "local"))) for upload in uploads}
        if len(critic_modes) != 1:
            raise ValueError(f"Mixed critic_sync_mode values are not supported in one aggregation: {critic_modes}")
        critic_sync_mode = next(iter(critic_modes))

        dual_modes = {cls._normalize_dual_update_mode(str(upload.get("dual_update_mode", "adaptive"))) for upload in uploads}
        if len(dual_modes) != 1:
            raise ValueError(f"Mixed dual_update_mode values are not supported in one aggregation: {dual_modes}")
        dual_update_mode = next(iter(dual_modes))

        first_actor_state = uploads[0]["actor_state"]
        payload: FederatedPayload = {
            "client_actor_gradients": actor_gradients,
            "client_returns": returns,
            "reference_actor_state": cls._clone_static_modules(first_actor_state),
            "dual_update_mode": dual_update_mode,
            "critic_sync_mode": critic_sync_mode,
            "num_clients": len(uploads),
        }

        if critic_sync_mode == "fedavg":
            payload["aggregated_critic_state"] = cls._average_static_modules(
                [upload["critic_state"] for upload in uploads],
                weights=None,
            )
        elif critic_sync_mode == "actor_like":
            payload["client_critic_deltas"] = [upload["critic_delta"] for upload in uploads]

        return payload

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        """Apply either a server broadcast payload or a server update payload."""
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        # Broadcast path: clients always receive the updated global actor.
        # Critic parameters are included only when critic_sync_mode is not local.
        if "actor_state" in payload and "client_actor_gradients" not in payload:
            incoming_actor = payload["actor_state"]
            if mix_weight < 1.0:
                incoming_actor = self._mix_static_modules(self._get_actor_state(), incoming_actor, mix_weight)
            self._set_actor_state(incoming_actor)

            if self._uses_global_critic() and "critic_state" in payload:
                incoming_critic = payload["critic_state"]
                if mix_weight < 1.0:
                    incoming_critic = self._mix_static_modules(self._get_critic_state(), incoming_critic, mix_weight)
                self._set_critic_state(incoming_critic)
            return

        # Server update path.
        payload_critic_mode = self._normalize_critic_sync_mode(str(payload.get("critic_sync_mode", "local")))
        if payload_critic_mode != self.critic_sync_mode:
            raise ValueError(
                f"Server critic_sync_mode={self.critic_sync_mode!r} does not match "
                f"payload critic_sync_mode={payload_critic_mode!r}."
            )

        payload_dual_mode = self._normalize_dual_update_mode(str(payload.get("dual_update_mode", "adaptive")))
        if payload_dual_mode != self.dual_update_mode:
            raise ValueError(
                f"Server dual_update_mode={self.dual_update_mode!r} does not match "
                f"payload dual_update_mode={payload_dual_mode!r}."
            )

        client_actor_gradients: list[FederatedModules] = payload["client_actor_gradients"]
        client_returns = np.asarray(payload["client_returns"], dtype=np.float64)
        num_clients = int(payload["num_clients"])

        self._ensure_lambda(num_clients)
        assert self.lambda_weights is not None

        actor_state = self._get_actor_state()
        module_name = self.federated_actor_module_name

        aggregated_grad: FederatedModules = {module_name: OrderedDict()}
        for key, value in actor_state[module_name].items():
            if th.is_floating_point(value):
                g = th.zeros_like(value, dtype=value.dtype)
                for lam, client_grad in zip(self.lambda_weights, client_actor_gradients, strict=True):
                    g += client_grad[module_name][key].to(value.dtype) * float(lam)
                aggregated_grad[module_name][key] = g
            else:
                aggregated_grad[module_name][key] = th.zeros_like(value)

        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSP-PG PPO server update.")

        updated_actor = self._add_static_modules(actor_state, aggregated_grad)
        if mix_weight < 1.0:
            updated_actor = self._mix_static_modules(actor_state, updated_actor, mix_weight)
        self._set_actor_state(updated_actor)

        if self.critic_sync_mode == "fedavg":
            if "aggregated_critic_state" not in payload:
                raise KeyError("Missing 'aggregated_critic_state' for critic_sync_mode='fedavg'.")
            current_critic = self._get_critic_state()
            updated_critic = payload["aggregated_critic_state"]
            if mix_weight < 1.0:
                updated_critic = self._mix_static_modules(current_critic, updated_critic, mix_weight)
            self._set_critic_state(updated_critic)
        elif self.critic_sync_mode == "actor_like":
            client_critic_deltas: list[FederatedModules] = payload["client_critic_deltas"]
            critic_state = self._get_critic_state()
            critic_module_name = self.federated_critic_module_name
            aggregated_critic_delta: FederatedModules = {critic_module_name: OrderedDict()}
            for key, value in critic_state[critic_module_name].items():
                if th.is_floating_point(value):
                    delta = th.zeros_like(value, dtype=value.dtype)
                    for lam, client_delta in zip(self.lambda_weights, client_critic_deltas, strict=True):
                        delta += client_delta[critic_module_name][key].to(value.dtype) * float(lam)
                    aggregated_critic_delta[critic_module_name][key] = delta
                else:
                    aggregated_critic_delta[critic_module_name][key] = th.zeros_like(value)

            updated_critic = self._add_static_modules(critic_state, aggregated_critic_delta)
            if mix_weight < 1.0:
                updated_critic = self._mix_static_modules(critic_state, updated_critic, mix_weight)
            self._set_critic_state(updated_critic)

        # Dual update. In uniform mode, lambda is fixed to 1/K and no return-based
        # projected-gradient step is applied.
        if self._uses_fixed_uniform_lambda():
            self.lambda_weights = self._uniform_lambda(num_clients)
        else:
            dual_signal = self._normalized_dual_update_signal(client_returns)
            self.lambda_weights = project_to_simplex(self.lambda_weights - self.dual_lr * dual_signal)
        self._fedsp_num_clients_hint = num_clients

    def get_client_weight(self) -> float:
        return 1.0

    # ------------------------------------------------------------------
    # Broadcast helper
    # ------------------------------------------------------------------
    def get_broadcast_payload(self) -> FederatedPayload:
        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "dual_update_mode": self.dual_update_mode,
            "critic_sync_mode": self.critic_sync_mode,
        }
        if self._uses_global_critic():
            payload["critic_state"] = self._get_critic_state()
        return payload
