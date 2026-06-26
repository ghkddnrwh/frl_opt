from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.type_aliases import RolloutBufferSamples
from stable_baselines3.common import utils as sb3_utils
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.ppo import PPO

from rl_zoo3.algorithms.federate.common.federated_algorithm import (
    FederatedAlgorithmMixin,
    FederatedModules,
    FederatedPayload,
)
from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo import project_to_simplex


class FedSPPGPPOPaperAligned(FederatedAlgorithmMixin, PPO):
    """Paper-aligned AMPO/PPO variant.

    This class intentionally does not inherit from ``FedSPPGPPO``. It keeps the
    same federated PPO backbone but follows the paper-level AMPO update rules
    more directly:

    1. clients upload an actor-gradient estimate instead of an actor delta,
    2. the server applies ``theta <- theta + eta_pi * sum_k lambda_k g_k``,
    3. the dual update uses raw per-client return estimates before simplex
       projection, unless ``dual_update_mode='uniform'`` fixes lambda to 1/K.

    By default, the critic remains client-local. ``critic_sync_mode`` can be
    set to ``'fedavg'`` or ``'actor_like'`` for ablation experiments.
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
        "server_actor_lr",
        "actor_gradient_mode",
        "server_actor_optimizer",
        "critic_sync_mode",
        "dual_update_mode",
        "fixed_uniform_lambda",
        "dual_fixed_uniform",
        "log_wandb",
    )

    federated_actor_module_name = "policy"
    federated_critic_module_name = "policy"
    valid_critic_sync_modes: tuple[str, ...] = ("local", "fedavg", "actor_like")
    valid_dual_update_modes: tuple[str, ...] = ("adaptive", "uniform")

    def __init__(self, *args, **kwargs):
        self.dual_lr = float(kwargs.pop("dual_lr", 0.05))
        self.initial_lambda = kwargs.pop("initial_lambda", None)
        server_actor_lr = kwargs.pop("server_actor_lr", None)
        self.server_actor_lr = None if server_actor_lr is None else float(server_actor_lr)
        self.actor_gradient_mode = str(kwargs.pop("actor_gradient_mode", "cumulative")).strip().lower()
        if self.actor_gradient_mode not in {"mean", "cumulative"}:
            raise ValueError(
                f"actor_gradient_mode must be 'mean' or 'cumulative', got {self.actor_gradient_mode!r}"
            )
        self.server_actor_optimizer = str(kwargs.pop("server_actor_optimizer", "adam")).strip().lower()
        if self.server_actor_optimizer not in {"adam", "sgd"}:
            raise ValueError(
                f"server_actor_optimizer must be 'adam' or 'sgd', got {self.server_actor_optimizer!r}"
            )

        self.critic_sync_mode = self._normalize_critic_sync_mode(kwargs.pop("critic_sync_mode", "local"))
        raw_dual_update_mode = kwargs.pop("dual_update_mode", None)
        fixed_uniform_lambda = self._as_bool(kwargs.pop("fixed_uniform_lambda", False)) or self._as_bool(
            kwargs.pop("dual_fixed_uniform", False)
        )
        if fixed_uniform_lambda:
            raw_dual_update_mode = "uniform"
        if raw_dual_update_mode is None:
            raw_dual_update_mode = "adaptive"
        self.dual_update_mode = self._normalize_dual_update_mode(raw_dual_update_mode)

        for key in self.federated_manager_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        self.lambda_weights: np.ndarray | None = None
        self._fedsp_last_gradient: FederatedModules | None = None
        self._fedsp_last_return: float | None = None
        self._fedsp_last_critic_state: FederatedModules | None = None
        self._fedsp_last_critic_delta: FederatedModules | None = None
        self._fedsp_num_clients_hint: int | None = None
        self._fedsp_last_local_gradient_norm: float | None = None
        self._fedsp_last_aggregated_gradient_norm: float | None = None
        self._fedsp_last_actor_step_norm: float | None = None
        self._fedsp_last_num_actor_batches: int | None = None
        self._server_actor_optimizer_instance: th.optim.Optimizer | None = None
        self._server_actor_optimizer_param_names: tuple[str, ...] | None = None
        self._last_federated_metrics: dict[str, float] = {}

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        return True

    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"1", "true", "yes", "y", "on"}:
                return True
            if normalized in {"0", "false", "no", "n", "off", "none", "null", ""}:
                return False
        return bool(value)

    @classmethod
    def _normalize_critic_sync_mode(cls, mode: str) -> str:
        """Normalize user-facing critic synchronization mode aliases.

        Modes:
          - local: keep critic fully client-local (default, paper-aligned).
          - fedavg: average final client critic parameters on the server.
          - actor_like: aggregate client critic deltas using the same lambda
            weights as actor-gradient aggregation and broadcast the resulting
            global critic.
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

    @classmethod
    def _normalize_dual_update_mode(cls, mode: str) -> str:
        """Normalize dual update modes.

        Modes:
          - adaptive: paper-aligned projected dual update.
          - uniform: fix lambda_k = 1 / K and skip the dual update.
        """
        normalized = str(mode).strip().lower().replace("-", "_")
        aliases = {
            "ampo": "adaptive",
            "learned": "adaptive",
            "learn": "adaptive",
            "update": "adaptive",
            "fixed": "uniform",
            "fixed_uniform": "uniform",
            "uniform_fixed": "uniform",
            "1/k": "uniform",
            "1_over_k": "uniform",
            "one_over_k": "uniform",
            "mean": "uniform",
            "average": "uniform",
        }
        normalized = aliases.get(normalized, normalized)
        if normalized not in cls.valid_dual_update_modes:
            raise ValueError(
                f"Unsupported dual_update_mode={mode!r}. "
                f"Choose one of {cls.valid_dual_update_modes}."
            )
        return normalized

    def _uses_global_critic(self) -> bool:
        return self.critic_sync_mode in {"fedavg", "actor_like"}

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
            return FedSPPGPPOPaperAligned._clone_static_modules(new_modules)

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
        weights = np.asarray(weights, dtype=np.float64)
        denom = float(np.sum(weights * weights))
        if denom <= 0.0:
            return 0.0
        return 1.0 / denom

    def _ensure_lambda(self, num_clients: int) -> None:
        if self.dual_update_mode == "uniform":
            self.lambda_weights = np.ones(num_clients, dtype=np.float64) / float(num_clients)
            return

        if self.lambda_weights is not None and len(self.lambda_weights) == num_clients:
            return

        if self.initial_lambda is None:
            self.lambda_weights = np.ones(num_clients, dtype=np.float64) / float(num_clients)
        else:
            init = np.asarray(self.initial_lambda, dtype=np.float64)
            if init.shape != (num_clients,):
                raise ValueError(f"initial_lambda shape mismatch: expected {(num_clients,)}, got {init.shape}")
            self.lambda_weights = project_to_simplex(init)

    # ------------------------------------------------------------------
    # SB3 rollout/training helpers
    # ------------------------------------------------------------------
    def _init_fedsp_training_state(self) -> None:
        if self.ep_info_buffer is None or self.ep_success_buffer is None:
            total_timesteps = max(int(getattr(self, "_total_timesteps", 0)), 1)
            self._setup_learn(
                total_timesteps=total_timesteps,
                callback=None,
                reset_num_timesteps=False,
                tb_log_name="fedsp_pg_ppo_paper_aligned",
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

    def _current_server_actor_lr(self) -> float:
        if self.server_actor_lr is not None:
            return float(self.server_actor_lr)
        return self._current_learning_rate()

    def _get_or_create_server_actor_optimizer(self) -> th.optim.Optimizer:
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSP-PG PPO server optimizer.")

        param_names = tuple(actor_params.keys())
        if (
            self._server_actor_optimizer_instance is not None
            and self._server_actor_optimizer_param_names == param_names
        ):
            return self._server_actor_optimizer_instance

        params = list(actor_params.values())
        lr = self._current_server_actor_lr()
        if self.server_actor_optimizer == "adam":
            optimizer = th.optim.Adam(params, lr=lr)
        else:
            optimizer = th.optim.SGD(params, lr=lr)

        self._server_actor_optimizer_instance = optimizer
        self._server_actor_optimizer_param_names = param_names
        return optimizer

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

    def _compute_actor_gradient(self) -> FederatedModules:
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSP-PG PPO.")

        self.policy.set_training_mode(True)
        clip_range = self._current_clip_range()
        gradient_sum = self._zero_like_actor_state()
        module_name = self.federated_actor_module_name
        num_actor_batches = 0

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

                # SB3 minimizes this loss. Negate its gradient to recover the
                # ascent direction used in the paper-level server update.
                actor_minimization_loss = policy_loss + self.ent_coef * entropy_loss

                self.policy.optimizer.zero_grad(set_to_none=True)
                actor_minimization_loss.backward()

                for name, parameter in self.policy.named_parameters():
                    if name not in actor_params:
                        parameter.grad = None

                th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)

                for name, parameter in actor_params.items():
                    if parameter.grad is not None and name in gradient_sum[module_name]:
                        gradient_sum[module_name][name] += -parameter.grad.detach().cpu()

                num_actor_batches += 1
                self.policy.optimizer.zero_grad(set_to_none=True)

        if num_actor_batches == 0:
            raise RuntimeError("No PPO minibatches were available for actor-gradient computation.")

        self._fedsp_last_num_actor_batches = num_actor_batches
        if self.actor_gradient_mode == "mean":
            for key in gradient_sum[module_name].keys():
                gradient_sum[module_name][key] /= float(num_actor_batches)
        return gradient_sum

    def _train_local_critic_and_compute_actor_gradient(self) -> FederatedModules:
        rollout_snapshot = self._snapshot_rollout_buffer()
        self._update_local_critic()
        self._restore_rollout_buffer(rollout_snapshot)
        self._refresh_rollout_advantages()
        return self._compute_actor_gradient()

    def _record_server_diagnostics(
        self,
        *,
        client_returns: np.ndarray,
        client_gradient_norms: list[float],
        aggregated_gradient_norm: float,
        actor_step_norm: float,
        actor_lr: float,
        num_actor_batches: int,
    ) -> None:
        if not hasattr(self, "_logger") or self._logger is None:
            self._logger = sb3_utils.configure_logger(
                self.verbose,
                self.tensorboard_log,
                "fedsp_pg_ppo_paper_aligned_server",
                False,
            )

        lambda_weights = np.asarray(self.lambda_weights, dtype=np.float64)
        metrics: dict[str, float] = {
            "server/num_clients": float(len(client_returns)),
            "server/fsp_paper/server_actor_lr": float(actor_lr),
            "server/fsp_paper/dual_lr": float(self.dual_lr),
            "server/fsp_paper/dual_uniform": float(self.dual_update_mode == "uniform"),
            "server/fsp_paper/critic_local": float(self.critic_sync_mode == "local"),
            "server/fsp_paper/critic_fedavg": float(self.critic_sync_mode == "fedavg"),
            "server/fsp_paper/critic_actor_like": float(self.critic_sync_mode == "actor_like"),
            "server/fsp_paper/optimizer_adam": float(self.server_actor_optimizer == "adam"),
            "server/fsp_paper/gradient_cumulative": float(self.actor_gradient_mode == "cumulative"),
            "server/fsp_paper/actor_batches": float(num_actor_batches),
            "server/fsp_paper/return_mean": float(np.mean(client_returns)),
            "server/fsp_paper/return_std": float(np.std(client_returns)),
            "server/fsp_paper/return_min": float(np.min(client_returns)),
            "server/fsp_paper/return_max": float(np.max(client_returns)),
            "server/fsp_paper/client_grad_norm_mean": float(np.mean(client_gradient_norms)),
            "server/fsp_paper/client_grad_norm_std": float(np.std(client_gradient_norms)),
            "server/fsp_paper/client_grad_norm_min": float(np.min(client_gradient_norms)),
            "server/fsp_paper/client_grad_norm_max": float(np.max(client_gradient_norms)),
            "server/fsp_paper/aggregated_grad_norm": float(aggregated_gradient_norm),
            "server/fsp_paper/actor_step_norm": float(actor_step_norm),
            "server/fsp_paper/lambda_entropy": self._simplex_entropy(lambda_weights),
            "server/fsp_paper/lambda_max": float(np.max(lambda_weights)),
            "server/fsp_paper/lambda_min": float(np.min(lambda_weights)),
            "server/fsp_paper/effective_clients": self._effective_num_clients(lambda_weights),
        }
        for client_idx, (client_return, lambda_weight, grad_norm) in enumerate(
            zip(client_returns, lambda_weights, client_gradient_norms, strict=True)
        ):
            metrics[f"server/fsp_paper/client_{client_idx}/return"] = float(client_return)
            metrics[f"server/fsp_paper/client_{client_idx}/lambda"] = float(lambda_weight)
            metrics[f"server/fsp_paper/client_{client_idx}/grad_norm"] = float(grad_norm)
        self._last_federated_metrics = metrics

        # Keep logger keys short: SB3 HumanOutputFormat truncates long keys
        # and raises if two keys collide after truncation.
        self._logger.record("train/fsp_paper/server_lr", float(actor_lr))
        self._logger.record("train/fsp_paper/dual_lr", float(self.dual_lr))
        self._logger.record("train/fsp_paper/dual_uniform", float(self.dual_update_mode == "uniform"))
        self._logger.record("train/fsp_paper/crit_local", float(self.critic_sync_mode == "local"))
        self._logger.record("train/fsp_paper/crit_fedavg", float(self.critic_sync_mode == "fedavg"))
        self._logger.record("train/fsp_paper/crit_actorlike", float(self.critic_sync_mode == "actor_like"))
        self._logger.record("train/fsp_paper/opt_adam", float(self.server_actor_optimizer == "adam"))
        self._logger.record("train/fsp_paper/grad_cumul", float(self.actor_gradient_mode == "cumulative"))
        self._logger.record("train/fsp_paper/actor_batches", float(num_actor_batches))
        self._logger.record("train/fsp_paper/ret_mean", float(np.mean(client_returns)))
        self._logger.record("train/fsp_paper/ret_std", float(np.std(client_returns)))
        self._logger.record("train/fsp_paper/ret_min", float(np.min(client_returns)))
        self._logger.record("train/fsp_paper/ret_max", float(np.max(client_returns)))
        self._logger.record("train/fsp_paper/client_gn_mean", float(np.mean(client_gradient_norms)))
        self._logger.record("train/fsp_paper/client_gn_std", float(np.std(client_gradient_norms)))
        self._logger.record("train/fsp_paper/client_gn_min", float(np.min(client_gradient_norms)))
        self._logger.record("train/fsp_paper/client_gn_max", float(np.max(client_gradient_norms)))
        self._logger.record("train/fsp_paper/agg_gn", float(aggregated_gradient_norm))
        self._logger.record("train/fsp_paper/actor_step", float(actor_step_norm))
        self._logger.record("train/fsp_paper/lambda_ent", self._simplex_entropy(lambda_weights))
        self._logger.record("train/fsp_paper/lambda_max", float(np.max(lambda_weights)))
        self._logger.record("train/fsp_paper/lambda_min", float(np.min(lambda_weights)))
        self._logger.record("train/fsp_paper/eff_clients", self._effective_num_clients(lambda_weights))
        self._logger.dump(step=int(self.num_timesteps))
        if self.verbose > 0:
            print(
                "[FedSP-Paper]"
                f" step={int(self.num_timesteps)}"
                f" actor_lr={float(actor_lr):.3e}"
                f" grad_norm={float(aggregated_gradient_norm):.3e}"
                f" actor_step_norm={float(actor_step_norm):.3e}"
                f" lambda_entropy={self._simplex_entropy(lambda_weights):.3f}"
                f" eff_clients={self._effective_num_clients(lambda_weights):.2f}"
                f" return_mean={float(np.mean(client_returns)):.3f}"
            )

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------
    def federated_local_update(self, local_steps: int, **kwargs) -> None:
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
        self._fedsp_last_local_gradient_norm = self._module_l2_norm(gradient_accumulator)

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

    def get_upload_payload(self) -> FederatedPayload:
        if self._fedsp_last_gradient is None:
            self._fedsp_last_gradient = self._zero_like_actor_state()
        if self._fedsp_last_return is None:
            self._fedsp_last_return = 0.0

        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "actor_gradient": self._clone_modules(self._fedsp_last_gradient),
            "return": float(self._fedsp_last_return),
            "num_actor_batches": 0 if self._fedsp_last_num_actor_batches is None else int(self._fedsp_last_num_actor_batches),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "critic_sync_mode": self.critic_sync_mode,
            "dual_update_mode": self.dual_update_mode,
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
        del weights

        if len(uploads) == 0:
            raise ValueError("At least one upload is required.")

        actor_gradients = [upload["actor_gradient"] for upload in uploads]
        returns = np.asarray([float(upload["return"]) for upload in uploads], dtype=np.float64)
        num_actor_batches = np.asarray([int(upload.get("num_actor_batches", 0)) for upload in uploads], dtype=np.int32)

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
            "client_num_actor_batches": num_actor_batches,
            "reference_actor_state": cls._clone_static_modules(first_actor_state),
            "critic_sync_mode": critic_sync_mode,
            "dual_update_mode": dual_update_mode,
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
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

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
        client_num_actor_batches = np.asarray(payload.get("client_num_actor_batches", []), dtype=np.int32)
        num_clients = int(payload["num_clients"])

        self._ensure_lambda(num_clients)
        assert self.lambda_weights is not None

        actor_state = self._get_actor_state()
        module_name = self.federated_actor_module_name
        actor_lr = self._current_server_actor_lr()
        client_gradient_norms = [self._module_l2_norm(client_grad) for client_grad in client_actor_gradients]
        actor_before = self._get_actor_state()

        aggregated_grad: FederatedModules = {module_name: OrderedDict()}
        for key, value in actor_state[module_name].items():
            if th.is_floating_point(value):
                grad = th.zeros_like(value, dtype=value.dtype)
                for lam, client_grad in zip(self.lambda_weights, client_actor_gradients, strict=True):
                    grad += client_grad[module_name][key].to(value.dtype) * float(lam)
                aggregated_grad[module_name][key] = grad
            else:
                aggregated_grad[module_name][key] = th.zeros_like(value)

        aggregated_gradient_norm = self._module_l2_norm(aggregated_grad)
        actor_params = self._actor_named_parameters()
        server_optimizer = self._get_or_create_server_actor_optimizer()
        self._set_optimizer_learning_rate(server_optimizer, actor_lr)
        server_optimizer.zero_grad(set_to_none=True)

        for name, parameter in actor_params.items():
            parameter.grad = -aggregated_grad[module_name][name].to(self.device)

        server_optimizer.step()
        server_optimizer.zero_grad(set_to_none=True)

        updated_actor = self._get_actor_state()
        if mix_weight < 1.0:
            mixed_actor: FederatedModules = {module_name: OrderedDict()}
            for key, value in actor_state[module_name].items():
                if th.is_floating_point(value):
                    mixed_actor[module_name][key] = mix_weight * updated_actor[module_name][key] + (1.0 - mix_weight) * value
                else:
                    mixed_actor[module_name][key] = value.clone()
            updated_actor = mixed_actor
            self._set_actor_state(updated_actor)

        actor_step: FederatedModules = {module_name: OrderedDict()}
        for key, before_value in actor_before[module_name].items():
            after_value = updated_actor[module_name][key]
            if th.is_floating_point(after_value):
                actor_step[module_name][key] = after_value - before_value
            else:
                actor_step[module_name][key] = th.zeros_like(after_value)
        actor_step_norm = self._module_l2_norm(actor_step)
        self._fedsp_last_aggregated_gradient_norm = aggregated_gradient_norm
        self._fedsp_last_actor_step_norm = actor_step_norm

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

        # Paper-aligned dual update: lambda <- Proj_Delta(lambda - eta_lambda * J).
        # Optional uniform mode fixes lambda_k = 1 / K and skips the dual update.
        if self.dual_update_mode == "uniform":
            self.lambda_weights = np.ones(num_clients, dtype=np.float64) / float(num_clients)
        else:
            self.lambda_weights = project_to_simplex(self.lambda_weights - self.dual_lr * client_returns)
        self._fedsp_num_clients_hint = num_clients
        self._record_server_diagnostics(
            client_returns=client_returns,
            client_gradient_norms=client_gradient_norms,
            aggregated_gradient_norm=aggregated_gradient_norm,
            actor_step_norm=actor_step_norm,
            actor_lr=actor_lr,
            num_actor_batches=0 if client_num_actor_batches.size == 0 else int(np.mean(client_num_actor_batches)),
        )

    def get_client_weight(self) -> float:
        return 1.0

    def get_broadcast_payload(self) -> FederatedPayload:
        payload: FederatedPayload = {
            "actor_state": self._get_actor_state(),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
            "critic_sync_mode": self.critic_sync_mode,
            "dual_update_mode": self.dual_update_mode,
        }
        if self._uses_global_critic():
            payload["critic_state"] = self._get_critic_state()
        return payload


# from __future__ import annotations

# from collections import OrderedDict
# from collections.abc import Sequence
# from typing import Any

# import numpy as np
# import torch as th
# import torch.nn.functional as F
# from gymnasium import spaces
# from stable_baselines3.common.buffers import RolloutBuffer
# from stable_baselines3.common.callbacks import CallbackList
# from stable_baselines3.common.type_aliases import RolloutBufferSamples
# from stable_baselines3.common import utils as sb3_utils
# from stable_baselines3.common.utils import obs_as_tensor
# from stable_baselines3.ppo import PPO

# from rl_zoo3.algorithms.federate.common.federated_algorithm import (
#     FederatedAlgorithmMixin,
#     FederatedModules,
#     FederatedPayload,
# )
# from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo import project_to_simplex


# class FedSPPGPPOPaperAligned(FederatedAlgorithmMixin, PPO):
#     """Paper-aligned AMPO/PPO variant.

#     This class intentionally does not inherit from ``FedSPPGPPO``. It keeps the
#     same federated PPO backbone but follows the paper-level AMPO update rules
#     more directly:

#     1. clients upload an actor-gradient estimate instead of an actor delta,
#     2. the server applies ``theta <- theta + eta_pi * sum_k lambda_k g_k``,
#     3. the dual update uses raw per-client return estimates before simplex
#        projection.
#     """

#     federated_manager_keys: tuple[str, ...] = (
#         "num_clients",
#         "local_steps",
#         "server_update_weight",
#         "perturb_noise_type",
#         "perturb_noise_range",
#         "eval_local_episodes",
#         "eval_nominal_episodes",
#         "eval_round_freq",
#         "eval_deterministic",
#         "dual_lr",
#         "initial_lambda",
#         "server_actor_lr",
#         "actor_gradient_mode",
#         "server_actor_optimizer",
#     )

#     federated_actor_module_name = "policy"

#     def __init__(self, *args, **kwargs):
#         self.dual_lr = float(kwargs.pop("dual_lr", 0.05))
#         self.initial_lambda = kwargs.pop("initial_lambda", None)
#         server_actor_lr = kwargs.pop("server_actor_lr", None)
#         self.server_actor_lr = None if server_actor_lr is None else float(server_actor_lr)
#         self.actor_gradient_mode = str(kwargs.pop("actor_gradient_mode", "cumulative")).strip().lower()
#         if self.actor_gradient_mode not in {"mean", "cumulative"}:
#             raise ValueError(
#                 f"actor_gradient_mode must be 'mean' or 'cumulative', got {self.actor_gradient_mode!r}"
#             )
#         self.server_actor_optimizer = str(kwargs.pop("server_actor_optimizer", "adam")).strip().lower()
#         if self.server_actor_optimizer not in {"adam", "sgd"}:
#             raise ValueError(
#                 f"server_actor_optimizer must be 'adam' or 'sgd', got {self.server_actor_optimizer!r}"
#             )

#         for key in self.federated_manager_keys:
#             kwargs.pop(key, None)

#         super().__init__(*args, **kwargs)

#         self.lambda_weights: np.ndarray | None = None
#         self._fedsp_last_gradient: FederatedModules | None = None
#         self._fedsp_last_return: float | None = None
#         self._fedsp_num_clients_hint: int | None = None
#         self._fedsp_last_local_gradient_norm: float | None = None
#         self._fedsp_last_aggregated_gradient_norm: float | None = None
#         self._fedsp_last_actor_step_norm: float | None = None
#         self._fedsp_last_num_actor_batches: int | None = None
#         self._server_actor_optimizer_instance: th.optim.Optimizer | None = None
#         self._server_actor_optimizer_param_names: tuple[str, ...] | None = None

#     @classmethod
#     def uses_federated_client_n_envs(cls) -> bool:
#         return True

#     # ------------------------------------------------------------------
#     # Actor/critic key filtering utilities
#     # ------------------------------------------------------------------
#     @staticmethod
#     def _is_critic_key(key: str) -> bool:
#         return (
#             key.startswith("value_net.")
#             or key.startswith("mlp_extractor.value_net.")
#             or key.startswith("vf_features_extractor.")
#         )

#     @staticmethod
#     def _is_explicit_actor_key(key: str) -> bool:
#         return (
#             key == "log_std"
#             or key.startswith("action_net.")
#             or key.startswith("mlp_extractor.policy_net.")
#             or key.startswith("pi_features_extractor.")
#             or key.startswith("features_extractor.")
#         )

#     def _actor_state_keys(self) -> tuple[str, ...]:
#         state = self.policy.state_dict()
#         explicit_actor_keys = [key for key in state.keys() if self._is_explicit_actor_key(key)]
#         if explicit_actor_keys:
#             return tuple(explicit_actor_keys)

#         return tuple(
#             key for key, value in state.items() if th.is_floating_point(value) and not self._is_critic_key(key)
#         )

#     def _actor_named_parameters(self) -> dict[str, th.nn.Parameter]:
#         actor_keys = set(self._actor_state_keys())
#         return {
#             name: parameter
#             for name, parameter in self.policy.named_parameters()
#             if name in actor_keys and parameter.requires_grad
#         }

#     def _critic_named_parameters(self) -> dict[str, th.nn.Parameter]:
#         return {
#             name: parameter
#             for name, parameter in self.policy.named_parameters()
#             if self._is_critic_key(name) and parameter.requires_grad
#         }

#     # ------------------------------------------------------------------
#     # Federated actor-state helpers
#     # ------------------------------------------------------------------
#     def _get_actor_state(self) -> FederatedModules:
#         policy_state = self.policy.state_dict()
#         actor_keys = self._actor_state_keys()
#         return {
#             self.federated_actor_module_name: OrderedDict(
#                 (key, policy_state[key].detach().cpu().clone()) for key in actor_keys
#             )
#         }

#     def _set_actor_state(self, modules: FederatedModules) -> None:
#         module_name = self.federated_actor_module_name
#         if module_name not in modules:
#             raise KeyError(f"Missing '{module_name}' in federated actor payload.")

#         current_state = self.policy.state_dict()
#         incoming_state = modules[module_name]
#         expected_keys = set(self._actor_state_keys())
#         incoming_keys = set(incoming_state.keys())
#         missing = expected_keys - incoming_keys
#         if missing:
#             raise KeyError(f"Actor payload is missing keys: {sorted(missing)}")

#         for key in expected_keys:
#             current_state[key] = incoming_state[key].to(self.device)
#         self.policy.load_state_dict(current_state, strict=True)

#     def _zero_like_actor_state(self) -> FederatedModules:
#         current = self._get_actor_state()
#         return {
#             module_name: OrderedDict(
#                 (key, th.zeros_like(value, dtype=value.dtype)) for key, value in module_state.items()
#             )
#             for module_name, module_state in current.items()
#         }

#     @staticmethod
#     def _clone_static_modules(modules: FederatedModules) -> FederatedModules:
#         return {
#             module_name: OrderedDict(
#                 (key, value.detach().cpu().clone()) for key, value in module_state.items()
#             )
#             for module_name, module_state in modules.items()
#         }

#     def _clone_modules(self, modules: FederatedModules) -> FederatedModules:
#         return self._clone_static_modules(modules)

#     @staticmethod
#     def _module_l2_norm(modules: FederatedModules) -> float:
#         total = 0.0
#         for module_state in modules.values():
#             for value in module_state.values():
#                 if th.is_floating_point(value):
#                     tensor = value.detach().to(dtype=th.float64)
#                     total += float(th.sum(tensor * tensor).cpu().item())
#         return float(np.sqrt(total))

#     @staticmethod
#     def _simplex_entropy(weights: np.ndarray) -> float:
#         safe = np.clip(np.asarray(weights, dtype=np.float64), 1e-12, 1.0)
#         return float(-np.sum(safe * np.log(safe)))

#     @staticmethod
#     def _effective_num_clients(weights: np.ndarray) -> float:
#         weights = np.asarray(weights, dtype=np.float64)
#         denom = float(np.sum(weights * weights))
#         if denom <= 0.0:
#             return 0.0
#         return 1.0 / denom

#     def _ensure_lambda(self, num_clients: int) -> None:
#         if self.lambda_weights is not None and len(self.lambda_weights) == num_clients:
#             return

#         if self.initial_lambda is None:
#             self.lambda_weights = np.ones(num_clients, dtype=np.float64) / float(num_clients)
#         else:
#             init = np.asarray(self.initial_lambda, dtype=np.float64)
#             if init.shape != (num_clients,):
#                 raise ValueError(f"initial_lambda shape mismatch: expected {(num_clients,)}, got {init.shape}")
#             self.lambda_weights = project_to_simplex(init)

#     # ------------------------------------------------------------------
#     # SB3 rollout/training helpers
#     # ------------------------------------------------------------------
#     def _init_fedsp_training_state(self) -> None:
#         if self.ep_info_buffer is None or self.ep_success_buffer is None:
#             total_timesteps = max(int(getattr(self, "_total_timesteps", 0)), 1)
#             self._setup_learn(
#                 total_timesteps=total_timesteps,
#                 callback=None,
#                 reset_num_timesteps=False,
#                 tb_log_name="fedsp_pg_ppo_paper_aligned",
#                 progress_bar=False,
#             )

#         if self._last_obs is None:
#             assert self.env is not None
#             self._last_obs = self.env.reset()
#             self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

#         if self.rollout_buffer is None:
#             self.rollout_buffer = RolloutBuffer(
#                 self.n_steps,
#                 self.observation_space,
#                 self.action_space,
#                 device=self.device,
#                 gamma=self.gamma,
#                 gae_lambda=self.gae_lambda,
#                 n_envs=self.n_envs,
#             )

#     def _estimate_discounted_mc_return(self) -> float:
#         rewards = np.asarray(self.rollout_buffer.rewards.copy(), dtype=np.float64)
#         episode_starts = np.asarray(self.rollout_buffer.episode_starts.copy(), dtype=bool)
#         if rewards.ndim == 1:
#             rewards = rewards[:, None]
#         if episode_starts.ndim == 1:
#             episode_starts = episode_starts[:, None]

#         returns: list[float] = []
#         fallback_partial_returns: list[float] = []
#         n_steps, n_envs = rewards.shape
#         for env_idx in range(n_envs):
#             running_return = 0.0
#             discount = 1.0
#             has_steps = False
#             for step_idx in range(n_steps):
#                 if bool(episode_starts[step_idx, env_idx]) and has_steps:
#                     returns.append(running_return)
#                     running_return = 0.0
#                     discount = 1.0
#                     has_steps = False

#                 running_return += discount * float(rewards[step_idx, env_idx])
#                 discount *= float(self.gamma)
#                 has_steps = True

#             if has_steps:
#                 fallback_partial_returns.append(running_return)

#         if not returns:
#             if not fallback_partial_returns:
#                 return 0.0
#             return float(np.mean(fallback_partial_returns))
#         return float(np.mean(returns))

#     def _collect_one_rollout(self) -> float:
#         self._init_fedsp_training_state()

#         callback = CallbackList([])
#         callback.init_callback(self)

#         success = self.collect_rollouts(
#             self.env,
#             callback,
#             self.rollout_buffer,
#             n_rollout_steps=self.n_steps,
#         )
#         if not success:
#             raise RuntimeError("collect_rollouts() returned False.")

#         total_timesteps = int(getattr(self, "_total_timesteps", 0))
#         if total_timesteps > 0:
#             self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

#         return self._estimate_discounted_mc_return()

#     def _prepare_rollout_actions(self, actions: th.Tensor) -> th.Tensor:
#         if isinstance(self.action_space, spaces.Discrete):
#             return actions.long().flatten()
#         return actions

#     def _current_clip_range(self) -> float:
#         return float(self.clip_range(self._current_progress_remaining))

#     def _current_learning_rate(self) -> float:
#         lr = self.lr_schedule(self._current_progress_remaining) if callable(self.lr_schedule) else self.learning_rate
#         return float(lr) if lr is not None else 1e-3

#     def _current_server_actor_lr(self) -> float:
#         if self.server_actor_lr is not None:
#             return float(self.server_actor_lr)
#         return self._current_learning_rate()

#     def _get_or_create_server_actor_optimizer(self) -> th.optim.Optimizer:
#         actor_params = self._actor_named_parameters()
#         if not actor_params:
#             raise RuntimeError("Could not identify actor parameters for FedSP-PG PPO server optimizer.")

#         param_names = tuple(actor_params.keys())
#         if (
#             self._server_actor_optimizer_instance is not None
#             and self._server_actor_optimizer_param_names == param_names
#         ):
#             return self._server_actor_optimizer_instance

#         params = list(actor_params.values())
#         lr = self._current_server_actor_lr()
#         if self.server_actor_optimizer == "adam":
#             optimizer = th.optim.Adam(params, lr=lr)
#         else:
#             optimizer = th.optim.SGD(params, lr=lr)

#         self._server_actor_optimizer_instance = optimizer
#         self._server_actor_optimizer_param_names = param_names
#         return optimizer

#     @staticmethod
#     def _set_optimizer_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
#         for param_group in optimizer.param_groups:
#             param_group["lr"] = learning_rate

#     def _current_clip_range_vf(self) -> float | None:
#         if self.clip_range_vf is None:
#             return None
#         return float(self.clip_range_vf(self._current_progress_remaining))

#     def _ppo_value_loss(self, values: th.Tensor, rollout_data: Any, clip_range_vf: float | None) -> th.Tensor:
#         values = values.flatten()
#         if clip_range_vf is None:
#             values_pred = values
#         else:
#             values_pred = rollout_data.old_values + th.clamp(
#                 values - rollout_data.old_values,
#                 -clip_range_vf,
#                 clip_range_vf,
#             )
#         return F.mse_loss(rollout_data.returns, values_pred)

#     def _snapshot_rollout_buffer(self) -> dict[str, Any]:
#         return {
#             "observations": np.array(self.rollout_buffer.observations, copy=True),
#             "actions": np.array(self.rollout_buffer.actions, copy=True),
#             "rewards": np.array(self.rollout_buffer.rewards, copy=True),
#             "episode_starts": np.array(self.rollout_buffer.episode_starts, copy=True),
#             "values": np.array(self.rollout_buffer.values, copy=True),
#             "log_probs": np.array(self.rollout_buffer.log_probs, copy=True),
#             "advantages": np.array(self.rollout_buffer.advantages, copy=True),
#             "returns": np.array(self.rollout_buffer.returns, copy=True),
#             "generator_ready": bool(self.rollout_buffer.generator_ready),
#         }

#     def _restore_rollout_buffer(self, snapshot: dict[str, Any]) -> None:
#         self.rollout_buffer.observations = snapshot["observations"]
#         self.rollout_buffer.actions = snapshot["actions"]
#         self.rollout_buffer.rewards = snapshot["rewards"]
#         self.rollout_buffer.episode_starts = snapshot["episode_starts"]
#         self.rollout_buffer.values = snapshot["values"]
#         self.rollout_buffer.log_probs = snapshot["log_probs"]
#         self.rollout_buffer.advantages = snapshot["advantages"]
#         self.rollout_buffer.returns = snapshot["returns"]
#         self.rollout_buffer.generator_ready = snapshot["generator_ready"]

#     def _iter_rollout_minibatches(self) -> Sequence[RolloutBufferSamples]:
#         total_size = self.rollout_buffer.buffer_size * self.rollout_buffer.n_envs
#         batch_size = self.batch_size or total_size
#         indices = np.random.permutation(total_size)

#         observations = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.observations)
#         actions = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.actions).astype(np.float32, copy=False)
#         old_values = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.values).flatten()
#         old_log_prob = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.log_probs).flatten()
#         advantages = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.advantages).flatten()
#         returns = self.rollout_buffer.swap_and_flatten(self.rollout_buffer.returns).flatten()

#         minibatches: list[RolloutBufferSamples] = []
#         start_idx = 0
#         while start_idx < total_size:
#             batch_inds = indices[start_idx : start_idx + batch_size]
#             minibatches.append(
#                 RolloutBufferSamples(
#                     observations=self.rollout_buffer.to_torch(observations[batch_inds]),
#                     actions=self.rollout_buffer.to_torch(actions[batch_inds]),
#                     old_values=self.rollout_buffer.to_torch(old_values[batch_inds]),
#                     old_log_prob=self.rollout_buffer.to_torch(old_log_prob[batch_inds]),
#                     advantages=self.rollout_buffer.to_torch(advantages[batch_inds]),
#                     returns=self.rollout_buffer.to_torch(returns[batch_inds]),
#                 )
#             )
#             start_idx += batch_size
#         return minibatches

#     def _refresh_rollout_advantages(self) -> None:
#         obs = self.rollout_buffer.observations
#         flat_obs = obs.reshape((-1, *obs.shape[2:]))
#         with th.no_grad():
#             value_tensor = self.policy.predict_values(obs_as_tensor(flat_obs, self.device))
#             last_values = self.policy.predict_values(obs_as_tensor(self._last_obs, self.device))

#         values = value_tensor.detach().cpu().numpy().reshape(self.rollout_buffer.buffer_size, self.rollout_buffer.n_envs)
#         self.rollout_buffer.values = values
#         self.rollout_buffer.generator_ready = False
#         self.rollout_buffer.compute_returns_and_advantage(
#             last_values=last_values,
#             dones=np.asarray(self._last_episode_starts, dtype=bool),
#         )

#     def _update_local_critic(self) -> None:
#         critic_params = self._critic_named_parameters()
#         if not critic_params:
#             return

#         self.policy.set_training_mode(True)
#         self._set_optimizer_learning_rate(self.policy.optimizer, self._current_learning_rate())
#         clip_range_vf = self._current_clip_range_vf()

#         for _ in range(self.n_epochs):
#             for rollout_data in self._iter_rollout_minibatches():
#                 actions = self._prepare_rollout_actions(rollout_data.actions)
#                 values, _, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
#                 value_loss = self._ppo_value_loss(values, rollout_data, clip_range_vf)
#                 loss = self.vf_coef * value_loss

#                 self.policy.optimizer.zero_grad(set_to_none=True)
#                 loss.backward()

#                 for name, parameter in self.policy.named_parameters():
#                     if name not in critic_params:
#                         parameter.grad = None

#                 th.nn.utils.clip_grad_norm_(list(critic_params.values()), self.max_grad_norm)
#                 self.policy.optimizer.step()

#         self.policy.optimizer.zero_grad(set_to_none=True)

#     def _compute_actor_gradient(self) -> FederatedModules:
#         actor_params = self._actor_named_parameters()
#         if not actor_params:
#             raise RuntimeError("Could not identify actor parameters for FedSP-PG PPO.")

#         self.policy.set_training_mode(True)
#         clip_range = self._current_clip_range()
#         gradient_sum = self._zero_like_actor_state()
#         module_name = self.federated_actor_module_name
#         num_actor_batches = 0

#         for _ in range(self.n_epochs):
#             for rollout_data in self._iter_rollout_minibatches():
#                 actions = self._prepare_rollout_actions(rollout_data.actions)
#                 if self.use_sde:
#                     self.policy.reset_noise(self.batch_size)

#                 _, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
#                 advantages = rollout_data.advantages
#                 if self.normalize_advantage and len(advantages) > 1:
#                     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#                 ratio = th.exp(log_prob - rollout_data.old_log_prob)
#                 policy_loss_1 = advantages * ratio
#                 policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
#                 policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

#                 if entropy is None:
#                     entropy_loss = -th.mean(-log_prob)
#                 else:
#                     entropy_loss = -th.mean(entropy)

#                 # SB3 minimizes this loss. Negate its gradient to recover the
#                 # ascent direction used in the paper-level server update.
#                 actor_minimization_loss = policy_loss + self.ent_coef * entropy_loss

#                 self.policy.optimizer.zero_grad(set_to_none=True)
#                 actor_minimization_loss.backward()

#                 for name, parameter in self.policy.named_parameters():
#                     if name not in actor_params:
#                         parameter.grad = None

#                 th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)

#                 for name, parameter in actor_params.items():
#                     if parameter.grad is not None and name in gradient_sum[module_name]:
#                         gradient_sum[module_name][name] += -parameter.grad.detach().cpu()

#                 num_actor_batches += 1
#                 self.policy.optimizer.zero_grad(set_to_none=True)

#         if num_actor_batches == 0:
#             raise RuntimeError("No PPO minibatches were available for actor-gradient computation.")

#         self._fedsp_last_num_actor_batches = num_actor_batches
#         if self.actor_gradient_mode == "mean":
#             for key in gradient_sum[module_name].keys():
#                 gradient_sum[module_name][key] /= float(num_actor_batches)
#         return gradient_sum

#     def _train_local_critic_and_compute_actor_gradient(self) -> FederatedModules:
#         rollout_snapshot = self._snapshot_rollout_buffer()
#         self._update_local_critic()
#         self._restore_rollout_buffer(rollout_snapshot)
#         self._refresh_rollout_advantages()
#         return self._compute_actor_gradient()

#     def _record_server_diagnostics(
#         self,
#         *,
#         client_returns: np.ndarray,
#         client_gradient_norms: list[float],
#         aggregated_gradient_norm: float,
#         actor_step_norm: float,
#         actor_lr: float,
#         num_actor_batches: int,
#     ) -> None:
#         if not hasattr(self, "_logger") or self._logger is None:
#             self._logger = sb3_utils.configure_logger(
#                 self.verbose,
#                 self.tensorboard_log,
#                 "fedsp_pg_ppo_paper_aligned_server",
#                 False,
#             )

#         lambda_weights = np.asarray(self.lambda_weights, dtype=np.float64)
#         self._logger.record("train/fedsp_paper/server_actor_lr", float(actor_lr))
#         self._logger.record("train/fedsp_paper/dual_lr", float(self.dual_lr))
#         self._logger.record("train/fedsp_paper/server_actor_optimizer_is_adam", float(self.server_actor_optimizer == "adam"))
#         self._logger.record("train/fedsp_paper/actor_gradient_mode_is_cumulative", float(self.actor_gradient_mode == "cumulative"))
#         self._logger.record("train/fedsp_paper/num_actor_batches", float(num_actor_batches))
#         self._logger.record("train/fedsp_paper/return_mean", float(np.mean(client_returns)))
#         self._logger.record("train/fedsp_paper/return_std", float(np.std(client_returns)))
#         self._logger.record("train/fedsp_paper/return_min", float(np.min(client_returns)))
#         self._logger.record("train/fedsp_paper/return_max", float(np.max(client_returns)))
#         self._logger.record("train/fedsp_paper/client_grad_norm_mean", float(np.mean(client_gradient_norms)))
#         self._logger.record("train/fedsp_paper/client_grad_norm_std", float(np.std(client_gradient_norms)))
#         self._logger.record("train/fedsp_paper/client_grad_norm_min", float(np.min(client_gradient_norms)))
#         self._logger.record("train/fedsp_paper/client_grad_norm_max", float(np.max(client_gradient_norms)))
#         self._logger.record("train/fedsp_paper/agg_grad_norm", float(aggregated_gradient_norm))
#         self._logger.record("train/fedsp_paper/actor_step_norm", float(actor_step_norm))
#         self._logger.record("train/fedsp_paper/lambda_entropy", self._simplex_entropy(lambda_weights))
#         self._logger.record("train/fedsp_paper/lambda_max", float(np.max(lambda_weights)))
#         self._logger.record("train/fedsp_paper/lambda_min", float(np.min(lambda_weights)))
#         self._logger.record("train/fedsp_paper/effective_clients", self._effective_num_clients(lambda_weights))
#         self._logger.dump(step=int(self.num_timesteps))
#         if self.verbose > 0:
#             print(
#                 "[FedSP-Paper]"
#                 f" step={int(self.num_timesteps)}"
#                 f" actor_lr={float(actor_lr):.3e}"
#                 f" grad_norm={float(aggregated_gradient_norm):.3e}"
#                 f" actor_step_norm={float(actor_step_norm):.3e}"
#                 f" lambda_entropy={self._simplex_entropy(lambda_weights):.3f}"
#                 f" eff_clients={self._effective_num_clients(lambda_weights):.2f}"
#                 f" return_mean={float(np.mean(client_returns)):.3f}"
#             )

#     # ------------------------------------------------------------------
#     # Federated interface
#     # ------------------------------------------------------------------
#     def federated_local_update(self, local_steps: int, **kwargs) -> None:
#         del kwargs

#         target_steps = int(local_steps)
#         if target_steps <= 0:
#             raise ValueError(f"local_steps must be positive, got {local_steps}")

#         collected_steps = 0
#         returns: list[float] = []
#         gradient_accumulator: FederatedModules | None = None
#         num_updates = 0

#         while collected_steps < target_steps:
#             rollout_return = self._collect_one_rollout()
#             local_grad = self._train_local_critic_and_compute_actor_gradient()

#             if gradient_accumulator is None:
#                 gradient_accumulator = self._clone_modules(local_grad)
#             else:
#                 for module_name in gradient_accumulator.keys():
#                     for key in gradient_accumulator[module_name].keys():
#                         gradient_accumulator[module_name][key] += local_grad[module_name][key]

#             returns.append(rollout_return)
#             collected_steps += self.n_steps * self.n_envs
#             num_updates += 1

#         assert gradient_accumulator is not None
#         for module_name in gradient_accumulator.keys():
#             for key in gradient_accumulator[module_name].keys():
#                 gradient_accumulator[module_name][key] /= float(num_updates)

#         self._fedsp_last_gradient = gradient_accumulator
#         self._fedsp_last_return = float(np.mean(returns))
#         self._fedsp_last_local_gradient_norm = self._module_l2_norm(gradient_accumulator)

#     def get_upload_payload(self) -> FederatedPayload:
#         if self._fedsp_last_gradient is None:
#             self._fedsp_last_gradient = self._zero_like_actor_state()
#         if self._fedsp_last_return is None:
#             self._fedsp_last_return = 0.0

#         return {
#             "actor_state": self._get_actor_state(),
#             "actor_gradient": self._clone_modules(self._fedsp_last_gradient),
#             "return": float(self._fedsp_last_return),
#             "num_actor_batches": 0 if self._fedsp_last_num_actor_batches is None else int(self._fedsp_last_num_actor_batches),
#             "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
#         }

#     @classmethod
#     def aggregate_uploads(
#         cls,
#         uploads: Sequence[FederatedPayload],
#         weights: Sequence[float] | None = None,
#     ) -> FederatedPayload:
#         del weights

#         if len(uploads) == 0:
#             raise ValueError("At least one upload is required.")

#         actor_gradients = [upload["actor_gradient"] for upload in uploads]
#         returns = np.asarray([float(upload["return"]) for upload in uploads], dtype=np.float64)
#         num_actor_batches = np.asarray([int(upload.get("num_actor_batches", 0)) for upload in uploads], dtype=np.int32)

#         first_actor_state = uploads[0]["actor_state"]
#         return {
#             "client_actor_gradients": actor_gradients,
#             "client_returns": returns,
#             "client_num_actor_batches": num_actor_batches,
#             "reference_actor_state": cls._clone_static_modules(first_actor_state),
#             "num_clients": len(uploads),
#         }

#     def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
#         if not (0.0 < mix_weight <= 1.0):
#             raise ValueError("mix_weight must be in (0, 1].")

#         if "actor_state" in payload and "client_actor_gradients" not in payload:
#             incoming_actor = payload["actor_state"]
#             if mix_weight < 1.0:
#                 current = self._get_actor_state()
#                 mixed: FederatedModules = {}
#                 for module_name in incoming_actor.keys():
#                     mixed[module_name] = OrderedDict()
#                     for key, value in incoming_actor[module_name].items():
#                         cur = current[module_name][key]
#                         if th.is_floating_point(value):
#                             mixed[module_name][key] = mix_weight * value + (1.0 - mix_weight) * cur.to(value.dtype)
#                         else:
#                             mixed[module_name][key] = value.clone()
#                 incoming_actor = mixed
#             self._set_actor_state(incoming_actor)
#             return

#         client_actor_gradients: list[FederatedModules] = payload["client_actor_gradients"]
#         client_returns = np.asarray(payload["client_returns"], dtype=np.float64)
#         client_num_actor_batches = np.asarray(payload.get("client_num_actor_batches", []), dtype=np.int32)
#         num_clients = int(payload["num_clients"])

#         self._ensure_lambda(num_clients)
#         assert self.lambda_weights is not None

#         actor_state = self._get_actor_state()
#         module_name = self.federated_actor_module_name
#         actor_lr = self._current_server_actor_lr()
#         client_gradient_norms = [self._module_l2_norm(client_grad) for client_grad in client_actor_gradients]
#         actor_before = self._get_actor_state()

#         aggregated_grad: FederatedModules = {module_name: OrderedDict()}
#         for key, value in actor_state[module_name].items():
#             if th.is_floating_point(value):
#                 grad = th.zeros_like(value, dtype=value.dtype)
#                 for lam, client_grad in zip(self.lambda_weights, client_actor_gradients, strict=True):
#                     grad += client_grad[module_name][key].to(value.dtype) * float(lam)
#                 aggregated_grad[module_name][key] = grad
#             else:
#                 aggregated_grad[module_name][key] = th.zeros_like(value)

#         aggregated_gradient_norm = self._module_l2_norm(aggregated_grad)
#         actor_params = self._actor_named_parameters()
#         server_optimizer = self._get_or_create_server_actor_optimizer()
#         self._set_optimizer_learning_rate(server_optimizer, actor_lr)
#         server_optimizer.zero_grad(set_to_none=True)

#         for name, parameter in actor_params.items():
#             parameter.grad = -aggregated_grad[module_name][name].to(self.device)

#         server_optimizer.step()
#         server_optimizer.zero_grad(set_to_none=True)

#         updated_actor = self._get_actor_state()
#         if mix_weight < 1.0:
#             mixed_actor: FederatedModules = {module_name: OrderedDict()}
#             for key, value in actor_state[module_name].items():
#                 if th.is_floating_point(value):
#                     mixed_actor[module_name][key] = mix_weight * updated_actor[module_name][key] + (1.0 - mix_weight) * value
#                 else:
#                     mixed_actor[module_name][key] = value.clone()
#             updated_actor = mixed_actor
#             self._set_actor_state(updated_actor)

#         actor_step: FederatedModules = {module_name: OrderedDict()}
#         for key, before_value in actor_before[module_name].items():
#             after_value = updated_actor[module_name][key]
#             if th.is_floating_point(after_value):
#                 actor_step[module_name][key] = after_value - before_value
#             else:
#                 actor_step[module_name][key] = th.zeros_like(after_value)
#         actor_step_norm = self._module_l2_norm(actor_step)
#         self._fedsp_last_aggregated_gradient_norm = aggregated_gradient_norm
#         self._fedsp_last_actor_step_norm = actor_step_norm

#         # Paper-aligned dual update: lambda <- Proj_Delta(lambda - eta_lambda * J).
#         self.lambda_weights = project_to_simplex(self.lambda_weights - self.dual_lr * client_returns)
#         self._fedsp_num_clients_hint = num_clients
#         self._record_server_diagnostics(
#             client_returns=client_returns,
#             client_gradient_norms=client_gradient_norms,
#             aggregated_gradient_norm=aggregated_gradient_norm,
#             actor_step_norm=actor_step_norm,
#             actor_lr=actor_lr,
#             num_actor_batches=0 if client_num_actor_batches.size == 0 else int(np.mean(client_num_actor_batches)),
#         )

#     def get_client_weight(self) -> float:
#         return 1.0

#     def get_broadcast_payload(self) -> FederatedPayload:
#         return {
#             "actor_state": self._get_actor_state(),
#             "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
#         }
