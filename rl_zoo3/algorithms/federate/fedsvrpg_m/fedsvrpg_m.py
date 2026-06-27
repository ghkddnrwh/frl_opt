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


class FedSVRPGM(FederatedAlgorithmMixin, PPO):
    """PPO-based practical FedSVRPG-M implemented directly on top of SB3 PPO.

    This class does not inherit from FedSPPGPPO.  It uses SB3's PPO as the RL
    backbone and implements only the federated FedSVRPG-M-style communication
    logic locally in this file.

    The paper's raw REINFORCE/GPOMDP estimator is approximated by an
    actor-only score/advantage gradient direction computed from PPO rollouts,
    not by a temporary PPO optimizer delta.  For each local rollout k on
    client i, the implemented direction is

        u_{r,k} = beta * g_score(theta_{r,k}; B_{r,k})
                  + (1 - beta) * [u_r
                                   + g_score(theta_{r,k}; B_{r,k})
                                   - w(B_{r,k}) g_score(theta_{r-1}; B_{r,k})],

    where w(B_{r,k}) is a trajectory-level importance-sampling weight computed
    from log pi_{theta_{r-1}} - log pi_{theta_{r,k}} on the rollout.  Setting
    ``importance_ratio_clip=None`` leaves this IS weight unclipped; a positive
    value clips it to [1 / clip, clip], which is a practical biased variant.

    The implementation uses the following scale convention.
    ``_ppo_actor_gradient_delta_on_current_buffer`` returns an actor delta that
    already includes SB3's PPO learning rate.  Therefore ``local_lr`` is the
    paper's local step-size eta in the units of that PPO-gradient delta, while
    the experiment manager's ``server_update_weight`` is the global step-size
    lambda.  The server computes u_{r+1} by dividing averaged client actor
    deltas by ``local_lr * num_local_updates``.

    The initial server momentum/anchor u_0 is estimated once in
    ``prepare_federated_training`` by averaging ``init_grad_episodes`` initial
    rollout gradient directions from each client.  The actor parameters are not
    mutated while estimating the current and previous directions; the real
    local actor update is applied once through u_{r,k}.  The critic/value branch
    remains client-local and is updated on the local rollout before estimating
    the actor direction.
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
        # Accepted for backward compatibility with older configs.  This PPO
        # version keeps the critic local, so these do not change synchronization.
        "critic_sync_mode",
    )

    def __init__(self, *args, **kwargs):
        self.local_lr = float(kwargs.pop("local_lr", 1.0))
        self.momentum_beta = float(kwargs.pop("momentum_beta", 0.9))
        self.init_grad_episodes = int(kwargs.pop("init_grad_episodes", 1))
        self.max_update_norm = kwargs.pop("max_update_norm", None)

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

        FedSVRPG-M's variance-reduction correction must subtract two
        estimators of the same form.  The FedSVRPG-M path therefore calls this
        function with ``use_ppo_clip=False`` for both the current local actor
        and the previous global actor.  In that mode the loss is the
        score/advantage surrogate ``-A log pi_theta(a|s)``; for the previous
        global actor it is additionally multiplied by trajectory-level IS
        weights.
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

        batch_size = self.batch_size or total_size
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
        """Compute a PPO actor-gradient delta without mutating actor parameters.

        This returns -lr * grad(actor_loss) accumulated over the same minibatch
        schedule PPO would use.  Unlike the old temporary-update proxy, no Adam
        state or repeated in-place actor update is mixed into the estimator.
        """
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSVRPG-M.")

        original_actor_state = self._get_actor_state()
        module_name = self.federated_actor_module_name
        accumulated_delta = self._zero_like_actor_state()

        try:
            self._set_actor_state(actor_state)
            self.policy.set_training_mode(True)
            learning_rate = self._current_learning_rate()
            clip_range = self._current_clip_range()

            for _ in range(self.n_epochs):
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
                            -float(learning_rate) * parameter.grad.detach().cpu().to(accumulated_delta[module_name][name].dtype)
                        )
                    self.policy.optimizer.zero_grad(set_to_none=True)

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
        # Use the same score/advantage estimator for both terms in
        # g(theta_{r,k}) - w g(theta_{r-1}).  PPO clipping is intentionally
        # disabled inside the SVRPG correction to preserve this pairing.
        current_delta = self._ppo_actor_gradient_delta_on_current_buffer(
            current_actor_state,
            sample_weights=None,
            use_ppo_clip=False,
        )
        self._restore_rng_state(rng_snapshot)

        trajectory_weights = self._trajectory_importance_weights(previous_global_actor_state)
        previous_global_delta = self._ppo_actor_gradient_delta_on_current_buffer(
            previous_global_actor_state,
            sample_weights=trajectory_weights,
            use_ppo_clip=False,
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
                use_ppo_clip=False,
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

        for client in clients:
            if not isinstance(client, FedSVRPGM):
                raise TypeError("FedSVRPGM.prepare_federated_training expects FedSVRPGM clients.")
            client._set_actor_state(theta0_actor)
            client._fedsvrpg_prev_global_actor_state = client._clone_modules(theta0_actor)
            client._fedsvrpg_round_start_actor_state = client._clone_modules(theta0_actor)
            direction, metrics = client._estimate_initial_actor_direction()
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

        u0_client_norms = [m["frl/fedsvrpg_u0_client_direction_norm"] for m in client_metrics]
        u0_client_returns = [m["frl/fedsvrpg_u0_client_return"] for m in client_metrics]
        self._last_federated_metrics = {
            "frl/fedsvrpg_u0_initialized": 1.0,
            "frl/fedsvrpg_u0_episodes_per_client": float(self.init_grad_episodes),
            "frl/fedsvrpg_u0_direction_norm": self._actor_delta_norm(u0),
            "frl/fedsvrpg_u0_client_direction_norm_mean": float(np.mean(u0_client_norms)) if u0_client_norms else 0.0,
            "frl/fedsvrpg_u0_client_return_mean": float(np.mean(u0_client_returns)) if u0_client_returns else 0.0,
        }

    # ------------------------------------------------------------------
    # Federated client side
    # ------------------------------------------------------------------
    def federated_local_update(self, local_steps: int, **kwargs) -> None:
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
        current_delta_norms: list[float] = []
        correction_norms: list[float] = []
        is_weight_means: list[float] = []
        is_weight_maxs: list[float] = []

        while collected_steps < target_steps:
            rollout_return = self._collect_one_rollout()

            # Keep the value function client-local.  Then recompute advantages
            # so the actor proxy uses the updated local critic.
            self._update_local_critic()
            self._refresh_rollout_advantages()

            current_actor = self._get_actor_state()
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

            local_returns.append(float(rollout_return))
            update_norms.append(self._actor_delta_norm(momentum_direction))
            current_delta_norms.append(self._actor_delta_norm(current_delta))
            correction_norms.append(self._actor_delta_norm(correction))
            is_weight_means.append(float(is_metrics["frl/fedsvrpg_is_weight_mean"]))
            is_weight_maxs.append(float(is_metrics["frl/fedsvrpg_is_weight_max"]))
            collected_steps += self.n_steps * self.n_envs
            num_local_updates += 1

        actor_after_round = self._get_actor_state()
        self._fedsvrpg_last_actor_delta = self._subtract_static_modules(actor_after_round, actor_before_round)
        self._fedsvrpg_last_return = float(np.mean(local_returns)) if local_returns else 0.0
        self._fedsvrpg_last_num_local_updates = int(num_local_updates)
        self._last_federated_metrics = {
            "frl/fedsvrpg_local_updates": float(num_local_updates),
            "frl/fedsvrpg_mean_update_norm": float(np.mean(update_norms)) if update_norms else 0.0,
            "frl/fedsvrpg_mean_current_delta_norm": float(np.mean(current_delta_norms)) if current_delta_norms else 0.0,
            "frl/fedsvrpg_mean_correction_norm": float(np.mean(correction_norms)) if correction_norms else 0.0,
            "frl/fedsvrpg_mean_is_weight": float(np.mean(is_weight_means)) if is_weight_means else 1.0,
            "frl/fedsvrpg_max_is_weight": float(np.max(is_weight_maxs)) if is_weight_maxs else 1.0,
        }

    def get_upload_payload(self) -> FederatedPayload:
        return {
            "round_start_actor_state": self._clone_modules(self._fedsvrpg_round_start_actor_state),
            "actor_delta": self._clone_modules(self._fedsvrpg_last_actor_delta),
            "return": float(self._fedsvrpg_last_return),
            "num_local_updates": int(self._fedsvrpg_last_num_local_updates),
            "local_lr": float(self.local_lr),
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

        actor_deltas = [upload["actor_delta"] for upload in uploads]
        aggregated_actor_delta = cls.average_module_states(actor_deltas, weights=weights)
        normalized_weights = cls.normalize_weights(len(uploads), weights)

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
            "return": mean_return,
            "num_clients": len(uploads),
            "mean_local_update_scale": denominator,
        }

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        # Server update path: reconstruct u_{r+1} from client deltas, then
        # update theta_{r+1} = theta_r + lambda * u_{r+1}.  The federated
        # manager passes lambda as server_update_weight via mix_weight.
        if payload.get("aggregation_type") == "fedsvrpg_m_update":
            reference_actor = payload["reference_actor_state"]
            server_direction = payload["server_direction"]
            server_step = self._scale_actor_delta(server_direction, mix_weight)
            updated_actor = self._add_static_modules(reference_actor, server_step)

            self._set_actor_state(updated_actor)
            self._fedsvrpg_prev_global_actor_state = self._clone_modules(reference_actor)
            self._fedsvrpg_server_direction = self._clone_modules(server_direction)
            self._fedsvrpg_round_start_actor_state = self._clone_modules(updated_actor)
            self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
            self._fedsvrpg_last_return = float(payload.get("return", 0.0))
            self._fedsvrpg_last_num_local_updates = 0
            self._reset_actor_optimizer_state()
            self._last_federated_metrics = {
                "frl/fedsvrpg_server_direction_norm": self._actor_delta_norm(server_direction),
                "frl/fedsvrpg_server_step_norm": self._actor_delta_norm(server_step),
                "frl/fedsvrpg_mean_return": float(payload.get("return", 0.0)),
                "frl/fedsvrpg_mean_local_update_scale": float(payload.get("mean_local_update_scale", 1.0)),
            }
            return

        # Broadcast path: clients receive theta_r, theta_{r-1}, and u_r.
        incoming_actor = payload["actor_state"]
        if mix_weight < 1.0:
            incoming_actor = self._mix_static_modules(self._get_actor_state(), incoming_actor, mix_weight)
        self._set_actor_state(incoming_actor)

        previous_actor = payload.get("prev_actor_state", incoming_actor)
        server_direction = payload.get("server_direction", self._zero_like_actor_state())
        self._fedsvrpg_prev_global_actor_state = self._clone_modules(previous_actor)
        self._fedsvrpg_server_direction = self._clone_modules(server_direction)
        self._fedsvrpg_round_start_actor_state = self._clone_modules(incoming_actor)
        self._fedsvrpg_last_actor_delta = self._zero_like_actor_state()
        self._fedsvrpg_last_return = float(payload.get("return", 0.0))
        self._fedsvrpg_last_num_local_updates = 0
        self._reset_actor_optimizer_state()

    def get_broadcast_payload(self) -> FederatedPayload:
        actor_state = self._get_actor_state()
        previous_actor = self._fedsvrpg_prev_global_actor_state
        if previous_actor is None:
            previous_actor = actor_state
        server_direction = self._fedsvrpg_server_direction
        if server_direction is None:
            server_direction = self._zero_like_actor_state()
        return {
            "aggregation_type": "fedsvrpg_m_broadcast",
            "actor_state": self._clone_modules(actor_state),
            "prev_actor_state": self._clone_modules(previous_actor),
            "server_direction": self._clone_modules(server_direction),
            "return": float(self._fedsvrpg_last_return),
        }

    def get_client_weight(self) -> float:
        return 1.0
