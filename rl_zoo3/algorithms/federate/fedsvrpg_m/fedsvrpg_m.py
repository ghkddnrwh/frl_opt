from __future__ import annotations

from collections import defaultdict
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.type_aliases import RolloutBufferSamples

from rl_zoo3.algorithms.federate.common.federated_algorithm import FederatedModules, FederatedPayload
from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo import FedSPPGPPO


class FedSVRPGM(FedSPPGPPO):
    """Practical FedSVRPG-M on top of the stable FedSP-PG PPO backbone.

    The original paper defines a variance-reduced recursive momentum estimator.
    On SB3, directly using raw trajectory REINFORCE gradients is too unstable,
    so this implementation keeps the paper's recursion but uses a practical
    PPO-style actor-update delta as the gradient proxy on each rollout:

      u_0 = g(theta_0; xi_0)
      u_k = beta * g(theta_k; xi_k) + (1 - beta) * (u_{k-1} + g(theta_k; xi_k) - g(theta_{k-1}; xi_k))
      theta_{k+1} = theta_k + eta * u_k

    where each ``g`` is estimated from the same rollout buffer using the local
    critic and a policy-gradient-style actor-only update direction.
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
        "local_lr",
        "momentum_beta",
        "init_grad_episodes",
        "max_update_norm",
    )

    def __init__(self, *args, **kwargs):
        self.local_lr = float(kwargs.pop("local_lr", 1.0))
        self.momentum_beta = float(kwargs.pop("momentum_beta", 0.9))
        self.init_grad_episodes = int(kwargs.pop("init_grad_episodes", 1))
        self.max_update_norm = kwargs.pop("max_update_norm", None)

        # Keep compatibility with the old config file.
        kwargs.pop("local_iteration_horizon", None)
        kwargs.pop("importance_ratio_clip", None)

        super().__init__(*args, **kwargs)

        if self.local_lr <= 0.0:
            raise ValueError("local_lr must be positive")
        if not (0.0 <= self.momentum_beta <= 1.0):
            raise ValueError("momentum_beta must be in [0, 1]")
        if self.init_grad_episodes < 1:
            raise ValueError("init_grad_episodes must be >= 1")
        if self.max_update_norm is not None and float(self.max_update_norm) <= 0.0:
            raise ValueError("max_update_norm must be positive when provided")

        self._fedsvrpg_last_actor_state = self._get_actor_state()
        self._fedsvrpg_last_return = 0.0

    def resolve_federated_local_steps(
        self,
        configured_local_steps: int,
        remaining_timesteps: int,
        num_clients: int,
    ) -> int:
        per_client_budget = int(np.ceil(max(int(remaining_timesteps), 1) / float(max(int(num_clients), 1))))
        return max(1, min(int(configured_local_steps), per_client_budget))

    def _reset_actor_optimizer_state(self) -> None:
        optimizer = self.policy.optimizer
        optimizer.state = defaultdict(dict)
        optimizer.zero_grad(set_to_none=True)

    def _actor_delta_norm(self, modules: FederatedModules) -> float:
        total_sq_norm = 0.0
        for module_state in modules.values():
            for value in module_state.values():
                if th.is_floating_point(value):
                    total_sq_norm += float(th.sum(value.float() ** 2).item())
        return total_sq_norm ** 0.5

    def _scale_actor_delta(self, modules: FederatedModules, scale: float) -> FederatedModules:
        return {
            module_name: OrderedDict((key, value * scale) for key, value in module_state.items())
            for module_name, module_state in modules.items()
        }

    def _add_actor_deltas(
        self,
        lhs: FederatedModules,
        rhs: FederatedModules,
        *,
        rhs_scale: float = 1.0,
    ) -> FederatedModules:
        out: FederatedModules = {}
        for module_name in lhs.keys():
            out[module_name] = OrderedDict()
            for key in lhs[module_name].keys():
                out[module_name][key] = lhs[module_name][key] + rhs[module_name][key].to(lhs[module_name][key].dtype) * rhs_scale
        return out

    def _clip_actor_delta(self, modules: FederatedModules) -> FederatedModules:
        if self.max_update_norm is None:
            return modules
        delta_norm = self._actor_delta_norm(modules)
        max_norm = float(self.max_update_norm)
        if delta_norm <= max_norm or delta_norm == 0.0:
            return modules
        return self._scale_actor_delta(modules, max_norm / delta_norm)

    def _policy_gradient_actor_delta(self) -> FederatedModules:
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for FedSVRPG-M.")

        actor_before = self._get_actor_state()
        optimizer_snapshot = deepcopy(self.policy.optimizer.state_dict())
        self.policy.set_training_mode(True)
        self._set_optimizer_learning_rate(self.policy.optimizer, self._current_learning_rate())
        self._reset_actor_optimizer_state()
        module_name = self.federated_actor_module_name

        for _ in range(self.n_epochs):
            for rollout_data in self._iter_rollout_minibatches():
                self._actor_only_policy_gradient_step(rollout_data, actor_params)

        actor_after = self._get_actor_state()
        self._set_actor_state(actor_before)
        self.policy.optimizer.load_state_dict(optimizer_snapshot)
        self.policy.optimizer.zero_grad(set_to_none=True)

        actor_delta: FederatedModules = {module_name: OrderedDict()}
        for key, before_value in actor_before[module_name].items():
            after_value = actor_after[module_name][key]
            if th.is_floating_point(after_value):
                actor_delta[module_name][key] = after_value - before_value
            else:
                actor_delta[module_name][key] = th.zeros_like(after_value)
        return actor_delta

    def _actor_only_policy_gradient_step(
        self,
        rollout_data: RolloutBufferSamples,
        actor_params: dict[str, th.nn.Parameter],
    ) -> None:
        actions = self._prepare_rollout_actions(rollout_data.actions)
        if self.use_sde:
            self.policy.reset_noise(self.batch_size)

        _, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
        advantages = rollout_data.advantages
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(advantages * log_prob).mean()
        if entropy is None:
            entropy_bonus = -th.mean(-log_prob)
        else:
            entropy_bonus = -th.mean(entropy)
        loss = policy_loss + self.ent_coef * entropy_bonus

        self.policy.optimizer.zero_grad(set_to_none=True)
        loss.backward()

        for name, parameter in self.policy.named_parameters():
            if name not in actor_params:
                parameter.grad = None

        th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)
        self.policy.optimizer.step()

    def _estimate_actor_delta_on_current_buffer(self, actor_state: FederatedModules) -> FederatedModules:
        current_actor_state = self._get_actor_state()
        self._set_actor_state(actor_state)
        try:
            return self._policy_gradient_actor_delta()
        finally:
            self._set_actor_state(current_actor_state)
            self.policy.optimizer.zero_grad(set_to_none=True)

    def _apply_actor_delta(self, actor_delta: FederatedModules) -> None:
        scaled_delta = self._scale_actor_delta(actor_delta, self.local_lr)
        scaled_delta = self._clip_actor_delta(scaled_delta)

        current_actor = self._get_actor_state()
        module_name = self.federated_actor_module_name
        updated_actor: FederatedModules = {module_name: OrderedDict()}
        for key, value in current_actor[module_name].items():
            if th.is_floating_point(value):
                updated_actor[module_name][key] = value + scaled_delta[module_name][key].to(value.dtype)
            else:
                updated_actor[module_name][key] = value.clone()

        self._set_actor_state(updated_actor)
        self._reset_actor_optimizer_state()

    def _ppo_local_actor_delta_step(self) -> tuple[FederatedModules, int]:
        actor_before = self._get_actor_state()
        timesteps_before = int(self.num_timesteps)

        self.learn(
            total_timesteps=self.n_steps * self.n_envs,
            callback=None,
            log_interval=4,
            tb_log_name="fedsvrpg_m_local",
            reset_num_timesteps=False,
            progress_bar=False,
        )

        actor_after = self._get_actor_state()
        actor_delta: FederatedModules = {self.federated_actor_module_name: OrderedDict()}
        for key, before_value in actor_before[self.federated_actor_module_name].items():
            after_value = actor_after[self.federated_actor_module_name][key]
            if th.is_floating_point(after_value):
                actor_delta[self.federated_actor_module_name][key] = after_value - before_value
            else:
                actor_delta[self.federated_actor_module_name][key] = th.zeros_like(after_value)

        return actor_delta, max(0, int(self.num_timesteps) - timesteps_before)

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        del kwargs

        target_steps = int(local_steps)
        if target_steps <= 0:
            raise ValueError(f"local_steps must be positive, got {local_steps}")

        collected_steps = 0
        local_returns: list[float] = []
        momentum_estimator: FederatedModules | None = None

        while collected_steps < target_steps:
            current_delta, step_increment = self._ppo_local_actor_delta_step()

            if momentum_estimator is None:
                momentum_estimator = self._clone_modules(current_delta)
            else:
                momentum_estimator = self._add_actor_deltas(
                    self._scale_actor_delta(current_delta, self.momentum_beta),
                    self._scale_actor_delta(momentum_estimator, 1.0 - self.momentum_beta),
                )

            self._apply_actor_delta(momentum_estimator)
            if self.ep_info_buffer is not None and len(self.ep_info_buffer) > 0:
                rewards = [float(ep_info["r"]) for ep_info in self.ep_info_buffer if "r" in ep_info]
                if rewards:
                    local_returns.append(float(np.mean(rewards)))
            collected_steps += max(step_increment, self.n_steps * self.n_envs)

        self._fedsvrpg_last_actor_state = self._get_actor_state()
        self._fedsvrpg_last_return = float(np.mean(local_returns)) if local_returns else 0.0

    def get_upload_payload(self) -> FederatedPayload:
        return {
            "actor_state": self._get_actor_state(),
            "return": float(self._fedsvrpg_last_return),
        }

    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        if len(uploads) == 0:
            raise ValueError("At least one upload is required for federated aggregation.")

        actor_states = [upload["actor_state"] for upload in uploads]
        aggregated_actor = cls.average_module_states(actor_states, weights=weights)
        mean_return = float(np.mean([float(upload["return"]) for upload in uploads]))
        return {
            "actor_state": aggregated_actor,
            "return": mean_return,
            "num_clients": len(uploads),
        }

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        incoming_actor = payload["actor_state"]
        if mix_weight < 1.0:
            current_actor = self._get_actor_state()
            mixed_actor: FederatedModules = {}
            for module_name in incoming_actor.keys():
                mixed_actor[module_name] = OrderedDict()
                for key, value in incoming_actor[module_name].items():
                    current_value = current_actor[module_name][key]
                    if th.is_floating_point(value):
                        mixed_actor[module_name][key] = (
                            mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
                        )
                    else:
                        mixed_actor[module_name][key] = value.clone()
            incoming_actor = mixed_actor

        self._set_actor_state(incoming_actor)
        self._reset_actor_optimizer_state()

    def get_client_weight(self) -> float:
        return 1.0
