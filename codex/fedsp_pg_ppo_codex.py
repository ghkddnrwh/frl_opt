from __future__ import annotations

import math
from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy

import numpy as np
import torch as th

from rl_zoo3.algorithms.federate.common.federated_algorithm import FederatedModules, FederatedPayload
from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo import project_to_simplex
from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo_paper_aligned import FedSPPGPPOPaperAligned


class CodexFedSPPGPPO(FedSPPGPPOPaperAligned):
    """FedSP-PG/PPO variant tuned for short-budget heterogeneous FRL.

    This class deliberately lives outside ``rl_zoo3/federate/fedsp_pg`` so the
    original two implementations stay untouched. It keeps the paper-aligned
    actor-gradient upload path, but adds three practical stabilizers:

    - decouple the server communication interval from large configured
      ``local_steps`` by allowing a smaller ``fedsp_rollout_steps``;
    - use a soft worst-client dual update based on an EMA of client returns;
    - optionally scale/clip the aggregated server actor gradient.
    """

    federated_manager_keys: tuple[str, ...] = FedSPPGPPOPaperAligned.federated_manager_keys + (
        "fedsp_rollout_steps",
        "max_local_steps_per_round",
        "local_gradient_aggregation",
        "dual_mode",
        "dual_temperature",
        "dual_ema",
        "lambda_mix",
        "min_lambda",
        "return_scale_floor",
        "server_gradient_scale",
        "server_gradient_clip",
        "actor_update_estimator",
        "server_update_rule",
        "server_delta_scale",
    )

    def __init__(self, *args, **kwargs):
        fedsp_rollout_steps = kwargs.pop("fedsp_rollout_steps", None)
        if fedsp_rollout_steps is not None:
            fedsp_rollout_steps = int(fedsp_rollout_steps)
            if fedsp_rollout_steps <= 0:
                raise ValueError("fedsp_rollout_steps must be positive.")
            kwargs["n_steps"] = fedsp_rollout_steps

        max_local_steps_per_round = kwargs.pop("max_local_steps_per_round", None)
        self.max_local_steps_per_round = (
            None if max_local_steps_per_round is None else int(max_local_steps_per_round)
        )
        self.local_gradient_aggregation = str(kwargs.pop("local_gradient_aggregation", "mean")).lower()
        if self.local_gradient_aggregation not in {"mean", "sum", "sqrt"}:
            raise ValueError("local_gradient_aggregation must be one of: mean, sum, sqrt")

        self.dual_mode = str(kwargs.pop("dual_mode", "softmin_ema")).lower()
        if self.dual_mode not in {"softmin_ema", "normalized_projected", "raw_projected"}:
            raise ValueError("dual_mode must be softmin_ema, normalized_projected, or raw_projected")

        self.dual_temperature = float(kwargs.pop("dual_temperature", 0.7))
        self.dual_ema = float(kwargs.pop("dual_ema", 0.9))
        self.lambda_mix = float(kwargs.pop("lambda_mix", 0.35))
        self.min_lambda = float(kwargs.pop("min_lambda", 0.03))
        self.return_scale_floor = float(kwargs.pop("return_scale_floor", 1.0))
        self.server_gradient_scale = float(kwargs.pop("server_gradient_scale", 1.0))
        self.server_gradient_clip = float(kwargs.pop("server_gradient_clip", 0.0))
        self.actor_update_estimator = str(kwargs.pop("actor_update_estimator", "ppo_delta")).lower()
        self.server_update_rule = str(kwargs.pop("server_update_rule", "direct_delta")).lower()
        self.server_delta_scale = float(kwargs.pop("server_delta_scale", 1.0))

        if self.dual_temperature <= 0.0:
            raise ValueError("dual_temperature must be positive.")
        if not (0.0 <= self.dual_ema < 1.0):
            raise ValueError("dual_ema must be in [0, 1).")
        if not (0.0 < self.lambda_mix <= 1.0):
            raise ValueError("lambda_mix must be in (0, 1].")
        if self.min_lambda < 0.0:
            raise ValueError("min_lambda must be non-negative.")
        if self.return_scale_floor <= 0.0:
            raise ValueError("return_scale_floor must be positive.")
        if self.actor_update_estimator not in {"gradient", "ppo_delta"}:
            raise ValueError("actor_update_estimator must be gradient or ppo_delta")
        if self.server_update_rule not in {"adam_gradient", "direct_delta"}:
            raise ValueError("server_update_rule must be adam_gradient or direct_delta")

        super().__init__(*args, **kwargs)

        if self.max_local_steps_per_round is None:
            self.max_local_steps_per_round = int(self.n_steps * self.n_envs)
        if self.max_local_steps_per_round <= 0:
            raise ValueError("max_local_steps_per_round must be positive.")

        max_floor = 1.0 / float(max(int(getattr(self, "_fedsp_num_clients_hint", 1) or 1), 1))
        if self.min_lambda >= max_floor:
            self.min_lambda = 0.0

        self._codex_return_ema: np.ndarray | None = None

    def resolve_federated_local_steps(
        self,
        configured_local_steps: int,
        remaining_timesteps: int,
        num_clients: int,
    ) -> int:
        del remaining_timesteps, num_clients
        rollout_quantum = max(1, int(self.n_steps * self.n_envs))
        target_steps = max(1, int(configured_local_steps))
        if self.max_local_steps_per_round is not None:
            target_steps = min(target_steps, int(self.max_local_steps_per_round))
        target_steps = max(target_steps, rollout_quantum)
        return int(math.ceil(target_steps / rollout_quantum) * rollout_quantum)

    @staticmethod
    def _scale_modules_in_place(modules: FederatedModules, scale: float) -> None:
        if scale == 1.0:
            return
        for module_state in modules.values():
            for key, value in module_state.items():
                if th.is_floating_point(value):
                    module_state[key] = value * float(scale)

    def _clip_modules_in_place(self, modules: FederatedModules, max_norm: float) -> float:
        norm = self._module_l2_norm(modules)
        if max_norm > 0.0 and norm > max_norm:
            self._scale_modules_in_place(modules, max_norm / (norm + 1e-12))
            return float(max_norm)
        return float(norm)

    @staticmethod
    def _apply_lambda_floor(weights: np.ndarray, floor: float) -> np.ndarray:
        weights = np.asarray(weights, dtype=np.float64)
        n = weights.size
        if n == 0:
            return weights
        max_floor = (1.0 - 1e-12) / float(n)
        floor = min(max(float(floor), 0.0), max_floor)
        if floor == 0.0:
            return project_to_simplex(weights)
        weights = project_to_simplex(weights)
        return np.full(n, floor, dtype=np.float64) + (1.0 - floor * n) * weights

    def _normalized_projected_lambda_update(self, client_returns: np.ndarray) -> np.ndarray:
        centered = client_returns - float(np.mean(client_returns))
        scale = max(float(np.std(centered)), self.return_scale_floor)
        return project_to_simplex(self.lambda_weights - self.dual_lr * centered / scale)

    def _softmin_ema_lambda_update(self, client_returns: np.ndarray) -> np.ndarray:
        if self._codex_return_ema is None or self._codex_return_ema.shape != client_returns.shape:
            self._codex_return_ema = client_returns.astype(np.float64).copy()
        else:
            self._codex_return_ema = (
                self.dual_ema * self._codex_return_ema + (1.0 - self.dual_ema) * client_returns
            )

        centered = self._codex_return_ema - float(np.mean(self._codex_return_ema))
        scale = max(float(np.std(centered)), self.return_scale_floor)
        logits = -centered / (scale * self.dual_temperature)
        logits = logits - float(np.max(logits))
        target = np.exp(logits)
        target = target / max(float(np.sum(target)), 1e-12)
        target = self._apply_lambda_floor(target, self.min_lambda)
        mixed = (1.0 - self.lambda_mix) * self.lambda_weights + self.lambda_mix * target
        return self._apply_lambda_floor(mixed, self.min_lambda)

    def _update_lambda_weights(self, client_returns: np.ndarray) -> None:
        assert self.lambda_weights is not None
        if self.dual_mode == "raw_projected":
            updated = project_to_simplex(self.lambda_weights - self.dual_lr * client_returns)
        elif self.dual_mode == "normalized_projected":
            updated = self._normalized_projected_lambda_update(client_returns)
        else:
            updated = self._softmin_ema_lambda_update(client_returns)
        self.lambda_weights = self._apply_lambda_floor(updated, self.min_lambda)

    def _snapshot_optimizer_state(self) -> dict:
        return deepcopy(self.policy.optimizer.state_dict())

    def _restore_optimizer_state(self, snapshot: dict) -> None:
        self.policy.optimizer.load_state_dict(snapshot)

    def _compute_actor_delta(self) -> FederatedModules:
        actor_params = self._actor_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for Codex FedSP-PG PPO.")

        actor_before = self._get_actor_state()
        optimizer_snapshot = self._snapshot_optimizer_state()
        self.policy.set_training_mode(True)
        self._set_optimizer_learning_rate(self.policy.optimizer, self._current_learning_rate())
        clip_range = self._current_clip_range()
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

                actor_minimization_loss = policy_loss + self.ent_coef * entropy_loss
                self.policy.optimizer.zero_grad(set_to_none=True)
                actor_minimization_loss.backward()

                for name, parameter in self.policy.named_parameters():
                    if name not in actor_params:
                        parameter.grad = None

                th.nn.utils.clip_grad_norm_(list(actor_params.values()), self.max_grad_norm)
                self.policy.optimizer.step()
                num_actor_batches += 1

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

        self._fedsp_last_num_actor_batches = num_actor_batches
        return actor_delta

    def _compute_actor_gradient(self) -> FederatedModules:
        if self.actor_update_estimator == "ppo_delta":
            return self._compute_actor_delta()
        return super()._compute_actor_gradient()

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        del kwargs

        target_steps = int(local_steps)
        if target_steps <= 0:
            raise ValueError(f"local_steps must be positive, got {local_steps}")

        collected_steps = 0
        returns: list[float] = []
        gradient_accumulator: FederatedModules | None = None
        num_updates = 0
        num_actor_batches = 0

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
            num_actor_batches += int(self._fedsp_last_num_actor_batches or 0)

        assert gradient_accumulator is not None
        if self.local_gradient_aggregation == "mean":
            scale = 1.0 / float(num_updates)
        elif self.local_gradient_aggregation == "sqrt":
            scale = 1.0 / math.sqrt(float(num_updates))
        else:
            scale = 1.0
        self._scale_modules_in_place(gradient_accumulator, scale)

        self._fedsp_last_gradient = gradient_accumulator
        self._fedsp_last_return = float(np.mean(returns))
        self._fedsp_last_local_gradient_norm = self._module_l2_norm(gradient_accumulator)
        self._fedsp_last_num_actor_batches = num_actor_batches

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        if "actor_state" in payload and "client_actor_gradients" not in payload:
            incoming_actor = payload["actor_state"]
            if mix_weight < 1.0:
                current = self._get_actor_state()
                mixed: FederatedModules = {}
                for module_name in incoming_actor.keys():
                    mixed[module_name] = OrderedDict()
                    for key, value in incoming_actor[module_name].items():
                        cur = current[module_name][key]
                        if th.is_floating_point(value):
                            mixed[module_name][key] = mix_weight * value + (1.0 - mix_weight) * cur.to(value.dtype)
                        else:
                            mixed[module_name][key] = value.clone()
                incoming_actor = mixed
            self._set_actor_state(incoming_actor)
            return

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

        self._scale_modules_in_place(aggregated_grad, self.server_gradient_scale)
        aggregated_gradient_norm = self._clip_modules_in_place(aggregated_grad, self.server_gradient_clip)

        if self.server_update_rule == "direct_delta":
            updated_actor: FederatedModules = {module_name: OrderedDict()}
            for key, value in actor_state[module_name].items():
                if th.is_floating_point(value):
                    updated_actor[module_name][key] = (
                        value + self.server_delta_scale * aggregated_grad[module_name][key]
                    )
                else:
                    updated_actor[module_name][key] = value.clone()
            self._set_actor_state(updated_actor)
        else:
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

        self._update_lambda_weights(client_returns)
        self._fedsp_num_clients_hint = num_clients
        self._record_server_diagnostics(
            client_returns=client_returns,
            client_gradient_norms=client_gradient_norms,
            aggregated_gradient_norm=aggregated_gradient_norm,
            actor_step_norm=actor_step_norm,
            actor_lr=actor_lr,
            num_actor_batches=0 if client_num_actor_batches.size == 0 else int(np.mean(client_num_actor_batches)),
        )
