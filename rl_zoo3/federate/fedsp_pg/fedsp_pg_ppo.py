from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from copy import deepcopy
from typing import Any

import numpy as np
import torch as th
from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.type_aliases import Schedule
from stable_baselines3.ppo import PPO

from rl_zoo3.federate.buff.buff.federated_algorithm import (
    FederatedAlgorithmMixin,
    FederatedModuleState,
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
    """
    PPO-based FedSP-PG.

    Local client:
      - collects rollout data on its own environment,
      - updates its local critic/value function,
      - computes a local actor gradient estimate,
      - estimates the environment return.

    Server:
      - aggregates local actor gradients with dual weights lambda,
      - updates the shared actor,
      - updates lambda via projected gradient descent using local returns.

    Notes:
      - actor is global/shared,
      - critic is local-only,
      - only actor parameters are synchronized.
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
    )

    federated_actor_modules: tuple[str, ...] = ("policy",)

    def __init__(self, *args, **kwargs):
        self.dual_lr = float(kwargs.pop("dual_lr", 0.05))
        self.initial_lambda = kwargs.pop("initial_lambda", None)

        for key in self.federated_manager_keys:
            kwargs.pop(key, None)

        super().__init__(*args, **kwargs)

        self.lambda_weights: np.ndarray | None = None
        self._fedsp_last_gradient: FederatedModules | None = None
        self._fedsp_last_return: float | None = None
        self._fedsp_num_clients_hint: int | None = None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _get_actor_state(self) -> FederatedModules:
        modules: FederatedModules = {}
        for module_name in self.federated_actor_modules:
            module = getattr(self, module_name)
            modules[module_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
            )
        return modules

    def _set_actor_state(self, modules: FederatedModules) -> None:
        device = self.device
        for module_name in self.federated_actor_modules:
            module = getattr(self, module_name)
            state_dict = OrderedDict(
                (key, value.to(device)) for key, value in modules[module_name].items()
            )
            module.load_state_dict(state_dict)

    def _zero_like_actor_state(self) -> FederatedModules:
        current = self._get_actor_state()
        zeros: FederatedModules = {}
        for module_name, module_state in current.items():
            zeros[module_name] = OrderedDict(
                (key, th.zeros_like(value, dtype=value.dtype)) for key, value in module_state.items()
            )
        return zeros

    def _clone_modules(self, modules: FederatedModules) -> FederatedModules:
        cloned: FederatedModules = {}
        for module_name, module_state in modules.items():
            cloned[module_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module_state.items()
            )
        return cloned

    def _ensure_lambda(self, num_clients: int) -> None:
        if self.lambda_weights is not None and len(self.lambda_weights) == num_clients:
            return

        if self.initial_lambda is None:
            self.lambda_weights = np.ones(num_clients, dtype=np.float64) / float(num_clients)
        else:
            init = np.asarray(self.initial_lambda, dtype=np.float64)
            if init.shape != (num_clients,):
                raise ValueError(
                    f"initial_lambda shape mismatch: expected {(num_clients,)}, got {init.shape}"
                )
            self.lambda_weights = project_to_simplex(init)

    def _init_fedsp_training_state(self) -> None:
        """
        Initialize the internal SB3 on-policy training state before calling collect_rollouts().
        This is normally done inside learn(), but FedSP-PG uses a custom local loop.
        """
        if self.ep_info_buffer is None or self.ep_success_buffer is None:
            self._setup_learn(
                total_timesteps=0,
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

        rewards = self.rollout_buffer.rewards.copy()
        episode_return_estimate = float(np.mean(np.sum(rewards, axis=0)))
        return episode_return_estimate

    def _train_local_critic_and_compute_actor_gradient(self) -> FederatedModules:
        """
        Run one PPO train() pass and extract the actor parameter delta as a gradient proxy.

        We approximate the local actor gradient by:
            grad ~= (theta_after - theta_before) / learning_rate
        This keeps the implementation simple and compatible with SB3 PPO.
        """
        actor_before = self._get_actor_state()

        # Standard PPO update over the rollout buffer:
        self.train()

        actor_after = self._get_actor_state()
        lr = self.lr_schedule(1.0) if callable(self.lr_schedule) else self.learning_rate
        lr = float(lr) if lr is not None else 1e-3
        lr = max(lr, 1e-12)

        grad_modules: FederatedModules = {}
        for module_name in actor_before.keys():
            grad_modules[module_name] = OrderedDict()
            for key in actor_before[module_name].keys():
                before = actor_before[module_name][key]
                after = actor_after[module_name][key]
                if th.is_floating_point(after):
                    grad_modules[module_name][key] = (after - before) / lr
                else:
                    grad_modules[module_name][key] = th.zeros_like(after)
        return grad_modules

    # ------------------------------------------------------------------
    # Federated interface
    # ------------------------------------------------------------------
    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        """
        Perform enough PPO rollout/update cycles to cover local_steps.

        local_steps is treated as the target amount of environment interaction.
        Since PPO collects fixed-size rollouts, the actual number of local steps
        is quantized by n_steps * n_envs.
        """
        del kwargs  # unused here

        target_steps = int(local_steps)
        if target_steps <= 0:
            raise ValueError(f"local_steps must be positive, got {local_steps}")

        collected_steps = 0
        returns: list[float] = []
        gradient_accumulator: FederatedModules | None = None
        num_updates = 0

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
        self.num_timesteps += collected_steps

    def get_upload_payload(self) -> FederatedPayload:
        if self._fedsp_last_gradient is None:
            self._fedsp_last_gradient = self._zero_like_actor_state()
        if self._fedsp_last_return is None:
            self._fedsp_last_return = 0.0

        return {
            "actor_state": self._get_actor_state(),
            "actor_gradient": self._clone_modules(self._fedsp_last_gradient),
            "return": float(self._fedsp_last_return),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
        }

    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        """
        Server-side aggregation payload.

        We do not use the generic weights here.
        FedSP-PG uses dual weights lambda stored on the server instead.
        This method simply packages client gradients and returns together.
        """
        del weights

        if len(uploads) == 0:
            raise ValueError("At least one upload is required.")

        actor_gradients = [upload["actor_gradient"] for upload in uploads]
        returns = np.asarray([float(upload["return"]) for upload in uploads], dtype=np.float64)

        first_actor_state = uploads[0]["actor_state"]
        return {
            "client_actor_gradients": actor_gradients,
            "client_returns": returns,
            "reference_actor_state": cls._clone_static_modules(first_actor_state),
            "num_clients": len(uploads),
        }

    @staticmethod
    def _clone_static_modules(modules: FederatedModules) -> FederatedModules:
        cloned: FederatedModules = {}
        for module_name, module_state in modules.items():
            cloned[module_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module_state.items()
            )
        return cloned

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        """
        On the server model:
          - aggregate client gradients with lambda,
          - update actor,
          - update lambda.

        On the client model:
          - receive actor_state and load it.
        """
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        # Broadcast path: payload already contains actor_state only
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

        # Server update path
        client_actor_gradients: list[FederatedModules] = payload["client_actor_gradients"]
        client_returns = np.asarray(payload["client_returns"], dtype=np.float64)
        num_clients = int(payload["num_clients"])

        self._ensure_lambda(num_clients)

        assert self.lambda_weights is not None

        actor_state = self._get_actor_state()
        lr = self.lr_schedule(1.0) if callable(self.lr_schedule) else self.learning_rate
        lr = float(lr) if lr is not None else 1e-3

        aggregated_grad: FederatedModules = {}
        for module_name, module_state in actor_state.items():
            aggregated_grad[module_name] = OrderedDict()
            for key, value in module_state.items():
                if th.is_floating_point(value):
                    g = th.zeros_like(value, dtype=value.dtype)
                    for lam, client_grad in zip(self.lambda_weights, client_actor_gradients, strict=True):
                        g += client_grad[module_name][key].to(value.dtype) * float(lam)
                    aggregated_grad[module_name][key] = g
                else:
                    aggregated_grad[module_name][key] = th.zeros_like(value)

        updated_actor: FederatedModules = {}
        for module_name, module_state in actor_state.items():
            updated_actor[module_name] = OrderedDict()
            for key, value in module_state.items():
                if th.is_floating_point(value):
                    updated_actor[module_name][key] = value + lr * aggregated_grad[module_name][key]
                else:
                    updated_actor[module_name][key] = value.clone()

        if mix_weight < 1.0:
            mixed_actor: FederatedModules = {}
            for module_name, module_state in actor_state.items():
                mixed_actor[module_name] = OrderedDict()
                for key, value in module_state.items():
                    if th.is_floating_point(value):
                        mixed_actor[module_name][key] = (
                            mix_weight * updated_actor[module_name][key]
                            + (1.0 - mix_weight) * value
                        )
                    else:
                        mixed_actor[module_name][key] = value.clone()
            updated_actor = mixed_actor

        self._set_actor_state(updated_actor)

        # Dual update: lambda <- Proj( lambda - dual_lr * J )
        self.lambda_weights = project_to_simplex(self.lambda_weights - self.dual_lr * client_returns)

        # Save latest server-side values so broadcast uses updated actor/lambda
        self._fedsp_num_clients_hint = num_clients

    def get_client_weight(self) -> float:
        return 1.0

    # ------------------------------------------------------------------
    # Broadcast helper
    # ------------------------------------------------------------------
    def get_broadcast_payload(self) -> FederatedPayload:
        return {
            "actor_state": self._get_actor_state(),
            "lambda_weights": None if self.lambda_weights is None else self.lambda_weights.copy(),
        }