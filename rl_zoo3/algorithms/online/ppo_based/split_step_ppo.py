from __future__ import annotations

from typing import Any, Literal, TypeVar

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from stable_baselines3.common.type_aliases import MaybeCallback
from stable_baselines3.common.utils import explained_variance
from stable_baselines3.ppo import PPO


SelfSplitStepPPO = TypeVar("SelfSplitStepPPO", bound="SplitStepPPO")
ActorGradientMode = Literal["mean", "cumulative"]


class SplitStepPPO(PPO):
    """PPO variant derived from the full-local update rule of Fed-AMPO-GroupPPO.

    The algorithm keeps the rollout/GAE/PPO surrogate machinery of SB3 PPO, but
    decouples actor and critic optimization.  Version 2 also anchors learning-rate
    and clipping schedules at the beginning of each actor stage, matching the
    full-local Fed-AMPO progress-lock semantics:

    1. Collect a rollout with the current actor.
    2. For ``n_epochs`` and all minibatches:
       - compute the PPO actor gradient, clip it, and accumulate it;
       - update only critic parameters with the value loss.
    3. Keep the actor frozen during the whole inner loop.
    4. After ``actor_update_rollouts`` rollouts, average the per-rollout actor
       gradients and apply one explicit actor step.

    With ``actor_update_rollouts=1`` this is the natural non-federated analogue
    of one full-local Fed-AMPO local stage containing one rollout.

    If an old federated configuration used ``local_steps`` interactions per local
    stage, use approximately

        actor_update_rollouts = ceil(local_steps / (n_steps * n_envs))

    to reproduce the same actor-update cadence.

    Notes
    -----
    * ``actor_gradient_mode='cumulative'`` sums clipped minibatch gradients.
    * ``actor_gradient_mode='mean'`` divides that sum by the number of actor
      minibatches. For a fixed number of minibatches, the two modes differ only
      by a scalar factor and can be matched by rescaling ``actor_step_size``.
    * ``actor_step_size`` must be provided explicitly and is independent of the
      PPO ``learning_rate`` used by the critic optimizer.
    * The critic uses the existing SB3 policy optimizer, but gradients for all
      non-critic parameters are removed before ``optimizer.step()``.
    * The actor update bypasses Adam, exactly like the full-local Fed-AMPO actor
      update: parameters are changed directly by ``+ step_size * gradient``.
    * ``target_kl`` from PPO is not used for early stopping because the actor is
      frozen while minibatch gradients are estimated. Post-step KL is logged as
      a diagnostic instead.
    """

    def __init__(
        self,
        *args: Any,
        actor_step_size: float | None = None,
        actor_gradient_mode: ActorGradientMode = "mean",
        actor_mix_weight: float = 1.0,
        actor_update_rollouts: int = 1,
        flush_actor_on_learn_end: bool = True,
        **kwargs: Any,
    ) -> None:
        if actor_step_size is None:
            raise ValueError(
                "actor_step_size must be explicitly provided; it is independent of learning_rate."
            )
        self.actor_step_size = float(actor_step_size)
        if self.actor_step_size <= 0.0:
            raise ValueError("actor_step_size must be positive.")

        normalized_mode = str(actor_gradient_mode).strip().lower()
        if normalized_mode not in {"mean", "cumulative"}:
            raise ValueError("actor_gradient_mode must be 'mean' or 'cumulative'.")
        self.actor_gradient_mode: ActorGradientMode = normalized_mode  # type: ignore[assignment]

        self.actor_mix_weight = float(actor_mix_weight)
        if not (0.0 < self.actor_mix_weight <= 1.0):
            raise ValueError("actor_mix_weight must be in (0, 1].")

        self.actor_update_rollouts = int(actor_update_rollouts)
        if self.actor_update_rollouts <= 0:
            raise ValueError("actor_update_rollouts must be positive.")

        self.flush_actor_on_learn_end = bool(flush_actor_on_learn_end)

        self._pending_actor_gradient: dict[str, th.Tensor] | None = None
        self._pending_actor_rollouts = 0
        self._split_step_progress_anchor: float | None = None
        self._last_actor_num_batches = 0
        self._last_actor_delta_norm = 0.0
        self._last_post_update_kl = 0.0
        self._last_post_update_clip_fraction = 0.0

        # Stage-schedule diagnostics.  These cache the values that were actually
        # used by the current/most recent split-step actor stage.  Keeping these
        # values explicitly avoids accidentally re-evaluating a schedule after
        # the stage anchor has been released.
        self._last_stage_progress = 1.0
        self._last_stage_critic_lr = 0.0
        self._last_stage_clip_range = 0.0
        self._last_stage_clip_range_vf: float | None = None
        self._last_stage_actor_step_size = 0.0
        self._last_applied_actor_step_size = 0.0

        super().__init__(*args, **kwargs)

    # ---------------------------------------------------------------------
    # Parameter partitioning: faithful to the full-local Fed-AMPO code.
    # ---------------------------------------------------------------------
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
        explicit_keys = [key for key in state if self._is_explicit_actor_key(key)]
        if explicit_keys:
            return tuple(explicit_keys)
        return tuple(
            key
            for key, value in state.items()
            if th.is_floating_point(value) and not self._is_critic_key(key)
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

    # ---------------------------------------------------------------------
    # Schedules and PPO losses.
    # ---------------------------------------------------------------------
    def _update_current_progress_remaining(
        self,
        num_timesteps: int,
        total_timesteps: int,
    ) -> None:
        # SB3 calls this after collecting a rollout and before train().
        # Preserve the value from immediately before that update as the
        # split-step stage anchor.  For actor_update_rollouts > 1 the anchor
        # remains unchanged until the explicit actor step completes.
        previous_progress = float(self._current_progress_remaining)
        super()._update_current_progress_remaining(num_timesteps, total_timesteps)
        if self._split_step_progress_anchor is None:
            self._split_step_progress_anchor = previous_progress

    def _schedule_progress(self) -> float:
        if self._split_step_progress_anchor is not None:
            return float(self._split_step_progress_anchor)
        return float(self._current_progress_remaining)

    def _current_actor_step_size(self) -> float:
        return float(self.actor_step_size)

    def _current_clip_range(self) -> float:
        return float(self.clip_range(self._schedule_progress()))

    def _current_clip_range_vf(self) -> float | None:
        if self.clip_range_vf is None:
            return None
        return float(self.clip_range_vf(self._schedule_progress()))

    @staticmethod
    def _set_optimizer_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

    def _prepare_actions(self, actions: th.Tensor) -> th.Tensor:
        if isinstance(self.action_space, spaces.Discrete):
            return actions.long().flatten()
        return actions

    def _actor_loss(
        self,
        log_prob: th.Tensor,
        entropy: th.Tensor | None,
        advantages: th.Tensor,
        old_log_prob: th.Tensor,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if self.normalize_advantage and len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        ratio = th.exp(log_prob - old_log_prob)
        clip_range = self._current_clip_range()
        policy_loss_1 = advantages * ratio
        policy_loss_2 = advantages * th.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)
        policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()
        entropy_loss = -th.mean(entropy) if entropy is not None else -th.mean(-log_prob)
        actor_minimization_loss = policy_loss + self.ent_coef * entropy_loss
        return actor_minimization_loss, policy_loss, entropy_loss

    def _value_loss(
        self,
        values: th.Tensor,
        old_values: th.Tensor,
        returns: th.Tensor,
    ) -> th.Tensor:
        values = values.flatten()
        clip_range_vf = self._current_clip_range_vf()
        if clip_range_vf is None:
            values_pred = values
        else:
            values_pred = old_values + th.clamp(
                values - old_values,
                -clip_range_vf,
                clip_range_vf,
            )
        return F.mse_loss(returns, values_pred)

    # ---------------------------------------------------------------------
    # Gradient helpers.
    # ---------------------------------------------------------------------
    @staticmethod
    def _zeros_like_named_parameters(params: dict[str, th.nn.Parameter]) -> dict[str, th.Tensor]:
        return {
            name: th.zeros_like(parameter, memory_format=th.preserve_format)
            for name, parameter in params.items()
        }

    @staticmethod
    def _gradient_l2_norm(gradient: dict[str, th.Tensor]) -> float:
        if not gradient:
            return 0.0
        total = th.zeros((), dtype=th.float64)
        for value in gradient.values():
            total += th.sum(value.detach().to(device="cpu", dtype=th.float64) ** 2)
        return float(th.sqrt(total).item())

    @staticmethod
    def _parameter_delta_l2_norm(
        before: dict[str, th.Tensor],
        after: dict[str, th.nn.Parameter],
    ) -> float:
        total = th.zeros((), dtype=th.float64)
        for name, parameter in after.items():
            delta = parameter.detach().to(device="cpu", dtype=th.float64) - before[name].to(
                device="cpu", dtype=th.float64
            )
            total += th.sum(delta * delta)
        return float(th.sqrt(total).item())

    def _clear_actor_optimizer_state(self, actor_params: dict[str, th.nn.Parameter]) -> None:
        optimizer = self.policy.optimizer
        for parameter in actor_params.values():
            optimizer.state.pop(parameter, None)

    def _accumulate_pending_actor_gradient(self, gradient: dict[str, th.Tensor]) -> None:
        if self._pending_actor_gradient is None:
            self._pending_actor_gradient = {
                name: value.detach().clone() for name, value in gradient.items()
            }
        else:
            for name, value in gradient.items():
                self._pending_actor_gradient[name].add_(value)
        self._pending_actor_rollouts += 1

    def _apply_pending_actor_update(self) -> bool:
        if self._pending_actor_gradient is None or self._pending_actor_rollouts == 0:
            return False

        actor_params = self._actor_named_parameters()
        if set(actor_params) != set(self._pending_actor_gradient):
            raise RuntimeError("Actor parameter set changed while gradients were pending.")

        averaged_gradient = {
            name: value / float(self._pending_actor_rollouts)
            for name, value in self._pending_actor_gradient.items()
        }

        actor_before = {
            name: parameter.detach().cpu().clone()
            for name, parameter in actor_params.items()
        }
        actor_step_size = self._current_actor_step_size()
        effective_step_size = self.actor_mix_weight * actor_step_size
        self._last_stage_actor_step_size = float(actor_step_size)
        self._last_applied_actor_step_size = float(effective_step_size)

        with th.no_grad():
            for name, parameter in actor_params.items():
                parameter.add_(averaged_gradient[name].to(parameter.device), alpha=effective_step_size)

        self._clear_actor_optimizer_state(actor_params)
        self._last_actor_delta_norm = self._parameter_delta_l2_norm(actor_before, actor_params)

        self._pending_actor_gradient = None
        self._pending_actor_rollouts = 0
        self._split_step_progress_anchor = None
        return True

    # ---------------------------------------------------------------------
    # Diagnostics after the explicit actor step.
    # ---------------------------------------------------------------------
    def _post_actor_step_diagnostics(self, clip_range: float) -> tuple[float, float]:
        approx_kls: list[float] = []
        clip_fractions: list[float] = []
        clip_range = float(clip_range)

        self.policy.set_training_mode(False)
        with th.no_grad():
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._prepare_actions(rollout_data.actions)
                _, log_prob, _ = self.policy.evaluate_actions(rollout_data.observations, actions)
                log_ratio = log_prob - rollout_data.old_log_prob
                ratio = th.exp(log_ratio)
                approx_kl = th.mean((ratio - 1.0) - log_ratio)
                clip_fraction = th.mean((th.abs(ratio - 1.0) > clip_range).float())
                approx_kls.append(float(approx_kl.cpu().item()))
                clip_fractions.append(float(clip_fraction.cpu().item()))

        self.policy.set_training_mode(True)
        return (
            float(np.mean(approx_kls)) if approx_kls else 0.0,
            float(np.mean(clip_fractions)) if clip_fractions else 0.0,
        )

    # ---------------------------------------------------------------------
    # Main optimization step called by SB3 after each rollout.
    # ---------------------------------------------------------------------
    def train(self) -> None:
        self.policy.set_training_mode(True)

        # Under OnPolicyAlgorithm.learn(), the anchor is installed by
        # _update_current_progress_remaining() from the progress value that
        # existed before the just-finished rollout.  Keep this fallback only
        # for direct/manual train() calls outside the normal learn() loop.
        if self._split_step_progress_anchor is None:
            self._split_step_progress_anchor = float(self._current_progress_remaining)

        stage_progress = self._schedule_progress()
        critic_learning_rate = float(self.lr_schedule(stage_progress))
        stage_clip_range = float(self.clip_range(stage_progress))
        stage_clip_range_vf = (
            None
            if self.clip_range_vf is None
            else float(self.clip_range_vf(stage_progress))
        )
        stage_actor_step_size = float(self.actor_step_size)

        self._last_stage_progress = float(stage_progress)
        self._last_stage_critic_lr = float(critic_learning_rate)
        self._last_stage_clip_range = float(stage_clip_range)
        self._last_stage_clip_range_vf = stage_clip_range_vf
        self._last_stage_actor_step_size = float(stage_actor_step_size)
        self._set_optimizer_learning_rate(self.policy.optimizer, critic_learning_rate)

        actor_params = self._actor_named_parameters()
        critic_params = self._critic_named_parameters()
        if not actor_params:
            raise RuntimeError("Could not identify actor parameters for SplitStepPPO.")
        if not critic_params:
            raise RuntimeError("Could not identify critic parameters for SplitStepPPO.")

        actor_gradient = self._zeros_like_named_parameters(actor_params)

        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_losses: list[float] = []
        pre_step_approx_kls: list[float] = []
        pre_step_clip_fractions: list[float] = []
        actor_grad_norms: list[float] = []
        critic_grad_norms: list[float] = []
        num_batches = 0

        for _epoch in range(self.n_epochs):
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = self._prepare_actions(rollout_data.actions)
                if self.use_sde:
                    self.policy.reset_noise(self.batch_size)

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                )

                actor_loss, policy_loss, entropy_loss = self._actor_loss(
                    log_prob,
                    entropy,
                    rollout_data.advantages,
                    rollout_data.old_log_prob,
                )
                value_loss = self._value_loss(
                    values,
                    rollout_data.old_values,
                    rollout_data.returns,
                )
                critic_loss = self.vf_coef * value_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    ratio = th.exp(log_ratio)
                    approx_kl = th.mean((ratio - 1.0) - log_ratio)
                    clip_fraction = th.mean(
                        (th.abs(ratio - 1.0) > stage_clip_range).float()
                    )

                # Actor: estimate and store gradient, but do not optimizer.step().
                self.policy.optimizer.zero_grad(set_to_none=True)
                actor_loss.backward(retain_graph=True)
                for name, parameter in self.policy.named_parameters():
                    if name not in actor_params:
                        parameter.grad = None

                actor_grad_norm = th.nn.utils.clip_grad_norm_(
                    list(actor_params.values()),
                    self.max_grad_norm,
                )
                actor_grad_norms.append(float(actor_grad_norm.detach().cpu().item()))

                for name, parameter in actor_params.items():
                    if parameter.grad is not None:
                        # Store ascent direction, matching full-local Fed-AMPO.
                        actor_gradient[name].add_(-parameter.grad.detach())

                # Critic: same minibatch, value loss only, optimizer step now.
                self.policy.optimizer.zero_grad(set_to_none=True)
                critic_loss.backward()
                for name, parameter in self.policy.named_parameters():
                    if name not in critic_params:
                        parameter.grad = None

                critic_grad_norm = th.nn.utils.clip_grad_norm_(
                    list(critic_params.values()),
                    self.max_grad_norm,
                )
                critic_grad_norms.append(float(critic_grad_norm.detach().cpu().item()))
                self.policy.optimizer.step()
                self.policy.optimizer.zero_grad(set_to_none=True)

                policy_losses.append(float(policy_loss.detach().cpu().item()))
                value_losses.append(float(value_loss.detach().cpu().item()))
                entropy_losses.append(float(entropy_loss.detach().cpu().item()))
                pre_step_approx_kls.append(float(approx_kl.detach().cpu().item()))
                pre_step_clip_fractions.append(float(clip_fraction.detach().cpu().item()))
                num_batches += 1

            self._n_updates += 1

        if num_batches == 0:
            raise RuntimeError("No rollout minibatches were available for actor-gradient estimation.")

        if self.actor_gradient_mode == "mean":
            for name in actor_gradient:
                actor_gradient[name].div_(float(num_batches))

        self._last_actor_num_batches = num_batches
        rollout_actor_gradient_norm = self._gradient_l2_norm(actor_gradient)
        self._accumulate_pending_actor_gradient(actor_gradient)

        actor_update_applied = False
        if self._pending_actor_rollouts >= self.actor_update_rollouts:
            actor_update_applied = self._apply_pending_actor_update()

        if actor_update_applied:
            post_kl, post_clip_fraction = self._post_actor_step_diagnostics(
                self._last_stage_clip_range
            )
            self._last_post_update_kl = post_kl
            self._last_post_update_clip_fraction = post_clip_fraction
        else:
            self._last_actor_delta_norm = 0.0
            self._last_applied_actor_step_size = 0.0

        explained_var = explained_variance(
            self.rollout_buffer.values.flatten(),
            self.rollout_buffer.returns.flatten(),
        )

        # Standard PPO-like diagnostics.
        self.logger.record("train/entropy_loss", float(np.mean(entropy_losses)))
        self.logger.record("train/policy_gradient_loss", float(np.mean(policy_losses)))
        self.logger.record("train/value_loss", float(np.mean(value_losses)))
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/clip_range", self._last_stage_clip_range)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", self._last_stage_clip_range_vf)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        # SplitStepPPO-specific diagnostics.
        self.logger.record("train/split_step/pre_update_approx_kl", float(np.mean(pre_step_approx_kls)))
        self.logger.record("train/split_step/pre_update_clip_fraction", float(np.mean(pre_step_clip_fractions)))
        self.logger.record("train/split_step/post_update_approx_kl", self._last_post_update_kl)
        self.logger.record("train/split_step/post_update_clip_fraction", self._last_post_update_clip_fraction)
        self.logger.record("train/split_step/actor_gradient_norm", rollout_actor_gradient_norm)
        self.logger.record("train/split_step/actor_delta_norm", self._last_actor_delta_norm)
        self.logger.record("train/split_step/actor_grad_norm_preclip", float(np.mean(actor_grad_norms)))
        self.logger.record("train/split_step/critic_grad_norm_preclip", float(np.mean(critic_grad_norms)))
        self.logger.record("train/split_step/num_actor_batches", float(num_batches))
        self.logger.record("train/split_step/actor_update_applied", float(actor_update_applied))
        self.logger.record("train/split_step/pending_actor_rollouts", float(self._pending_actor_rollouts))
        self.logger.record("train/split_step/stage_progress", self._last_stage_progress)
        self.logger.record("train/split_step/critic_lr", self._last_stage_critic_lr)
        self.logger.record("train/split_step/stage_clip_range", self._last_stage_clip_range)
        if self._last_stage_clip_range_vf is not None:
            self.logger.record(
                "train/split_step/stage_clip_range_vf",
                self._last_stage_clip_range_vf,
            )
        self.logger.record(
            "train/split_step/actor_step_size",
            self._last_stage_actor_step_size,
        )
        self.logger.record(
            "train/split_step/effective_actor_step_size",
            self._last_applied_actor_step_size,
        )
        self.logger.record(
            "train/split_step/gradient_mode_mean",
            float(self.actor_gradient_mode == "mean"),
        )
        self.logger.record(
            "train/split_step/gradient_mode_cumulative",
            float(self.actor_gradient_mode == "cumulative"),
        )

    def learn(
        self: SelfSplitStepPPO,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "SplitStepPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfSplitStepPPO:
        result = super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

        if self.flush_actor_on_learn_end and self._pending_actor_rollouts > 0:
            self._apply_pending_actor_update()

        return result
