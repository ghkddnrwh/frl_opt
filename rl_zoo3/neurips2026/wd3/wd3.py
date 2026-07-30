"""Weighted Delayed Deep Deterministic Policy Gradient (WD3)."""

from typing import Any, ClassVar, TypeVar

import torch as th

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.noise import ActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.td3.td3 import TD3
from rl_zoo3.neurips2026.wd3.policies import CnnPolicy, MlpPolicy, MultiInputPolicy, WD3Policy

SelfWD3 = TypeVar("SelfWD3", bound="WD3")


class WD3(TD3):
    """Weighted Delayed Deep Deterministic Policy Gradient (WD3).

    WD3 changes TD3's clipped double-Q target. Given two target critic
    estimates ``Q1`` and ``Q2``, it uses

    ``beta * min(Q1, Q2) + (1 - beta) * (Q1 + Q2) / 2``.

    ``beta=1`` is equivalent to TD3's target aggregation, while ``beta=0``
    uses the mean of the two critics. The paper uses ``beta=0.45`` for all
    reported environments.

    Paper: https://arxiv.org/abs/2006.12622

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...).
    :param env: The environment to learn from (if registered in Gymnasium, can be a string).
    :param learning_rate: Learning rate for the actor and critic Adam optimizers.
    :param buffer_size: Replay buffer capacity.
    :param learning_starts: Number of environment steps collected before learning begins.
    :param batch_size: Minibatch size for each gradient update.
    :param tau: Polyak soft-update coefficient.
    :param gamma: Discount factor.
    :param train_freq: Update frequency in environment steps or episodes.
    :param gradient_steps: Gradient steps after each rollout; ``-1`` matches rollout steps.
    :param action_noise: Noise added during environment interaction.
    :param replay_buffer_class: Replay buffer implementation.
    :param replay_buffer_kwargs: Arguments passed to the replay buffer.
    :param optimize_memory_usage: Use the memory-efficient replay-buffer variant.
    :param n_steps: Number of steps used by n-step replay.
    :param policy_delay: Actor and target update delay.
    :param target_policy_noise: Standard deviation of target policy smoothing noise.
    :param target_noise_clip: Absolute clipping limit for target policy smoothing noise.
    :param beta: Weight of the minimum target critic estimate, in ``[0, 1]``.
    :param stats_window_size: Window size for rollout statistics.
    :param tensorboard_log: TensorBoard log directory.
    :param policy_kwargs: Arguments passed to the policy constructor.
    :param verbose: Verbosity level.
    :param seed: Random seed.
    :param device: PyTorch device.
    :param _init_setup_model: Whether to create networks immediately.
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    policy: WD3Policy

    def __init__(
        self,
        policy: str | type[WD3Policy],
        env: GymEnv | str,
        learning_rate: float | Schedule = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 25_000,
        batch_size: int = 100,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int | tuple[int, str] = 1,
        gradient_steps: int = 1,
        action_noise: ActionNoise | None = None,
        replay_buffer_class: type[ReplayBuffer] | None = None,
        replay_buffer_kwargs: dict[str, Any] | None = None,
        optimize_memory_usage: bool = False,
        n_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        beta: float = 0.45,
        stats_window_size: int = 100,
        tensorboard_log: str | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        verbose: int = 0,
        seed: int | None = None,
        device: th.device | str = "auto",
        _init_setup_model: bool = True,
    ):
        if not 0.0 <= beta <= 1.0:
            raise ValueError(f"beta must be in [0, 1], got {beta}")

        self.beta = float(beta)

        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            action_noise=action_noise,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            n_steps=n_steps,
            policy_delay=policy_delay,
            target_policy_noise=target_policy_noise,
            target_noise_clip=target_noise_clip,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
        )

        configured_n_critics = self.policy_kwargs.get("n_critics", 2)
        if configured_n_critics != 2:
            raise ValueError(f"WD3 requires exactly two critics, got n_critics={configured_n_critics}")

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        n_critics = len(self.critic.q_networks)
        if n_critics != 2:
            raise ValueError(f"WD3 requires exactly two critics, but the policy created {n_critics}")

    def _aggregate_target_q_values(self, next_q_values: th.Tensor) -> th.Tensor:
        """Apply the WD3 weighted target update from Equation (8) of the paper."""
        if next_q_values.ndim != 2 or next_q_values.shape[1] != 2:
            raise ValueError(
                "WD3 target aggregation expects shape (batch_size, 2), "
                f"got {tuple(next_q_values.shape)}"
            )
        minimum = th.min(next_q_values, dim=1, keepdim=True).values
        average = th.mean(next_q_values, dim=1, keepdim=True)
        return self.beta * minimum + (1.0 - self.beta) * average

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:
        super().train(gradient_steps=gradient_steps, batch_size=batch_size)
        self.logger.record("train/beta", self.beta)

    def learn(
        self: SelfWD3,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "WD3",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfWD3:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
