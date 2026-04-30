
from __future__ import annotations

import math
import os
from copy import deepcopy
from typing import Any

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.federate.td3_avg.td3_avg import TD3Avg


class TD3AvgExperimentManager(ExperimentManager):
    """
    Federated training manager for TD3Avg.

    Design:
    - each client owns a local TD3Avg model and replay buffer,
    - each client trains locally for ``local_steps``,
    - the server averages actor/critic/target parameters,
    - averaged parameters are broadcast back to all clients.

    Extensions in this version:
    - heterogeneous client environments via perturbation sampling,
    - evaluation on both local perturb environments and nominal environments,
    - evaluation results saved to a single NPZ file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_model: TD3Avg | None = None
        self.clients: list[TD3Avg] = []
        self.client_weights: list[float] | None = None
        self.num_clients = 1
        self.local_steps = 1_000
        self.server_update_weight = 1.0
        self._base_hyperparams: dict[str, Any] = {}

        # Heterogeneous perturbation config
        self.perturb_noise_type: str | None = None
        self.perturb_noise_mean: float = 0.0
        self.perturb_noise_std: float = 0.0
        self.perturb_noise_clip: float | None = None
        self.client_noises: list[float] = []

        # Eval config
        self.eval_round_freq: int = 1
        self.eval_local_episodes: int = 5
        self.eval_nominal_episodes: int = 5
        self.eval_deterministic: bool = True
        self.local_eval_envs: list[Any] = []
        self.nominal_eval_envs: list[Any] = []

        # NPZ eval storage
        self.eval_npz_path: str | None = None
        self.eval_history: dict[str, Any] = {}

    def _sample_client_noises(self) -> list[float]:
        if self.perturb_noise_type is None:
            return [0.0 for _ in range(self.num_clients)]

        rng = np.random.default_rng(self.seed)
        samples = rng.normal(
            loc=self.perturb_noise_mean,
            scale=self.perturb_noise_std,
            size=self.num_clients,
        ).astype(np.float32)

        if self.perturb_noise_clip is not None:
            samples = np.clip(samples, -self.perturb_noise_clip, self.perturb_noise_clip)

        return [float(x) for x in samples]

    def _client_env_kwargs(self, client_idx: int) -> dict[str, Any]:
        env_kwargs = deepcopy(self.env_kwargs) if self.env_kwargs is not None else {}
        if self.perturb_noise_type is not None:
            env_kwargs["noise_type"] = self.perturb_noise_type
            env_kwargs["noise"] = float(self.client_noises[client_idx])
        return env_kwargs

    def _nominal_env_kwargs(self) -> dict[str, Any]:
        env_kwargs = deepcopy(self.env_kwargs) if self.env_kwargs is not None else {}
        if "noise_type" in env_kwargs:
            env_kwargs["noise_type"] = None
        if "noise" in env_kwargs:
            env_kwargs["noise"] = 0.0
        return env_kwargs

    def _create_single_env(self, env_kwargs: dict[str, Any] | None = None, no_log: bool = True):
        original_env_kwargs = deepcopy(self.env_kwargs) if self.env_kwargs is not None else None
        try:
            self.env_kwargs = deepcopy(env_kwargs) if env_kwargs is not None else None
            return self.create_envs(1, no_log=no_log)
        finally:
            self.env_kwargs = original_env_kwargs

    def _init_eval_storage(self) -> None:
        assert self.save_path is not None
        self.eval_npz_path = os.path.join(self.save_path, "evaluations.npz")
        self.eval_history = {
            "rounds": [],
            "client_noises": np.asarray(self.client_noises, dtype=np.float32),
            "local_mean": [],
            "local_std": [],
            "nominal_mean": [],
            "nominal_std": [],
            "local_mean_across_clients": [],
            "local_min_across_clients": [],
            "local_max_across_clients": [],
            "nominal_mean_across_clients": [],
            "nominal_min_across_clients": [],
            "nominal_max_across_clients": [],
            "perturb_noise_type": np.asarray("" if self.perturb_noise_type is None else self.perturb_noise_type),
            "perturb_noise_mean": np.asarray(self.perturb_noise_mean, dtype=np.float32),
            "perturb_noise_std": np.asarray(self.perturb_noise_std, dtype=np.float32),
            "perturb_noise_clip": np.asarray(
                np.nan if self.perturb_noise_clip is None else self.perturb_noise_clip, dtype=np.float32
            ),
            "num_clients": np.asarray(self.num_clients, dtype=np.int32),
            "local_steps": np.asarray(self.local_steps, dtype=np.int32),
            "eval_local_episodes": np.asarray(self.eval_local_episodes, dtype=np.int32),
            "eval_nominal_episodes": np.asarray(self.eval_nominal_episodes, dtype=np.int32),
        }

    def _record_eval_results(
        self,
        round_idx: int,
        local_results: list[tuple[float, float]],
        nominal_results: list[tuple[float, float]],
    ) -> None:
        local_means = np.asarray([m for m, _ in local_results], dtype=np.float32)
        local_stds = np.asarray([s for _, s in local_results], dtype=np.float32)
        nominal_means = np.asarray([m for m, _ in nominal_results], dtype=np.float32)
        nominal_stds = np.asarray([s for _, s in nominal_results], dtype=np.float32)

        self.eval_history["rounds"].append(round_idx)
        self.eval_history["local_mean"].append(local_means)
        self.eval_history["local_std"].append(local_stds)
        self.eval_history["nominal_mean"].append(nominal_means)
        self.eval_history["nominal_std"].append(nominal_stds)

        self.eval_history["local_mean_across_clients"].append(float(local_means.mean()))
        self.eval_history["local_min_across_clients"].append(float(local_means.min()))
        self.eval_history["local_max_across_clients"].append(float(local_means.max()))

        self.eval_history["nominal_mean_across_clients"].append(float(nominal_means.mean()))
        self.eval_history["nominal_min_across_clients"].append(float(nominal_means.min()))
        self.eval_history["nominal_max_across_clients"].append(float(nominal_means.max()))

    def _save_eval_npz(self) -> None:
        assert self.eval_npz_path is not None

        payload = {
            "rounds": np.asarray(self.eval_history["rounds"], dtype=np.int32),
            "client_noises": np.asarray(self.eval_history["client_noises"], dtype=np.float32),
            "local_mean": np.asarray(self.eval_history["local_mean"], dtype=np.float32),
            "local_std": np.asarray(self.eval_history["local_std"], dtype=np.float32),
            "nominal_mean": np.asarray(self.eval_history["nominal_mean"], dtype=np.float32),
            "nominal_std": np.asarray(self.eval_history["nominal_std"], dtype=np.float32),
            "local_mean_across_clients": np.asarray(
                self.eval_history["local_mean_across_clients"], dtype=np.float32
            ),
            "local_min_across_clients": np.asarray(
                self.eval_history["local_min_across_clients"], dtype=np.float32
            ),
            "local_max_across_clients": np.asarray(
                self.eval_history["local_max_across_clients"], dtype=np.float32
            ),
            "nominal_mean_across_clients": np.asarray(
                self.eval_history["nominal_mean_across_clients"], dtype=np.float32
            ),
            "nominal_min_across_clients": np.asarray(
                self.eval_history["nominal_min_across_clients"], dtype=np.float32
            ),
            "nominal_max_across_clients": np.asarray(
                self.eval_history["nominal_max_across_clients"], dtype=np.float32
            ),
            "perturb_noise_type": self.eval_history["perturb_noise_type"],
            "perturb_noise_mean": self.eval_history["perturb_noise_mean"],
            "perturb_noise_std": self.eval_history["perturb_noise_std"],
            "perturb_noise_clip": self.eval_history["perturb_noise_clip"],
            "num_clients": self.eval_history["num_clients"],
            "local_steps": self.eval_history["local_steps"],
            "eval_local_episodes": self.eval_history["eval_local_episodes"],
            "eval_nominal_episodes": self.eval_history["eval_nominal_episodes"],
        }
        np.savez(self.eval_npz_path, **payload)

    def _evaluate_round(self, round_idx: int, aggregated_timesteps: int) -> None:
        local_results: list[tuple[float, float]] = []
        nominal_results: list[tuple[float, float]] = []

        for client_idx, client_model in enumerate(self.clients):
            local_mean, local_std = evaluate_policy(
                client_model,
                self.local_eval_envs[client_idx],
                n_eval_episodes=self.eval_local_episodes,
                deterministic=self.eval_deterministic,
            )
            nominal_mean, nominal_std = evaluate_policy(
                client_model,
                self.nominal_eval_envs[client_idx],
                n_eval_episodes=self.eval_nominal_episodes,
                deterministic=self.eval_deterministic,
            )

            local_results.append((float(local_mean), float(local_std)))
            nominal_results.append((float(nominal_mean), float(nominal_std)))

            if self.verbose > 0:
                print(
                    f"[TD3Avg][Eval] round={round_idx} client={client_idx} "
                    f"local_mean={local_mean:.3f} local_std={local_std:.3f} "
                    f"nominal_mean={nominal_mean:.3f} nominal_std={nominal_std:.3f}"
                )

            if hasattr(self, "logger") and self.logger is not None:
                self.logger.record(f"frl_eval/local/client_{client_idx}_mean", float(local_mean))
                self.logger.record(f"frl_eval/local/client_{client_idx}_std", float(local_std))
                self.logger.record(f"frl_eval/nominal/client_{client_idx}_mean", float(nominal_mean))
                self.logger.record(f"frl_eval/nominal/client_{client_idx}_std", float(nominal_std))

        local_means = np.asarray([m for m, _ in local_results], dtype=np.float32)
        nominal_means = np.asarray([m for m, _ in nominal_results], dtype=np.float32)

        if hasattr(self, "logger") and self.logger is not None:
            self.logger.record("frl_eval/local/mean_across_clients", float(local_means.mean()))
            self.logger.record("frl_eval/local/min_across_clients", float(local_means.min()))
            self.logger.record("frl_eval/local/max_across_clients", float(local_means.max()))
            self.logger.record("frl_eval/nominal/mean_across_clients", float(nominal_means.mean()))
            self.logger.record("frl_eval/nominal/min_across_clients", float(nominal_means.min()))
            self.logger.record("frl_eval/nominal/max_across_clients", float(nominal_means.max()))
            self.logger.dump(step=aggregated_timesteps)

        self._record_eval_results(round_idx, local_results, nominal_results)
        self._save_eval_npz()

    def setup_experiment(self) -> tuple[BaseAlgorithm, dict[str, Any]] | None:
        unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(
            unprocessed_hyperparams
        )

        self.create_log_folder()

        # Build once to infer action-noise shapes.
        probe_env = self.create_envs(1, no_log=True)
        hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, probe_env)
        probe_env.close()

        self.num_clients = int(hyperparams.pop("num_clients", 4))
        self.local_steps = int(hyperparams.pop("local_steps", 1_000))
        self.server_update_weight = float(hyperparams.pop("server_update_weight", 1.0))

        self.perturb_noise_type = hyperparams.pop("perturb_noise_type", None)
        self.perturb_noise_mean = float(hyperparams.pop("perturb_noise_mean", 0.0))
        self.perturb_noise_std = float(hyperparams.pop("perturb_noise_std", 0.0))
        perturb_noise_clip = hyperparams.pop("perturb_noise_clip", None)
        self.perturb_noise_clip = None if perturb_noise_clip is None else float(perturb_noise_clip)

        self.eval_round_freq = int(hyperparams.pop("eval_round_freq", 1))
        self.eval_local_episodes = int(hyperparams.pop("eval_local_episodes", 5))
        self.eval_nominal_episodes = int(hyperparams.pop("eval_nominal_episodes", 5))
        self.eval_deterministic = bool(hyperparams.pop("eval_deterministic", True))

        if self.num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        if self.local_steps < 1:
            raise ValueError("local_steps must be >= 1")
        if not (0.0 < self.server_update_weight <= 1.0):
            raise ValueError("server_update_weight must be in (0, 1].")
        if self.eval_round_freq < 1:
            raise ValueError("eval_round_freq must be >= 1")

        self.client_weights = [1.0 / self.num_clients] * self.num_clients
        self._base_hyperparams = deepcopy(hyperparams)
        self.client_noises = self._sample_client_noises()

        self.server_model = TD3Avg(
            env=self._create_single_env(env_kwargs=self._nominal_env_kwargs(), no_log=True),
            tensorboard_log=self.tensorboard_log,
            seed=self.seed,
            verbose=self.verbose,
            device=self.device,
            **deepcopy(self._base_hyperparams),
        )

        self.clients = []
        self.local_eval_envs = []
        self.nominal_eval_envs = []

        for client_idx in range(self.num_clients):
            client_env_kwargs = self._client_env_kwargs(client_idx)
            client_model = TD3Avg(
                env=self._create_single_env(env_kwargs=client_env_kwargs, no_log=True),
                tensorboard_log=None,
                seed=self.seed + client_idx,
                verbose=self.verbose,
                device=self.device,
                **deepcopy(self._base_hyperparams),
            )
            client_model.sync_from(self.server_model)
            self.clients.append(client_model)

            self.local_eval_envs.append(self._create_single_env(env_kwargs=client_env_kwargs, no_log=True))
            self.nominal_eval_envs.append(self._create_single_env(env_kwargs=self._nominal_env_kwargs(), no_log=True))

        self._init_eval_storage()
        self._save_config(saved_hyperparams)
        return self.server_model, saved_hyperparams

    def learn(self, model: BaseAlgorithm) -> None:
        assert isinstance(model, TD3Avg)
        assert self.server_model is not None

        remaining_timesteps = int(self.n_timesteps)
        round_idx = 0
        aggregated_timesteps = 0

        while remaining_timesteps > 0:
            round_idx += 1
            current_local_steps = min(self.local_steps, max(1, math.ceil(remaining_timesteps / self.num_clients)))

            if self.verbose > 0:
                print(
                    f"[TD3Avg] round={round_idx} local_steps={current_local_steps} "
                    f"clients={self.num_clients} remaining={remaining_timesteps}"
                )

            for client_idx, client_model in enumerate(self.clients):
                if self.verbose > 1:
                    print(
                        f"[TD3Avg] local update client={client_idx} "
                        f"noise_type={self.perturb_noise_type} noise={self.client_noises[client_idx]:+.4f}"
                    )
                client_model.learn(
                    total_timesteps=current_local_steps,
                    callback=None,
                    log_interval=self.log_interval if self.log_interval > -1 else 4,
                    tb_log_name=f"td3_avg_client_{client_idx}",
                    reset_num_timesteps=False,
                    progress_bar=False,
                )

            averaged_state = TD3Avg.average_models(self.clients, weights=self.client_weights)
            if self.server_update_weight < 1.0:
                current_server_state = self.server_model.get_federated_state()
                mixed_state = {}
                for module_name, module_state in averaged_state.items():
                    mixed_state[module_name] = module_state.__class__()
                    for key, value in module_state.items():
                        server_value = current_server_state[module_name][key]
                        if value.dtype.is_floating_point:
                            mixed_state[module_name][key] = (
                                self.server_update_weight * value
                                + (1.0 - self.server_update_weight) * server_value.to(value.dtype)
                            )
                        else:
                            mixed_state[module_name][key] = value
                averaged_state = mixed_state

            self.server_model.set_federated_state(averaged_state)
            self.server_model.broadcast_to(self.clients)

            aggregated_timesteps += current_local_steps * self.num_clients
            remaining_timesteps -= current_local_steps * self.num_clients
            self.server_model.num_timesteps = aggregated_timesteps

            if round_idx % self.eval_round_freq == 0:
                self._evaluate_round(round_idx=round_idx, aggregated_timesteps=aggregated_timesteps)

        if self.verbose > 0:
            print(f"[TD3Avg] finished after {round_idx} rounds, aggregated_timesteps={aggregated_timesteps}")

        if len(self.eval_history.get("rounds", [])) > 0:
            self._save_eval_npz()

    def save_trained_model(self, model: BaseAlgorithm) -> None:
        super().save_trained_model(model)

    def close(self) -> None:
        if self.server_model is not None and self.server_model.env is not None:
            self.server_model.env.close()
        for client in self.clients:
            if client.env is not None:
                client.env.close()
        for env in self.local_eval_envs:
            env.close()
        for env in self.nominal_eval_envs:
            env.close()
