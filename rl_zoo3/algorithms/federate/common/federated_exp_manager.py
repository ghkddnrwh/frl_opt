from __future__ import annotations

import json
import math
import os
import time
from copy import deepcopy
from typing import Any, cast

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv, sync_envs_normalization

from rl_zoo3.exp_manager import ExperimentManager
from rl_zoo3.algorithms.federate.common.federated_algorithm import FederatedAlgorithmMixin
from rl_zoo3.utils import ALGOS


class FederatedExperimentManager(ExperimentManager):
    """Generic federated experiment manager.

    This manager owns the FRL orchestration only:
    - create the server and local client models,
    - trigger local client updates,
    - collect upload payloads,
    - aggregate them on the server,
    - broadcast the global payload back to clients.

    It also supports heterogeneous client environments by sampling client-specific
    perturbation strengths and evaluating both:
    - each client's own local perturbation environment,
    - the nominal non-perturbed environment.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.server_model: BaseAlgorithm | None = None
        self.clients: list[BaseAlgorithm] = []
        self.num_clients = 1
        self.local_steps = 1_000
        self.server_update_weight = 1.0
        self._base_hyperparams: dict[str, Any] = {}

        self.perturb_noise_type: str | None = None
        self.perturb_noise_range = 0.0
        self.client_noise_values: list[float] = []
        self.client_env_kwargs: list[dict[str, Any]] = []

        self.eval_local_episodes = 10
        self.eval_nominal_episodes = 10
        self.eval_round_freq = 1
        self.eval_deterministic = True
        self.local_eval_envs: list[VecEnv] = []
        self.nominal_eval_env: VecEnv | None = None

        self.eval_npz_path: str | None = None
        self.eval_history: dict[str, Any] = {}
        self.perturbation_info_path: str | None = None
        arg_log_wandb = getattr(self.args, "log_wandb", None)
        self.log_wandb = bool(getattr(self.args, "track", False)) if arg_log_wandb is None else self._as_bool(arg_log_wandb)
        self._wandb_metrics_defined = False

    def _get_federated_algo_class(self) -> type[BaseAlgorithm]:
        algo_cls = ALGOS[self.algo]
        if not issubclass(algo_cls, FederatedAlgorithmMixin):
            raise ValueError(
                f"Algorithm '{self.algo}' does not implement the federated interface. "
                "Use --frl only with federated algorithms."
            )
        return algo_cls

    @staticmethod
    def _normalize_perturb_noise_type(noise_type: Any) -> str | None:
        if noise_type is None:
            return None
        if isinstance(noise_type, str) and noise_type.strip().lower() in {"", "none", "null"}:
            return None
        return str(noise_type)

    @staticmethod
    def _parse_client_noise_values(value: Any) -> list[float]:
        """Parse explicitly configured perturbation values for each client."""
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    "client_noise_values must be a list of numbers, for example "
                    "[-0.5, -0.25, 0.0, 0.25, 0.5]"
                ) from exc

        if not isinstance(value, (list, tuple, np.ndarray)):
            raise ValueError("client_noise_values must be a list, tuple, or numpy array")

        parsed_values = [float(noise) for noise in value]
        if not all(math.isfinite(noise) for noise in parsed_values):
            raise ValueError("client_noise_values must contain only finite numbers")
        if not all(-1.0 < noise < 1.0 for noise in parsed_values):
            raise ValueError("Each client_noise_values entry must be in (-1, 1)")
        return parsed_values

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

    def _ensure_wandb_run(self, saved_hyperparams: dict[str, Any] | None = None) -> Any | None:
        if not self.log_wandb:
            return None

        try:
            import wandb
        except ImportError as e:
            raise ImportError(
                "log_wandb=True requires Weights & Biases. Install it with `pip install wandb`."
            ) from e

        run = wandb.run
        if run is None:
            run_name = f"{self.env_name}__{self.algo}__{self.seed}__{int(time.time())}"
            tags = [*getattr(self.args, "wandb_tags", []), "frl", self.algo]
            config: dict[str, Any] = {
                "algo": self.algo,
                "env": str(self.env_name),
                "seed": int(self.seed),
                "frl": True,
                "log_folder": self.log_folder,
                "save_path": self.save_path,
            }
            if saved_hyperparams is not None:
                config["saved_hyperparams"] = dict(saved_hyperparams)
            run = wandb.init(
                name=run_name,
                project=getattr(self.args, "wandb_project_name", "sb3"),
                entity=getattr(self.args, "wandb_entity", None),
                group=getattr(self.args, "wandb_group", None),
                tags=tags,
                config=config,
                sync_tensorboard=self.tensorboard_log is not None,
                monitor_gym=True,
                save_code=True,
            )
        elif saved_hyperparams is not None:
            run.config.update({"saved_hyperparams": dict(saved_hyperparams)}, allow_val_change=True)

        if not self._wandb_metrics_defined:
            wandb.define_metric("frl/round")
            wandb.define_metric("frl/*", step_metric="frl/round")
            wandb.define_metric("eval/*", step_metric="frl/round")
            wandb.define_metric("server/*", step_metric="frl/round")
            self._wandb_metrics_defined = True

        return run

    def _wandb_log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self.log_wandb:
            return

        run = self._ensure_wandb_run()
        if run is None:
            return

        clean_metrics: dict[str, Any] = {}
        for key, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                continue
            clean_metrics[key] = value

        if clean_metrics:
            run.log(clean_metrics, step=step)

    def _create_envs_with_kwargs(
        self,
        n_envs: int,
        env_kwargs_override: dict[str, Any] | None = None,
        *,
        eval_env: bool = False,
        no_log: bool = False,
    ) -> VecEnv:
        attr_name = "eval_env_kwargs" if eval_env else "env_kwargs"
        original_kwargs = deepcopy(getattr(self, attr_name))
        try:
            if env_kwargs_override is not None:
                setattr(self, attr_name, deepcopy(env_kwargs_override))
            return self.create_envs(n_envs=n_envs, eval_env=eval_env, no_log=no_log)
        finally:
            setattr(self, attr_name, original_kwargs)

    def _get_client_n_envs(self) -> int:
        algo_cls = self._get_federated_algo_class()
        if algo_cls.uses_federated_client_n_envs():
            return max(1, int(self.n_envs))
        return 1

    def _build_model(
        self,
        hyperparams: dict[str, Any],
        client_idx: int | None = None,
        env_kwargs_override: dict[str, Any] | None = None,
        *,
        n_envs: int = 1,
    ) -> BaseAlgorithm:
        algo_cls = self._get_federated_algo_class()
        seed = self.seed if client_idx is None else self.seed + client_idx
        tensorboard_log = self.tensorboard_log if client_idx is None else None
        env = self._create_envs_with_kwargs(n_envs, env_kwargs_override, no_log=True)
        return algo_cls(
            env=env,
            tensorboard_log=tensorboard_log,
            seed=seed,
            verbose=self.verbose,
            device=self.device,
            **deepcopy(hyperparams),
        )

    def _init_client_from_server(self, client: FederatedAlgorithmMixin, server: FederatedAlgorithmMixin) -> None:
        client.apply_global_payload(server.get_broadcast_payload(), mix_weight=1.0)

    def _set_model_total_timesteps_hint(self, model: BaseAlgorithm, total_timesteps: int) -> None:
        if hasattr(model, "_total_timesteps"):
            model._total_timesteps = max(int(total_timesteps), 1)

    def _set_model_progress_hint(self, model: BaseAlgorithm, num_timesteps: int, total_timesteps: int) -> None:
        if hasattr(model, "_update_current_progress_remaining"):
            model._update_current_progress_remaining(int(num_timesteps), max(int(total_timesteps), 1))

    def _sample_client_noises(self) -> list[float]:
        if self.perturb_noise_type is None or self.perturb_noise_range <= 0.0:
            return [0.0 for _ in range(self.num_clients)]

        if not (0.0 < self.perturb_noise_range < 1.0):
            raise ValueError("perturb_noise_range must be in (0, 1)")

        rng = np.random.default_rng(self.seed)
        samples = rng.uniform(
            low=-self.perturb_noise_range,
            high=self.perturb_noise_range,
            size=self.num_clients,
        )
        return [float(sample) for sample in samples]

    def _build_client_env_kwargs(self, client_noise: float) -> dict[str, Any]:
        env_kwargs = deepcopy(self.env_kwargs)
        if self.perturb_noise_type is not None:
            env_kwargs["noise_type"] = self.perturb_noise_type
            env_kwargs["noise"] = client_noise
        return env_kwargs

    def _build_nominal_env_kwargs(self) -> dict[str, Any]:
        base_kwargs = deepcopy(self.eval_env_kwargs if len(self.eval_env_kwargs) > 0 else self.env_kwargs)
        base_kwargs["noise_type"] = None
        base_kwargs["noise"] = 0.0
        return base_kwargs

    @staticmethod
    def _sync_eval_env_normalization(model: BaseAlgorithm, eval_env: VecEnv) -> None:
        """Copy VecNormalize statistics from the training env to the eval env.

        Stable-Baselines3 stores VecNormalize running statistics in the
        environment wrapper, not in the policy weights. Because PPOAvg creates
        separate eval envs, their normalization statistics start fresh unless we
        explicitly copy the trained statistics before evaluation.
        """
        train_env = model.get_env()
        if train_env is None or model.get_vec_normalize_env() is None:
            return

        try:
            sync_envs_normalization(train_env, eval_env)
        except AttributeError:
            # If eval_env is not wrapped with VecNormalize, leave evaluation as-is.
            return

        # Evaluation must use fixed normalization statistics.
        # Do not let eval episodes update running mean/std or normalize rewards.
        if hasattr(eval_env, "training"):
            eval_env.training = False
        if hasattr(eval_env, "norm_reward"):
            eval_env.norm_reward = False

    def _setup_eval_envs(self) -> None:
        self.local_eval_envs = []
        if self.eval_local_episodes > 0:
            for client_env_kwargs in self.client_env_kwargs:
                self.local_eval_envs.append(
                    self._create_envs_with_kwargs(1, client_env_kwargs, eval_env=True, no_log=True)
                )

        self.nominal_eval_env = None
        if self.eval_nominal_episodes > 0:
            self.nominal_eval_env = self._create_envs_with_kwargs(
                1, self._build_nominal_env_kwargs(), eval_env=True, no_log=True
            )

    def _init_eval_history(self) -> None:
        self.eval_npz_path = os.path.join(self.save_path, "evaluations.npz")
        self.eval_history = {
            "rounds": [],
            "client_noises": np.asarray(self.client_noise_values, dtype=np.float32),
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
            "num_clients": int(self.num_clients),
            "local_steps": int(self.local_steps),
            "eval_local_episodes": int(self.eval_local_episodes),
            "eval_nominal_episodes": int(self.eval_nominal_episodes),
            "eval_round_freq": int(self.eval_round_freq),
            "eval_deterministic": bool(self.eval_deterministic),
            "perturb_noise_type": "" if self.perturb_noise_type is None else str(self.perturb_noise_type),
            "perturb_noise_range": float(self.perturb_noise_range),
        }

    def _save_perturbation_info(self) -> None:
        self.perturbation_info_path = os.path.join(self.save_path, "perturbations.json")
        payload = {
            "perturb_noise_type": self.perturb_noise_type,
            "perturb_noise_range": float(self.perturb_noise_range),
            "num_clients": int(self.num_clients),
            "clients": [
                {
                    "client_id": client_idx,
                    "noise_type": self.perturb_noise_type,
                    "noise": float(noise_value),
                    "env_kwargs": env_kwargs,
                }
                for client_idx, (noise_value, env_kwargs) in enumerate(
                    zip(self.client_noise_values, self.client_env_kwargs, strict=True)
                )
            ],
            "nominal_env_kwargs": self._build_nominal_env_kwargs(),
        }
        with open(self.perturbation_info_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2)

    def _record_eval_results(
        self,
        round_idx: int,
        local_results: list[tuple[float, float]],
        nominal_results: list[tuple[float, float]],
    ) -> None:
        self.eval_history["rounds"].append(int(round_idx))

        if local_results:
            local_means = np.asarray([mean for mean, _ in local_results], dtype=np.float32)
            local_stds = np.asarray([std for _, std in local_results], dtype=np.float32)
            self.eval_history["local_mean"].append(local_means)
            self.eval_history["local_std"].append(local_stds)
            self.eval_history["local_mean_across_clients"].append(float(np.mean(local_means)))
            self.eval_history["local_min_across_clients"].append(float(np.min(local_means)))
            self.eval_history["local_max_across_clients"].append(float(np.max(local_means)))
        else:
            self.eval_history["local_mean"].append(np.full((self.num_clients,), np.nan, dtype=np.float32))
            self.eval_history["local_std"].append(np.full((self.num_clients,), np.nan, dtype=np.float32))
            self.eval_history["local_mean_across_clients"].append(np.nan)
            self.eval_history["local_min_across_clients"].append(np.nan)
            self.eval_history["local_max_across_clients"].append(np.nan)

        if nominal_results:
            nominal_means = np.asarray([mean for mean, _ in nominal_results], dtype=np.float32)
            nominal_stds = np.asarray([std for _, std in nominal_results], dtype=np.float32)
            self.eval_history["nominal_mean"].append(nominal_means)
            self.eval_history["nominal_std"].append(nominal_stds)
            self.eval_history["nominal_mean_across_clients"].append(float(np.mean(nominal_means)))
            self.eval_history["nominal_min_across_clients"].append(float(np.min(nominal_means)))
            self.eval_history["nominal_max_across_clients"].append(float(np.max(nominal_means)))
        else:
            self.eval_history["nominal_mean"].append(np.full((self.num_clients,), np.nan, dtype=np.float32))
            self.eval_history["nominal_std"].append(np.full((self.num_clients,), np.nan, dtype=np.float32))
            self.eval_history["nominal_mean_across_clients"].append(np.nan)
            self.eval_history["nominal_min_across_clients"].append(np.nan)
            self.eval_history["nominal_max_across_clients"].append(np.nan)

    def _save_eval_npz(self) -> None:
        if self.eval_npz_path is None:
            return

        payload = {
            "rounds": np.asarray(self.eval_history["rounds"], dtype=np.int32),
            "client_noises": np.asarray(self.eval_history["client_noises"], dtype=np.float32),
            "local_mean": np.asarray(self.eval_history["local_mean"], dtype=np.float32),
            "local_std": np.asarray(self.eval_history["local_std"], dtype=np.float32),
            "nominal_mean": np.asarray(self.eval_history["nominal_mean"], dtype=np.float32),
            "nominal_std": np.asarray(self.eval_history["nominal_std"], dtype=np.float32),
            "local_mean_across_clients": np.asarray(self.eval_history["local_mean_across_clients"], dtype=np.float32),
            "local_min_across_clients": np.asarray(self.eval_history["local_min_across_clients"], dtype=np.float32),
            "local_max_across_clients": np.asarray(self.eval_history["local_max_across_clients"], dtype=np.float32),
            "nominal_mean_across_clients": np.asarray(
                self.eval_history["nominal_mean_across_clients"], dtype=np.float32
            ),
            "nominal_min_across_clients": np.asarray(
                self.eval_history["nominal_min_across_clients"], dtype=np.float32
            ),
            "nominal_max_across_clients": np.asarray(
                self.eval_history["nominal_max_across_clients"], dtype=np.float32
            ),
            "num_clients": np.asarray(self.eval_history["num_clients"], dtype=np.int32),
            "local_steps": np.asarray(self.eval_history["local_steps"], dtype=np.int32),
            "eval_local_episodes": np.asarray(self.eval_history["eval_local_episodes"], dtype=np.int32),
            "eval_nominal_episodes": np.asarray(self.eval_history["eval_nominal_episodes"], dtype=np.int32),
            "eval_round_freq": np.asarray(self.eval_history["eval_round_freq"], dtype=np.int32),
            "eval_deterministic": np.asarray(self.eval_history["eval_deterministic"], dtype=np.bool_),
            "perturb_noise_type": np.asarray(self.eval_history["perturb_noise_type"]),
            "perturb_noise_range": np.asarray(self.eval_history["perturb_noise_range"], dtype=np.float32),
        }
        np.savez(self.eval_npz_path, **payload)

    def setup_experiment(self) -> tuple[BaseAlgorithm, dict[str, Any]] | None:
        unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
        hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(
            unprocessed_hyperparams
        )

        self.create_log_folder()

        probe_env = self.create_envs(1, no_log=True)
        hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, probe_env)
        probe_env.close()

        if "num_clients" in hyperparams:
            self.num_clients = int(hyperparams.pop("num_clients"))
        if "local_steps" in hyperparams:
            self.local_steps = int(hyperparams.pop("local_steps"))
        if "server_update_weight" in hyperparams:
            self.server_update_weight = float(hyperparams.pop("server_update_weight"))

        configured_client_noise_values: list[float] | None = None
        if "perturb_noise_type" in hyperparams:
            self.perturb_noise_type = self._normalize_perturb_noise_type(hyperparams.pop("perturb_noise_type"))
        if "perturb_noise_range" in hyperparams:
            self.perturb_noise_range = float(hyperparams.pop("perturb_noise_range"))
        if "client_noise_values" in hyperparams:
            configured_client_noise_values = self._parse_client_noise_values(
                hyperparams.pop("client_noise_values")
            )

        if "eval_local_episodes" in hyperparams:
            self.eval_local_episodes = int(hyperparams.pop("eval_local_episodes"))
        if "eval_nominal_episodes" in hyperparams:
            self.eval_nominal_episodes = int(hyperparams.pop("eval_nominal_episodes"))
        if "eval_round_freq" in hyperparams:
            self.eval_round_freq = int(hyperparams.pop("eval_round_freq"))
        if "eval_deterministic" in hyperparams:
            self.eval_deterministic = bool(hyperparams.pop("eval_deterministic"))
        if "log_wandb" in hyperparams:
            self.log_wandb = self._as_bool(hyperparams.pop("log_wandb"))

        if self.num_clients < 1:
            raise ValueError("num_clients must be >= 1")
        if self.local_steps < 1:
            raise ValueError("local_steps must be >= 1")
        if not (0.0 < self.server_update_weight <= 1.0):
            raise ValueError("server_update_weight must be in (0, 1].")
        if self.eval_round_freq < 1:
            raise ValueError("eval_round_freq must be >= 1")
        if not (0.0 <= self.perturb_noise_range < 1.0):
            raise ValueError("perturb_noise_range must be in [0, 1)")
        if configured_client_noise_values is not None:
            if self.perturb_noise_type is None:
                raise ValueError(
                    "perturb_noise_type must be set when client_noise_values is provided"
                )
            if len(configured_client_noise_values) != self.num_clients:
                raise ValueError(
                    "client_noise_values length must match num_clients: "
                    f"got {len(configured_client_noise_values)} values for "
                    f"{self.num_clients} clients"
                )

        algo_cls = self._get_federated_algo_class()
        if hasattr(algo_cls, "reset_federated_state"):
            algo_cls.reset_federated_state()

        self._ensure_wandb_run(saved_hyperparams)

        self._base_hyperparams = deepcopy(hyperparams)
        client_n_envs = self._get_client_n_envs()
        server_n_envs = client_n_envs
        if configured_client_noise_values is not None:
            self.client_noise_values = configured_client_noise_values
            noise_assignment = "explicit"
        else:
            self.client_noise_values = self._sample_client_noises()
            noise_assignment = "sampled"
        self.client_env_kwargs = [self._build_client_env_kwargs(noise) for noise in self.client_noise_values]

        if self.verbose > 0 and client_n_envs > 1:
            print(f"[FRL] using n_envs={client_n_envs} for each local client")

        if self.continue_training:
            server_env = self._create_envs_with_kwargs(server_n_envs, self._build_nominal_env_kwargs(), no_log=True)
            self.server_model = self._load_pretrained_agent(self._base_hyperparams, server_env)
            if not isinstance(self.server_model, FederatedAlgorithmMixin):
                raise ValueError(f"Loaded model for '{self.algo}' is not a federated algorithm.")
        else:
            self.server_model = self._build_model(
                self._base_hyperparams,
                env_kwargs_override=self._build_nominal_env_kwargs(),
                n_envs=server_n_envs,
            )
        self._set_model_total_timesteps_hint(self.server_model, int(self.n_timesteps))

        server_algo = cast(FederatedAlgorithmMixin, self.server_model)
        self.clients = []
        for client_idx, client_env_kwargs in enumerate(self.client_env_kwargs):
            client_model = self._build_model(
                self._base_hyperparams,
                client_idx=client_idx,
                env_kwargs_override=client_env_kwargs,
                n_envs=client_n_envs,
            )
            self._set_model_total_timesteps_hint(client_model, int(self.n_timesteps))
            client_algo = cast(FederatedAlgorithmMixin, client_model)
            self._init_client_from_server(client_algo, server_algo)
            self.clients.append(client_model)

        client_algos = [cast(FederatedAlgorithmMixin, client) for client in self.clients]
        server_algo.prepare_federated_training(client_algos)

        self._setup_eval_envs()
        self._init_eval_history()
        self._save_perturbation_info()

        initial_metrics: dict[str, Any] = {
            "frl/round": 0,
            "frl/num_clients": int(self.num_clients),
            "frl/local_steps": int(self.local_steps),
            "frl/server_update_weight": float(self.server_update_weight),
            "frl/perturb_noise_range": float(self.perturb_noise_range),
        }
        for client_idx, noise_value in enumerate(self.client_noise_values):
            initial_metrics[f"frl/client_{client_idx}/noise"] = float(noise_value)
        self._wandb_log(initial_metrics, step=0)

        if self.verbose > 0 and self.perturb_noise_type is not None:
            print(
                f"[FRL] client perturbation type={self.perturb_noise_type} "
                f"assignment={noise_assignment} range={self.perturb_noise_range} "
                f"values={self.client_noise_values}"
            )

        self._save_config(saved_hyperparams)
        return self.server_model, saved_hyperparams

    def _evaluate_clients(self, model: BaseAlgorithm, round_idx: int) -> None:
        local_results: list[tuple[float, float]] = []
        nominal_results: list[tuple[float, float]] = []

        for client_idx, client_model in enumerate(self.clients):
            if self.eval_local_episodes > 0:
                local_eval_env = self.local_eval_envs[client_idx]
                self._sync_eval_env_normalization(client_model, local_eval_env)
                local_mean, local_std = evaluate_policy(
                    client_model,
                    local_eval_env,
                    n_eval_episodes=self.eval_local_episodes,
                    deterministic=self.eval_deterministic,
                    render=False,
                    warn=False,
                )
                local_results.append((float(local_mean), float(local_std)))

            if self.eval_nominal_episodes > 0 and self.nominal_eval_env is not None:
                self._sync_eval_env_normalization(client_model, self.nominal_eval_env)
                nominal_mean, nominal_std = evaluate_policy(
                    client_model,
                    self.nominal_eval_env,
                    n_eval_episodes=self.eval_nominal_episodes,
                    deterministic=self.eval_deterministic,
                    render=False,
                    warn=False,
                )
                nominal_results.append((float(nominal_mean), float(nominal_std)))

        local_means = [mean for mean, _ in local_results]
        nominal_means = [mean for mean, _ in nominal_results]

        self._record_eval_results(round_idx, local_results, nominal_results)
        self._save_eval_npz()

        if self.verbose > 0:
            summary = [f"[FRL][Eval] round={round_idx}"]
            if local_means:
                summary.append(f"local_mean={float(np.mean(local_means)):.3f}")
                summary.append(f"local_min={float(np.min(local_means)):.3f}")
            if nominal_means:
                summary.append(f"nominal_mean={float(np.mean(nominal_means)):.3f}")
                summary.append(f"nominal_min={float(np.min(nominal_means)):.3f}")
            print(" ".join(summary))

        metrics: dict[str, Any] = {"frl/round": int(round_idx)}
        if local_means:
            local_array = np.asarray(local_means, dtype=np.float64)
            metrics.update(
                {
                    "eval/local_mean_across_clients": float(np.mean(local_array)),
                    "eval/local_min_across_clients": float(np.min(local_array)),
                    "eval/local_max_across_clients": float(np.max(local_array)),
                    "eval/local_std_across_clients": float(np.std(local_array)),
                }
            )
            for client_idx, (mean, std) in enumerate(local_results):
                metrics[f"eval/client_{client_idx}/local_mean"] = float(mean)
                metrics[f"eval/client_{client_idx}/local_std"] = float(std)
        if nominal_means:
            nominal_array = np.asarray(nominal_means, dtype=np.float64)
            metrics.update(
                {
                    "eval/nominal_mean_across_clients": float(np.mean(nominal_array)),
                    "eval/nominal_min_across_clients": float(np.min(nominal_array)),
                    "eval/nominal_max_across_clients": float(np.max(nominal_array)),
                    "eval/nominal_std_across_clients": float(np.std(nominal_array)),
                }
            )
            for client_idx, (mean, std) in enumerate(nominal_results):
                metrics[f"eval/client_{client_idx}/nominal_mean"] = float(mean)
                metrics[f"eval/client_{client_idx}/nominal_std"] = float(std)

        self._wandb_log(metrics, step=int(getattr(model, "num_timesteps", round_idx)))

    def learn(self, model: BaseAlgorithm) -> None:
        if not isinstance(model, FederatedAlgorithmMixin):
            raise ValueError(f"Model '{type(model).__name__}' does not implement the federated interface.")

        remaining_timesteps = int(self.n_timesteps)
        round_idx = 0
        aggregated_timesteps = 0
        eval_enabled = self.eval_local_episodes > 0 or self.eval_nominal_episodes > 0
        last_eval_round = 0

        try:
            while remaining_timesteps > 0:
                round_idx += 1
                self._set_model_progress_hint(model, aggregated_timesteps, int(self.n_timesteps))
                for client_model in self.clients:
                    self._set_model_progress_hint(client_model, aggregated_timesteps, int(self.n_timesteps))
                current_local_steps = cast(
                    FederatedAlgorithmMixin,
                    model,
                ).resolve_federated_local_steps(
                    configured_local_steps=self.local_steps,
                    remaining_timesteps=remaining_timesteps,
                    num_clients=self.num_clients,
                )
                if current_local_steps <= 0:
                    raise ValueError(f"Resolved local_steps must be positive, got {current_local_steps}")

                if self.verbose > 0:
                    print(
                        f"[FRL] round={round_idx} local_steps={current_local_steps} "
                        f"clients={self.num_clients} remaining={remaining_timesteps}"
                    )

                round_timesteps = 0
                for client_idx, client_model in enumerate(self.clients):
                    client_algo = cast(FederatedAlgorithmMixin, client_model)
                    if self.verbose > 1:
                        client_noise = (
                            self.client_noise_values[client_idx] if client_idx < len(self.client_noise_values) else 0.0
                        )
                        print(f"[FRL] local update client={client_idx} noise={client_noise:+.4f}")
                    before_timesteps = int(client_model.num_timesteps)
                    client_algo.federated_local_update(
                        current_local_steps,
                        callback=None,
                        log_interval=self.log_interval if self.log_interval > -1 else 4,
                        tb_log_name=f"{self.algo}_client_{client_idx}",
                        reset_num_timesteps=False,
                        progress_bar=False,
                    )
                    round_timesteps += max(0, int(client_model.num_timesteps) - before_timesteps)

                client_algos = [cast(FederatedAlgorithmMixin, client) for client in self.clients]
                uploads = [client.get_upload_payload() for client in client_algos]
                weights = [client.get_client_weight() for client in client_algos]
                global_payload = type(model).aggregate_uploads(uploads, weights=weights)
                self._set_model_progress_hint(model, aggregated_timesteps + round_timesteps, int(self.n_timesteps))
                model.num_timesteps = aggregated_timesteps + round_timesteps

                model.apply_global_payload(global_payload, mix_weight=self.server_update_weight)
                broadcast_payload = cast(FederatedAlgorithmMixin, model).get_broadcast_payload()
                for client in client_algos:
                    client.apply_global_payload(broadcast_payload, mix_weight=1.0)

                aggregated_timesteps += round_timesteps
                remaining_timesteps -= round_timesteps
                model.num_timesteps = aggregated_timesteps

                round_metrics: dict[str, Any] = {
                    "frl/round": int(round_idx),
                    "frl/local_steps": int(current_local_steps),
                    "frl/round_timesteps": int(round_timesteps),
                    "frl/aggregated_timesteps": int(aggregated_timesteps),
                    "frl/remaining_timesteps": int(max(remaining_timesteps, 0)),
                    "frl/num_clients": int(self.num_clients),
                    "frl/server_update_weight": float(self.server_update_weight),
                }
                for client_idx, noise_value in enumerate(self.client_noise_values):
                    round_metrics[f"frl/client_{client_idx}/noise"] = float(noise_value)
                round_metrics.update(getattr(model, "_last_federated_metrics", {}))
                self._wandb_log(round_metrics, step=int(aggregated_timesteps))

                do_eval = eval_enabled and round_idx % self.eval_round_freq == 0
                if do_eval:
                    self._evaluate_clients(model, round_idx)
                    last_eval_round = round_idx

            if eval_enabled and round_idx > 0 and last_eval_round != round_idx:
                if self.verbose > 0:
                    print(f"[FRL] final evaluation at round={round_idx}")
                self._evaluate_clients(model, round_idx)

            if self.verbose > 0:
                print(f"[FRL] finished after {round_idx} rounds, aggregated_timesteps={aggregated_timesteps}")
        finally:
            self.close()

    def close(self) -> None:
        if self.server_model is not None and self.server_model.env is not None:
            self.server_model.env.close()
        for client in self.clients:
            if client.env is not None:
                client.env.close()
        for eval_env in self.local_eval_envs:
            eval_env.close()
        if self.nominal_eval_env is not None:
            self.nominal_eval_env.close()
