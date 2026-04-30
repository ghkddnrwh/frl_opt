# from __future__ import annotations

# import math
# from copy import deepcopy
# from typing import Any

# from stable_baselines3.common.base_class import BaseAlgorithm

# from rl_zoo3.exp_manager import ExperimentManager
# from rl_zoo3.federate.td3_avg.td3_avg import TD3Avg


# class TD3AvgExperimentManager(ExperimentManager):
#     """
#     Federated training manager for TD3Avg.

#     Design:
#     - each client owns a local TD3Avg model and replay buffer,
#     - each client trains locally for ``local_steps``,
#     - the server averages actor/critic/target parameters,
#     - averaged parameters are broadcast back to all clients.

#     This is intentionally a minimal synchronous FedAvg-style implementation.
#     """

#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.server_model: TD3Avg | None = None
#         self.clients: list[TD3Avg] = []
#         self.client_weights: list[float] | None = None
#         self.num_clients = 1
#         self.local_steps = 1_000
#         self.server_update_weight = 1.0
#         self._base_hyperparams: dict[str, Any] = {}

#     def setup_experiment(self) -> tuple[BaseAlgorithm, dict[str, Any]] | None:
#         unprocessed_hyperparams, saved_hyperparams = self.read_hyperparameters()
#         hyperparams, self.env_wrapper, self.callbacks, self.vec_env_wrapper = self._preprocess_hyperparams(
#             unprocessed_hyperparams
#         )

#         self.create_log_folder()

#         # Build once to infer action-noise shapes.
#         probe_env = self.create_envs(1, no_log=True)
#         hyperparams = self._preprocess_action_noise(hyperparams, saved_hyperparams, probe_env)
#         probe_env.close()

#         self.num_clients = int(hyperparams.pop("num_clients", 4))
#         self.local_steps = int(hyperparams.pop("local_steps", 1_000))
#         self.server_update_weight = float(hyperparams.pop("server_update_weight", 1.0))

#         if self.num_clients < 1:
#             raise ValueError("num_clients must be >= 1")
#         if self.local_steps < 1:
#             raise ValueError("local_steps must be >= 1")
#         if not (0.0 < self.server_update_weight <= 1.0):
#             raise ValueError("server_update_weight must be in (0, 1].")

#         self.client_weights = [1.0 / self.num_clients] * self.num_clients
#         self._base_hyperparams = deepcopy(hyperparams)

#         self.server_model = TD3Avg(
#             env=self.create_envs(1, no_log=True),
#             tensorboard_log=self.tensorboard_log,
#             seed=self.seed,
#             verbose=self.verbose,
#             device=self.device,
#             **deepcopy(self._base_hyperparams),
#         )

#         self.clients = []
#         for client_idx in range(self.num_clients):
#             client_model = TD3Avg(
#                 env=self.create_envs(1, no_log=True),
#                 tensorboard_log=None,
#                 seed=self.seed + client_idx,
#                 verbose=self.verbose,
#                 device=self.device,
#                 **deepcopy(self._base_hyperparams),
#             )
#             client_model.sync_from(self.server_model)
#             self.clients.append(client_model)

#         self._save_config(saved_hyperparams)
#         return self.server_model, saved_hyperparams

#     def learn(self, model: BaseAlgorithm) -> None:
#         assert isinstance(model, TD3Avg)
#         assert self.server_model is not None

#         remaining_timesteps = int(self.n_timesteps)
#         round_idx = 0
#         aggregated_timesteps = 0

#         while remaining_timesteps > 0:
#             round_idx += 1
#             current_local_steps = min(self.local_steps, max(1, math.ceil(remaining_timesteps / self.num_clients)))

#             if self.verbose > 0:
#                 print(
#                     f"[TD3Avg] round={round_idx} local_steps={current_local_steps} "
#                     f"clients={self.num_clients} remaining={remaining_timesteps}"
#                 )

#             for client_idx, client_model in enumerate(self.clients):
#                 if self.verbose > 1:
#                     print(f"[TD3Avg] local update client={client_idx}")
#                 client_model.learn(
#                     total_timesteps=current_local_steps,
#                     callback=None,
#                     log_interval=self.log_interval if self.log_interval > -1 else 4,
#                     tb_log_name=f"td3_avg_client_{client_idx}",
#                     reset_num_timesteps=False,
#                     progress_bar=False,
#                 )

#             averaged_state = TD3Avg.average_models(self.clients, weights=self.client_weights)
#             if self.server_update_weight < 1.0:
#                 current_server_state = self.server_model.get_federated_state()
#                 mixed_state = {}
#                 for module_name, module_state in averaged_state.items():
#                     mixed_state[module_name] = module_state.__class__()
#                     for key, value in module_state.items():
#                         server_value = current_server_state[module_name][key]
#                         if value.dtype.is_floating_point:
#                             mixed_state[module_name][key] = (
#                                 self.server_update_weight * value
#                                 + (1.0 - self.server_update_weight) * server_value.to(value.dtype)
#                             )
#                         else:
#                             mixed_state[module_name][key] = value
#                 averaged_state = mixed_state

#             self.server_model.set_federated_state(averaged_state)
#             self.server_model.broadcast_to(self.clients)

#             aggregated_timesteps += current_local_steps * self.num_clients
#             remaining_timesteps -= current_local_steps * self.num_clients
#             self.server_model.num_timesteps = aggregated_timesteps

#         if self.verbose > 0:
#             print(f"[TD3Avg] finished after {round_idx} rounds, aggregated_timesteps={aggregated_timesteps}")

#     def save_trained_model(self, model: BaseAlgorithm) -> None:
#         super().save_trained_model(model)

#     def close(self) -> None:
#         if self.server_model is not None and self.server_model.env is not None:
#             self.server_model.env.close()
#         for client in self.clients:
#             if client.env is not None:
#                 client.env.close()
