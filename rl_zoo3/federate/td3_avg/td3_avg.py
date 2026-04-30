# from __future__ import annotations

# from collections import OrderedDict
# from collections.abc import Iterable, Sequence
# from typing import Any

# import torch as th
# from stable_baselines3.td3 import TD3


# class TD3Avg(TD3):
#     """
#     Minimal TD3 extension for federated parameter averaging.

#     The local optimization logic is still the original SB3 TD3 implementation.
#     This class only adds utilities for extracting, averaging and loading the
#     actor/critic/target parameters that are synchronized by the federated server.
#     """

#     federated_modules: tuple[str, ...] = ("actor", "critic", "actor_target", "critic_target")

#     def __init__(self, *args, **kwargs):
#         # Accept federated-only kwargs so rl_zoo hyperparams can contain them.
#         kwargs.pop("num_clients", None)
#         kwargs.pop("local_steps", None)
#         kwargs.pop("server_update_weight", None)
#         super().__init__(*args, **kwargs)

#     def get_federated_state(self) -> dict[str, OrderedDict[str, th.Tensor]]:
#         state: dict[str, OrderedDict[str, th.Tensor]] = {}
#         for module_name in self.federated_modules:
#             module = getattr(self, module_name)
#             state[module_name] = OrderedDict(
#                 (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
#             )
#         return state

#     def set_federated_state(self, state: dict[str, OrderedDict[str, th.Tensor]]) -> None:
#         device = self.device
#         for module_name in self.federated_modules:
#             module = getattr(self, module_name)
#             module_state = OrderedDict((key, value.to(device)) for key, value in state[module_name].items())
#             module.load_state_dict(module_state)
#         self._create_aliases()

#     def sync_from(self, other: "TD3Avg") -> None:
#         self.set_federated_state(other.get_federated_state())

#     @classmethod
#     def average_federated_states(
#         cls,
#         states: Sequence[dict[str, OrderedDict[str, th.Tensor]]],
#         weights: Sequence[float] | None = None,
#     ) -> dict[str, OrderedDict[str, th.Tensor]]:
#         if len(states) == 0:
#             raise ValueError("At least one client state is required for federated averaging.")

#         if weights is None:
#             weights = [1.0 / len(states)] * len(states)
#         elif len(weights) != len(states):
#             raise ValueError("The number of weights must match the number of client states.")
#         else:
#             weight_sum = float(sum(weights))
#             if weight_sum <= 0:
#                 raise ValueError("The sum of client weights must be strictly positive.")
#             weights = [float(weight) / weight_sum for weight in weights]

#         averaged_state: dict[str, OrderedDict[str, th.Tensor]] = {}
#         reference_state = states[0]

#         for module_name in cls.federated_modules:
#             averaged_state[module_name] = OrderedDict()
#             for key, value in reference_state[module_name].items():
#                 if th.is_floating_point(value):
#                     avg_tensor = th.zeros_like(value, dtype=value.dtype)
#                     for client_state, weight in zip(states, weights):
#                         avg_tensor = avg_tensor + client_state[module_name][key].to(value.dtype) * weight
#                     averaged_state[module_name][key] = avg_tensor
#                 else:
#                     averaged_state[module_name][key] = value.clone()

#         return averaged_state

#     @classmethod
#     def average_models(cls, models: Sequence["TD3Avg"], weights: Sequence[float] | None = None):
#         client_states = [model.get_federated_state() for model in models]
#         return cls.average_federated_states(client_states, weights=weights)

#     def aggregate_from(self, clients: Sequence["TD3Avg"], weights: Sequence[float] | None = None) -> None:
#         averaged_state = self.average_models(clients, weights=weights)
#         self.set_federated_state(averaged_state)

#     def broadcast_to(self, clients: Iterable["TD3Avg"]) -> None:
#         server_state = self.get_federated_state()
#         for client in clients:
#             client.set_federated_state(server_state)
