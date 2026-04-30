from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import torch as th
from stable_baselines3.td3 import TD3

from rl_zoo3.federate.buff.buff.federated_algorithm import FederatedAlgorithmMixin, FederatedModules, FederatedPayload


class TD3Avg(FederatedAlgorithmMixin, TD3):
    """TD3 with a FedAvg-style synchronization rule.

    - local client optimization is delegated to SB3's original TD3 implementation,
    - clients upload actor/critic/target network parameters,
    - the server averages those parameters,
    - the averaged parameters are pushed back to all clients.
    """

    federated_modules: tuple[str, ...] = ("actor", "critic", "actor_target", "critic_target")
    federated_manager_keys: tuple[str, ...] = ("num_clients", "local_steps", "server_update_weight")

    def __init__(self, *args, **kwargs):
        # Accept generic federated-manager kwargs so they can live in the YAML
        # config without leaking into SB3's TD3 constructor.
        for key in self.federated_manager_keys:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        self.learn(total_timesteps=local_steps, **kwargs)

    def _get_module_states(self) -> FederatedModules:
        module_states: FederatedModules = {}
        for module_name in self.federated_modules:
            module = getattr(self, module_name)
            module_states[module_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
            )
        return module_states

    def _set_module_states(self, module_states: FederatedModules) -> None:
        device = self.device
        for module_name in self.federated_modules:
            module = getattr(self, module_name)
            state_dict = OrderedDict((key, value.to(device)) for key, value in module_states[module_name].items())
            module.load_state_dict(state_dict)
        self._create_aliases()

    def get_upload_payload(self) -> FederatedPayload:
        return {
            "modules": self._get_module_states(),
            "meta": {
                "client_weight": self.get_client_weight(),
                "num_timesteps": int(self.num_timesteps),
            },
        }

    @classmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        if len(uploads) == 0:
            raise ValueError("At least one upload is required for federated aggregation.")

        module_states = [upload["modules"] for upload in uploads]
        aggregated_modules = cls.average_module_states(module_states, weights=weights)
        return {
            "modules": aggregated_modules,
            "meta": {
                "num_clients": len(uploads),
            },
        }

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        incoming_modules = payload["modules"]
        if mix_weight < 1.0:
            current_modules = self._get_module_states()
            mixed_modules: FederatedModules = {}
            for module_name, module_state in incoming_modules.items():
                mixed_modules[module_name] = OrderedDict()
                for key, value in module_state.items():
                    current_value = current_modules[module_name][key]
                    if th.is_floating_point(value):
                        mixed_modules[module_name][key] = (
                            mix_weight * value + (1.0 - mix_weight) * current_value.to(value.dtype)
                        )
                    else:
                        mixed_modules[module_name][key] = value.clone()
            incoming_modules = mixed_modules

        self._set_module_states(incoming_modules)

    def get_client_weight(self) -> float:
        return 1.0
