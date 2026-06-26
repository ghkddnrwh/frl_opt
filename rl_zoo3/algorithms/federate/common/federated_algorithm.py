from __future__ import annotations

from abc import ABC, abstractmethod
from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch as th

FederatedModuleState = OrderedDict[str, th.Tensor]
FederatedModules = dict[str, FederatedModuleState]
FederatedPayload = dict[str, Any]


class FederatedAlgorithmMixin(ABC):
    """Common interface for federated RL algorithms.

    The :class:`FederatedExperimentManager` only orchestrates the federated loop:
    local update -> client upload -> server aggregation -> broadcast.
    Algorithm-specific details such as which parameters are uploaded,
    how aggregation is performed, and how the global payload is applied
    are implemented by each federated algorithm.
    """

    @abstractmethod
    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        """Run one local client update step."""

    @abstractmethod
    def get_upload_payload(self) -> FederatedPayload:
        """Return the client payload that will be sent to the server."""

    def get_broadcast_payload(self) -> FederatedPayload:
        """Return the server payload that will be broadcast to clients.

        Algorithms with asymmetric upload/broadcast payloads can override this.
        """
        return self.get_upload_payload()

    @classmethod
    @abstractmethod
    def aggregate_uploads(
        cls,
        uploads: Sequence[FederatedPayload],
        weights: Sequence[float] | None = None,
    ) -> FederatedPayload:
        """Aggregate client uploads into a single global payload."""

    @abstractmethod
    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        """Apply a server payload to the local model."""

    def get_client_weight(self) -> float:
        """Return the client aggregation weight used by the server."""
        return 1.0

    def prepare_federated_training(self, clients: Sequence["FederatedAlgorithmMixin"]) -> None:
        """Optional hook executed once after all clients are initialized."""
        del clients

    @classmethod
    def uses_federated_client_n_envs(cls) -> bool:
        """Whether FRL clients should honor the experiment ``n_envs`` setting.

        Most federated algorithms in this repo have historically created one
        environment per client. PPO-style methods can optionally opt in so that
        each client uses its own vectorized env and ``n_envs`` keeps its usual
        SB3 meaning inside the local learner.
        """
        return False

    def resolve_federated_local_steps(
        self,
        configured_local_steps: int,
        remaining_timesteps: int,
        num_clients: int,
    ) -> int:
        """Optional hook for algorithm-specific local-step scheduling."""
        del remaining_timesteps, num_clients
        return int(configured_local_steps)

    @staticmethod
    def normalize_weights(num_uploads: int, weights: Sequence[float] | None = None) -> list[float]:
        if num_uploads <= 0:
            raise ValueError("num_uploads must be strictly positive.")

        if weights is None:
            return [1.0 / num_uploads] * num_uploads
        if len(weights) != num_uploads:
            raise ValueError("The number of weights must match the number of uploads.")

        weight_sum = float(sum(weights))
        if weight_sum <= 0:
            raise ValueError("The sum of client weights must be strictly positive.")
        return [float(weight) / weight_sum for weight in weights]

    @classmethod
    def average_module_states(
        cls,
        states: Sequence[FederatedModules],
        weights: Sequence[float] | None = None,
    ) -> FederatedModules:
        """Average a list of module ``state_dict`` objects.

        This helper performs weighted averaging for floating tensors and copies
        non-floating tensors from the reference state.
        """
        if len(states) == 0:
            raise ValueError("At least one state is required for federated averaging.")

        normalized_weights = cls.normalize_weights(len(states), weights)
        reference_state = states[0]
        averaged_state: FederatedModules = {}

        for module_name, module_state in reference_state.items():
            averaged_state[module_name] = OrderedDict()
            for key, value in module_state.items():
                if th.is_floating_point(value):
                    avg_tensor = th.zeros_like(value, dtype=value.dtype)
                    for client_state, weight in zip(states, normalized_weights, strict=True):
                        avg_tensor = avg_tensor + client_state[module_name][key].to(value.dtype) * weight
                    averaged_state[module_name][key] = avg_tensor
                else:
                    averaged_state[module_name][key] = value.clone()

        return averaged_state

    @staticmethod
    def clone_module_states(states: Mapping[str, Mapping[str, th.Tensor]]) -> FederatedModules:
        cloned: FederatedModules = {}
        for module_name, module_state in states.items():
            cloned[module_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module_state.items()
            )
        return cloned
