from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping, Sequence
from typing import Any

import torch as th

from rl_zoo3.algorithms.federate.common.federated_algorithm import (
    FederatedAlgorithmMixin,
    FederatedModules,
    FederatedPayload,
)


class OffPolicyFedAvgAlgorithm(FederatedAlgorithmMixin):
    """Reusable FedAvg implementation for SB3 off-policy algorithms.

    Child classes only need to define ``federated_modules`` and optionally
    override the extra-state hooks when an algorithm has non-module trainable
    tensors, e.g. SAC's entropy coefficient.
    """

    federated_modules: tuple[str, ...] = ()
    federated_manager_keys: tuple[str, ...] = (
        "num_clients",
        "local_steps",
        "server_update_weight",
    )

    def __init__(self, *args, **kwargs):
        self.reset_optimizer_on_broadcast = self._as_bool(
            kwargs.pop(
                "reset_optimizer_on_broadcast",
                kwargs.pop("reset_optimizer", False),
            )
        )

        self._consume_federated_kwargs(kwargs)
        for key in self.federated_manager_keys:
            kwargs.pop(key, None)
        super().__init__(*args, **kwargs)

    def _reset_optimizer_state(self, optimizer: Any) -> None:
        if optimizer is None:
            return
        if hasattr(optimizer, "state"):
            optimizer.state.clear()

    def _reset_federated_optimizer_states(self) -> None:
        seen_optimizer_ids: set[int] = set()

        def reset_once(optimizer: Any) -> None:
            if optimizer is None:
                return
            optimizer_id = id(optimizer)
            if optimizer_id in seen_optimizer_ids:
                return
            seen_optimizer_ids.add(optimizer_id)
            self._reset_optimizer_state(optimizer)

        for module_name in self.federated_modules:
            module = getattr(self, module_name, None)
            reset_once(getattr(module, "optimizer", None))

        # SAC / AR-SAC entropy coefficient optimizer.
        reset_once(getattr(self, "ent_coef_optimizer", None))
        
    def _consume_federated_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Consume child-specific FRL kwargs before SB3 sees them."""
        del kwargs

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

    def federated_local_update(self, local_steps: int, **kwargs) -> None:
        self.learn(total_timesteps=local_steps, **kwargs)

    def _get_module_states(self) -> FederatedModules:
        if len(self.federated_modules) == 0:
            raise ValueError(f"{type(self).__name__}.federated_modules must not be empty.")

        module_states: FederatedModules = {}
        for module_name in self.federated_modules:
            module = getattr(self, module_name)
            module_states[module_name] = OrderedDict(
                (key, value.detach().cpu().clone()) for key, value in module.state_dict().items()
            )
        return module_states

    def _set_module_states(self, module_states: FederatedModules) -> None:
        device = self.device
        missing_modules = set(self.federated_modules) - set(module_states)
        if missing_modules:
            raise KeyError(f"Missing modules in federated payload: {sorted(missing_modules)}")

        for module_name in self.federated_modules:
            module = getattr(self, module_name)
            state_dict = OrderedDict((key, value.to(device)) for key, value in module_states[module_name].items())
            module.load_state_dict(state_dict)

    def _get_extra_federated_states(self) -> FederatedModules:
        return {}

    def _set_extra_federated_states(self, extra_states: FederatedModules) -> None:
        if extra_states:
            raise KeyError(f"Unknown extra federated states: {sorted(extra_states)}")

    def _after_set_federated_states(self) -> None:
        if hasattr(self, "_create_aliases"):
            self._create_aliases()

    def _get_federated_states(self) -> FederatedModules:
        states = self._get_module_states()
        extra_states = self._get_extra_federated_states()
        overlap = set(states) & set(extra_states)
        if overlap:
            raise KeyError(f"Extra federated states overlap with module states: {sorted(overlap)}")
        states.update(extra_states)
        return states

    def _set_federated_states(self, states: FederatedModules) -> None:
        module_names = set(self.federated_modules)
        unknown_required_modules = module_names - set(states)
        if unknown_required_modules:
            raise KeyError(f"Missing modules in federated payload: {sorted(unknown_required_modules)}")

        unknown_non_extra_states = set(states) - module_names - set(self._get_extra_state_names())
        if unknown_non_extra_states:
            raise KeyError(f"Unknown states in federated payload: {sorted(unknown_non_extra_states)}")

        module_states: FederatedModules = {
            module_name: states[module_name]
            for module_name in self.federated_modules
        }
        extra_states: FederatedModules = {
            state_name: state
            for state_name, state in states.items()
            if state_name not in module_names
        }

        self._set_module_states(module_states)
        self._set_extra_federated_states(extra_states)
        self._after_set_federated_states()

        if self.reset_optimizer_on_broadcast:
            self._reset_federated_optimizer_states()

    def _get_extra_state_names(self) -> tuple[str, ...]:
        return ()

    def get_upload_payload(self) -> FederatedPayload:
        return {
            "modules": self._get_federated_states(),
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

    @staticmethod
    def _mix_federated_states(
        current_states: Mapping[str, Mapping[str, th.Tensor]],
        incoming_states: Mapping[str, Mapping[str, th.Tensor]],
        mix_weight: float,
    ) -> FederatedModules:
        mixed_states: FederatedModules = {}
        for state_name, incoming_state in incoming_states.items():
            if state_name not in current_states:
                raise KeyError(f"Current model is missing federated state {state_name!r}.")
            mixed_states[state_name] = OrderedDict()
            for key, incoming_value in incoming_state.items():
                if key not in current_states[state_name]:
                    raise KeyError(f"Current model is missing key {state_name}.{key}.")
                current_value = current_states[state_name][key]
                if th.is_floating_point(incoming_value):
                    mixed_states[state_name][key] = (
                        mix_weight * incoming_value
                        + (1.0 - mix_weight) * current_value.to(incoming_value.dtype)
                    )
                else:
                    mixed_states[state_name][key] = incoming_value.clone()
        return mixed_states

    def apply_global_payload(self, payload: FederatedPayload, mix_weight: float = 1.0) -> None:
        if not (0.0 < mix_weight <= 1.0):
            raise ValueError("mix_weight must be in (0, 1].")

        incoming_states = payload["modules"]
        if mix_weight < 1.0:
            incoming_states = self._mix_federated_states(
                current_states=self._get_federated_states(),
                incoming_states=incoming_states,
                mix_weight=mix_weight,
            )

        self._set_federated_states(incoming_states)
