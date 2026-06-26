from __future__ import annotations

from collections import OrderedDict

import torch as th
from stable_baselines3.common.utils import get_parameters_by_name

from rl_zoo3.algorithms.federate.common.federated_algorithm import FederatedModules


class EntropyCoefFedAvgMixin:
    """FedAvg hook for SAC-style entropy coefficient state.

    This mixin is intentionally small: it only adds the non-module entropy
    coefficient to the payload. The actor/critic/protester modules are still
    handled by OffPolicyFedAvgAlgorithm.
    """

    federated_manager_keys: tuple[str, ...] = ()
    entropy_state_name = "entropy_coef"

    def _consume_federated_kwargs(self, kwargs: dict) -> None:
        super()._consume_federated_kwargs(kwargs)
        self.sync_entropy_coef = self._as_bool(kwargs.pop("sync_entropy_coef", True))

    def _get_extra_state_names(self) -> tuple[str, ...]:
        parent_extra_names = super()._get_extra_state_names()
        return (*parent_extra_names, self.entropy_state_name)

    def _get_extra_federated_states(self) -> FederatedModules:
        extra_states = super()._get_extra_federated_states()
        if not self.sync_entropy_coef:
            return extra_states

        entropy_state: OrderedDict[str, th.Tensor] = OrderedDict()
        if getattr(self, "log_ent_coef", None) is not None:
            entropy_state["log_ent_coef"] = self.log_ent_coef.detach().cpu().clone()
        elif hasattr(self, "ent_coef_tensor"):
            entropy_state["ent_coef_tensor"] = self.ent_coef_tensor.detach().cpu().clone()

        if len(entropy_state) > 0:
            extra_states[self.entropy_state_name] = entropy_state
        return extra_states

    def _set_extra_federated_states(self, extra_states: FederatedModules) -> None:
        if self.entropy_state_name not in extra_states:
            super()._set_extra_federated_states(extra_states)
            return

        if not self.sync_entropy_coef:
            raise KeyError(
                "Received entropy coefficient state while sync_entropy_coef=False. "
                "Make sure all SAC-style FedAvg clients use the same sync_entropy_coef setting."
            )

        entropy_state = extra_states.pop(self.entropy_state_name)
        device = self.device

        if "log_ent_coef" in entropy_state:
            if getattr(self, "log_ent_coef", None) is None:
                raise ValueError(
                    "Received a learnable entropy coefficient, but this instance uses fixed ent_coef."
                )
            with th.no_grad():
                self.log_ent_coef.copy_(entropy_state["log_ent_coef"].to(device))
        elif "ent_coef_tensor" in entropy_state:
            if not hasattr(self, "ent_coef_tensor"):
                raise ValueError(
                    "Received a fixed entropy coefficient, but this instance uses auto ent_coef."
                )
            self.ent_coef_tensor = entropy_state["ent_coef_tensor"].to(device)
        else:
            raise KeyError(f"Unknown entropy coefficient payload keys: {list(entropy_state.keys())}")

        super()._set_extra_federated_states(extra_states)

    def _after_set_federated_states(self) -> None:
        super()._after_set_federated_states()
        if hasattr(self, "critic") and hasattr(self, "critic_target"):
            self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
            self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
