from __future__ import annotations

from collections import OrderedDict

import torch as th
from stable_baselines3.common.utils import get_parameters_by_name
from stable_baselines3.sac import SAC

from rl_zoo3.algorithms.federate.common.federated_algorithm import FederatedModules
from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm


class SACAvg(OffPolicyFedAvgAlgorithm, SAC):
    """SAC with FedAvg synchronization."""

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "critic_target",
    )
    federated_manager_keys: tuple[str, ...] = (
        *OffPolicyFedAvgAlgorithm.federated_manager_keys,
        "sync_entropy_coef",
    )
    entropy_state_name = "entropy_coef"

    def _consume_federated_kwargs(self, kwargs: dict) -> None:
        self.sync_entropy_coef = self._as_bool(kwargs.pop("sync_entropy_coef", True))

    def _get_extra_state_names(self) -> tuple[str, ...]:
        return (self.entropy_state_name,)

    def _get_extra_federated_states(self) -> FederatedModules:
        if not self.sync_entropy_coef:
            return {}

        entropy_state: OrderedDict[str, th.Tensor] = OrderedDict()
        if self.log_ent_coef is not None:
            entropy_state["log_ent_coef"] = self.log_ent_coef.detach().cpu().clone()
        elif hasattr(self, "ent_coef_tensor"):
            entropy_state["ent_coef_tensor"] = self.ent_coef_tensor.detach().cpu().clone()

        if len(entropy_state) == 0:
            return {}
        return {self.entropy_state_name: entropy_state}

    def _set_extra_federated_states(self, extra_states: FederatedModules) -> None:
        if not self.sync_entropy_coef:
            if extra_states:
                raise KeyError(
                    "Received entropy coefficient state while sync_entropy_coef=False. "
                    "Make sure all SACAvg clients use the same sync_entropy_coef setting."
                )
            return

        if len(extra_states) == 0:
            return
        if set(extra_states) != {self.entropy_state_name}:
            raise KeyError(f"Unknown SACAvg extra states: {sorted(extra_states)}")

        entropy_state = extra_states[self.entropy_state_name]
        device = self.device
        if "log_ent_coef" in entropy_state:
            if self.log_ent_coef is None:
                raise ValueError(
                    "Received a learnable entropy coefficient, but this SACAvg instance uses fixed ent_coef."
                )
            with th.no_grad():
                self.log_ent_coef.copy_(entropy_state["log_ent_coef"].to(device))
            return

        if "ent_coef_tensor" in entropy_state:
            if not hasattr(self, "ent_coef_tensor"):
                raise ValueError(
                    "Received a fixed entropy coefficient, but this SACAvg instance uses auto ent_coef."
                )
            self.ent_coef_tensor = entropy_state["ent_coef_tensor"].to(device)
            return

        raise KeyError(f"Unknown entropy coefficient payload keys: {list(entropy_state.keys())}")

    def _after_set_federated_states(self) -> None:
        super()._after_set_federated_states()
        self.batch_norm_stats = get_parameters_by_name(self.critic, ["running_"])
        self.batch_norm_stats_target = get_parameters_by_name(self.critic_target, ["running_"])
