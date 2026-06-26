from __future__ import annotations

from rl_zoo3.algorithms.federate.common.entropy_coef_fedavg_mixin import EntropyCoefFedAvgMixin
from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm
from rl_zoo3.algorithms.protester.smc.smc import SMC


class SMCAvg(EntropyCoefFedAvgMixin, OffPolicyFedAvgAlgorithm, SMC):
    """SMC with FedAvg synchronization.

    The original SMC update rule is unchanged. FedAvg synchronizes the actor,
    critic, target critic, protester, and optionally the SAC-style entropy
    coefficient.
    """

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "critic_target",
        "protester",
    )
    federated_manager_keys: tuple[str, ...] = (
        *OffPolicyFedAvgAlgorithm.federated_manager_keys,
        "sync_entropy_coef",
    )
