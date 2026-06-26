from __future__ import annotations

from rl_zoo3.algorithms.federate.common.entropy_coef_fedavg_mixin import EntropyCoefFedAvgMixin
from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm
from rl_zoo3.algorithms.protester.ar_sac.ar_sac import ARSAC


class ARSACAvg(EntropyCoefFedAvgMixin, OffPolicyFedAvgAlgorithm, ARSAC):
    """ARSAC with FedAvg synchronization.

    The original ARSAC update rule is unchanged. FedAvg synchronizes the actor,
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
