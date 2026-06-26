from __future__ import annotations

from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm
from rl_zoo3.algorithms.protester.ar_ddpg.ar_ddpg import ARDDPG


class ARDDPGAvg(OffPolicyFedAvgAlgorithm, ARDDPG):
    """ARDDPG with FedAvg synchronization.

    The original ARDDPG update rule is unchanged. This class only adds FedAvg
    synchronization for the model modules used by the underlying TD3-style code.
    """

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
        "protester",
    )
