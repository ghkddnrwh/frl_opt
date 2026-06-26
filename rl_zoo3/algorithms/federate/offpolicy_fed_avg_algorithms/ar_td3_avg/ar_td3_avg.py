from __future__ import annotations

from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm
from rl_zoo3.algorithms.protester.ar_td3.ar_td3 import ARTD3


class ARTD3Avg(OffPolicyFedAvgAlgorithm, ARTD3):
    """ARTD3 with FedAvg synchronization.

    The original ARTD3 update rule is unchanged. FedAvg synchronizes the actor,
    critic, target networks, and protester network across clients.
    """

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
        "protester",
    )
