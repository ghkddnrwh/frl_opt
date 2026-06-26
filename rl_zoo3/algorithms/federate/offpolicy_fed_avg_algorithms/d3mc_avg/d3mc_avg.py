from __future__ import annotations

from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm
from rl_zoo3.algorithms.protester.d3mc.d3mc import D3MC


class D3MCAvg(OffPolicyFedAvgAlgorithm, D3MC):
    """D3MC with FedAvg synchronization.

    The original D3MC update rule is unchanged. FedAvg only controls which
    trainable modules are uploaded, averaged, and broadcast between clients.
    """

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
        "protester",
    )
