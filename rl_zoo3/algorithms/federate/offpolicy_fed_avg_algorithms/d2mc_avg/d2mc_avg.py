from __future__ import annotations

from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm
from rl_zoo3.algorithms.protester.d2mc.d2mc import D2MC


class D2MCAvg(OffPolicyFedAvgAlgorithm, D2MC):
    """D2MC with FedAvg synchronization.

    The original D2MC update rule is unchanged. This class only adds the
    off-policy FedAvg upload/aggregate/broadcast interface.
    """

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
        "protester",
    )
