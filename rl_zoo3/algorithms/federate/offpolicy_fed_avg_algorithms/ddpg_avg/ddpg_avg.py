from __future__ import annotations

from stable_baselines3.ddpg import DDPG

from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm


class DDPGAvg(OffPolicyFedAvgAlgorithm, DDPG):
    """DDPG with FedAvg synchronization.

    The original DDPG update rule is unchanged. This class only adds FedAvg
    synchronization for the actor/critic modules and their target networks.
    """

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
    )
