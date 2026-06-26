from __future__ import annotations

from stable_baselines3.td3 import TD3

from rl_zoo3.algorithms.federate.common.offpolicy_fedavg_algorithm import OffPolicyFedAvgAlgorithm


class TD3Avg(OffPolicyFedAvgAlgorithm, TD3):
    """TD3 with FedAvg synchronization."""

    federated_modules: tuple[str, ...] = (
        "actor",
        "critic",
        "actor_target",
        "critic_target",
    )
