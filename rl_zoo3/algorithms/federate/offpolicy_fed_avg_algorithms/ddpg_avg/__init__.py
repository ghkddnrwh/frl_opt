from stable_baselines3.ddpg.policies import CnnPolicy, MlpPolicy, MultiInputPolicy

from rl_zoo3.algorithms.federate.offpolicy_fed_avg_algorithms.ddpg_avg.ddpg_avg import DDPGAvg

__all__ = ["CnnPolicy", "MlpPolicy", "MultiInputPolicy", "DDPGAvg"]
