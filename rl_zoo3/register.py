import gymnasium as gym
from rl_zoo3.perturb_gym.pendulum.perturb_pendulum import PerturbPendulumEnv

from rl_zoo3.perturb_gym.ant.perturb_ant import PerturbAntEnv
from rl_zoo3.perturb_gym.hopper.perturb_hopper import PerturbHopperEnv
from rl_zoo3.perturb_gym.half_cheetah.perturb_half_cheetah import PerturbHalfCheetahEnv
from rl_zoo3.perturb_gym.walker2d.perturb_walker2d import PerturbWalker2dEnv
from rl_zoo3.perturb_gym.humanoid.perturb_humanoid import PerturbHumanoidEnv

gym.register(
    id="PerturbPendulum-v1",
    entry_point=PerturbPendulumEnv,
    max_episode_steps=200,
)


gym.register(
    id="PerturbAnt-v4",
    entry_point=PerturbAntEnv,
    max_episode_steps=1000,
    reward_threshold=6000.0,
)

gym.register(
    id="PerturbHopper-v4",
    max_episode_steps=1000,
    entry_point=PerturbHopperEnv,
)

gym.register(
    id="PerturbHalfCheetah-v4",
    entry_point=PerturbHalfCheetahEnv,
    max_episode_steps=1000,
    reward_threshold=3800.0,
)

gym.register(
    id="PerturbHumanoid-v4",
    entry_point=PerturbHumanoidEnv,
    max_episode_steps=1000,
    reward_threshold=4800.0,
)

gym.register(
    id="PerturbWalker2d-v4",
    entry_point=PerturbWalker2dEnv,
    max_episode_steps=1000,
)