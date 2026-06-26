import gymnasium as gym
from rl_zoo3.perturb_gym.pendulum.perturb_pendulum import PerturbPendulumEnv
from rl_zoo3.perturb_gym.mountain_car.perturb_mountain_car import PerturbMountainCarEnv
from rl_zoo3.perturb_gym.mountaincar_continuous.perturb_mountaincar_continuous import PerturbMountainCarContinuousEnv
from rl_zoo3.perturb_gym.cartpole.perturb_cartpole import PerturbCartPoleEnv
from rl_zoo3.perturb_gym.acrobot.perturb_acrobot import PerturbAcrobotEnv

# try:
#     from rl_zoo3.perturb_gym.lunar_lander.perturb_lunar_lander import PerturbLunarLander
#     from rl_zoo3.perturb_gym.lunar_lander_wind.perturb_lunar_lander_wind import PerturbLunarLanderWind
# except Exception:
#     PerturbLunarLander = None
#     PerturbLunarLanderWind = None

from rl_zoo3.perturb_gym.lunar_lander.perturb_lunar_lander import PerturbLunarLander
from rl_zoo3.perturb_gym.lunar_lander_wind.perturb_lunar_lander_wind import PerturbLunarLanderWind

from rl_zoo3.perturb_gym.lunar_lander_continuous.perturb_lunar_lander_continuous import PerturbLunarLanderContinuous
from rl_zoo3.perturb_gym.lunar_lander_wind_continuous.perturb_lunar_lander_wind_continuous import PerturbLunarLanderWindContinuous

try:
    from rl_zoo3.perturb_gym.ant.perturb_ant import PerturbAntEnv
    from rl_zoo3.perturb_gym.hopper.perturb_hopper import PerturbHopperEnv
    from rl_zoo3.perturb_gym.half_cheetah.perturb_half_cheetah import PerturbHalfCheetahEnv
    from rl_zoo3.perturb_gym.walker2d.perturb_walker2d import PerturbWalker2dEnv
    from rl_zoo3.perturb_gym.humanoid.perturb_humanoid import PerturbHumanoidEnv
except Exception:
    PerturbAntEnv = None
    PerturbHopperEnv = None
    PerturbHalfCheetahEnv = None
    PerturbWalker2dEnv = None
    PerturbHumanoidEnv = None

gym.register(
    id="PerturbPendulum-v1",
    entry_point=PerturbPendulumEnv,
    max_episode_steps=200,
)

gym.register(
    id="PerturbMountainCarContinuous-v0",
    entry_point=PerturbMountainCarContinuousEnv,
    max_episode_steps=999,
    reward_threshold=90.0,
)

gym.register(
    id="PerturbMountainCar-v0",
    entry_point=PerturbMountainCarEnv,
    max_episode_steps=200,
    reward_threshold=-110.0,
)

gym.register(
    id="PerturbCartPole-v1",
    entry_point=PerturbCartPoleEnv,
    max_episode_steps=500,
    reward_threshold=475.0,
)

gym.register(
    id="PerturbAcrobot-v1",
    entry_point=PerturbAcrobotEnv,
    reward_threshold=-100.0,
    max_episode_steps=500,
)

if PerturbLunarLander is not None:
    gym.register(
        id="PerturbLunarLander-v3",
        entry_point=PerturbLunarLander,
        max_episode_steps=1000,
        reward_threshold=200.0,
    )

if PerturbLunarLanderWind is not None:
    gym.register(
        id="PerturbLunarLanderWind-v3",
        entry_point=PerturbLunarLanderWind,
        max_episode_steps=1000,
        reward_threshold=200.0,
    )



gym.register(
    id="PerturbLunarLanderContinuous-v3",
    entry_point=PerturbLunarLanderContinuous,
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)

gym.register(
    id="PerturbLunarLanderWindContinuous-v3",
    entry_point=PerturbLunarLanderWindContinuous,
    kwargs={"continuous": True},
    max_episode_steps=1000,
    reward_threshold=200,
)



if PerturbAntEnv is not None:
    gym.register(
        id="PerturbAnt-v4",
        entry_point=PerturbAntEnv,
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )

if PerturbHopperEnv is not None:
    gym.register(
        id="PerturbHopper-v4",
        max_episode_steps=1000,
        entry_point=PerturbHopperEnv,
    )

if PerturbHalfCheetahEnv is not None:
    gym.register(
        id="PerturbHalfCheetah-v4",
        entry_point=PerturbHalfCheetahEnv,
        max_episode_steps=1000,
        reward_threshold=3800.0,
    )

if PerturbHumanoidEnv is not None:
    gym.register(
        id="PerturbHumanoid-v4",
        entry_point=PerturbHumanoidEnv,
        max_episode_steps=1000,
        reward_threshold=4800.0,
    )

if PerturbWalker2dEnv is not None:
    gym.register(
        id="PerturbWalker2d-v4",
        entry_point=PerturbWalker2dEnv,
        max_episode_steps=1000,
    )
