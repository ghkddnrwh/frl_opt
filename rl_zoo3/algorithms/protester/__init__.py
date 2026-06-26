from rl_zoo3.algorithms.protester.d3mc import D3MC
from rl_zoo3.algorithms.protester.d2mc import D2MC
from rl_zoo3.algorithms.protester.smc import SMC
from rl_zoo3.algorithms.protester.ar_ddpg import ARDDPG
from rl_zoo3.algorithms.protester.ar_td3 import ARTD3
from rl_zoo3.algorithms.protester.ar_sac import ARSAC

PROTESTER_ALGOS = {
    "d3mc" : D3MC,
    "d2mc" : D2MC,
    "smc" : SMC,
    "ar_ddpg" : ARDDPG,
    "ar_sac" : ARSAC,
    "ar_td3" : ARTD3,
}

__all__ = [
    "PROTESTER_ALGOS",
    "D3MC",
    "SMC",
    "D2MC",
    "ARTD3",
    "ARDDPG",
    "ARSAC",
]
