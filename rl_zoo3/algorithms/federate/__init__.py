from rl_zoo3.algorithms.federate.ampo import FedAMPPO, FedAMPOPPO, FedAMPOLocalPPO
from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo import FedSPPGPPO
from rl_zoo3.algorithms.federate.fedsp_pg.fedsp_pg_ppo_paper_aligned import FedSPPGPPOPaperAligned
from rl_zoo3.algorithms.federate.fedsvrpg_m.fedsvrpg_m import FedSVRPGM
from rl_zoo3.algorithms.federate.ppo_avg.ppo_avg import PPOAvg
from rl_zoo3.algorithms.federate.offpolicy_fed_avg_algorithms import TD3Avg, DDPGAvg, SACAvg, ARDDPGAvg, ARTD3Avg, ARSACAvg, D2MCAvg, D3MCAvg, SMCAvg

FEDERATE_ALGOS = {
    "td3_avg" : TD3Avg,
    "sac_avg" : SACAvg,
    "ddpg_avg" : DDPGAvg,
    "ar_td3_avg" : ARTD3Avg,
    "ar_sac_avg" : ARSACAvg,
    "ar_ddpg_avg" : ARDDPGAvg,
    "d3mc_avg" : D3MCAvg,
    "d2mc_avg" : D2MCAvg,
    "smc_avg" : SMCAvg, 

    "fed_ampo_ppo" : FedAMPPO,
    "fed_ampo_local_ppo": FedAMPOLocalPPO,
    "ppo_avg" : PPOAvg,
    "fed_svrpg_m" : FedSVRPGM,
}

__all__ = [
    "TD3Avg",
    "DDPGAvg",
    "SACAvg",
    "ARTD3Avg",
    "ARDDPGAvg",
    "ARSACAvg",
    "D2MCAvg",
    "D3MCAvg",
    "SMCAvg",

    "PPOAvg",
    "FedSPPGPPO",
    "FedSPPGPPOPaperAligned",
    "FedAMPPO",
    "FedAMPOPPO",
    "FedSVRPGM",
    "FedAMPOLocalPPO",
]
