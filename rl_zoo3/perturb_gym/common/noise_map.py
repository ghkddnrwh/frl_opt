from typing import Dict, List
        # noise_type이 적용될 변수 매핑

noise_map: Dict[str, Dict[str, List[str]]] = {
    "PerturbAcrobot-v1": {
        "gravity": ["GRAVITY"],
        "length": ["LINK_LENGTH_1", "LINK_LENGTH_2"],
        "mass": ["LINK_MASS_1", "LINK_MASS_2"],
        "pos": ["LINK_COM_POS_1", "LINK_COM_POS_2"],
        "length_pos": ["LINK_LENGTH_1", "LINK_LENGTH_2", "LINK_COM_POS_1", "LINK_COM_POS_2"],
        "length_mass": ["LINK_LENGTH_1", "LINK_LENGTH_2", "LINK_MASS_1", "LINK_MASS_2"],
        "length_pos_mass": ["LINK_LENGTH_1", "LINK_LENGTH_2", "LINK_MASS_1", "LINK_MASS_2",  "LINK_COM_POS_1", "LINK_COM_POS_2"],
    },
    "PerturbCartPole-v1" : {
        "gravity": ["gravity"],
        "mass_cart": ["masscart"],
        "mass_pole": ["masspole"],
        "length": ["length"],
        "force_mag": ["force_mag"],
        "all" : ["gravity", "masscart", "masspole", "length", "force_mag"],
    },
    "PerturbLunarLanderWind-v3" : {
        "gravity": ["gravity"],
        "wind_power": ["wind_power"],
        "turbulence_power": ["turbulence_power"],
        "gravity_wind_power": ["gravity", "wind_power"],
        "gravity_turbulence_power": ["gravity", "turbulence_power"],
        "wind_power_turbulence_power": ["wind_power", "turbulence_power"],
        "all" : ["gravity", "wind_power", "turbulence_power"],
    },
    "PerturbLunarLanderWindContinuous-v3" : {
        "gravity": ["gravity"],
        "wind_power": ["wind_power"],
        "turbulence_power": ["turbulence_power"],
        "gravity_wind_power": ["gravity", "wind_power"],
        "gravity_turbulence_power": ["gravity", "turbulence_power"],
        "wind_power_turbulence_power": ["wind_power", "turbulence_power"],
        "all" : ["gravity", "wind_power", "turbulence_power"],
    },
    "PerturbLunarLander-v3" : {
        "gravity": ["gravity"],
    },
    "PerturbLunarLanderContinuous-v3" : {
        "gravity": ["gravity"],
    },
    "PerturbMountainCarContinuous-v0" : {
        "gravity": ["gravity"],
        "power": ["power"],      
        "all" : ["gravity", "power"],  
    },
    "PerturbMountainCar-v0" : {
        "gravity": ["gravity"],
        "force": ["force"],
        "all" : ["gravity", "force"],
    },
    "PerturbPendulum-v1": {
        "gravity": ["g"],
        "mass": ["m"],
        "length": ["l"],
        "dt": ["dt"],
        "gravity_length_mass": ["g", "m", "l"],
        "all": ["g", "m", "l", "dt"],
    },
}   
