from typing import Dict, List

noise_map: Dict[str, Dict[str, List[str]]] = {
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
    "PerturbMountainCarContinuous-v0" : {
        "gravity": ["gravity"],
        "power": ["power"],      
        "all" : ["gravity", "power"],  
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