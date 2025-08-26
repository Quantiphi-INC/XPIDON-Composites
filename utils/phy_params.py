import jax.numpy as np
import json
from pathlib import Path
import pickle 
import numpy
import matplotlib.pyplot as plt
from jax import random,vmap
import itertools
from tqdm import trange
import jax


"""
Configuration holder for material and process parameters.

Default parameter values are taken from:
"Johnston, Andrew A. An Integrated Model of the Development of Process-Induced Deformation 
in Autoclave Processing of Composite Structures. PhD Dissertation, 
University of British Columbia, 1997."

"""

class PhyParams:
    def __init__(self, config_path):
        # Load config file
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            config = json.load(f)

        # Assign with inline defaults
        self.A = config.get("A", 1.528e5)              # Pre-exponential factor in Arrhenius equation [1/s]
        self.dE = config.get("dE", 6.650e4)            # Activation energy [J/mol]
        self.MM = config.get("MM", 0.8129)             # Model constant M (material property)
        self.NN = config.get("NN", 2.7360)             # Model constant N (material property)
        self.CC = config.get("CC", 43.09)              # Heat capacity [J/(g*K)]
        self.ALCT = config.get("ALCT", 5.475e-3)       # Linear coefficient for temperature effect
        self.ALC = config.get("ALC", -1.6840)          # Linear coefficient for cure kinetics
        self.RR = config.get("RR", 8.314)              # Universal gas constant [J/(mol*K)]
        self.a_c = config.get("a_c", 3.7388048e-07)    # Thermal diffusivity [m²/s]
        self.a_c_t = config.get("a_c_t", 3.1276314e-06)# Thermal conductivity [W/(m*K)]
        self.k_bot = config.get("k_bot", 13)           # Heat transfer coefficient at bottom [W/(m²*K)]
        self.k_top = config.get("k_top", 0.6386316)    # Heat transfer coefficient at top [W/(m²*K)]
        self.part_len = config.get("part_len", 0.03)   # Part length [m]

    def __repr__(self):
        return f"PhyParams({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"