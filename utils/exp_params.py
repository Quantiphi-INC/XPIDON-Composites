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
Configuration holder for Initial Conditions and Range of input functions.
"""

class ExpParams:
    def __init__(self, config_path):
        # Load config file
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            config = json.load(f)

       # Minimam Range of Input Functions
        self.r1_min = config.get("r1_min", 1.5)   # Min ramp1 (C/min)
        self.ht1_min = config.get("ht1_min", 100) # Min hold tempearture 1 (C)
        self.hd1_min = config.get("hd1_min", 50)  # Min hold duration 1 (min)
        self.r2_min = config.get("r2_min", 1.5)   # Min ramp2 (C/min)
        self.ht2_min = config.get("ht2_min", 175) # Min hold tempearture 2 (C)
        self.hd2_min = config.get("hd2_min", 100) # Min hold duration 2 (min)
        self.hcb_min = config.get("hcb_min", 40)  # Min bottom heat transfer coefficient W/m2.K
        self.hct_min = config.get("hct_min", 70)  # Min top heat transfer coefficient W/m2.K
        self.lt_min  = config.get("lt_min", 0.02) # Min tool thickness (m)

        # Maximum Range of Input Functions
        self.r1_max = config.get("r1_max", 3)     # Max ramp1 (C/min)
        self.ht1_max = config.get("ht1_max", 120) # Max hold tempearture 1 (C)
        self.hd1_max = config.get("hd1_max", 70)  # Max hold duration 1 (min)
        self.r2_max = config.get("r2_max", 3)     # Max ramp2 (C/min)
        self.ht2_max = config.get("ht2_max", 185) # Max hold tempearture 2 (C)
        self.hd2_max = config.get("hd2_max", 110) # Max hold duration 2 (min)
        self.hcb_max = config.get("hcb_max", 90)  # Max bottom heat transfer coefficient W/m2.K
        self.hct_max = config.get("hct_max", 120) # Max top heat transfer coefficient W/m2.K
        self.lt_max  = config.get("lt_max", 0.04) # Max tool thickness (m)

        # Process limits
        self.T_max = config.get("T_max", 250)     # Max allowable temperature [°C]
        self.t_max = config.get("t_max", 18800)   # Max processing time [min]
        self.l_max = config.get("l_max", 0.05)    # Total length limit [m]
        self.h_max = config.get("h_max", 120)     # Max heat transfer coefficient

        # Initial conditions
        self.T_ini = config.get("T_ini", 20)      # Initial temperature [°C]
        self.alpha_ini = config.get("alpha_ini", 0.05)  # Initial alpha value (dimensionless)

        # Calculate Scaling Factor
        self.T_scaler = 1 / self.T_max        # Temperature scaling factor
        self.t_scaler = 1 / self.t_max        # Scaling factor for interacting with boundary conditions
        self.len_scaler = 1 / self.l_max      # Length scaling factor
        self.h_scaler = 1 / self.h_max        # Heat transfer coefficient scaler
        self.a_diff = 1 - self.alpha_ini 

        # Calculate Scaled Initial Conditions
        self.T_ini = self.T_ini * self.T_scaler

        # Calculate Vector for minimum level for input function
        self.minvals = [(self.r1_min/60) * (self.T_scaler * self.t_max), (100) * self.T_scaler, (50*60) / self.t_max, (1.5/60) * (self.T_scaler * self.t_max), (175) * self.T_scaler, (100*60) / self.t_max, 40 * self.h_scaler, 70 * self.h_scaler, 0.02 * self.len_scaler]
        # Calculate Vector for minimum level for input function
        self.maxvals = [(self.r1_max/60) * (self.T_scaler * self.t_max), (120) * self.T_scaler, (70*60) / self.t_max, (3/60) * (self.T_scaler * self.t_max), (185) * self.T_scaler, (110*60) / self.t_max, 90 * self.h_scaler, 120 * self.h_scaler, 0.04 * self.len_scaler]

    def __repr__(self):
        return f"ExpParams({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

