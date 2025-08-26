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
Configuration holder for Training Parameters.
"""

class TrainParams:
    def __init__(self, config_path):
        # Load config file
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r") as f:
            config = json.load(f)

        # Number of collocation points
        self.N = config.get("N", 500)   # Number of input samples (different design conditions)
        self.P_ic = config.get("P_ic", 100) # Number of output sensors for initial conditions
        self.P_bc = config.get("P_bc", 100)  # Number of collocation points for boundary conditions
        self.Q_train = config.get("Q_train", 500)   # Number of collocation points for PDE/ODE residuals


        # Define batch sizes for different types of training data
        self.batch_size_ic = config.get("batch_size_ic", 64)     # Batch size for initial condition data
        self.batch_size_inf = config.get("batch_size_inf", 128)  
        self.batch_size_bc = config.get("batch_size_bc", 128)    # Batch size for boundary condition data
        self.batch_size_res = config.get("batch_size_res", 256)  # Batch size for residual points (PDE/ODE)


        # Define weights for different loss terms
        self.w_ic_T = config.get("w_ic_T", 1)        # Weight for initial condition loss (Temperature)
        self.w_bcb = config.get("w_bcb", 5)          # Weight for bottom boundary condition loss
        self.w_bct = config.get("w_bct", 5)          # Weight for top boundary condition loss
        self.w_ic_a = config.get("w_ic_a", 1)        # Weight for initial condition loss (Alpha)
        self.w_ode = config.get("w_ode", 1)          # Weight for ODE residual loss


    def __repr__(self):
        return f"TrainParams({', '.join(f'{k}={v}' for k, v in self.__dict__.items())})"

