# Import all necessary components from their respective modules
from utils.phy_params import PhyParams
from utils.exp_params import ExpParams
from utils.train_params import TrainParams
from utils.air_temp import Temp_air
from utils.data_utils import DataGenerator, generate_training_data
from trainer.train import train
from models.pidon import XPIDON
from loss.loss import XPIDONLoss

def main():
    # --- Load Parameters from JSON Files ---
    # Assuming exp_params.json and phy_params.json are in the main directory
    params = PhyParams('phy_params.json')
    exp_params = ExpParams('exp_params.json')
    train_params = TrainParams('train_params.json')

    # --- Define Model and Training Hyperparameters ---
    m_inp = 9
    branch_layers = [m_inp, 50, 50, 50, 100]
    trunk_layers = [2, 50, 50, 50, 50, 50, 100]
    nomad_layers_T = [100, 50, 50, 50, 50, 1] # Example: 1 output for temperature
    
    init_sub_domain = 5 # Initial number of subdomains
    Tolerance_level = 2e-5 # Loss tolerance for subdomain convergence

    # --- Call the Training Function ---
    train(XPIDON, XPIDONLoss, generate_training_data, DataGenerator, Temp_air,
          params, exp_params, train_params, init_sub_domain,Tolerance_level,m_inp,
          branch_layers ,trunk_layers, nomad_layers_T)

if __name__ == "__main__":
    main()