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
Domain decomposition training for XPIDON.
"""

def train(model_class, model_loss_class, generate_training_data, DataGenerator, Temp_air,phy_params,exp_params,train_params,init_sub_domain,Tolerance_level,n_ips,
          branch_layers ,trunk_layers, nomad_layers_T):
    
    """
    Domain decomposition training for XPIDON.

    This function performs sequential subdomain training for the Extended Physics-Informed DeepONet (XPIDON) model. 
    The time domain is decomposed into smaller subdomains, and the model is trained on each subdomain 
    separately, potentially splitting further if the loss tolerance criteria are not met.

    Key steps:
    1. Initialize subdomain boundaries based on the number of initial subdomains.
    2. For each subdomain:
        - Initialize the XPIDON model for the subdomain's time range.
        - Load pre-trained parameters from the previous subdomain if available, or create dummy ones for the first subdomain.
        - Generate training data for PDE residuals, boundary conditions (BC), initial conditions (IC), and ODE terms.
        - Train the model in alternating phases for temperature parameters (T) and alpha parameters (a).
        - Log and monitor training losses.
        - If the losses exceed the given tolerance, further split the subdomain into two smaller ones and retrain.
        - Otherwise, save the trained parameters for use in the next subdomain.

    Parameters
    ----------
    model_class : class
        XPIDON class or compatible model constructor.
    model_loss_class : class
        Class to calculate all physics constraint losses (XPIDONLoss).
    generate_training_data : callable
        Function to generate training data for PDE, IC, BC, and ODE.
    DataGenerator : callable
        Class to create iterable batches of training data.
    Temp_air : class
        Ambient temperature calculation.
    phy_params : dict
        Dictionary containing physical parameters for the problem setup.
    exp_params : dict
        Dictionary containing experimental parameters for data generation.
    train_params : dict
        Dictionary containing training parameters.
    init_sub_domain : int
        Initial number of subdomains to split the time domain into.
    Tolerance_level : float
        Threshold for loss convergence to determine if a subdomain needs further splitting.
    n_ips : int
        Number of input function for training.
    branch_layers : list[int]
        Layer sizes for the branch network of XPIDON.
    trunk_layers : list[int]
        Layer sizes for the trunk network of XPIDON.
    nomad_layers_T : list[int]
        Layer sizes for the temperature prediction head (nomad_T) of XPIDON.

    Returns
    -------
    None
        The function saves model parameters for each converged subdomain to .pkl files and may print/log training progress.
    """

    # --- 1. Domain Decomposition Initialization ---
    # Create a list of normalized time points (0 to 1) that define the subdomain boundaries.
    # For example, if init_sub_domain = 2, dd_list will be [0.0, 0.5, 1.0].
    dd_list = numpy.linspace(0, 1, init_sub_domain + 1).tolist()  
    sub_count = 0 # Counter for the current subdomain being processed

    # Loop through each subdomain
    while sub_count < (len(dd_list) - 1):
        print(f'\n--- Processing Sub-domain: {sub_count + 1}/{len(dd_list) - 1} ---')
        print(f'Time range: {exp_params.t_max * dd_list[sub_count]:.2f} to {exp_params.t_max * dd_list[sub_count + 1]:.2f} seconds')
        
        # Determine if this is the first subdomain (for initial parameter loading)
        first_sub = True if sub_count == 0 else False

        # --- 2. Model and Loss Class Initialization for Current Subdomain ---
        # Initialize the XPIDON model for the current subdomain's time range.
        # The t_min and t_max for the model are scaled by exp_params.t_max.
        md = model_class(phy_params, exp_params, branch_layers, trunk_layers, nomad_layers_T, 
                         exp_params.t_max * dd_list[sub_count], exp_params.t_max * dd_list[sub_count + 1])
        
        # Initialize the XPIDONLoss class, passing the model instance and subdomain time range.
        # This instance will handle all loss calculations and physics-based residuals.
        loss = model_loss_class(md, phy_params, exp_params,
                                exp_params.t_max * dd_list[sub_count], exp_params.t_max * dd_list[sub_count + 1])
        
        # Define vmapped versions of physics-based functions for plotting/inspection.
        # These now correctly reference the methods within the 'loss' object.
        res_pred_fn = vmap(loss.pde_net, (None, None, 0, 0, 0, None))
        ode_pred_fn = vmap(loss.ode_net, (None, None, 0, 0, 0))
        bcb_fn = vmap(loss.bcb_net, (None, 0, 0, 0, 0))
        bct_fn = vmap(loss.bct_net, (None, 0, 0, 0, 0))

        # Get initial (or pre-trained) parameters from the model
        T_params, a_params = md.get_weights()
        
        # Get time parameters from the model 
        t_min_l, t_max_l, del_t_l = md.get_t_params()

        # --- 3. Load Pre-trained Parameters (if not first subdomain) ---
        if first_sub:
            # For the first subdomain, use the randomly initialized parameters
            T_params_pre, a_params_pre = md.get_weights()
        else:
            # For subsequent subdomains, load parameters from the end of the previous subdomain
            # The filename uses the end time of the previous subdomain (which is the start time of current)
            prev_sub_end_time_str = str(dd_list[sub_count])[2:] # e.g., '05' for 0.5
            file_T = Path(f"xpidon_class_0{prev_sub_end_time_str}_T.pkl")
            file_a = Path(f"xpidon_class_0{prev_sub_end_time_str}_a.pkl")
            
            if file_T.exists() and file_a.exists():
                with open(file_T, 'rb') as f:
                    T_params_pre = pickle.load(f)
                with open(file_a, 'rb') as f:
                    a_params_pre = pickle.load(f)
                print(f"Loaded pre-trained parameters from {file_T.name} and {file_a.name}")
            else:
                print(f"Warning: Pre-trained files not found for subdomain start {dd_list[sub_count]}. Initializing from scratch.")
                T_params_pre, a_params_pre = md.get_weights() # Fallback to random init

        # --- 4. Generate Training Data ---
        # Define parameters for data generation
        taskkey = random.PRNGKey(65203)
        reskey = random.PRNGKey(12301) # Not directly used in this snippet, but good to keep
        N = train_params.N # Number of input samples (different design conditions)
        P_ic = train_params.P_ic # Number of output sensors for initial conditions
        P_bc = train_params.P_bc # Number of collocation points for boundary conditions
        Q_train = train_params.Q_train  # Number of collocation points for PDE/ODE residuals
        
        # Time interval for data generation (should match current subdomain)
        t_int_min = t_min_l 
        t_int_max = t_max_l 

        # Call the data generation function
        u_bc_train, y_bc_train, s_bc_train, \
        u_ic_train, y_ic_train, s_ic_train, \
        u_res_train, y_res_train_orig, s_res_train, \
        u_inf_train, y_inf_train_orig, s_inf_train = generate_training_data(
            taskkey, exp_params.minvals, exp_params.maxvals,
            t_int_min, t_int_max, T_params_pre, a_params_pre, md, Temp_air, exp_params, N,
            P_ic, P_bc, Q_train, hold = 'two', first_subdomain = first_sub
        )

        # --- 5. Data Batching Setup ---
        # Define batch sizes for different types of training data
        batch_size_ic = train_params.batch_size_ic
        batch_size_inf = train_params.batch_size_inf
        batch_size_bc = train_params.batch_size_bc
        batch_size_res = train_params.batch_size_res
        
        # Initial iteration counter for PRNG keys
        it_init_key = 1 

        # Define weights for different loss terms
        w_ic_T = train_params.w_ic_T      # Weight for initial condition loss (Temperature)
        w_bcb = train_params.w_bcb        # Weight for bottom boundary condition loss
        w_bct = train_params.w_bct        # Weight for top boundary condition loss
        w_ic_a = train_params.w_ic_a      # Weight for initial condition loss (Alpha)
        w_ode = train_params.w_ode        # Weight for ODE residual loss
        weights = [w_ic_T, w_bcb, w_bct]  # Weights for T-phase optimization
        weights_ode = [w_ic_a, w_ode]     # Weights for alpha-phase optimization

        # --- 6. Logger Initialization ---
        # Lists to store loss values for logging and plotting
        loss_log = [] # Total T loss
        loss_a_log = [] # Total alpha loss
        loss_ics_T_log = []
        loss_ics_tool_log = []
        loss_ics_a_log = []
        loss_bct_log = []
        loss_bcb_log = []
        loss_res_log = [] # PDE part loss
        loss_res_tool_log = [] # PDE tool loss
        loss_ode_log = [] # ODE cure part loss
        loss_inf_log = [] # Interface temperature loss
        loss_flux_log = [] # Interface flux loss
        loss_lr_log = [] # Learning rate log (not used in current pbar, but useful)

        # Training loop parameters
        seq_id = 0 # Sequence ID for alternating T and alpha training
        nIter = 200 # Number of outer iterations (epochs)
        batch_count = 1000 # Number of batches per outer iteration
        seq_print = ['** Training T **', '** Training a **'] # Status messages
        seq_pid = 0 # Index for seq_print

        # Create data generators for each type of data
        ics_dataset = DataGenerator(u_ic_train, y_ic_train, s_ic_train, None, batch_size_ic, rng_key=random.PRNGKey(it_init_key))
        bcs_dataset = DataGenerator(u_bc_train, y_bc_train, s_bc_train, None, batch_size_bc, rng_key=random.PRNGKey(it_init_key + 1))
        inf_dataset = DataGenerator(u_inf_train, y_inf_train_orig, s_inf_train, None, batch_size_bc, rng_key=random.PRNGKey(it_init_key + 4))
        res_dataset = DataGenerator(u_res_train, y_res_train_orig,s_res_train, None, batch_size_res, rng_key=random.PRNGKey(it_init_key + 2))
        ode_dataset = DataGenerator(u_res_train, y_res_train_orig,s_res_train, None, batch_size_res, rng_key=random.PRNGKey(it_init_key + 2))

        # Create iterators for the datasets to fetch batches
        ics_data = iter(ics_dataset)
        bcs_data = iter(bcs_dataset)
        inf_data = iter(inf_dataset)
        res_data = iter(res_dataset)
        ode_data = iter(ode_dataset)

        # Parameters for dynamic data permutation (if needed)
        key_counter = 0
        key = random.PRNGKey(12255)
        key_train = random.split(key, nIter + 100) # Ensure enough keys for permutations

        # Fixed beta value
        b_val = 0.7003080410794345

        # Initialize loss values for the first printout (before any training steps)
        # These will be updated after the first batch.
        loss_T_value = float('inf')
        loss_a_value = float('inf')
        loss_ics_tool_value = float('inf')
        loss_ics_T_value = float('inf')
        loss_bct_value = float('inf')
        loss_bcb_value = float('inf')
        loss_res_value = float('inf')
        loss_res_tool_value = float('inf')
        loss_inf_value = float('inf')
        loss_flux_value = float('inf')
        loss_ode_value = float('inf')
        loss_ics_a_value = float('inf')
        lr_T = 0.0
        lr_a = 0.0

        # --- 7. Main Training Loop ---
        pbar = trange(nIter, ncols=500, desc="Training Progress") # Progress bar
        for it in pbar: # Outer loop for epochs
            key_counter += 1
            
            # Save temporary parameters (useful for debugging or resuming)
            with open('T_temp.pkl', 'wb') as f:
                pickle.dump(T_params, f)
            with open('a_temp.pkl', 'wb') as f:
                pickle.dump(a_params, f)
            
            # Permute residual and interface data
            y_res_train = random.permutation(key_train[key_counter], y_res_train_orig)
            y_inf_train = random.permutation(key_train[key_counter], y_inf_train_orig)
            
            # Re-initialize data generators with permuted data
            res_dataset = DataGenerator(u_res_train,y_res_train,s_res_train, None, batch_size_res, rng_key=random.PRNGKey(it + 2))
            res_data = iter(res_dataset)
        

            # --- Plotting Predictions (every 10 epochs) ---
            if np.logical_and(it % 10 == 0, it > 0):
                # Fetch a batch for plotting (ensure these batches are available)
                # Note: These batches are from the current iterators, not necessarily the ones used for training in this specific step
                try:
                    inputs_bc, outputs_bc = next(bcs_data)
                    u_bc, y_bcs_batch = inputs_bc
                    inputs_res, _ = next(res_data)
                    u_res, y_res_batch = inputs_res
                    inputs_ode, _ = next(ode_data)
                    u_ode, y_ode_batch = inputs_ode
                except StopIteration:
                    # If iterators are exhausted, re-create them for plotting
                    ics_data = iter(ics_dataset)
                    bcs_data = iter(bcs_dataset)
                    inf_data = iter(inf_dataset)
                    res_data = iter(res_dataset)
                    ode_data = iter(ode_dataset)
                    inputs_bc, outputs_bc = next(bcs_data)
                    u_bc, y_bcs_batch = inputs_bc
                    inputs_res, _ = next(res_data)
                    u_res, y_res_batch = inputs_res
                    inputs_ode, _ = next(ode_data)
                    u_ode, y_ode_batch = inputs_ode


                # Get predictions for plotting from the 'loss' object's physics functions
                res_preds = np.abs(res_pred_fn(T_params, a_params, u_res, y_res_batch[:,0], y_res_batch[:,1], b_val))
                ode_preds = np.abs(ode_pred_fn(T_params, a_params, u_ode, y_ode_batch[:,0], y_ode_batch[:,1]))
                bc_preds_t = np.abs(bct_fn(T_params, outputs_bc, u_bc, y_bcs_batch[:,0], y_bcs_batch[:,2]))
                bc_preds_b = np.abs(bcb_fn(T_params, outputs_bc, u_bc, y_bcs_batch[:,0], y_bcs_batch[:,1]))

                # Plotting code
                cm = plt.cm.get_cmap('viridis')
                fig, axs = plt.subplots(1, 3, figsize=(24, 3))
                
                # BC Plot
                sc1 = axs[0].scatter(y_bcs_batch[:,0], y_bcs_batch[:,1], c=bc_preds_b, cmap=cm, s=5, alpha=0.7)
                sc = axs[0].scatter(y_bcs_batch[:,0], y_bcs_batch[:,2], c=bc_preds_t, cmap=cm, s=5, alpha=0.7)
                axs[0].set_title('BC Residuals')
                plt.colorbar(sc, ax=axs[0], label='Top BC Residual')
                plt.colorbar(sc1, ax=axs[0], label='Bottom BC Residual')
                
                # PDE Plot
                sc = axs[1].scatter(y_res_batch[:,0], y_res_batch[:,1], c=res_preds, cmap=cm, s=5, alpha=0.7)
                axs[1].set_title('PDE Residuals')
                plt.colorbar(sc, ax=axs[1], label='PDE Residual')
                
                # ODE Plot
                sc = axs[2].scatter(y_ode_batch[:,0], y_ode_batch[:,1], c=ode_preds, cmap=cm, s=5, alpha=0.7)
                axs[2].set_title('ODE Residuals')
                plt.colorbar(sc, ax=axs[2], label='ODE Residual')
                
                plt.tight_layout()
                plt.show()
            
            # --- Alternating Training Phase Logic ---
            # Switch between training T-parameters and alpha-parameters every 10 epochs
            if np.logical_and(it % 10 == 0, it > 0):
                seq_id += 1
                seq_pid = int(seq_id % 2) # 0 for T, 1 for alpha

            # --- Inner Loop: Batch Training ---
            for btch in np.arange(batch_count + 1):
                
                # --- Phase 1: Train Temperature (T) Parameters ---
                if seq_id % 2 == 1: # If seq_id is odd, train T
                    
                    # On the very first batch of the first epoch, initialize loss values
                    # This ensures all loss variables are defined before the first printout
                    if it == 0 and btch == 0:
                        # Fetch initial batches to calculate initial losses
                        ics_batch = next(ics_data)
                        res_batch = next(res_data)
                        ode_batch = next(ode_data)
                        bcs_batch = next(bcs_data)
                        inf_batch = next(inf_data)
                        
                        # Calculate initial loss values using the 'loss' object
                        loss_a_value = loss.total_loss_a(T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, inf_batch, weights_ode)
                        loss_ics_a_value = loss.loss_ics_a(a_params, ics_batch)
                        loss_ode_value = loss.loss_ode(T_params, a_params, ode_batch)
                        
                        # Get initial learning rate for alpha
                        count, _ = md.opt_state_a
                        lr_a = md.schedule(count[0])

                    # Fetch new batches for the current training step
                    # Use try-except StopIteration to re-initialize iterators if exhausted
                    try:
                        ics_batch = next(ics_data)
                        bcs_batch = next(bcs_data)
                        res_batch = next(res_data)
                        inf_batch = next(inf_data)
                        ode_batch = next(ode_data) # Ensure ode_batch is available for T loss calculation
                    except StopIteration:
                        # Re-initialize all iterators if any is exhausted
                        ics_data = iter(ics_dataset)
                        bcs_data = iter(bcs_dataset)
                        inf_data = iter(inf_dataset)
                        res_data = iter(res_dataset)
                        ode_data = iter(ode_dataset)
                        ics_batch = next(ics_data)
                        bcs_batch = next(bcs_data)
                        res_batch = next(res_data)
                        inf_batch = next(inf_data)
                        ode_batch = next(ode_data)

                    # Perform one optimization step for T parameters
                    # The step_temp method is now part of the 'md' (XPIDON) object
                    md.opt_state_T, T_params, loss_value_T = loss.step_temp(
                        md.opt_state_T, 
                        T_params, 
                        a_params, # a_params are passed but gradients are stopped internally
                        ics_batch, 
                        bcs_batch, 
                        res_batch, 
                        ode_batch, 
                        inf_batch, 
                        inf_batch, # ext_batch is passed as inf_batch in original code
                        weights, 
                        b_val
                    )
                    
                    # Log and print losses every 100 batches
                    if btch % 100 == 0:
                        # Calculate all individual loss components for logging
                        loss_T_value = loss.total_loss_temp(T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, inf_batch, weights, b_val)
                        loss_ics_tool_value = loss.loss_ics_tool(T_params, ics_batch)
                        loss_ics_T_value = loss.loss_ics_T(T_params, ics_batch)
                        loss_bct_value = loss.loss_bct(T_params, bcs_batch)
                        loss_bcb_value = loss.loss_bcb(T_params, bcs_batch)
                        loss_res_value = loss.loss_res(T_params, a_params, res_batch, b_val)
                        loss_res_tool_value = loss.loss_res_tool(T_params, inf_batch)
                        loss_inf_value = loss.loss_inf(T_params, inf_batch)
                        loss_flux_value = loss.loss_flux(T_params, inf_batch)
                        # Recalculate alpha-related losses for display, as they might have changed due to T_params update affecting cure_kinetics
                        loss_a_value = loss.total_loss_a(T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, inf_batch, weights_ode)
                        loss_ics_a_value = loss.loss_ics_a(a_params, ics_batch)
                        loss_ode_value = loss.loss_ode(T_params, a_params, ode_batch)


                        # Append to log lists
                        loss_log.append(loss_T_value)
                        loss_ics_T_log.append(loss_ics_T_value)
                        loss_ics_tool_log.append(loss_ics_tool_value)
                        loss_bct_log.append(loss_bct_value)
                        loss_bcb_log.append(loss_bcb_value)
                        loss_res_log.append(loss_res_value)
                        loss_res_tool_log.append(loss_res_tool_value)
                        loss_inf_log.append(loss_inf_value)
                        loss_flux_log.append(loss_flux_value)
                        
                        # Get current learning rate for T
                        count, _ = md.opt_state_T
                        lr_T = md.schedule(count[0])

                        # Update progress bar postfix with current loss values
                        pbar.set_postfix({
                            'Status': seq_print[seq_pid],
                            'Loss T': f'{loss_T_value:.2e}',
                            'Loss a': f'{loss_a_value:.2e}',
                            'ICs T': f'{loss_ics_T_value:.2e}',
                            'ICs tool': f'{loss_ics_tool_value:.2e}',
                            'ICs Alpha': f'{loss_ics_a_value:.2e}',
                            'BCs Bot': f'{loss_bcb_value:.2e}',
                            'BCs Top': f'{loss_bct_value:.2e}',
                            'PDE Part': f'{loss_res_value:.2e}',
                            'PDE Tool': f'{loss_res_tool_value:.2e}',
                            'Cure Part': f'{loss_ode_value:.2e}',
                            'INF': f'{loss_inf_value:.2e}',
                            'FLUX': f'{loss_flux_value:.2e}',
                            'T lr': f'{lr_T:.2e}',
                            'a lr': f'{lr_a:.2e}'
                        })
                    
                # --- Phase 2: Train Alpha (a) Parameters ---
                if seq_id % 2 == 0: # If seq_id is even, train alpha
                    
                    # On the very first batch of the first epoch, initialize loss values
                    if it == 0 and btch == 0:
                        # Fetch initial batches to calculate initial losses
                        ics_batch = next(ics_data)
                        bcs_batch = next(bcs_data)
                        res_batch = next(res_data)
                        ode_batch = next(ode_data)
                        inf_batch = next(inf_data)
                        
                        # Calculate initial loss values for T
                        loss_T_value = loss.total_loss_temp(T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, inf_batch, weights, b_val)
                        loss_ics_tool_value = loss.loss_ics_tool(T_params, ics_batch)
                        loss_ics_T_value = loss.loss_ics_T(T_params, ics_batch)
                        loss_bct_value = loss.loss_bct(T_params, bcs_batch)
                        loss_bcb_value = loss.loss_bcb(T_params, bcs_batch)
                        loss_res_value = loss.loss_res(T_params, a_params, res_batch, b_val)
                        loss_res_tool_value = loss.loss_res_tool(T_params, inf_batch)
                        loss_inf_value = loss.loss_inf(T_params, inf_batch)
                        loss_flux_value = loss.loss_flux(T_params, inf_batch)
                        
                        # Get initial learning rate for T
                        count, _ = md.opt_state_T
                        lr_T = md.schedule(count[0])
                    
                    # Fetch new batches for the current training step
                    try:
                        ics_batch = next(ics_data)
                        # res_batch = next(res_data) # res_batch is needed for loss_res in printout
                        ode_batch = next(ode_data)
                        bcs_batch = next(bcs_data) # bcs_batch is needed for loss_bcb/bct in printout
                        inf_batch = next(inf_data) # inf_batch is needed for loss_inf/flux in printout
                        res_batch = next(res_data) # Ensure res_batch is available
                    except StopIteration:
                        # Re-initialize all iterators if any is exhausted
                        ics_data = iter(ics_dataset)
                        bcs_data = iter(bcs_dataset)
                        inf_data = iter(inf_dataset)
                        res_data = iter(res_dataset)
                        ode_data = iter(ode_dataset)
                        ics_batch = next(ics_data)
                        bcs_batch = next(bcs_data)
                        inf_batch = next(inf_data)
                        res_batch = next(res_data)
                        ode_batch = next(ode_data)

                    # Perform one optimization step for alpha parameters
                    # The step_alpha method is now part of the 'md' (XPIDON) object
                    md.opt_state_a, a_params, loss_value_a = loss.step_alpha(
                        md.opt_state_a, 
                        T_params, # T_params are passed but gradients are stopped internally
                        a_params, 
                        ics_batch, 
                        bcs_batch, 
                        res_batch, 
                        ode_batch, 
                        inf_batch, 
                        inf_batch, # ext_batch is passed as inf_batch in original code
                        weights_ode
                    )
                    
                    # Log and print losses every 100 batches
                    if btch % 100 == 0:
                        # Calculate all individual loss components for logging
                        loss_a_value = loss.total_loss_a(T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, inf_batch, weights_ode)
                        loss_ics_a_value = loss.loss_ics_a(a_params, ics_batch)
                        loss_ode_value = loss.loss_ode(T_params, a_params, ode_batch)
                        # Recalculate T-related losses for display, as they might have changed due to a_params update affecting cure_kinetics
                        loss_T_value = loss.total_loss_temp(T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, inf_batch, weights, b_val)
                        loss_ics_tool_value = loss.loss_ics_tool(T_params, ics_batch)
                        loss_ics_T_value = loss.loss_ics_T(T_params, ics_batch)
                        loss_bct_value = loss.loss_bct(T_params, bcs_batch)
                        loss_bcb_value = loss.loss_bcb(T_params, bcs_batch)
                        loss_res_value = loss.loss_res(T_params, a_params, res_batch, b_val)
                        loss_res_tool_value = loss.loss_res_tool(T_params, inf_batch)
                        loss_inf_value = loss.loss_inf(T_params, inf_batch)
                        loss_flux_value = loss.loss_flux(T_params, inf_batch)


                        # Append to log lists
                        loss_a_log.append(loss_a_value)
                        loss_ics_a_log.append(loss_ics_a_value)
                        loss_ode_log.append(loss_ode_value)
            
                        # Get current learning rate for alpha
                        count, _ = md.opt_state_a
                        lr_a = md.schedule(count[0])

                        # Update progress bar postfix with current loss values
                        pbar.set_postfix({
                            'Status': seq_print[seq_pid],
                            'Loss T': f'{loss_T_value:.2e}',
                            'Loss a': f'{loss_a_value:.2e}',
                            'ICs T': f'{loss_ics_T_value:.2e}',
                            'ICs tool': f'{loss_ics_tool_value:.2e}',
                            'ICs Alpha': f'{loss_ics_a_value:.2e}',
                            'BCs Bot': f'{loss_bcb_value:.2e}',
                            'BCs Top': f'{loss_bct_value:.2e}',
                            'PDE Part': f'{loss_res_value:.2e}',
                            'PDE Tool': f'{loss_res_tool_value:.2e}',
                            'Cure Part': f'{loss_ode_value:.2e}',
                            'INF': f'{loss_inf_value:.2e}',
                            'FLUX': f'{loss_flux_value:.2e}',
                            'T lr': f'{lr_T:.2e}',
                            'a lr': f'{lr_a:.2e}'
                        })
                        
        # --- 8. Subdomain Convergence Check and Saving/Splitting ---
        # After completing all training iterations for the current subdomain
        if loss_T_value > Tolerance_level or loss_a_value > Tolerance_level:
            # If losses are too high, split the current subdomain into two
            # Insert a new midpoint into the dd_list to refine the decomposition
            new_midpoint = round((dd_list[sub_count] + dd_list[sub_count+1])/2, 4)
            dd_list.insert(sub_count + 1, new_midpoint)
            print(f'\nLosses {loss_T_value:.2e} (T), {loss_a_value:.2e} (a) exceed tolerance {Tolerance_level:.2e}.')
            print(f'Subdomain split. New decomposition: {dd_list}')
            # The loop will re-process the current 'sub_count' with the new, smaller range
            # and then proceed to the next segment.
        else:
            # If losses are within tolerance, the subdomain is considered converged
            # Save the trained parameters for this subdomain
            current_sub_end_time_str = str(dd_list[sub_count+1])[2:] # e.g., '05' for 0.5, '10' for 1.0
            filename_T =  f"xpidon_class_0{current_sub_end_time_str}_T.pkl"
            filename_a =  f"xpidon_class_0{current_sub_end_time_str}_a.pkl"
            
            with open(filename_T, 'wb') as f:
                pickle.dump(T_params, f)
            with open(filename_a, 'wb') as f:
                pickle.dump(a_params, f)
                
            print(f"\nModels saved for converged subdomain {dd_list[sub_count]:.2f} - {dd_list[sub_count+1]:.2f}")
            print(f"Saved as {filename_T} and {filename_a}")
            
            # Move to the next subdomain in the list
            sub_count += 1

    print("\n--- Domain Decomposition Training Complete ---")