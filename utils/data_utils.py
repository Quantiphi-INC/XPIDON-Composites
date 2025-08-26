from torch.utils import data
import jax.numpy as np
from jax import random, jit, vmap
from functools import partial


"""
The class `DataGenerator` will generate dataset and batches comprising input functions, collocation points and solution.
"""
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s, prob,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u
        self.y = y
        self.s = s
        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key
        self.prob = prob

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        'Generates data containing batch_size samples'
        idx = random.choice(key, self.N, (self.batch_size,), replace = False, p = self.prob)
        s = self.s[idx,:]
        y = self.y[idx,:]
        u = self.u[idx,:]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs
    

def generate_one_operator_training_data(
                                    key, 
                                    minvals, 
                                    maxvals, 
                                    t_int_min, 
                                    t_int_max,
                                    T_params,
                                    a_params,
                                    model,
                                    Temp_air,
                                    exp_params, 
                                    P_ic = 100, 
                                    P_bc = 200,
                                    Q = 1000, 
                                    hold = 'one',
                                    first_subdomain = True): # True: Training first subdomain, False: Training subsequent subdomains.
    
    subkeys = random.split(key, 30)
    
    ramp1 = random.uniform(subkeys[1], minval = minvals[0], maxval = maxvals[0])    # ramp1 (C/min)
    hold1_T = random.uniform(subkeys[2], minval = minvals[1], maxval = maxvals[1])  # hold tempearture 1 (C)
    hold1_d = random.uniform(subkeys[3], minval = minvals[2], maxval = maxvals[2])  # hold duration 1 (min)
    ramp2 = random.uniform(subkeys[4], minval = minvals[3], maxval = maxvals[3])    # ramp2 (C/min)
    hold2_T = random.uniform(subkeys[5], minval = minvals[4], maxval = maxvals[4])  # hold tempearture 2 (C)
    hold2_d = random.uniform(subkeys[6], minval = minvals[5], maxval = maxvals[5])  # hold duration 2 (min)
    htc_bot = random.uniform(subkeys[7], minval = minvals[6], maxval = maxvals[6])  # HTC at the bottom surface (W/m^2K)
    htc_top = random.uniform(subkeys[8], minval = minvals[7], maxval = maxvals[7])  # HTC at the top surface (W/m^2K)
    tool_len = random.uniform(subkeys[9], minval = minvals[8], maxval = maxvals[8]) # Tool length
    u_scaled = np.array([ramp1, hold1_T, hold1_d, ramp2, hold2_T, hold2_d, htc_bot, htc_top, tool_len])
    
    # ICs
    x_ic = random.uniform(key = subkeys[11], shape = (P_ic, 1), minval=0.0, maxval=1.) # Tool IC
    t_ic = np.zeros((P_ic, 1))                                                                  # IC time
    y_ic = np.hstack([t_ic, x_ic])             
    u_ic = np.tile(u_scaled, (P_ic,1)) 

    if not first_subdomain:
        y_ic_end = np.hstack([np.ones((P_ic, 1)), x_ic])
        T_pred_test = model.pred_T(T_params, u_ic, y_ic_end)
        tool_pred_test = model.pred_tool(T_params, u_ic, y_ic_end)
        a_pred_test = model.pred_a(a_params, u_ic, y_ic_end)
        s_ic = np.hstack([T_pred_test.reshape([-1,1]), a_pred_test.reshape([-1,1]), tool_pred_test.reshape([-1,1])]) 
         
    elif first_subdomain:
        s_ic = np.hstack([np.ones((P_ic, 1))*exp_params.T_ini, np.ones((P_ic, 1))*exp_params.alpha_ini, np.ones((P_ic, 1))*exp_params.T_ini])                         # IC values [T, Alpha]

    # BCs
    t_bc = random.uniform(key = subkeys[12], shape = (P_bc, 1), minval=0.0, maxval=1.)                      # BC time
    #t_bc = np.vstack([t_bc1, t_bc2, t_bc3])
    x_bc1 = np.zeros_like(t_bc)                                     # Bottom BC x
    x_bc2 = np.ones_like(t_bc)                                      # Top BC x
    y_bc = np.hstack([t_bc, x_bc1, x_bc2])                           # BC points
    #s_bc = BC_gen(subkeys[0], minvals, maxvals, t_bc, hold = hold)   # Air temperature at BC points
    t_interval = random.uniform(key = subkeys[12], shape = (P_bc, 1), minval=t_int_min*exp_params.t_scaler, maxval=t_int_max*exp_params.t_scaler) # Time interval
    s_bc = Temp_air(exp_params.T_ini, ramp1, hold1_T, hold1_d, ramp2, hold2_T, hold2_d, 1.).two_hold(t_interval)
    
    u_bc = np.tile(u_scaled, (t_bc.shape[0],1))


    # Interface points
    t_inf = random.uniform(key = subkeys[13], shape = (P_bc, 1), minval=0.0, maxval=1.0)     # Inference time
    x_inf = np.ones_like(t_inf)                   # Inference x
    x_inf2 = np.zeros_like(t_inf)                 # Inference x
    y_inf = np.hstack([t_inf, x_inf, x_inf2]) # Inference points
    s_inf = np.zeros((t_inf.shape[0], 1))     # Inference values
    u_inf = np.tile(u_scaled, (t_inf.shape[0],1)) 


    
    # Residual points
    x_r= random.uniform(subkeys[1], minval = 0, maxval = 1, shape = (Q,1))
    t_r = random.uniform(subkeys[15], minval = 0, maxval = 1, shape = (Q,1))
    y_r = np.hstack([t_r, x_r])
    s_r = np.zeros((Q, 1))
    u_r = np.tile(u_scaled, (Q,1))

    
    return  u_bc, y_bc, s_bc,\
            u_ic, y_ic, s_ic,\
            u_r, y_r, s_r,\
            u_inf, y_inf, s_inf,


# Geneate training data corresponding to N input sample
def generate_training_data(
                        key, 
                        minvals, 
                        maxvals,
                        t_int_min, 
                        t_int_max,
                        T_params,
                        a_params,
                        model,
                        Temp_air,
                        exp_params,  
                        N, 
                        P_ic, 
                        P_bc, 
                        Q, 
                        hold,
                        first_subdomain):
    #config.update("jax_enable_x64", True)
    keys = random.split(key, N)
    
    u_bc, y_bc, s_bc,\
    u_ic, y_ic, s_ic,\
    u_r, y_r, s_r,\
    u_inf, y_inf, s_inf = vmap(generate_one_operator_training_data, (0, None, None, None, 
                                                                     None, None,None, None, 
                                                                     None, None,None, None,None,None,None))(keys, minvals, maxvals,
                                                                                           t_int_min, t_int_max, T_params,
                                                                                           a_params, model, Temp_air, exp_params, P_ic, P_bc, Q, hold, first_subdomain)

    u_bc = np.float32(u_bc.reshape(N * P_bc,-1))
    y_bc = np.float32(y_bc.reshape(N * P_bc,-1))
    s_bc = np.float32(s_bc.reshape(N * P_bc,-1))
    
    u_ic = np.float32(u_ic.reshape(N * P_ic,-1))
    y_ic = np.float32(y_ic.reshape(N * P_ic,-1))
    s_ic = np.float32(s_ic.reshape(N * P_ic ,-1))

    u_r = np.float32(u_r.reshape(N * Q,-1))
    y_r = np.float32(y_r.reshape(N * Q,-1))
    s_r = np.float32(s_r.reshape(N * Q,-1))
    
    u_inf = np.float32(u_inf.reshape(N * P_bc,-1))
    y_inf = np.float32(y_inf.reshape(N * P_bc,-1))
    s_inf = np.float32(s_inf.reshape(N * P_bc,-1))
    
    return  u_bc, y_bc, s_bc,\
            u_ic, y_ic, s_ic,\
            u_r, y_r, s_r,\
            u_inf, y_inf, s_inf