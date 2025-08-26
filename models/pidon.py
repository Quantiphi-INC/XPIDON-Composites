import jax
import optax
import jax.numpy as np
from jax import random, grad, vmap, jit, value_and_grad
from jax.example_libraries import optimizers
from jax.experimental.ode import odeint
from scipy.spatial import cKDTree
from functools import partial
from .mlp import MLP


"""
The `XPIDON` class is the main implementation of the Extended Physics-Informed DeepONet (XPIDON). 
It focuses on defining the neural network architecture and its forward prediction
capabilities. All physics-informed loss calculations are delegated to the
`XPIDONLoss` class.
"""

# Portions of this code are derived from:
# Physics-informed-DeepONets (Predictive Intelligence Lab)
# https://github.com/PredictiveIntelligenceLab/Physics-informed-DeepONets

class XPIDON:
    
    def __init__(
             self,
             phy_params,
             exp_params,
             branch_layers, 
             trunk_layers, 
             nd_layers,
             t_min,
             t_max):
        """
        Initializes the XPIDON model.

        Parameters
        ----------
        phy_params : dict
            Dictionary containing physical parameters.
        exp_params : dict
            Dictionary containing experimental parameters.
        branch_layers : list[int]
            Layer sizes for the branch network.
        trunk_layers : list[int]
            Layer sizes for the trunk network.
        nd_layers : list[int]
            Layer sizes for the nonlinear decoder network.
        t_min : float
            Minimum time of the current subdomain.
        t_max : float
            Maximum time of the current subdomain.
        """
        # Store physical and experimental parameters
        self.phy_params = phy_params
        self.exp_params = exp_params

        # Store subdomain time boundaries
        self.t_min = t_min                                                            
        self.t_max = t_max
        self.del_t = t_max - t_min                                                            

        # Initialize the neural network components (Branch, Trunk, Nonlinear Decoder)
        # These MLPs are shared across different DeepONets (T, tool, alpha)
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)
        self.nd_init, self.nd_apply = MLP(nd_layers, activation=np.tanh)

        # Initialize parameters for the shared MLP components
        branch_params = self.branch_init(random.PRNGKey(123456))
        trunk_params = self.trunk_init(random.PRNGKey(123456))
        nd_params = self.nd_init(random.PRNGKey(123456))

        # Group parameters for the DeepONets:
        # T_params: (branch_T, trunk_T, nd_T, branch_tool, trunk_tool, nd_tool)
        # This structure allows for storing DeepONets for part temperature and tool temperature
        self.T_params = (branch_params, trunk_params, nd_params, branch_params, trunk_params, nd_params)
        
        # a_params: (branch_alpha, trunk_alpha, nd_alpha)
        # This is for the DeepONet predicting degree of cure (alpha)
        self.a_params = (branch_params, trunk_params, nd_params)
        
        # Initialize Optax optimizer with an exponential decay learning rate schedule
        self.schedule = optax.exponential_decay(init_value=1e-3,transition_steps=2000,
                                    decay_rate=0.9,staircase=False, end_value = 1e-7)

        # Initialize separate optimizers and their states for T and alpha parameters
        self.optimizer_T = optax.adam(learning_rate=self.schedule)
        self.opt_state_T = self.optimizer_T.init(self.T_params)

        self.optimizer_a = optax.adam(learning_rate=self.schedule)
        self.opt_state_a = self.optimizer_a.init(self.a_params)
    
    def get_weights(self):
        """
        Returns the current parameters (weights) of the temperature and alpha DeepONets.

        Returns
        -------
        tuple
            A tuple containing (T_params, a_params).
        """
        return self.T_params, self.a_params
        
    # --- Neural Network Prediction Functions ---
    # These functions define the forward pass of the DeepONet for different outputs.

    def op_net_T(self, params, u, t, x):
        """
        Performs the forward pass for the part temperature DeepONet.

        Parameters
        ----------
        params : tuple
            Parameters for the temperature DeepONets (branch, trunk, nd).
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.

        Returns
        -------
        jax.numpy.ndarray
            Predicted part temperature.
        """
        # Unpack parameters for the part temperature DeepONet
        branch_params, trunk_params, nd_params, _, _, _ = params
        y = np.stack([t, x]) # Stack spatio-temporal coordinates for the trunk network
        
        B = self.branch_apply(branch_params, u) # Branch network output
        T = self.trunk_apply(trunk_params, y)   # Trunk network output
        
        out_total = B * T # Element-wise product of branch and trunk outputs
        out = self.nd_apply(nd_params, out_total) # Nonlinear decoder output
        
        # Apply a softplus activation and add initial temperature for physical bounds
        out_corrected = jax.nn.softplus(out[0]) + self.exp_params.T_ini
        return out_corrected
        
    def op_net_tool(self, params, u, t, x):
        """
        Performs the forward pass for the tool temperature DeepONet.

        Parameters
        ----------
        params : tuple
            Parameters for the temperature DeepONets (branch, trunk, nd).
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.

        Returns
        -------
        jax.numpy.ndarray
            Predicted tool temperature.
        """
        # Unpack parameters for the tool temperature DeepONet
        _, _, _, branch_params, trunk_params, nd_params = params
        y = np.stack([t, x])
        
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        
        out_total = B * T
        out = self.nd_apply(nd_params, out_total)
        
        # Apply a softplus activation and add initial temperature for physical bounds
        out_corrected = jax.nn.softplus(out[0]) + self.exp_params.T_ini
        return out_corrected
    
    def op_net_alpha(self, params, u, t, x):
        """
        Performs the forward pass for the degree of cure (alpha) DeepONet.

        Parameters
        ----------
        params : tuple
            Parameters for the alpha DeepONet (branch, trunk, nd).
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.

        Returns
        -------
        jax.numpy.ndarray
            Predicted degree of cure (alpha).
        """
        # Unpack parameters for the alpha DeepONet
        branch_params, trunk_params, nd_params = params
        y = np.stack([t, x])
        
        B = self.branch_apply(branch_params, u)
        T = self.trunk_apply(trunk_params, y)
        
        out_total = B * T
        out = self.nd_apply(nd_params, out_total)
        
        # Apply sigmoid activation and scale to the expected alpha range
        out_corrected = self.exp_params.a_diff*jax.nn.sigmoid(out[0]) + self.exp_params.alpha_ini
        return out_corrected

    # --- Prediction Functions for Evaluation ---
    # These functions are used to get predictions from the trained model.

    @partial(jit, static_argnums=(0,))
    def pred_T(self, params, U_star, Y_star):
        """
        Evaluates the part temperature DeepONet at given test points.

        Parameters
        ----------
        params : tuple
            Parameters of the part temperature DeepONet.
        U_star : jax.numpy.ndarray
            Input function (design variables) for test points.
        Y_star : jax.numpy.ndarray
            Spatio-temporal coordinates for test points (t, x).

        Returns
        -------
        jax.numpy.ndarray
            Predicted part temperatures at test points.
        """
        s_pred_fn = vmap(self.op_net_T, (None, 0, 0, 0)) # Vectorize over multiple test points
        s_star = s_pred_fn(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_star
    
    @partial(jit, static_argnums=(0,))
    def pred_tool(self, params, U_star, Y_star):
        """
        Evaluates the tool temperature DeepONet at given test points.

        Parameters
        ----------
        params : tuple
            Parameters of the tool temperature DeepONet.
        U_star : jax.numpy.ndarray
            Input function (design variables) for test points.
        Y_star : jax.numpy.ndarray
            Spatio-temporal coordinates for test points (t, x).

        Returns
        -------
        jax.numpy.ndarray
            Predicted tool temperatures at test points.
        """
        s_pred_fn = vmap(self.op_net_tool, (None, 0, 0, 0))
        s_star = s_pred_fn(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_star
    
    @partial(jit, static_argnums=(0,))
    def pred_a(self, params, U_star, Y_star):
        """
        Evaluates the degree of cure (alpha) DeepONet at given test points.

        Parameters
        ----------
        params : tuple
            Parameters of the alpha DeepONet.
        U_star : jax.numpy.ndarray
            Input function (design variables) for test points.
        Y_star : jax.numpy.ndarray
            Spatio-temporal coordinates for test points (t, x).

        Returns
        -------
        jax.numpy.ndarray
            Predicted alpha values at test points.
        """
        s_pred_fn = vmap(self.op_net_alpha, (None, 0, 0, 0))
        s_star = s_pred_fn(params, U_star, Y_star[:,0], Y_star[:,1])
        return s_star
    
    # --- Utility Functions ---

    def denorm(self, t):
        """
        Denormalizes a time value from the [0, 1] range of the subdomain
        back to its original physical time range [t_min, t_max].

        Parameters
        ----------
        t : jax.numpy.ndarray
            Normalized time value.

        Returns
        -------
        jax.numpy.ndarray
            Denormalized time value.
        """
        return t*(self.t_max - self.t_min) + self.t_min
    
    def get_t_params(self):
        """
        Returns the time parameters of the current subdomain.

        Returns
        -------
        tuple
            A tuple containing (t_min, t_max, del_t).
        """
        return self.t_min, self.t_max, self.del_t