import jax
import optax
import jax.numpy as np
from jax import random, grad, vmap, jit, value_and_grad
from functools import partial
"""
The `XPIDONLoss` class encapsulates all extended physics-informed residual calculations
and the aggregation of various loss terms for the XPIDON model.
It separates the concerns of defining the physical laws and computing their
residuals from the neural network's architecture and forward pass.
"""
class XPIDONLoss:
    def __init__(self, model_instance, phy_params, exp_params, t_min, t_max):
        """
        Initializes the XPIDONLoss calculator.

        Parameters
        ----------
        model_instance : XPIDON
            An instance of the XPIDON model, used to access its neural network
            prediction functions (op_net_T, op_net_tool, op_net_alpha).
        phy_params : dict
            Dictionary containing physical parameters (e.g., material properties).
        exp_params : dict
            Dictionary containing experimental parameters (e.g., initial temperature, scalers).
        t_min : float
            Minimum time of the current subdomain.
        t_max : float
            Maximum time of the current subdomain.
        """
        self.model = model_instance  # Reference to the XPIDON model for its prediction functions
        self.phy_params = phy_params
        self.exp_params = exp_params
        self.t_min = t_min
        self.t_max = t_max
        self.del_t = t_max - t_min # Time duration of the current subdomain

    # --- Physics-based Residual Functions ---
    # These functions define the governing equations and boundary/interface conditions.
    # They take model parameters and input coordinates, and return the residual (error)
    # of the equation at those points.

    def cure_kinetics(self, T_params, a_params, u, t, x):
        """
        Calculates the cure kinetics term (rate of cure) based on temperature and alpha.
        This is a key component of the ODE governing the degree of cure.

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        a_params : tuple
            Parameters of the alpha (degree of cure) DeepONet.
        u : jax.numpy.ndarray
            Input function (design variables) for the DeepONet.
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.

        Returns
        -------
        jax.numpy.ndarray
            The calculated cure kinetics term.
        """
        # Predict temperature and alpha using the XPIDON model's networks
        T = self.model.op_net_T(T_params, u, t, x)
        T_k = T / self.exp_params.T_scaler + 273.15 # Convert temperature to Kelvin
        alpha = self.model.op_net_alpha(a_params, u, t, x)

        # Apply the cure kinetics formula
        num = (self.phy_params.A * self.del_t) * np.exp(-self.phy_params.dE/(self.phy_params.RR*T_k))
        den = 1. + np.exp(self.phy_params.CC*(alpha - (self.phy_params.ALC + self.phy_params.ALCT*T_k)))
        cure_term = (num/den)*(alpha**self.phy_params.MM)*((1-alpha)**self.phy_params.NN)
        return cure_term
    
    def ode_net(self, T_params, a_params, u, t, x):
        """
        Calculates the residual of the Ordinary Differential Equation (ODE)
        governing the degree of cure (alpha).
        ODE: d(alpha)/dt - cure_kinetics = 0

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        a_params : tuple
            Parameters of the alpha (degree of cure) DeepONet.
        u : jax.numpy.ndarray
            Input function (design variables) for the DeepONet.
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.

        Returns
        -------
        jax.numpy.ndarray
            The ODE residual.
        """
        cure_term = self.cure_kinetics(T_params, a_params, u, t, x)
        # Calculate the time derivative of alpha using JAX's automatic differentiation
        a_t = grad(self.model.op_net_alpha, argnums=2)(a_params, u, t, x)
        f = a_t - cure_term
        return f

    def pde_net(self, T_params, a_params, u, t, x, b_val):
        """
        Calculates the residual of the Partial Differential Equation (PDE)
        governing the part temperature.
        PDE: dT/dt - alpha_c * d^2T/dx^2 - beta * cure_kinetics = 0

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        a_params : tuple
            Parameters of the alpha (degree of cure) DeepONet.
        u : jax.numpy.ndarray
            Input function (design variables) for the DeepONet.
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.
        b_val : float
            Beta value, a coefficient related to heat generation from curing.

        Returns
        -------
        jax.numpy.ndarray
            The PDE residual for part temperature.
        """
        part_len_var = self.phy_params.part_len
        ac_net = (self.phy_params.a_c * self.del_t) / (part_len_var**2) # Thermal diffusivity term
        b_net = b_val
        cure_term = self.cure_kinetics(T_params, a_params, u, t, x)
        
        # Calculate time and second spatial derivatives of temperature using JAX's AD
        T_t = grad(self.model.op_net_T, argnums=2)(T_params, u, t, x)
        T_xx= grad(grad(self.model.op_net_T, argnums=3), argnums=3)(T_params, u, t, x)
        
        f = T_t - ac_net*T_xx - b_net*cure_term
        return f

    def pde_tool_net(self, T_params, u, t, x):
        """
        Calculates the residual of the PDE governing the tool temperature.
        PDE: dT_tool/dt - alpha_c_tool * d^2T_tool/dx^2 = 0 (assuming no internal heat generation)

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet (specifically for tool temperature).
        u : jax.numpy.ndarray
            Input function (design variables) for the DeepONet.
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate.

        Returns
        -------
        jax.numpy.ndarray
            The PDE residual for tool temperature.
        """
        tool_len_var = u[8]/self.exp_params.len_scaler # Scaled tool length
        ac_net = (self.phy_params.a_c_t * self.del_t) / (tool_len_var**2) # Thermal diffusivity term for tool
        
        # Calculate time and second spatial derivatives of tool temperature
        T_t = grad(self.model.op_net_tool, argnums=2)(T_params, u, t, x)
        T_xx= grad(grad(self.model.op_net_tool, argnums=3), argnums=3)(T_params, u, t, x)
        
        f = T_t - ac_net*T_xx
        return f

    def bcb_net(self, params, ub, u, t, x):
        """
        Calculates the residual for the bottom boundary condition (tool side).
        This typically involves a convection/conduction balance.

        Parameters
        ----------
        params : tuple
            Parameters of the relevant DeepONet (tool temperature).
        ub : jax.numpy.ndarray
            Boundary condition value (e.g., ambient temperature).
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate at the boundary.

        Returns
        -------
        jax.numpy.ndarray
            The bottom boundary condition residual.
        """
        h_val = u[6]/self.exp_params.h_scaler # Scaled heat transfer coefficient
        tool_len_var = u[8]/self.exp_params.len_scaler # Scaled tool length
        C = self.phy_params.k_bot/(h_val*tool_len_var) # Conduction/convection ratio
        
        s = self.model.op_net_tool(params, u, t, x) # Predicted tool temperature at boundary
        T_x = grad(self.model.op_net_tool, argnums=3)(params, u, t, x) # Spatial derivative at boundary
        
        f = (ub - s) + C * T_x # Residual of the boundary condition equation
        return f
    
    def bct_net(self, params, ub, u, t, x):
        """
        Calculates the residual for the top boundary condition (part side).
        Similar to bcb_net, but for the part.

        Parameters
        ----------
        params : tuple
            Parameters of the relevant DeepONet (part temperature).
        ub : jax.numpy.ndarray
            Boundary condition value (e.g., ambient temperature).
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x : jax.numpy.ndarray
            Spatial coordinate at the boundary.

        Returns
        -------
        jax.numpy.ndarray
            The top boundary condition residual.
        """
        h_val = u[7]/self.exp_params.h_scaler # Scaled heat transfer coefficient
        part_len_var = self.phy_params.part_len
        C = self.phy_params.k_top/(h_val*part_len_var)
        
        s = self.model.op_net_T(params, u, t, x) # Predicted part temperature at boundary
        T_x = grad(self.model.op_net_T, argnums=3)(params, u, t, x) # Spatial derivative at boundary
        
        f = (s - ub) + C * T_x # Residual of the boundary condition equation
        return f
    
    def interface_net(self, T_params, u, t, x_bot, x_top):
        """
        Calculates the residual for the temperature continuity at the interface
        between the tool and the part.
        Ensures T_tool(interface) = T_part(interface).

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x_bot : jax.numpy.ndarray
            Spatial coordinate of the tool side of the interface.
        x_top : jax.numpy.ndarray
            Spatial coordinate of the part side of the interface.

        Returns
        -------
        tuple of jax.numpy.ndarray
            Residuals for temperature continuity from both sides of the interface.
        """
        T_b = self.model.op_net_tool(T_params, u, t, x_bot) # Tool temp at interface
        T_t = self.model.op_net_T(T_params, u, t, x_top) # Part temp at interface
        f_av = (T_b + T_t)/2 # Average temperature at interface
        f1 = T_b - f_av # Residual for tool side
        f2 = T_t - f_av # Residual for part side
        return f1, f2
    
    def flux_net(self, T_params, u, t, x_bot, x_top):
        """
        Calculates the residual for heat flux continuity at the interface.
        Ensures k_tool * dT_tool/dx = k_part * dT_part/dx at the interface.

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        u : jax.numpy.ndarray
            Input function (design variables).
        t : jax.numpy.ndarray
            Time coordinate.
        x_bot : jax.numpy.ndarray
            Spatial coordinate of the tool side of the interface.
        x_top : jax.numpy.ndarray
            Spatial coordinate of the part side of the interface.

        Returns
        -------
        tuple of jax.numpy.ndarray
            Residuals for heat flux continuity from both sides of the interface.
        """
        tool_len_var = u[8]/self.exp_params.len_scaler
        part_len_var =  self.phy_params.part_len
        
        # Spatial derivatives of temperature at the interface for both tool and part
        Tb_x = grad(self.model.op_net_tool, argnums=3)(T_params, u, t, x_bot)
        Tt_x = grad(self.model.op_net_T, argnums=3)(T_params, u, t, x_top)
        
        f1 = (part_len_var/tool_len_var)*(self.phy_params.k_bot * Tb_x) # Flux from tool side
        f2 = (self.phy_params.k_top * Tt_x) # Flux from part side
        f_av = (f1 + f2)/2 # Average flux at interface
        return f1 - f_av, f2 - f_av # Residuals for flux continuity

    # --- Individual Loss Term Calculations ---
    # These methods compute the mean squared error (MSE) for each specific
    # physics-informed constraint or initial condition.

    @partial(jit, static_argnums=(0,))
    def loss_ics_tool(self, params, batch):
        """Calculates the initial condition loss for tool temperature."""
        inputs, outputs = batch
        u, y = inputs
        tool_pred_fn = vmap(self.model.op_net_tool, (None, 0, 0, 0))
        s_pred = tool_pred_fn(params, u, y[:,0], y[:,1])
        loss_ics = np.mean((outputs[:,2].flatten() - s_pred.flatten())**2)
        return loss_ics
    
    @partial(jit, static_argnums=(0,))
    def loss_ics_T(self, params, batch):
        """Calculates the initial condition loss for part temperature."""
        inputs, outputs = batch
        u, y = inputs
        s_pred_fn = vmap(self.model.op_net_T, (None, 0, 0, 0))
        s_pred = s_pred_fn(params, u, y[:,0], y[:,1])
        loss_ics = np.mean((outputs[:,0].flatten() - s_pred.flatten())**2)
        return loss_ics
    
    @partial(jit, static_argnums=(0,))
    def loss_ics_a(self, params, batch):
        """Calculates the initial condition loss for degree of cure (alpha)."""
        inputs, outputs = batch
        u, y = inputs
        a_pred_fn = vmap(self.model.op_net_alpha, (None, 0, 0, 0))
        a_pred = a_pred_fn(params, u, y[:,0], y[:,1])
        loss_ics = np.mean((outputs[:,1].flatten() - a_pred.flatten())**2)
        return loss_ics
    
    @partial(jit, static_argnums=(0,))
    def loss_res(self, params_T, params_a, batch, b_val):
        """Calculates the PDE residual loss for part temperature."""
        inputs, _ = batch # Outputs are not needed for residual loss (should be zero)
        u, y = inputs
        res_pred_fn = vmap(self.pde_net, (None, None, 0, 0, 0, None)) # Vmap over inputs
        res_pred = res_pred_fn(params_T, params_a, u, y[:,0], y[:,1], b_val)
        return np.mean(res_pred**2)
    
    @partial(jit, static_argnums=(0,))
    def loss_res_tool(self, params, batch):
        """Calculates the PDE residual loss for tool temperature."""
        inputs, _ = batch
        u, y = inputs
        res_tool_pred_fn = vmap(self.pde_tool_net, (None, 0, 0, 0))
        res_tool_pred = res_tool_pred_fn(params, u, y[:,0], y[:,1])
        return np.mean(res_tool_pred**2)
    
    @partial(jit, static_argnums=(0,))
    def loss_ode(self, params_T, params_a, batch):
        """Calculates the ODE residual loss for degree of cure (alpha)."""
        inputs, _ = batch
        u, y = inputs
        ode_pred_fn = vmap(self.ode_net, (None, None, 0, 0, 0))
        ode_pred = ode_pred_fn(params_T, params_a, u, y[:,0], y[:,1])
        return np.mean(ode_pred**2)
    
    @partial(jit, static_argnums=(0,))
    def loss_bct(self, params, batch):
        """Calculates the boundary condition loss for the top surface (part)."""
        inputs, outputs = batch # outputs here are the boundary values (e.g., ambient temp)
        u, y = inputs
        bct_fn = vmap(self.bct_net, (None, 0, 0, 0, 0))
        bc_pred_top = bct_fn(params, outputs, u, y[:,0], y[:,2])
        return np.mean(bc_pred_top**2)
    
    @partial(jit, static_argnums=(0,))
    def loss_bcb(self, params, batch):
        """Calculates the boundary condition loss for the bottom surface (tool)."""
        inputs, outputs = batch
        u, y = inputs
        bcb_fn = vmap(self.bcb_net, (None, 0, 0, 0, 0))
        bc_pred_bot = bcb_fn(params, outputs, u, y[:,0], y[:,1])
        return np.mean(bc_pred_bot**2)
    
    @partial(jit, static_argnums=(0,))
    def loss_inf(self, params, batch):
        """Calculates the interface temperature continuity loss."""
        inputs, _ = batch
        u, y = inputs
        inf_fn = vmap(self.interface_net, (None, 0, 0, 0, 0))
        inf_pred1, inf_pred2 = inf_fn(params, u, y[:,0], y[:,1], y[:,2])
        return np.mean(inf_pred1**2) + np.mean(inf_pred2**2)
    
    @partial(jit, static_argnums=(0,))
    def loss_flux(self,params, batch):
        """Calculates the interface heat flux continuity loss."""
        inputs, _ = batch
        u, y = inputs
        flux_fn = vmap(self.flux_net, (None, 0, 0, 0, 0))
        flux_pred1, flux_pred2 = flux_fn(params, u, y[:,0], y[:,1], y[:,2])
        return np.mean(flux_pred1**2) + np.mean(flux_pred2**2)
    
    # --- Total Loss Aggregation Functions ---
    # These functions combine the individual loss terms, often with weighting,
    # to form the total loss for optimization.

    @partial(jit, static_argnums=(0,))
    def total_loss_temp(self, T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch, weights, b_val):
        """
        Calculates the total loss for optimizing the temperature (T) DeepONet.
        This loss includes initial conditions, boundary conditions, PDE residuals,
        and interface conditions related to temperature.
        The alpha parameters are treated as fixed (stop_gradient) during T optimization.

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        a_params : tuple
            Parameters of the alpha (degree of cure) DeepONet.
        ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch : tuple
            Batches of training data for different loss components.
        weights : tuple
            Weights for different loss terms (w_ic, w_bcb, w_bct).
        b_val : float
            Beta value for the PDE.

        Returns
        -------
        jax.numpy.ndarray
            The total temperature loss.
        """
        w_ic, w_bcb, w_bct = weights
        
        # Calculate individual loss components
        loss_ics_T_val = self.loss_ics_T(T_params, ics_batch)
        loss_ics_tool_val = self.loss_ics_tool(T_params, ics_batch)
        loss_bcb_val = self.loss_bcb(T_params, bcs_batch)
        loss_bct_val = self.loss_bct(T_params, bcs_batch)
        
        # PDE residual for temperature depends on alpha, but alpha's parameters are fixed during T optimization (jax.lax.stop_gradient)
        loss_res_val = self.loss_res(T_params, jax.lax.stop_gradient(a_params), res_batch, b_val)
        loss_res_tool_val = self.loss_res_tool(T_params, res_batch)
        
        loss_inf_val = self.loss_inf(T_params, inf_batch)
        loss_flux_val = self.loss_flux(T_params, inf_batch)
        
        # Combine losses with their respective weights
        loss = (w_ic * loss_ics_T_val + loss_ics_tool_val + 
                w_bcb * loss_bcb_val + w_bct * loss_bct_val + 
                loss_res_val + loss_res_tool_val + 
                loss_inf_val + loss_flux_val)
        return loss
    
    @partial(jit, static_argnums=(0,))
    def total_loss_a(self, T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch, weights):
        """
        Calculates the total loss for optimizing the degree of cure (alpha) DeepONet.
        This loss includes initial conditions and the ODE residual for alpha.
        The temperature parameters are treated as fixed (stop_gradient) during alpha optimization.

        Parameters
        ----------
        T_params : tuple
            Parameters of the temperature DeepONet.
        a_params : tuple
            Parameters of the alpha (degree of cure) DeepONet.
        ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch : tuple
            Batches of training data for different loss components.
        weights : tuple
            Weights for different loss terms (w_ic_a, w_ode).

        Returns
        -------
        jax.numpy.ndarray
            The total alpha loss.
        """
        w_ic_a, w_ode = weights
        
        # Calculate individual loss components
        loss_ics_a_val = self.loss_ics_a(a_params, ics_batch)
        
        # ODE residual for alpha depends on temperature, but temperature's parameters
        # are fixed during alpha optimization (jax.lax.stop_gradient)
        loss_o_val = self.loss_ode(jax.lax.stop_gradient(T_params), a_params, ode_batch)
        
        # Combine losses with their respective weights
        loss = w_ic_a * loss_ics_a_val + w_ode * loss_o_val
        return loss
    
    # --- Optimization Step Functions ---
    # These functions perform a single optimization step for the model parameters.
    # They delegate the loss calculation to the `XPIDONLoss` instance.

    @partial(jit, static_argnums=(0,))
    def step_temp(self, opt_state_T, T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch, weights, b_val):
        """
        Performs one optimization step for the temperature (T) DeepONet parameters.

        Parameters
        ----------
        opt_state_T : optax.OptState
            Current state of the T optimizer.
        T_params : tuple
            Current parameters of the T DeepONet.
        a_params : tuple
            Current parameters of the alpha DeepONet (used for loss calculation, but not updated).
        ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch : tuple
            Batches of training data.
        weights : tuple
            Weights for the T loss terms.
        b_val : float
            Beta value for the PDE.

        Returns
        -------
        tuple
            Updated (opt_state_T, T_params, loss_value).
        """
        # Calculate loss and its gradient with respect to T_params
        # The actual loss computation is delegated to self.loss_calculator.total_loss_temp
        loss, g = value_and_grad(self.total_loss_temp, argnums=0)(
            T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch, weights, b_val
        )
        # Apply gradients to update T_params
        updates, opt_state_T = self.model.optimizer_T.update(g, opt_state_T, T_params)
        T_params = optax.apply_updates(T_params, updates)
        return opt_state_T, T_params, loss

    @partial(jit, static_argnums=(0,))
    def step_alpha(self, opt_state_a, T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch, weights_ode):
        """
        Performs one optimization step for the degree of cure (alpha) DeepONet parameters.

        Parameters
        ----------
        opt_state_a : optax.OptState
            Current state of the alpha optimizer.
        T_params : tuple
            Current parameters of the T DeepONet (used for loss calculation, but not updated).
        a_params : tuple
            Current parameters of the alpha DeepONet.
        ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch : tuple
            Batches of training data.
        weights_ode : tuple
            Weights for the alpha loss terms.

        Returns
        -------
        tuple
            Updated (opt_state_a, a_params, loss_value).
        """
        # Calculate loss and its gradient with respect to a_params
        # The actual loss computation is delegated to self.loss_calculator.total_loss_a
        loss, g = value_and_grad(self.total_loss_a, argnums=1)(
            T_params, a_params, ics_batch, bcs_batch, res_batch, ode_batch, inf_batch, ext_batch, weights_ode
        )
        # Apply gradients to update a_params
        updates, opt_state_a = self.model.optimizer_a.update(g, opt_state_a, a_params)
        a_params = optax.apply_updates(a_params, updates)
        return opt_state_a, a_params, loss