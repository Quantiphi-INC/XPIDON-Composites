import jax.numpy as np
"""
Create a class for the air temperature (BCs) This Class will return 
"""
class Temp_air:
    def __init__(self, T_ini, ramp1, T_hold1, T_hold_duration1, ramp2, T_hold2, T_hold_duration2, t_end):
        """
        Initializes the air temperature profile for the composite curing cycle.

        Parameters:
        - T_ini (float): Initial temperature (°C).
        - ramp1 (float): Heating rate to first hold (°C/min).
        - T_hold1 (float): First hold temperature (°C).
        - T_hold_duration1 (float): Duration of first hold phase (min).
        - ramp2 (float): Heating rate to second hold (°C/min).
        - T_hold2 (float): Second hold temperature (°C).
        - T_hold_duration2 (float): Duration of second hold phase (min).
        - t_end (float): Total curing time (min).
        """
        self.T_ini = T_ini
        self.ramp1 = ramp1
        self.T_hold1 = T_hold1
        self.T_hold_duration1 = T_hold_duration1
        self.t_end = t_end
        self.ramp_end1 = (self.T_hold1 - self.T_ini)/self.ramp1
        self.t_hold_end1 = self.T_hold_duration1 + self.ramp_end1
        self.ramp2 = ramp2
        self.T_hold2 = T_hold2
        self.T_hold_duration2 = T_hold_duration2
        self.ramp_end2 = (self.T_hold2 - self.T_hold1)/self.ramp2 + self.t_hold_end1
        self.t_hold_end2 = self.T_hold_duration2 + self.ramp_end2
        
    
    def two_hold(self, t):
        """
        Computes the temperature at time t for a two-hold curing cycle.

        Returns:
        - Temperature at time t (°C).
        """
        ramp_rate_3 = (self.T_ini - self.T_hold2)/(self.t_end - self.t_hold_end2)
        return np.where(t < self.ramp_end1, self.T_ini + t*self.ramp1, #  Ramp 1
                        np.where(t < self.t_hold_end1, self.T_hold1, # Hold 1
                                 np.where(t < self.ramp_end2, self.T_hold1 + (t - self.t_hold_end1)*self.ramp2, # Ramp 2
                                          np.where(t < self.t_hold_end2, self.T_hold2, # Hold 2
                                                    self.T_hold2 + (t - self.t_hold_end2)*ramp_rate_3) # Ramp 3
                                            )))
    