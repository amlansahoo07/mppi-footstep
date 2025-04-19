import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from gym_quadruped.utils.quadruped_utils import LegsAttr

@dataclass
class GaitParameters:
    """Parameters defining gait constraints"""
    min_stance_duration: float = 0.3  # seconds
    max_stance_duration: float = 0.8  # seconds
    min_swing_duration: float = 0.2   # seconds
    max_swing_duration: float = 0.4   # seconds
    # min_support_legs: int = 2         # minimum legs in stance
    enforce_diagonal_coordination: bool = False
    
class RandomGaitGenerator:
    """Generates random but feasible gait patterns"""
    
    def __init__(self, params: Optional[GaitParameters] = None):
        self.params = params or GaitParameters()
        self.n_legs = 4
        self.leg_states = np.ones(self.n_legs)  # 1=stance, 0=swing
        self.leg_timers = np.zeros(self.n_legs)
        self.previous_sequence = None
        
    def generate_contact_sequence(self, 
                                horizon: int, 
                                dt: float,
                                current_contacts: Optional[np.ndarray] = None) -> np.ndarray:
        """Generate random but feasible contact sequence
        
        Args:
            horizon: Number of timesteps to generate
            dt: Timestep duration
            current_contacts: Current contact state (optional)
            
        Returns:
            contact_sequence: Array of shape (4, horizon) with binary contact states
        """
        # Initialize sequence
        sequence = np.ones((self.n_legs, horizon))
        
        # Use current contacts if provided
        if current_contacts is not None:
            self.leg_states = current_contacts.copy()
            
        # Convert time constraints to steps
        min_stance_steps = int(self.params.min_stance_duration / dt)
        max_stance_steps = int(self.params.max_stance_duration / dt)
        min_swing_steps = int(self.params.min_swing_duration / dt)
        max_swing_steps = int(self.params.max_swing_duration / dt)
        
        # Generate sequence
        for t in range(horizon):
            for leg in range(self.n_legs):
                if self.leg_states[leg] == 1:  # In stance
                    if (self.leg_timers[leg] >= min_stance_steps and 
                        self._can_start_swing(self.leg_states, leg)):
                        # Randomly decide to start swing
                        if np.random.random() > 0.7:  # 30% chance to start swing
                            self.leg_states[leg] = 0
                            self.leg_timers[leg] = 0
                            
                else:  # In swing
                    if self.leg_timers[leg] >= min_swing_steps:
                        self.leg_states[leg] = 1  # Return to stance
                        self.leg_timers[leg] = 0
                        
                self.leg_timers[leg] += 1
                sequence[leg, t] = self.leg_states[leg]
                
        self.previous_sequence = sequence
        return sequence
    
    def _can_start_swing(self, current_states: np.ndarray, leg_idx: int) -> bool:
        """Check if transitioning leg to swing maintains stability"""
        # Keep diagonal leg coordination (avoids flying phases):
        if self.params.enforce_diagonal_coordination:
            if leg_idx in [0, 3] and current_states[3-leg_idx] == 0:
                return False  # Don't swing both diagonal legs
            if leg_idx in [1, 2] and current_states[3-leg_idx] == 0:
                return False
            
        return True
    
    def reset(self):
        """Reset generator state"""
        self.leg_states = np.ones(self.n_legs)
        self.leg_timers = np.zeros(self.n_legs)
        self.previous_sequence = None