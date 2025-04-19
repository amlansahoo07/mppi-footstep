import numpy as np
from typing import Optional, Union

from quadruped_pympc.helpers.quadruped_utils import GaitType
from quadruped_pympc.helpers.periodic_gait_generator import PeriodicGaitGenerator
from quadruped_pympc.helpers.random_gait_generator import RandomGaitGenerator, GaitParameters
from gym_quadruped.utils.quadruped_utils import LegsAttr

class GaitAdapter:
    """Adapter interface between RandomGaitGenerator and PeriodicGaitGenerator
    
    This class provides a unified interface to use either periodic or random gait
    generation while maintaining compatibility with the existing system.
    """
    
    def __init__(self, 
                 duty_factor: float, 
                 step_freq: float, 
                 gait_type: GaitType, 
                 horizon: int,
                 use_random_gait: bool = False,
                 random_gait_params: Optional[GaitParameters] = None):
        """Initialize the gait adapter
        
        Args:
            duty_factor: Ratio of stance time to stride time
            step_freq: Frequency of the gait in Hz
            gait_type: Type of gait pattern (only used for periodic gaits)
            horizon: Prediction horizon length
            use_random_gait: Whether to use random gait generation
            random_gait_params: Parameters for random gait generator
        """
        self.use_random_gait = use_random_gait
        self.horizon = horizon
        self.duty_factor = duty_factor
        self.step_freq = step_freq
        
        # Initialize both generators
        self.periodic_generator = PeriodicGaitGenerator(
            duty_factor=duty_factor,
            step_freq=step_freq,
            gait_type=gait_type,
            horizon=horizon
        )
        
        # Set random generator parameters based on periodic parameters if not provided
        if random_gait_params is None:
            # Convert duty factor to stance/swing durations
            period = 1.0 / step_freq if step_freq > 0 else 1.0
            min_stance_duration = duty_factor * period * 0.8  # 80% of expected stance
            max_stance_duration = duty_factor * period * 1.2  # 120% of expected stance
            min_swing_duration = (1-duty_factor) * period * 0.8  # 80% of expected swing
            max_swing_duration = (1-duty_factor) * period * 1.2  # 120% of expected swing
            
            random_gait_params = GaitParameters(
                min_stance_duration=min_stance_duration,
                max_stance_duration=max_stance_duration,
                min_swing_duration=min_swing_duration,
                max_swing_duration=max_swing_duration,
                enforce_diagonal_coordination=True  # Start with coordination on
            )
        
        self.random_generator = RandomGaitGenerator(params=random_gait_params)
        
        # Store the current contact state
        self.current_contact = np.ones(4)
    
    def compute_contact_sequence(self, 
                            contact_sequence_dts=None, 
                            contact_sequence_lenghts=None) -> np.ndarray:
        """Generate contact sequence using the selected generator
        
        Args:
            contact_sequence_dts: List of timesteps (for periodic gait)
            contact_sequence_lenghts: Number of steps for each dt (for periodic gait)
            
        Returns:
            contact_sequence: Binary contact sequence for all legs
        """
        if self.use_random_gait:
            # For random gait, we use a uniform dt based on step frequency
            dt = 1.0 / (self.step_freq * 10) if self.step_freq > 0 else 0.02
            sequence = self.random_generator.generate_contact_sequence(
                horizon=self.horizon,
                dt=dt,
                current_contacts=self.current_contact
            )
            # Update current_contact for next call
            if sequence.shape[1] > 0:
                self.current_contact = sequence[:, 0]
        else:
            # Use periodic generator with its parameters
            # Provide default values if not specified
            if contact_sequence_dts is None:
                contact_sequence_dts = [0.02]  # Default timestep
            
            if contact_sequence_lenghts is None:
                contact_sequence_lenghts = [self.horizon]  # Use full horizon for single dt
                
            sequence = self.periodic_generator.compute_contact_sequence(
                contact_sequence_dts=contact_sequence_dts,
                contact_sequence_lenghts=contact_sequence_lenghts
            )
        
        return sequence
    
    def run(self, dt: float, new_step_freq: float) -> np.ndarray:
        """Update internal state based on time increment
        
        Args:
            dt: Time increment
            new_step_freq: New step frequency
            
        Returns:
            contact: Current contact state
        """
        if self.use_random_gait:
            # Random gait doesn't use run directly, but we can simulate it
            # by generating a short sequence and taking the first step
            self.step_freq = new_step_freq
            sequence = self.random_generator.generate_contact_sequence(
                horizon=1,
                dt=dt,
                current_contacts=self.current_contact
            )
            self.current_contact = sequence[:, 0]
            return self.current_contact
        else:
            return self.periodic_generator.run(dt, new_step_freq)
    
    def reset(self):
        """Reset both generators"""
        self.periodic_generator.reset()
        self.random_generator.reset()
        self.current_contact = np.ones(4)
    
    def set_full_stance(self):
        """Set all legs to full stance"""
        self.periodic_generator.set_full_stance()
        self.current_contact = np.ones(4)
    
    def restore_previous_gait(self):
        """Restore previous gait type for periodic generator"""
        self.periodic_generator.restore_previous_gait()
    
    def update_start_and_stop(self, *args, **kwargs):
        """Update start/stop behavior for periodic generator"""
        if not self.use_random_gait:
            self.periodic_generator.update_start_and_stop(*args, **kwargs)
    
    def set_gait_mode(self, use_random: bool):
        """Switch between random and periodic gait modes
        
        Args:
            use_random: Whether to use random gait
        """
        self.use_random_gait = use_random
        self.reset()