import numpy as np
import jax
import jax.numpy as jnp

from quadruped_pympc.controllers.sampling.centroidal_nmpc_jax import Sampling_MPC
from quadruped_pympc.helpers.gait_adapter import GaitAdapter
from quadruped_pympc.helpers.quadruped_utils import GaitType
from quadruped_pympc import config

class RandomGaitMPPI(Sampling_MPC):
    """MPPI-based controller that optimizes both GRF and gait patterns"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Gait optimization parameters
        self.num_gait_samples = config.mpc_params.get('num_gait_samples', 20)
        self.gait_stability_weight = config.mpc_params.get('gait_stability_weight', 1.0)
        self.min_support_legs = config.mpc_params.get('min_support_legs', 0)
        
        # Initialize gait adapter
        duty_factor = config.simulation_params['gait_params'][config.simulation_params['gait']]['duty_factor']
        step_freq = config.simulation_params['gait_params'][config.simulation_params['gait']]['step_freq']
        gait_type = config.simulation_params['gait_params'][config.simulation_params['gait']]['type']
        
        self.gait_adapter = GaitAdapter(
            duty_factor=duty_factor,
            step_freq=step_freq,
            gait_type=gait_type,
            horizon=self.horizon,
            use_random_gait=True
        )
        
        # Track current contacts for continuity
        self.current_contacts = np.ones(4)

        # Overriding jitted compute_control method
        self.jitted_compute_control = jax.jit(self.compute_control, device=self.device)

    def generate_contact_sequences(self, num_sequences, current_contacts=None):
        """Generate multiple candidate contact sequences
        
        Args:
            num_sequences: Number of sequences to generate
            current_contacts: Current contact state for continuity
            
        Returns:
            List of contact sequences
        """
        print("GENERATING CONTACT SEQUENCES....................")
        sequences = []
        
        # Use current contacts if provided, otherwise use stored contacts
        if current_contacts is not None:
            self.current_contacts = current_contacts.copy()
        
        # Generate candidate sequences
        for i in range(num_sequences):
            sequence = self.gait_adapter.compute_contact_sequence()
            sequences.append(sequence)
        
        # Update current contacts for next time
        if len(sequences) > 0:
            self.current_contacts = sequences[0][:, 0]
            
        return sequences

    def evaluate_sequence_stability(self, sequence, state):
        """Compute stability cost for a sequence
        
        Args:
            sequence: Contact sequence to evaluate
            state: Current state vector
            
        Returns:
            Stability cost (lower is better)
        """
        cost = 0.0
        
        # Check minimum number of supporting legs
        for t in range(sequence.shape[1]):
            legs_in_stance = np.sum(sequence[:, t])
            if legs_in_stance < self.min_support_legs:
                cost += 1000.0 * (self.min_support_legs - legs_in_stance)
        
        # Penalize rapid transitions (optional)
        for t in range(1, sequence.shape[1]):
            transitions = np.sum(np.abs(sequence[:, t] - sequence[:, t-1]))
            cost += transitions * 10.0
        
        return cost

    def compute_control_mppi_with_gait(self, state, reference, current_contacts, best_control_parameters, key):
        """Compute optimal control with integrated gait planning
        
        Args:
            state: Current state vector
            reference: Reference state trajectory
            current_contacts: Current contact state
            best_control_parameters: Previous best GRF parameters
            key: JAX random key
            
        Returns:
            GRF, footholds, predicted state, parameters, cost, frequency, costs
        """
        # Generate multiple candidate gait sequences
        candidate_sequences = self.generate_contact_sequences(self.num_gait_samples, current_contacts)
        
        best_cost = float('inf')
        best_result = None
        
        # For each candidate sequence, optimize forces
        for sequence in candidate_sequences:
            # Call standard MPPI to optimize forces for this sequence
            nmpc_GRFs, nmpc_footholds, predicted_state, parameters, cost, freq, costs = \
                super().compute_control_mppi(state, reference, sequence, best_control_parameters, key, None, None, None)
            
            # Add gait stability cost
            stability_cost = self.evaluate_sequence_stability(sequence, state)
            total_cost = cost + stability_cost * self.gait_stability_weight
            
            # Track best result
            if total_cost < best_cost:
                best_cost = total_cost
                best_result = (nmpc_GRFs, nmpc_footholds, predicted_state, parameters, cost, freq, costs)
                best_sequence = sequence
        
        # Store the best sequence for future use
        self.best_sequence = best_sequence
        self.current_contacts = best_sequence[:, 0]
        
        return best_result
    
    # def compute_control(self, state, reference, contact_sequence, best_control_parameters, key, *args, **kwargs):
    def compute_control(self, state, reference, contact_sequence, best_control_parameters, key, timing, nominal_step_frequency, optimize_swing):    
        """Main entry point for control computation
        
        This redirects to our gait-optimizing version instead of the standard one
        """
        print("ENTERING GAIT-OPTIMIZING MPPI..........................")
        # Use current contacts from the provided sequence (first timestep)
        current_contacts = contact_sequence[:, 0] if contact_sequence is not None else None
        
        # Call our custom implementation 
        return self.compute_control_mppi_with_gait(state, reference, current_contacts, best_control_parameters, key)