import numpy as np
import pytest
from quadruped_pympc.helpers.random_gait_generator import RandomGaitGenerator, GaitParameters

def test_gait_generator_initialization():
    generator = RandomGaitGenerator()
    assert generator.n_legs == 4
    assert np.all(generator.leg_states == 1)
    assert np.all(generator.leg_timers == 0)

def test_contact_sequence_generation():
    generator = RandomGaitGenerator()
    horizon = 12
    dt = 0.02
    
    sequence = generator.generate_contact_sequence(horizon, dt)
    print(f"Generated sequence: {sequence}")
    
    # Check shape
    assert sequence.shape == (4, horizon)
    
    # Check binary values
    assert np.all(np.logical_or(sequence == 0, sequence == 1))
    
def test_stance_duration_constraints():
    # Setup
    params = GaitParameters(min_stance_duration=0.3)
    generator = RandomGaitGenerator(params)
    horizon = 12
    dt = 0.02
    sequence = generator.generate_contact_sequence(horizon, dt)
    
    min_steps = int(params.min_stance_duration/dt)
    print(f"\nMinimum required stance steps: {min_steps}")
    
    for leg in range(4):
        print(f"\nLeg {leg} sequence: {sequence[leg]}")
        
        # Find all stance periods
        stance_periods = []
        current_stance = 0
        
        for t in range(horizon):
            if sequence[leg, t] == 1:
                current_stance += 1
            elif current_stance > 0:
                # End of a stance period
                stance_periods.append(current_stance)
                current_stance = 0
                
        # Add final stance if it exists
        if current_stance > 0:
            if current_stance == horizon:
                print(f"Full horizon stance detected: {current_stance} steps")
            else:
                stance_periods.append(current_stance)
        
        print(f"Detected stance periods: {stance_periods}")
        
        # Check all complete stance periods
        for duration in stance_periods[:-1]:  # Exclude last period
            print(f"Checking stance duration: {duration}")
            assert duration >= min_steps, f"Stance duration {duration} steps < minimum {min_steps} steps"

def test_diagonal_leg_coordination():
    generator = RandomGaitGenerator()
    horizon = 12
    dt = 0.02
    sequence = generator.generate_contact_sequence(horizon, dt)
    
    # Check diagonal legs aren't in swing simultaneously
    diagonal_pairs = [(0, 3), (1, 2)]  # FL-RR, FR-RL
    for pair in diagonal_pairs:
        leg1, leg2 = pair
        assert not np.any(np.logical_and(sequence[leg1] == 0, sequence[leg2] == 0))

def test_current_contacts():
    generator = RandomGaitGenerator()
    horizon = 12
    dt = 0.02
    current = np.array([1, 0, 1, 1])  # Initial contact state
    
    sequence = generator.generate_contact_sequence(horizon, dt, current_contacts=current)
    
    # Check if first timestep matches current contacts
    assert np.all(sequence[:, 0] == current)