import numpy as np
import pytest
import matplotlib.pyplot as plt
from quadruped_pympc.helpers.gait_adapter import GaitAdapter
from quadruped_pympc.helpers.quadruped_utils import GaitType
from quadruped_pympc.helpers.random_gait_generator import GaitParameters

def test_gait_adapter_initialization():
    # Test with default parameters
    adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                         gait_type=GaitType.TROT, horizon=20)
    
    assert adapter.use_random_gait == False
    assert adapter.horizon == 20
    assert adapter.duty_factor == 0.6
    assert adapter.step_freq == 2.0
    
    # Test with random gait enabled
    random_adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0,
                                gait_type=GaitType.TROT, horizon=20,
                                use_random_gait=True)
    
    assert random_adapter.use_random_gait == True

def test_periodic_mode():
    adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                         gait_type=GaitType.TROT, horizon=20)
    
    # Get sequence from adapter with proper dt and length params
    dt = [0.02]  # 50Hz sampling
    lengths = [20]  # Use all 20 steps
    sequence = adapter.compute_contact_sequence(
        contact_sequence_dts=dt,
        contact_sequence_lenghts=lengths
    )
    
    # Basic shape check
    assert sequence.shape[0] == 4  # 4 legs
    
    # Check binary values
    assert np.all(np.logical_or(sequence == 0, sequence == 1))
    
    # For trot, diagonal legs should be in the same phase
    # FL (0) and RR (3) should match, FR (1) and RL (2) should match
    assert np.allclose(sequence[0], sequence[3])
    assert np.allclose(sequence[1], sequence[2])
    
    # Diagonal pairs should be in opposite phase from each other
    assert not np.allclose(sequence[0], sequence[1])

def test_random_mode():
    adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                         gait_type=GaitType.TROT, horizon=20,
                         use_random_gait=True)
    
    # Get sequence from adapter
    sequence = adapter.compute_contact_sequence()
    
    # Basic shape check
    assert sequence.shape[0] == 4  # 4 legs
    assert sequence.shape[1] == 20  # horizon
    
    # Check binary values
    assert np.all(np.logical_or(sequence == 0, sequence == 1))

def test_mode_switching():
    # Start with periodic
    adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                         gait_type=GaitType.TROT, horizon=20)
    
    # Get periodic sequence
    dt = [0.02]
    lengths = [20]
    periodic_sequence = adapter.compute_contact_sequence(
        contact_sequence_dts=dt,
        contact_sequence_lenghts=lengths
    )
    
    # Switch to random
    adapter.set_gait_mode(True)
    assert adapter.use_random_gait == True
    
    # Get random sequence
    random_sequence = adapter.compute_contact_sequence()
    
    # Switch back to periodic
    adapter.set_gait_mode(False)
    assert adapter.use_random_gait == False
    
    # Get periodic sequence again
    periodic_sequence2 = adapter.compute_contact_sequence(
        contact_sequence_dts=dt,
        contact_sequence_lenghts=lengths
    )
    
    # The two periodic sequences should be similar (may not be identical due to state)
    # But random sequence should be different
    assert not np.allclose(periodic_sequence, random_sequence)

def test_run_method():
    adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                         gait_type=GaitType.TROT, horizon=20)
    
    # Run in periodic mode
    contacts_periodic = adapter.run(dt=0.01, new_step_freq=2.0)
    assert contacts_periodic.shape == (4,)
    
    # Switch to random mode
    adapter.set_gait_mode(True)
    
    # Run in random mode
    contacts_random = adapter.run(dt=0.01, new_step_freq=2.0)
    assert contacts_random.shape == (4,)

def visualize_adapter_comparison():
    """Visual comparison of periodic vs random gaits (not a test)"""
    # Create adapter with both modes
    periodic_adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                                  gait_type=GaitType.TROT, horizon=50)
    
    random_adapter = GaitAdapter(duty_factor=0.6, step_freq=2.0, 
                                gait_type=GaitType.TROT, horizon=50,
                                use_random_gait=True)
    
    # Get sequences
    periodic_sequence = periodic_adapter.compute_contact_sequence()
    random_sequence = random_adapter.compute_contact_sequence()
    
    # Setup visualization
    fig, axs = plt.subplots(2, 1, figsize=(12, 8))
    plt.subplots_adjust(hspace=0.4)
    
    leg_order = ['RL', 'FL', 'FR', 'RR']
    leg_indices = [2, 0, 1, 3]  # Indices for RL, FL, FR, RR
    colors = ['red', 'blue', 'green', 'purple']
    
    # Plot periodic sequence
    ax = axs[0]
    dt = 0.02
    for j, leg_idx in enumerate(leg_indices):
        for t in range(periodic_sequence.shape[1]):
            if periodic_sequence[leg_idx, t] == 1:  # Stance
                ax.add_patch(plt.Rectangle((t*dt, 4-j+0.2), dt, 0.6, 
                            color=colors[j], alpha=0.7))
    
    ax.set_yticks(range(1, len(leg_order)+1))
    ax.set_yticklabels(leg_order[::-1])
    ax.set_xlabel('Time (s)')
    ax.set_title(f"Periodic Gait (Trot)")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, periodic_sequence.shape[1]*dt)
    ax.set_ylim(0, len(leg_order)+1)
    
    # Plot random sequence
    ax = axs[1]
    for j, leg_idx in enumerate(leg_indices):
        for t in range(random_sequence.shape[1]):
            if random_sequence[leg_idx, t] == 1:  # Stance
                ax.add_patch(plt.Rectangle((t*dt, 4-j+0.2), dt, 0.6, 
                            color=colors[j], alpha=0.7))
    
    ax.set_yticks(range(1, len(leg_order)+1))
    ax.set_yticklabels(leg_order[::-1])
    ax.set_xlabel('Time (s)')
    ax.set_title(f"Random Gait")
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_xlim(0, random_sequence.shape[1]*dt)
    ax.set_ylim(0, len(leg_order)+1)
    
    # plt.savefig("gait_adapter_comparison.png")
    plt.show()

if __name__ == "__main__":
    # Run visualization
    visualize_adapter_comparison()