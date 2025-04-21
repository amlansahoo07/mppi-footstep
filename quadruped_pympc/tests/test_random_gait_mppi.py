import numpy as np
import matplotlib.pyplot as plt
from quadruped_pympc.controllers.sampling.random_gait_mppi import RandomGaitMPPI
from quadruped_pympc.helpers.quadruped_utils import GaitType

def test_random_gait_mppi_integration():
    """Test that random gait MPPI can be initialized and run"""
    
    # Create controller
    controller = RandomGaitMPPI()
    
    # Create dummy state and reference
    state = np.zeros(24)
    reference = np.zeros(24)
    
    # Create a basic contact sequence
    contact_sequence = np.ones((4, controller.horizon))
    
    # Try generating contact sequences
    sequences = controller.generate_contact_sequences(5)
    
    # Verify sequences are generated
    assert len(sequences) == 5
    assert sequences[0].shape == (4, controller.horizon)
    
    # Visualize the sequences
    fig, axs = plt.subplots(5, 1, figsize=(10, 12))
    leg_order = ['FL', 'FR', 'RL', 'RR']
    colors = ['red', 'blue', 'green', 'purple']
    dt = 0.02
    
    for i, sequence in enumerate(sequences):
        ax = axs[i]
        for j, leg_name in enumerate(leg_order):
            for t in range(sequence.shape[1]):
                if sequence[j, t] == 1:  # Stance
                    ax.add_patch(plt.Rectangle((t*dt, 4-j+0.2), dt, 0.6, 
                                color=colors[j], alpha=0.7))
        
        ax.set_yticks(range(1, len(leg_order)+1))
        ax.set_yticklabels(leg_order[::-1])
        ax.set_xlabel('Time (s)')
        ax.set_title(f"Random Gait Sequence {i+1}")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, sequence.shape[1]*dt)
        ax.set_ylim(0, len(leg_order)+1)
    
    plt.tight_layout()
    plt.savefig("random_gait_mppi_sequences.png")

if __name__ == "__main__":
    test_random_gait_mppi_integration()