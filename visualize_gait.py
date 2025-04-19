import numpy as np
import matplotlib.pyplot as plt
from quadruped_pympc.helpers.random_gait_generator import RandomGaitGenerator, GaitParameters

if __name__ == "__main__":
    # Generate different contact sequences
    generator = RandomGaitGenerator()
    dt = 0.02
    horizon = 12  # Increased for better visualization
    
    # Create a single figure with 3 subplots
    fig, axs = plt.subplots(3, 1, figsize=(10, 8))
    plt.subplots_adjust(hspace=0.4)  # Add space between subplots
    
    # Reorder for printing
    leg_order = ['RL', 'FL', 'FR', 'RR']
    leg_indices = [2, 0, 1, 3]  # Indices for RL, FL, FR, RR
    
    for i in range(3):  # Generate 3 different sequences
        sequence = generator.generate_contact_sequence(horizon, dt)
        
        # Print the sequence in desired order
        print(f"\nSequence {i+1}:")
        for j, leg in enumerate(leg_order):
            print(f"{leg}: {sequence[leg_indices[j]]}")
        
        # Plot directly to the correct subplot
        ax = axs[i]
        
        # Plot the sequence in the desired leg order
        for j, leg_idx in enumerate(leg_indices):
            for t in range(sequence.shape[1]):
                if sequence[leg_idx, t] == 1:  # Stance
                    ax.add_patch(plt.Rectangle((t*dt, 4-j+0.2), dt,.6, 
                                color=['red', 'blue', 'green', 'purple'][j], alpha=0.7))
        
        # Configure each subplot
        ax.set_yticks(range(1, len(leg_order)+1))
        ax.set_yticklabels(leg_order[::-1])
        ax.set_xlabel('Time (s)')
        ax.set_title(f"Random Gait Sequence {i+1}")
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax.set_xlim(0, sequence.shape[1]*dt)
        ax.set_ylim(0, len(leg_order)+1)
    
    plt.tight_layout()
    plt.savefig("random_gait_sequences.png")
    plt.show()
    
    print("\nGait visualization saved to random_gait_sequences.png")