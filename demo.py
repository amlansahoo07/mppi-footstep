import numpy as np

num_legs = 4
horizon = 12
min_contact = 3

gait_mask = np.zeros((num_legs, horizon))  
        
for t in range(horizon):
        contact_count = np.random.randint(min_contact, num_legs + 1)
        contact_legs = np.random.choice(num_legs, size=contact_count, replace=False) 
        gait_mask[contact_legs, t] = 1

print(gait_mask)

"""

I am working on a research project revolving around using MPPI to plan footstep for a quadruped. Here,
quadruped has a predefined gait pattern as per the config file. My first task is to randomize the gait 
such that the quadruped doesn't necessarily follow a fixed gaited pattern. I need to generate a random
gait mask such that it's realistic. That is for a foot having a contact sequence of 1, 0, 1, 0, etc. 
doesn't make sense as the foot should be in contact for a specific time before lifting off. So I kinda
need it to be pseudo random where the gait mask follows some kinf of primtives. For the predefined gaits
we are using a periodic gait generator and the foothold planning using Foothold reference generator. I 
think we need to modify the periodic gait generator to generate random gait masks. I am not sure how to 
exactly do it. Can you help me first discuss how such a technique can be implemented and what all things
we need to target to modify in this codebase?

When and where to place feet using MPPI? That is what both placement and periodicity should be determined
by MPPI.

After discussion with the professor I figured out that the research aim is flexible and the overall aim 
is to make a quadruped perform in real-time with a non-gaited pattern. That is we aim to enable the 
quadruped to plan its footstep in any terrain, and for it we want to randomize the gait mask of its 4 
legs and randomize the periodocity.

Since you are generating a randomized mask for gait, the reference footholds should adapt dynamically based 
on the contact sequence rather than relying on predefined gait patterns.

Can you tell me how to add an implementation of using random gaits? The objective is to develop a kind of 
non-gaited framework where we randomize the masks and then use MPPI to plan the most optimal footstep.
"""