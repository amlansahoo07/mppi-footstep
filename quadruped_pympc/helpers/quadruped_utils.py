from __future__ import annotations

import copy
from collections import namedtuple
from enum import Enum

import numpy as np
from mujoco.viewer import Handle



class GaitType(Enum):
    """Enumeration class to represent the different gaits that a quadruped robot can perform."""

    TROT = 0
    PACE = 1
    BOUNDING = 2
    CIRCULARCRAWL = 3
    BFDIAGONALCRAWL = 4
    BACKDIAGONALCRAWL = 5
    FRONTDIAGONALCRAWL = 6
    FULL_STANCE = 7


def plot_swing_mujoco(
    viewer: Handle,
    swing_traj_controller,
    swing_period,
    swing_time: namedtuple,
    lift_off_positions: namedtuple,
    nmpc_footholds: namedtuple,
    ref_feet_pos: namedtuple,
    early_stance_detector,
    geom_ids: namedtuple = None,
):
    """Function to plot the desired foot swing trajectory in Mujoco.

    Args:
    ----
    viewer (Handle): The Mujoco viewer.
    swing_traj_controller (SwingTrajectoryController): The swing trajectory controller.
    swing_period: The swing period.
    swing_time (LegsAttr): The swing time for each leg.
    lift_off_positions (LegsAttr): The lift-off positions for each leg.
    nmpc_footholds (LegsAttr): The footholds for each leg.
    ref_feet_pos (LegsAttr): The reference feet positions for each leg.
    geom_ids (LegsAttr, optional): The geometry ids for each leg. Defaults to None.

    Returns:
    -------
    LegsAttr: The geometry ids for each leg trajectory
    """
    from gym_quadruped.utils.mujoco.visual import render_line, render_sphere
    from gym_quadruped.utils.quadruped_utils import LegsAttr

    # Plot desired foot swing trajectory in Mujoco.
    NUM_TRAJ_POINTS = 6

    if geom_ids is None:
        geom_ids = LegsAttr(FL=[], FR=[], RL=[], RR=[])
        # Instantiate a new geometry
        for leg_id, leg_name in enumerate(["FL", "FR", "RL", "RR"]):
            viewer.user_scn.ngeom += NUM_TRAJ_POINTS
            geom_ids[leg_name] = list(range(viewer.user_scn.ngeom - NUM_TRAJ_POINTS - 1, viewer.user_scn.ngeom - 1))

    # viewer.user_scn.ngeom = 1
    # We first draw the trajectory of the feet
    des_foot_traj = LegsAttr(FL=[], FR=[], RL=[], RR=[])
    for leg_id, leg_name in enumerate(["FL", "FR", "RL", "RR"]):
        # TODO: This function should be vectorized rather than queried sequentially
        if swing_time[leg_name] == 0.0:
            continue
        for point_idx, foot_swing_time in enumerate(np.linspace(swing_time[leg_name], swing_period, NUM_TRAJ_POINTS)):
            ref_foot_pos, _, _ = swing_traj_controller.swing_generator.compute_trajectory_references(
                foot_swing_time, lift_off_positions[leg_name], nmpc_footholds[leg_name], 
                early_stance_detector.hitmoments[leg_name], early_stance_detector.hitpoints[leg_name]
            )
            des_foot_traj[leg_name].append(ref_foot_pos.squeeze())

        for point_idx in range(NUM_TRAJ_POINTS - 1):
            render_line(
                viewer=viewer,
                initial_point=des_foot_traj[leg_name][point_idx],
                target_point=des_foot_traj[leg_name][point_idx + 1],
                width=0.005,
                color=np.array([1, 0, 0, 1]),
                geom_id=geom_ids[leg_name][point_idx],
            )

        # Add a sphere at the the ref_feet_pos
        render_sphere(
            viewer=viewer,
            position=ref_feet_pos[leg_name],
            diameter=0.04,
            color=np.array([1, 0, 1, 0.5]),
            geom_id=geom_ids[leg_name][-1],
        )
    return geom_ids


def check_zmp_constraint_satisfaction(state, contact_status, forces):
    # TODO: This import should go
    from quadruped_pympc import config

    base_w = copy.deepcopy(state["position"])
    base_vel_w = copy.deepcopy(state["linear_velocity"])

    FL = copy.deepcopy(state["foot_FL"])
    FR = copy.deepcopy(state["foot_FR"])
    RL = copy.deepcopy(state["foot_RL"])
    RR = copy.deepcopy(state["foot_RR"])

    yaw = copy.deepcopy(state["orientation"][2])
    h_R_w = np.zeros((2, 2))
    h_R_w[0, 0] = np.cos(yaw)
    h_R_w[0, 1] = np.sin(yaw)
    h_R_w[1, 0] = -np.sin(yaw)
    h_R_w[1, 1] = np.cos(yaw)

    FL[0:2] = h_R_w @ (FL[0:2] - base_w[0:2])
    FR[0:2] = h_R_w @ (FR[0:2] - base_w[0:2])
    RL[0:2] = h_R_w @ (RL[0:2] - base_w[0:2])
    RR[0:2] = h_R_w @ (RR[0:2] - base_w[0:2])

    foot_force_fl = forces[0:3]  # @self.centroidal_model.param[0]
    foot_force_fr = forces[3:6]  # @self.centroidal_model.param[1]
    foot_force_rl = forces[6:9]  # @self.centroidal_model.param[2]
    foot_force_rr = forces[9:12]  # @self.centroidal_model.param[3]
    temp = foot_force_fl + foot_force_fr + foot_force_rl + foot_force_rr
    gravity = np.array([0, 0, -9.81])
    linear_com_acc = (1 / config.mass) * temp + gravity

    if config.mpc_params['use_zmp_stability']:
        gravity_z = 9.81
        robotHeight = base_w[2]
        zmp = base_w[0:2] - linear_com_acc[0:2] * (robotHeight / gravity_z)
        zmp = h_R_w @ (zmp - base_w[0:2])
        x = zmp[0]
        y = zmp[1]
    else:
        x = 0.0
        y = 0.0

    y_FL = FL[1]
    y_FR = FR[1]
    y_RL = RL[1]
    y_RR = RR[1]

    x_FL = FL[0]
    x_FR = FR[0]
    x_RL = RL[0]
    x_RR = RR[0]

    # LF - RF : x < (x2 - x1) (y - y1) / (y2 - y1) + x1
    # RF - RH: y > (y2 - y1) (x - x1) / (x2 - x1) + y1
    # RH - LH : x > (x2 - x1) (y - y1) / (y2 - y1) + x1
    # LH - LF: y < (y2 - y1) (x - x1) / (x2 - x1) + y1

    # FL and FR cannot stay at the same x! #constrint should be less than zero
    constraint_FL_FR = x - (x_FR - x_FL) * (y - y_FL) / (y_FR - y_FL + 0.001) - x_FL

    # FR and RR cannot stay at the same y! #constraint should be bigger than zero
    constraint_FR_RR = y - (y_RR - y_FR) * (x - x_FR) / (x_RR - x_FR + 0.001) - y_FR

    # RL and RR cannot stay at the same x! #constraint should be bigger than zero
    constraint_RR_RL = x - (x_RL - x_RR) * (y - y_RR) / (y_RL - y_RR + 0.001) - x_RR

    # FL and RL cannot stay at the same y! #constraint should be less than zero
    constraint_RL_FL = y - (y_FL - y_RL) * (x - x_RL) / (x_FL - x_RL + 0.001) - y_RL

    # the diagonal stuff can be at any point...
    constraint_FL_RR = y - (y_RR - y_FL) * (x - x_FL) / (x_RR - x_FL + 0.001) - y_FL  # bigger than zero
    constraint_FR_RL = y - (y_RL - y_FR) * (x - x_FR) / (x_RL - x_FR + 0.001) - y_FR  # bigger than zero

    FL_contact = contact_status[0]
    FR_contact = contact_status[1]
    RL_contact = contact_status[2]
    RR_contact = contact_status[3]

    violation = 0

    if FL_contact == 1:
        if FR_contact == 1:
            # ub_support_FL_FR = -0.0
            # lb_support_FL_FR = -1000
            if constraint_FL_FR > 0:
                violation = violation + 1

        else:
            # ub_support_FL_RR = 1000
            # lb_support_FL_RR = 0.0
            if constraint_FL_RR < 0:
                violation = violation + 1

    if FR_contact == 1:
        if RR_contact == 1:
            # ub_support_FR_RR = 1000
            # lb_support_FR_RR = 0.0
            if constraint_FR_RR < 0:
                violation = violation + 1

        else:
            ub_support_FR_RL = 1000
            lb_support_FR_RL = 0.0
            # if(constraint_FR_RL < 0):
            # violation = violation + 1
            # TOCHECK

    if RR_contact == 1:
        if RL_contact == 1:
            # ub_support_RR_RL = 1000
            # lb_support_RR_RL = 0.0
            if constraint_RR_RL < 0:
                violation = violation + 1

        else:
            # ub_support_FL_RR = -0.0
            # lb_support_FL_RR = -1000
            if constraint_FL_RR > 0:
                violation = violation + 1 * 0

    if RL_contact == 1:
        if FL_contact == 1:
            # ub_support_RL_FL = -0.0
            # lb_support_RL_FL = -1000
            if constraint_RL_FL > 0:
                violation = violation + 1
        else:
            # ub_support_FR_RL = -0.0
            # lb_support_FR_RL = -1000
            if constraint_FR_RL > 0:
                violation = violation + 1

    if FL_contact == 1 and FR_contact == 1 and RL_contact == 1 and RR_contact == 1:
        violation = 0

    if violation >= 1:
        return True
    else:
        return False

# Debug Amlan
def plot_contact_sequence_mujoco(
    viewer: Handle,
    contact_sequence: list,
    base_pos: np.ndarray,
    base_height: float,
    dt: float,
    geom_ids: list = None,
    color_in_contact=(0.0, 0.0, 1.0, 0.5),
    color_in_swing=(0.0, 1.0, 1.0, 0.5),
):
    """Plot the contact sequence from MPC in Mujoco.
    
    Args:
        viewer (Handle): The Mujoco viewer.
        contact_sequence (list): The contact sequence from MPC.
        base_pos (np.ndarray): Base position in world frame.
        base_height (float): Height of the base from ground.
        dt (float): Time step of the MPC.
        geom_ids (list, optional): List of geometry IDs to reuse. Defaults to None.
        color_in_contact (tuple, optional): Color for contact state. Defaults to green.
        color_in_swing (tuple, optional): Color for swing state. Defaults to red.
        
    Returns:
        list: The geometry IDs for the contact sequence visualization
    """
    from gym_quadruped.utils.mujoco.visual import render_sphere
    
    # Determine the number of legs and horizon length
    num_legs = len(contact_sequence)
    horizon = len(contact_sequence[0])
    
    # Spacing between indicators
    x_spacing = 0.12
    y_spacing = 0.1
    z_offset = -base_height * 0.7  # Position below the robot
    
    leg_names = ["FL", "FR", "RL", "RR"]
    
    # Calculate total number of spheres needed
    total_spheres = num_legs * horizon
    
    # Create or reuse geometry IDs
    if geom_ids is None:
        viewer.user_scn.ngeom += total_spheres
        geom_ids = list(range(viewer.user_scn.ngeom - total_spheres, viewer.user_scn.ngeom))
    
    sphere_idx = 0
    
    # For each leg
    for leg_idx in range(num_legs):
        # For each time step in the horizon
        for t in range(horizon):
            # Calculate position
            x_pos = base_pos[0] + (horizon/2 - t) * x_spacing
            y_pos = base_pos[1] + (leg_idx - 1.5) * y_spacing
            z_pos = base_pos[2] + z_offset
            
            # Set color based on contact state (1 = contact, 0 = swing)
            color = color_in_contact if contact_sequence[leg_idx][t] == 1 else color_in_swing
            
            # Render sphere
            geom_ids[sphere_idx] = render_sphere(
                viewer=viewer,
                position=np.array([x_pos, y_pos, z_pos]),
                diameter=0.03,
                color=np.array(color),
                geom_id=geom_ids[sphere_idx],
            )
            
            sphere_idx += 1
    
    return geom_ids