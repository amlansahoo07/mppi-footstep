from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Sequence

import mujoco
import numpy as np
from gymnasium import spaces

import logging

log = logging.getLogger(__name__)


@dataclass
class LegsAttr:
    """Dataclass to store attributes associated with the legs of a quadruped robot.

    This class is useful to deal with different leg's ordering and naming conventions, given by different robots,
    vendors, labs. This should allow you to be flexible enough so that code made with a given convention can be
    easily used with another one. :)

    Attributes
    ----------
    FR : Any object/scalar/vector/tensor/feature associated with the Front Right leg
    FL : Any object/scalar/vector/tensor/feature associated with the Front Left  leg
    RR : Any object/scalar/vector/tensor/feature associated with the Rear  Right leg
    RL : Any object/scalar/vector/tensor/feature associated with the Rear  Left  leg

    Examples
    --------
    >>> feet_pos = LegsAttr(FR=[1, 3, 5], FL=[2, 4, 6], RR=[7, 9, 11], RL=[8, 10, 12])
    >>> feet_pos["FR"] = [0.1, 0.1, 0.2]  # Set the value of the FR attribute
    >>> feet_pos.RR = [0.3, 0.1, 0.2]     # Set the value of the RR attribute
    >>> b = feet_pos["FR"]  # feet_pos.FR Get the value of the FR attribute
    >>> # Get the (4, 3) numpy array of the feet positions in the order FR, FL, RR, RL
    >>> import numpy as np
    >>> a = np.array([feet_pos.to_list(order=['FR', 'FL', 'RR', 'RL'])])
    >>> # Basic arithmetic operations are supported
    >>> c: LegsAttr = feet_pos + feet_pos
    >>> assert c.FR == feet_pos.FR + feet_pos.FR
    >>> d: LegsAttr = feet_pos - feet_pos
    >>> assert d.FL == feet_pos.FL - feet_pos.FL
    >>> e: LegsAttr = feet_pos / 2
    >>> assert e.RR == (feet_pos.RR / 2)
    """

    FR: Any
    FL: Any
    RR: Any
    RL: Any

    order = ['FL', 'FR', 'RL', 'RR']

    def to_list(self, order=None):
        """Return a list of the leg's attributes in the order specified (or self.order if order=None)."""
        order = order if order is not None else self.order
        return [getattr(self, leg) for leg in order]

    def __getitem__(self, key):
        """Get the value of the attribute associated with the leg key."""
        assert key in self.order, f"Key {key} is not a valid leg label. Expected any of {self.order}"
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Set the value of the attribute associated with the leg key."""
        setattr(self, key, value)

    def __iter__(self):
        """Iterate over the legs attributes in the order self.order."""
        return iter(self.to_list())

    def __add__(self, other):
        """Add the attributes of the legs with the attributes of the other LegsAttr object."""
        if isinstance(other, LegsAttr):
            return LegsAttr(FR=self.FR + other.FR, FL=self.FL + other.FL, RR=self.RR + other.RR, RL=self.RL + other.RL)
        elif isinstance(other, type(self.FR)):
            return LegsAttr(FR=self.FR + other, FL=self.FL + other, RR=self.RR + other, RL=self.RL + other)
        else:
            raise TypeError("Unsupported operand type for +: 'LegsAttr' and '{}'".format(type(other)))

    def __sub__(self, other):
        """Subtract the attributes of the legs with the attributes of the other LegsAttr object."""
        if isinstance(other, LegsAttr):
            return LegsAttr(FR=self.FR - other.FR, FL=self.FL - other.FL, RR=self.RR - other.RR, RL=self.RL - other.RL)
        elif isinstance(other, type(self.FR)):
            return LegsAttr(FR=self.FR - other, FL=self.FL - other, RR=self.RR - other, RL=self.RL - other)
        else:
            raise TypeError("Unsupported operand type for -: 'LegsAttr' and '{}'".format(type(other)))

    def __truediv__(self, other):
        """Divide the attributes of the legs with the attributes of the other LegsAttr object."""
        if isinstance(other, type(self.FR)) or isinstance(other, (int, float)):
            return LegsAttr(FR=self.FR / other, FL=self.FL / other, RR=self.RR / other, RL=self.RL / other)
        else:
            raise TypeError("Unsupported operand type for /: 'LegsAttr' and '{}'".format(type(other)))

    def __matmul__(self, other):
        """Matrix multiplication of the attributes of the legs with the attributes of the other LegsAttr object."""
        if isinstance(other, LegsAttr):
            return LegsAttr(FR=self.FR @ other.FR, FL=self.FL @ other.FL, RR=self.RR @ other.RR, RL=self.RL @ other.RL)
        elif isinstance(other, type(self.FR)):
            return LegsAttr(FR=self.FR @ other, FL=self.FL @ other, RR=self.RR @ other, RL=self.RL @ other)
        else:
            raise TypeError("Unsupported operand type for @: 'LegsAttr' and '{}'".format(type(other)))

    def __str__(self):
        """Return a string representation of the legs attributes."""
        return f"{', '.join([f'{leg}={getattr(self, leg)}' for leg in self.order])}"

    def __repr__(self):
        """Return a string representation of the legs attributes."""
        return self.__str__()


@dataclass
class JointInfo:
    """Dataclass to store information about the joints of a robot.

    Attributes
    ----------
    name : (str) The name of the joint.
    type : (int) The type of the joint.
    body_id : (int) The body id of the joint.
    nq : (int) The number of generalized coordinates.
    nv : (int) The number of generalized velocities.
    qpos_idx : (tuple) The indices of the joint's generalized coordinates.
    qvel_idx : (tuple) The indices of the joint's generalized velocities.
    tau_idx: (tuple) The indices of the joint's in the generalized forces vector.
    range : list(min, max) The range of the joint's generalized coordinates.
    """

    name: str
    type: int
    body_id: int
    nq: int
    nv: int
    qpos_idx: tuple
    qvel_idx: tuple
    range: list
    tau_idx: tuple = field(default_factory=tuple)
    actuator_id: int = field(default=-1)

    def __str__(self):
        """Return a string representation of the joint information."""
        return f"{', '.join([f'{key}={getattr(self, key)}' for key in self.__dict__.keys()])}"


def extract_mj_joint_info(model: mujoco.MjModel) -> OrderedDict[str, JointInfo]:
    """Returns the joint-space information of the model.

    Thanks to the obscure Mujoco API, this function tries to do the horrible hacks to get the joint information
    we need to do a minimum robotics project with a rigid body system.

    Returns
    -------
        A dictionary with the joint names as keys and the JointInfo namedtuple as values.
            each JointInfo namedtuple contains the following fields:
            - name: The joint name.
            - type: The joint type (mujoco.mjtJoint).
            - body_id: The body id to which the joint is attached.
            - range: The joint range.
            - nq: The number of joint position variables.
            - nv: The number of joint velocity variables.
            - qpos_idx: The indices of the joint position variables in the qpos array.
            - qvel_idx: The indices of the joint velocity variables in the qvel array.
    """
    joint_info = OrderedDict()
    for joint_id in range(model.njnt):
        # Get the starting index of the joint name in the model.names string
        name_start_index = model.name_jntadr[joint_id]
        # Extract the joint name from the model.names bytes and decode it
        joint_name = model.names[name_start_index:].split(b'\x00', 1)[0].decode('utf-8')
        joint_type = model.jnt_type[joint_id]
        qpos_idx_start = model.jnt_qposadr[joint_id]
        qvel_idx_start = model.jnt_dofadr[joint_id]

        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            joint_nq, joint_nv = 7, 6
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            joint_nq, joint_nv = 4, 3
        elif joint_type == mujoco.mjtJoint.mjJNT_SLIDE:
            joint_nq, joint_nv = 1, 1
        elif joint_type == mujoco.mjtJoint.mjJNT_HINGE:
            joint_nq, joint_nv = 1, 1
        else:
            raise RuntimeError(f"Unknown mujoco joint type: {joint_type} available {mujoco.mjtJoint}")

        qpos_idx = np.arange(qpos_idx_start, qpos_idx_start + joint_nq)
        qvel_idx = np.arange(qvel_idx_start, qvel_idx_start + joint_nv)

        joint_info[joint_name] = JointInfo(
            name=joint_name,
            type=joint_type,
            body_id=model.jnt_bodyid[joint_id],
            range=model.jnt_range[joint_id],
            nq=joint_nq,
            nv=joint_nv,
            qpos_idx=qpos_idx,
            qvel_idx=qvel_idx)

    # Iterate over all actuators
    current_dim = 0
    for acutator_idx in range(model.nu):
        name_start_index = model.name_actuatoradr[acutator_idx]
        act_name = model.names[name_start_index:].split(b'\x00', 1)[0].decode('utf-8')
        mj_actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
        # Get the joint index associated with the actuator
        joint_id = model.actuator_trnid[mj_actuator_id, 0]
        # Get the joint name from the joint index
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)

        # Add the actuator indx to the joint_info
        joint_info[joint_name].actuator_id = mj_actuator_id
        joint_info[joint_name].tau_idx = tuple(range(current_dim, current_dim + joint_info[joint_name].nv))
        current_dim += joint_info[joint_name].nv
    return joint_info


def configure_observation_space(
        mj_model: mujoco.MjModel, obs_names: Sequence[str]
        ) -> [spaces.Space, dict[str, slice]]:
    """Configures the observation space for the environment based on the provided state observation names.

    Args:
    ----
    state_obs_names (list[str]): A list of state observation names based on which the observation space is
    configured.

    Returns:
    -------
    gym.Space: The environment state observation space.
    dict: A dictionary mapping each state observation name to its indices in the observation space.
    """
    obs_dim, last_idx = 0, 0

    obs_lim_min, obs_lim_max = [], []
    qpos_lim_min, qpos_lim_max = mj_model.jnt_range[:, 0], mj_model.jnt_range[:, 1]
    tau_lim_min, tau_lim_max = mj_model.actuator_ctrlrange[:, 0], mj_model.actuator_ctrlrange[:, 1]

    obs_idx = {k: None for k in obs_names}
    for obs_name in obs_names:
        # Generalized position, velocity, and force (torque) spaces
        if obs_name == 'qpos':
            obs_dim += mj_model.nq
            obs_lim_max.extend([np.inf] * 7 + qpos_lim_max[1:].tolist())  # Ignore the base position
            obs_lim_min.extend([-np.inf] * 7 + qpos_lim_min[1:].tolist())  # Ignore the base position
        elif obs_name == 'qvel':
            obs_dim += mj_model.nv
            obs_lim_max.extend([np.inf] * mj_model.nv)
            obs_lim_min.extend([-np.inf] * mj_model.nv)
        elif obs_name == 'tau_ctrl_setpoint':
            obs_dim += mj_model.nu
            obs_lim_max.extend(tau_lim_max)
            obs_lim_min.extend(tau_lim_min)
        # Joint-space position and velocity spaces
        elif obs_name == 'qpos_js':  # Joint space position configuration
            obs_dim += mj_model.nq - 7
            obs_lim_max.extend(qpos_lim_max[1:])
            obs_lim_min.extend(qpos_lim_min[1:])
        elif obs_name == 'qvel_js':  # Joint space velocity configuration
            obs_dim += mj_model.nv - 6
            obs_lim_max.extend([np.inf] * (mj_model.nv - 6))
            obs_lim_min.extend([-np.inf] * (mj_model.nv - 6))
        # Base position and velocity configurations (in world frame)
        elif obs_name == 'base_pos':
            if "qpos" in obs_names:
                log.debug("base_pos is redundant with additional obs qpos. base_pos = qpos[0:3]")
            obs_dim += 3
            obs_lim_max.extend([np.inf] * 3)
            obs_lim_min.extend([-np.inf] * 3)
        elif 'base_lin_vel' in obs_name:  # base_lin_vel / base_lin_vel:base (base frame)
            if "qvel" in obs_names:
                log.debug("base_lin_vel is redundant with additional obs qvel. base_lin_vel = qvel[0:3]")
            obs_dim += 3
            obs_lim_max.extend([np.inf] * 3)
            obs_lim_min.extend([-np.inf] * 3)
        elif 'base_lin_acc' in obs_name:  # base_lin_acc / base_lin_acc:base (base frame)
            obs_dim += 3
            obs_lim_max.extend([np.inf] * 3)
            obs_lim_min.extend([-np.inf] * 3)
        elif 'base_ang_vel' in obs_name:
            if "qvel" in obs_names:
                log.debug("base_ang_vel is redundant with additional obs qvel. base_ang_vel = qvel[3:6]")
            obs_dim += 3
            obs_lim_max.extend([np.inf] * 3)
            obs_lim_min.extend([-np.inf] * 3)
        elif 'base_ori_euler_xyz' in obs_name:
            if "qpos" in obs_names:
                log.debug(
                    "base_ori_euler_xyz is redundant with additional obs qpos. base_ori_euler_xyz = qpos[3:6]")
            obs_dim += 3
            obs_lim_max.extend([np.inf] * 3)
            obs_lim_min.extend([-np.inf] * 3)
        elif obs_name == 'base_ori_quat_wxyz':
            if "qpos" in obs_names:
                log.debug(
                    "base_ori_quat_wxyz is redundant with additional obs qpos. base_ori_quat_wxyz = qpos[3:7]")
            obs_dim += 4
            obs_lim_max.extend([np.inf] * 4)
            obs_lim_min.extend([-np.inf] * 4)
        elif obs_name == 'base_ori_SO3':
            if "qpos" in obs_names:
                log.debug("base_ori_SO3 is redundant with additional obs qpos. base_ori_SO3 = qpos[3:7]")
            obs_dim += 9
            obs_lim_max.extend([np.inf] * 9)
            obs_lim_min.extend([-np.inf] * 9)
        # Feet positions and velocities
        elif 'feet_pos' in obs_name:  # feet_pos:frame := feet_pos:world or feet_pos:base
            obs_dim += 12
            obs_lim_max.extend([np.inf] * 12)
            obs_lim_min.extend([-np.inf] * 12)
        elif 'feet_vel' in obs_name:  # feet_vel:frame := feet_vel:world or feet_vel:base
            obs_dim += 12
            obs_lim_max.extend([np.inf] * 12)
            obs_lim_min.extend([-np.inf] * 12)
        elif obs_name == 'contact_state':
            obs_dim += 4
            obs_lim_max.extend([1] * 4)
            obs_lim_min.extend([0] * 4)
        elif 'contact_forces' in obs_name:
            obs_dim += 12
            obs_lim_max.extend([np.inf] * 12)
            obs_lim_min.extend([-np.inf] * 12)
        elif 'gravity_vector' in obs_name:
            obs_dim += 3
            obs_lim_max.extend([np.inf] * 3)
            obs_lim_min.extend([-np.inf] * 3)
        else:
            from gym_quadruped.quadruped_env import QuadrupedEnv
            raise ValueError(f"Invalid observation name: {obs_name}, available obs: {QuadrupedEnv.ALL_OBS}")
        obs_idx[obs_name] = range(last_idx, obs_dim)
        last_idx = obs_dim

        if obs_dim != len(obs_lim_max) or obs_dim != len(obs_lim_min):
            raise ValueError(
                f"Invalid configuration of observation {obs_name}: \n - obs_dim: {obs_dim} \n"
                f" - lower_lim_dim: {len(obs_lim_max)} \t - upper_lim_dim: {len(obs_lim_min)}"
                )

    obs_lim_min = np.array(obs_lim_min)
    obs_lim_max = np.array(obs_lim_max)
    observation_space = spaces.Box(low=obs_lim_min, high=obs_lim_max, shape=(obs_dim,), dtype=np.float32)
    return observation_space, obs_idx


def configure_observation_space_representations(
        robot_name: str,
        obs_names: Sequence[str]
        ) -> [spaces.Space, dict[str, slice]]:
    try:
        import morpho_symm
        from morpho_symm.utils.robot_utils import load_symmetric_system
        from morpho_symm.utils.rep_theory_utils import escnn_representation_form_mapping, group_rep_from_gens
    except ImportError as e:
        raise ImportError("morpho_symm package is required to configure observation group representations") from e

    G = load_symmetric_system(robot_name=robot_name, return_robot=False)
    rep_Q_js = G.representations['Q_js']  # Representation on joint space position coordinates
    rep_TqQ_js = G.representations['TqQ_js']  # Representation on joint space velocity coordinates
    rep_Rd = G.representations['R3']  # Representation on vectors in R^d
    rep_Rd_pseudo = G.representations['R3_pseudo']  # Representation on pseudo vectors in R^d
    rep_euler_xyz = G.representations['euler_xyz']  # Representation on Euler angles
    # TODO: Ensure the limb order in the configuration matches the used order by quadruped gym.
    rep_kin_three = G.representations['kin_chain']  # Permutation of legs
    rep_Rd_on_limbs = rep_kin_three.tensor(rep_Rd)  # Representation on signals R^d on the limbs
    rep_Rd_on_limbs.name = 'Rd_on_limbs'
    rep_Rd_pseudo_on_limbs = rep_kin_three.tensor(rep_Rd_pseudo)  # Representation on pseudo vect R^d on the limbs
    rep_Rd_pseudo_on_limbs.name = 'Rd_pseudo_on_limbs'
    rep_SO3_flat = {}
    for h in G.elements:
        rep_SO3_flat[h] = np.kron(rep_Rd(h), rep_Rd(~h).T)
    rep_SO3_flat = escnn_representation_form_mapping(G, rep_SO3_flat)
    rep_SO3_flat.name = 'SO3_flat'

    # Create a representation for the z dimension alone of the base position
    rep_z = G.irrep(*rep_Rd.irreps[-1])

    obs_reps = {k: None for k in obs_names}
    for obs_name in obs_names:
        # Generalized position, velocity, and force (torque) spaces
        if obs_name == 'qpos':
            continue  # Quaternion does not have a left-group action definition.
        elif obs_name == 'qvel':
            obs_reps[obs_name] = rep_Rd + rep_Rd_pseudo + rep_TqQ_js  # lin_vel , ang_vel, joint_vel
        elif obs_name == 'tau_ctrl_setpoint':
            obs_reps[obs_name] = rep_TqQ_js
        elif obs_name == 'qpos_js':  # Joint space position configuration
            obs_reps[obs_name] = rep_Q_js
        elif obs_name == 'qvel_js':  # Joint space velocity configuration
            obs_reps[obs_name] = rep_TqQ_js
        elif obs_name == 'base_pos':
            obs_reps[obs_name] = rep_Rd
        elif obs_name == 'base_pos_z':
            obs_reps[obs_name] = rep_z
        elif 'base_lin_vel' in obs_name:  # base_lin_vel / base_lin_vel:base (base frame)
            obs_reps[obs_name] = rep_Rd
        elif 'base_lin_acc' in obs_name:  # base_lin_acc / base_lin_acc:base (base frame)
            obs_reps[obs_name] = rep_Rd
        elif 'base_ang_vel' in obs_name:
            obs_reps[obs_name] = rep_Rd_pseudo
        elif 'base_ori_euler_xyz' in obs_name:
            obs_reps[obs_name] = rep_euler_xyz
        elif obs_name == 'base_ori_quat_wxyz':
            continue  # Quaternion does not have a left-group action definition.
        elif obs_name == 'base_ori_SO3':
            obs_reps[obs_name] = rep_SO3_flat
        elif 'feet_pos' in obs_name:  # feet_pos:frame := feet_pos:world or feet_pos:base
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif 'feet_vel' in obs_name:  # feet_vel:frame := feet_vel:world or feet_vel:base
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif obs_name == 'contact_state':
            obs_reps[obs_name] = rep_kin_three
        elif 'contact_forces' in obs_name:
            obs_reps[obs_name] = rep_Rd_on_limbs
        elif 'gravity_vector' in obs_name:
            obs_reps[obs_name] = rep_Rd
        else:
            from gym_quadruped.quadruped_env import QuadrupedEnv
            raise ValueError(f"Invalid observation name: {obs_name}, available obs: {QuadrupedEnv.ALL_OBS}")

    return obs_reps
