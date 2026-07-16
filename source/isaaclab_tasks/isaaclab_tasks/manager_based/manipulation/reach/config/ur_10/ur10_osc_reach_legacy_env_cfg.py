# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Legacy UR10 OSC reach config for checkpoints trained before the observation update."""

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.joint_pos_env_cfg import UR10ReachEnvCfg
from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.ur10_osc_reach_env_cfg import (
    UR10OperationalSpaceControllerActionCfg,
    commanded_joint_efforts_l2,
    end_effector_velocity_b,
    torque_saturation,
)


@configclass
class UR10OSCLegacyObservationsCfg:
    """Observation terms used by the original 2026-07-11_01-54-00 OSC checkpoint."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_vel = ObsTerm(
            func=end_effector_velocity_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR10OSCLegacyReachEnvCfg(UR10ReachEnvCfg):
    """UR10 OSC reach task matching the original 31D-observation OSC policy."""

    observations: UR10OSCLegacyObservationsCfg = UR10OSCLegacyObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0
        self.scene.robot.spawn.rigid_props.disable_gravity = True

        self.actions.arm_action = UR10OperationalSpaceControllerActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name="ee_link",
            body_offset=UR10OperationalSpaceControllerActionCfg.OffsetCfg(),
            controller_cfg=OperationalSpaceControllerCfg(
                target_types=["pose_rel"],
                impedance_mode="fixed",
                motion_control_axes_task=(1, 1, 1, 1, 1, 1),
                contact_wrench_control_axes_task=(0, 0, 0, 0, 0, 0),
                inertial_dynamics_decoupling=False,
                partial_inertial_dynamics_decoupling=False,
                gravity_compensation=False,
                motion_stiffness_task=(80.0, 80.0, 80.0, 25.0, 25.0, 25.0),
                motion_damping_ratio_task=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
                nullspace_control="none",
            ),
            position_scale=0.03,
            orientation_scale=0.15,
            effort_limit_scale=1.0,
        )

        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
        self.rewards.joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
        self.rewards.joint_torques = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.commanded_joint_efforts = RewTerm(func=commanded_joint_efforts_l2, weight=-1.0e-6)
        self.rewards.torque_saturation = RewTerm(func=torque_saturation, weight=-0.25)


@configclass
class UR10OSCLegacyReachEnvCfg_PLAY(UR10OSCLegacyReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

