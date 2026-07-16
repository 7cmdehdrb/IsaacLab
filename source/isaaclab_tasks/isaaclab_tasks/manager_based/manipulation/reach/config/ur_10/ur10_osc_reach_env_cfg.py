# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.controllers.operational_space_cfg import OperationalSpaceControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import OperationalSpaceControllerActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import OperationalSpaceControllerAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.joint_pos_env_cfg import (
    UR10ReachEnvCfg,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class UR10OperationalSpaceControllerAction(OperationalSpaceControllerAction):
    """OSC action term with RPY relative-pose actions and bounded effort commands."""

    cfg: "UR10OperationalSpaceControllerActionCfg"

    def __init__(self, cfg: "UR10OperationalSpaceControllerActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._torque_saturated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    @property
    def torque_saturated(self) -> torch.Tensor:
        return self._torque_saturated

    @property
    def joint_efforts(self) -> torch.Tensor:
        return self._joint_efforts

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._joint_efforts[env_ids] = 0.0
        self._torque_saturated[env_ids] = False

    def _preprocess_actions(self, actions: torch.Tensor):
        super()._preprocess_actions(actions)

        if self._pose_rel_idx is None:
            return

        rot_idx = slice(self._pose_rel_idx + 3, self._pose_rel_idx + 6)
        delta_quat = math_utils.quat_from_euler_xyz(
            self._processed_actions[:, rot_idx.start],
            self._processed_actions[:, rot_idx.start + 1],
            self._processed_actions[:, rot_idx.start + 2],
        )
        self._processed_actions[:, rot_idx] = math_utils.axis_angle_from_quat(delta_quat)

    def apply_actions(self):
        self._compute_dynamic_quantities()
        self._compute_ee_jacobian()
        self._compute_ee_pose()
        self._compute_ee_velocity()
        self._compute_ee_force()
        self._compute_joint_states()

        self._joint_efforts[:] = self._osc.compute(
            jacobian_b=self._jacobian_b,
            current_ee_pose_b=self._ee_pose_b,
            current_ee_vel_b=self._ee_vel_b,
            current_ee_force_b=self._ee_force_b,
            mass_matrix=self._mass_matrix,
            gravity=self._gravity,
            current_joint_pos=self._joint_pos,
            current_joint_vel=self._joint_vel,
            nullspace_joint_pos_target=self._nullspace_joint_pos_target,
        )

        finite_efforts = torch.isfinite(self._joint_efforts).all(dim=1)
        self._joint_efforts[:] = torch.where(finite_efforts.unsqueeze(-1), self._joint_efforts, 0.0)

        effort_limits = self._asset.data.joint_effort_limits[:, self._joint_ids] * self.cfg.effort_limit_scale
        effort_limits = torch.clamp(effort_limits, min=self.cfg.minimum_effort_limit)
        clamped_efforts = torch.clamp(self._joint_efforts, min=-effort_limits, max=effort_limits)
        self._torque_saturated[:] = (~finite_efforts) | torch.any(
            torch.abs(clamped_efforts - self._joint_efforts) > self.cfg.saturation_tolerance, dim=1
        )
        self._joint_efforts[:] = clamped_efforts

        self._asset.set_joint_effort_target(self._joint_efforts, joint_ids=self._joint_ids)


@configclass
class UR10OperationalSpaceControllerActionCfg(OperationalSpaceControllerActionCfg):
    """Configuration for UR10 OSC reach actions."""

    class_type: type = UR10OperationalSpaceControllerAction

    effort_limit_scale: float = 1.0
    """Fraction of the articulation effort limits available to OSC commands."""

    minimum_effort_limit: float = 1.0e-6
    """Lower bound used when clamping effort limits."""

    saturation_tolerance: float = 1.0e-6
    """Tolerance used to detect effort saturation."""


def end_effector_velocity_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector linear and angular velocity in the robot root frame."""
    asset = env.scene[asset_cfg.name]
    ee_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :]
    relative_vel_w = ee_vel_w - asset.data.root_vel_w
    linear_vel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, relative_vel_w[:, 0:3])
    angular_vel_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, relative_vel_w[:, 3:6])
    return torch.cat((linear_vel_b, angular_vel_b), dim=1)


def end_effector_position_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in the robot root frame."""
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_pos_b, _ = math_utils.subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def end_effector_to_target_b(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Vector from the current end-effector position to the target in the robot root frame."""
    command = env.command_manager.get_command(command_name)
    return command[:, :3] - end_effector_position_b(env, asset_cfg)


def position_command_success(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward whether the end-effector is within a target-position threshold."""
    return (mdp.position_command_error(env, command_name, asset_cfg) < threshold).float()


def position_command_error_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """Smooth dense position-tracking reward with a stronger near-target gradient."""
    distance = mdp.position_command_error(env, command_name, asset_cfg)
    return torch.exp(-torch.square(distance / std))


def end_effector_linear_velocity_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize fast end-effector linear motion for OSC stability."""
    ee_vel = end_effector_velocity_b(env, asset_cfg)
    return torch.sum(torch.square(ee_vel[:, :3]), dim=1)


def commanded_joint_efforts_l2(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalize OSC effort commands before they are written to the articulation."""
    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.joint_efforts), dim=1)


def torque_saturation(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalty indicator for OSC effort commands that hit configured effort limits."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.torque_saturated.float()


@configclass
class UR10OSCObservationsCfg:
    """Observation terms for the UR10 OSC reach task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_pos = ObsTerm(
            func=end_effector_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        ee_to_target = ObsTerm(
            func=end_effector_to_target_b,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
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
class UR10OSCReachEnvCfg(UR10ReachEnvCfg):
    """UR10 reach task using 6D relative pose operational-space control actions."""

    observations: UR10OSCObservationsCfg = UR10OSCObservationsCfg()

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

        self.rewards.end_effector_position_tracking.weight = -2.0
        self.rewards.end_effector_position_tracking_fine_grained.weight = 1.5
        self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.08
        self.rewards.end_effector_orientation_tracking.weight = -0.02

        self.rewards.end_effector_position_tracking_exp = RewTerm(
            func=position_command_error_exp,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                "command_name": "ee_pose",
                "std": 0.15,
            },
        )
        self.rewards.end_effector_position_success = RewTerm(
            func=position_command_success,
            weight=5.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
                "command_name": "ee_pose",
                "threshold": 0.04,
            },
        )

        self.rewards.action_rate.weight = -0.002
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.00005)
        self.rewards.joint_vel.weight = -0.0002
        self.rewards.joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
        self.rewards.joint_torques = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-2.0e-6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.commanded_joint_efforts = RewTerm(func=commanded_joint_efforts_l2, weight=-2.0e-7)
        self.rewards.end_effector_linear_velocity = RewTerm(
            func=end_effector_linear_velocity_l2,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        self.rewards.torque_saturation = RewTerm(func=torque_saturation, weight=-0.25)

        self.curriculum.action_rate.params["weight"] = -0.002
        self.curriculum.joint_vel.params["weight"] = -0.0002


@configclass
class UR10OSCReachEnvCfg_PLAY(UR10OSCReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
