# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""UR10 reach environment with torque-level policy actions.

The task keeps the Reach command/reward structure, exposes operational-space
state to the policy, and applies policy actions directly as joint torques.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.actions.actions_cfg import JointEffortActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointEffortAction
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


class UR10JointTorqueAction(JointEffortAction):
    """Joint-effort action that interprets policy outputs as normalized torques."""

    cfg: "UR10JointTorqueActionCfg"

    def __init__(self, cfg: "UR10JointTorqueActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._torque_saturated = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def joint_efforts(self) -> torch.Tensor:
        """Processed joint torques written to the articulation."""
        return self._processed_actions

    @property
    def torque_saturated(self) -> torch.Tensor:
        """Whether the raw policy action was invalid or outside [-1, 1]."""
        return self._torque_saturated

    def process_actions(self, actions: torch.Tensor):
        finite_actions = torch.isfinite(actions).all(dim=1)
        sanitized_actions = torch.where(torch.isfinite(actions), actions, 0.0)

        self._raw_actions[:] = sanitized_actions
        normalized_actions = torch.clamp(sanitized_actions, min=-1.0, max=1.0)

        effort_limits = (
            self._asset.data.joint_effort_limits[:, self._joint_ids]
            * self.cfg.effort_limit_scale
        )
        effort_limits = torch.clamp(effort_limits, min=self.cfg.minimum_effort_limit)
        self._processed_actions[:] = normalized_actions * effort_limits

        self._torque_saturated[:] = (~finite_actions) | torch.any(
            torch.abs(normalized_actions - sanitized_actions)
            > self.cfg.saturation_tolerance,
            dim=1,
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        super().reset(env_ids)
        self._processed_actions[env_ids] = 0.0
        self._torque_saturated[env_ids] = False


@configclass
class UR10JointTorqueActionCfg(JointEffortActionCfg):
    """Configuration for normalized UR10 joint torque actions."""

    class_type: type = UR10JointTorqueAction

    effort_limit_scale: float = 1.0
    """Fraction of the articulation effort limits used for policy torques."""

    minimum_effort_limit: float = 1.0e-6
    """Lower bound used when effort limits are read from the articulation."""

    saturation_tolerance: float = 1.0e-6
    """Tolerance used to detect clipped or invalid torque actions."""


def end_effector_pose_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """End-effector pose in the robot root frame as ``xyz + wxyz``."""
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    ee_pos_b, ee_quat_b = math_utils.subtract_frame_transforms(
        asset.data.root_pos_w, asset.data.root_quat_w, ee_pos_w, ee_quat_w
    )
    return torch.cat((ee_pos_b, ee_quat_b), dim=1)


def end_effector_velocity_b(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """End-effector linear and angular velocity in the robot root frame."""
    asset = env.scene[asset_cfg.name]
    ee_vel_w = asset.data.body_vel_w[:, asset_cfg.body_ids[0], :]
    relative_vel_w = ee_vel_w - asset.data.root_vel_w
    linear_vel_b = math_utils.quat_apply_inverse(
        asset.data.root_quat_w, relative_vel_w[:, 0:3]
    )
    angular_vel_b = math_utils.quat_apply_inverse(
        asset.data.root_quat_w, relative_vel_w[:, 3:6]
    )
    return torch.cat((linear_vel_b, angular_vel_b), dim=1)


def position_command_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward whether the end-effector is within a target-position threshold."""
    return (
        mdp.position_command_error(env, command_name, asset_cfg) < threshold
    ).float()


def orientation_command_success(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    threshold: float,
) -> torch.Tensor:
    """Reward whether the end-effector is within a target-orientation threshold."""
    return (
        mdp.orientation_command_error(env, command_name, asset_cfg) < threshold
    ).float()


def position_command_error_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """Smooth dense position-tracking reward with a stronger near-target gradient."""
    distance = mdp.position_command_error(env, command_name, asset_cfg)
    return torch.exp(-torch.square(distance / std))


def orientation_command_error_exp(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, std: float
) -> torch.Tensor:
    """Smooth dense orientation-tracking reward with a stronger near-target gradient."""
    distance = mdp.orientation_command_error(env, command_name, asset_cfg)
    return torch.exp(-torch.square(distance / std))


def end_effector_linear_velocity_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Penalize fast end-effector linear motion for torque-level stability."""
    ee_vel = end_effector_velocity_b(env, asset_cfg)
    return torch.sum(torch.square(ee_vel[:, :3]), dim=1)


def commanded_joint_torques_l2(
    env: ManagerBasedRLEnv, action_name: str = "arm_action"
) -> torch.Tensor:
    """Penalize processed joint torque commands."""
    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.joint_efforts), dim=1)


def torque_saturation(
    env: ManagerBasedRLEnv, action_name: str = "arm_action"
) -> torch.Tensor:
    """Penalty indicator for clipped or invalid torque actions."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.torque_saturated.float()


@configclass
class UR10OSCTorqueObservationsCfg:
    """Observation terms for the torque-level UR10 reach task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel,
            noise=Unoise(n_min=-0.01, n_max=0.01),
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        joint_effort = ObsTerm(
            func=mdp.joint_effort,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
        )
        eef_pose = ObsTerm(
            func=end_effector_pose_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        target_pose = ObsTerm(
            func=mdp.generated_commands, params={"command_name": "ee_pose"}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR10OSCTorqueReachEnvCfg(UR10ReachEnvCfg):
    """UR10 Reach task with operational-space observations and joint-torque actions."""

    observations: UR10OSCTorqueObservationsCfg = UR10OSCTorqueObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot.actuators["arm"].stiffness = 0.0
        self.scene.robot.actuators["arm"].damping = 0.0

        self.scene.robot.actuators["arm"].effort_limit_sim = 870.0

        self.scene.robot.spawn.rigid_props.disable_gravity = True

        self.actions.arm_action = UR10JointTorqueActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            effort_limit_scale=1.0,
        )

        # self.rewards.end_effector_position_tracking.weight = -2.0
        # self.rewards.end_effector_position_tracking_fine_grained.weight = 1.5
        # self.rewards.end_effector_position_tracking_fine_grained.params["std"] = 0.08
        # self.rewards.end_effector_orientation_tracking.weight = -0.02

        # self.rewards.end_effector_position_tracking_exp = RewTerm(
        #     func=position_command_error_exp,
        #     weight=2.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
        #         "command_name": "ee_pose",
        #         "std": 0.15,
        #     },
        # )
        # self.rewards.end_effector_position_success = RewTerm(
        #     func=position_command_success,
        #     weight=5.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
        #         "command_name": "ee_pose",
        #         "threshold": 0.04,
        #     },
        # )

        # self.rewards.end_effector_orientation_tracking.exp = RewTerm(
        #     func=orientation_command_error_exp,
        #     weight=2.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
        #         "command_name": "ee_pose",
        #         "std": 0.1,
        #     },
        # )

        # self.rewards.end_effector_orientation_success = RewTerm(
        #     func=orientation_command_success,
        #     weight=5.0,
        #     params={
        #         "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"]),
        #         "command_name": "ee_pose",
        #         "threshold": 0.1,
        #     },
        # )

        self.rewards.action_rate.weight = -0.002
        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.00005)
        self.rewards.joint_vel.weight = -0.0002
        self.rewards.joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-2.0)
        self.rewards.joint_torques = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-2.0e-6,
            params={"asset_cfg": SceneEntityCfg("robot")},
        )
        self.rewards.commanded_joint_torques = RewTerm(
            func=commanded_joint_torques_l2, weight=-2.0e-7
        )
        self.rewards.end_effector_linear_velocity = RewTerm(
            func=end_effector_linear_velocity_l2,
            weight=-0.005,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        self.rewards.torque_saturation = RewTerm(func=torque_saturation, weight=-0.25)

        self.curriculum.action_rate.params["weight"] = -0.002
        self.curriculum.joint_vel.params["weight"] = -0.0002


@configclass
class UR10OSCTorqueReachEnvCfg_PLAY(UR10OSCTorqueReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
