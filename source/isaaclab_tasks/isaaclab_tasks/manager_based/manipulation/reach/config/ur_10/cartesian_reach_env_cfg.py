# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp.actions.task_space_actions import DifferentialInverseKinematicsAction
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import axis_angle_from_quat, quat_from_euler_xyz, subtract_frame_transforms
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.agents.rsl_rl_ppo_cfg import (
    UR10ReachPPORunnerCfg,
)
from isaaclab_tasks.manager_based.manipulation.reach.config.ur_10.joint_pos_env_cfg import (
    UR10ReachEnvCfg,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class CartesianDifferentialInverseKinematicsAction(DifferentialInverseKinematicsAction):
    """Differential IK action with Cartesian workspace and joint-target safety clamps."""

    cfg: "CartesianDifferentialInverseKinematicsActionCfg"

    def __init__(self, cfg: "CartesianDifferentialInverseKinematicsActionCfg", env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._workspace_bounds = torch.tensor(cfg.workspace_bounds, device=self.device, dtype=torch.float32)
        self._ik_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._workspace_clamped = torch.zeros_like(self._ik_failure)
        self._joint_limit_clamped = torch.zeros_like(self._ik_failure)

    @property
    def ik_failure(self) -> torch.Tensor:
        return self._ik_failure

    @property
    def workspace_clamped(self) -> torch.Tensor:
        return self._workspace_clamped

    @property
    def joint_limit_clamped(self) -> torch.Tensor:
        return self._joint_limit_clamped

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        self._processed_actions[:] = self.raw_actions * self._scale
        # DifferentialIKController consumes rotational deltas as axis-angle, while the policy action is RPY.
        delta_quat = quat_from_euler_xyz(
            self._processed_actions[:, 3], self._processed_actions[:, 4], self._processed_actions[:, 5]
        )
        self._processed_actions[:, 3:6] = axis_angle_from_quat(delta_quat)

        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        self._ik_controller.set_command(self._processed_actions, ee_pos_curr, ee_quat_curr)

        desired_pos = self._ik_controller.ee_pos_des
        lower = self._workspace_bounds[:, 0]
        upper = self._workspace_bounds[:, 1]
        clamped_pos = torch.max(torch.min(desired_pos, upper), lower)
        self._workspace_clamped[:] = torch.any(
            torch.abs(clamped_pos - desired_pos) > self.cfg.clamp_tolerance, dim=1
        )
        self._ik_controller.ee_pos_des[:] = clamped_pos

    def apply_actions(self):
        ee_pos_curr, ee_quat_curr = self._compute_frame_pose()
        joint_pos = self._asset.data.joint_pos[:, self._joint_ids]

        quat_valid = torch.linalg.norm(ee_quat_curr, dim=1) > self.cfg.clamp_tolerance
        try:
            jacobian = self._compute_frame_jacobian()
            joint_pos_des = self._ik_controller.compute(ee_pos_curr, ee_quat_curr, jacobian, joint_pos)
            finite_solution = torch.isfinite(joint_pos_des).all(dim=1)
        except RuntimeError:
            joint_pos_des = joint_pos.clone()
            finite_solution = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        valid_solution = quat_valid & finite_solution
        joint_pos_des = torch.where(valid_solution.unsqueeze(-1), joint_pos_des, joint_pos)

        joint_limits = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, :]
        joint_pos_clamped = torch.max(torch.min(joint_pos_des, joint_limits[..., 1]), joint_limits[..., 0])
        self._joint_limit_clamped[:] = torch.any(
            torch.abs(joint_pos_clamped - joint_pos_des) > self.cfg.clamp_tolerance, dim=1
        )
        self._ik_failure[:] = ~valid_solution

        self._asset.set_joint_position_target(joint_pos_clamped, self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._processed_actions[env_ids] = 0.0
        self._ik_failure[env_ids] = False
        self._workspace_clamped[env_ids] = False
        self._joint_limit_clamped[env_ids] = False


@configclass
class CartesianDifferentialInverseKinematicsActionCfg(DifferentialInverseKinematicsActionCfg):
    """Configuration for a bounded 6D relative Cartesian EEF action."""

    class_type: type = CartesianDifferentialInverseKinematicsAction

    workspace_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.25, 0.75),
        (-0.35, 0.35),
        (0.05, 0.65),
    )
    """Root-frame Cartesian limits for the commanded EEF target position."""

    clamp_tolerance: float = 1.0e-6
    """Tolerance used to decide whether a clamp or invalid command occurred."""


def end_effector_position_b(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """End-effector position in the robot root frame."""
    asset = env.scene[asset_cfg.name]
    ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    ee_pos_b, _ = subtract_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def end_effector_to_target_b(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Vector from the current EEF position to the reach target in the robot root frame."""
    command = env.command_manager.get_command(command_name)
    return command[:, :3] - end_effector_position_b(env, asset_cfg)


def cartesian_ik_failure(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalty indicator for invalid IK output."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.ik_failure.float()


def cartesian_workspace_clamped(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalty indicator for Cartesian target commands outside the configured workspace."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.workspace_clamped.float()


def cartesian_joint_limit_clamped(env: ManagerBasedRLEnv, action_name: str = "arm_action") -> torch.Tensor:
    """Penalty indicator for IK joint targets clamped to soft joint limits."""
    action_term = env.action_manager.get_term(action_name)
    return action_term.joint_limit_clamped.float()


@configclass
class CartesianObservationsCfg:
    """Observation terms for the Cartesian UR10 reach task."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        ee_position = ObsTerm(
            func=end_effector_position_b,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        ee_to_target = ObsTerm(
            func=end_effector_to_target_b,
            params={"command_name": "ee_pose", "asset_cfg": SceneEntityCfg("robot", body_names=["ee_link"])},
        )
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class UR10CartesianReachEnvCfg(UR10ReachEnvCfg):
    """UR10 reach task with 6D relative Cartesian end-effector actions."""

    observations: CartesianObservationsCfg = CartesianObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.actions.arm_action = CartesianDifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[".*"],
            body_name="ee_link",
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=(0.05, 0.05, 0.05, 0.25, 0.25, 0.25),
            body_offset=CartesianDifferentialInverseKinematicsActionCfg.OffsetCfg(),
            workspace_bounds=((0.25, 0.75), (-0.35, 0.35), (0.05, 0.65)),
        )

        self.rewards.action_l2 = RewTerm(func=mdp.action_l2, weight=-0.0001)
        self.rewards.joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)
        self.rewards.cartesian_workspace_clamped = RewTerm(func=cartesian_workspace_clamped, weight=-0.05)
        self.rewards.cartesian_joint_limit_clamped = RewTerm(func=cartesian_joint_limit_clamped, weight=-0.25)
        self.rewards.cartesian_ik_failure = RewTerm(func=cartesian_ik_failure, weight=-1.0)


@configclass
class UR10CartesianReachEnvCfg_PLAY(UR10CartesianReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


@configclass
class UR10CartesianReachPPORunnerCfg(UR10ReachPPORunnerCfg):
    experiment_name = "reach_ur10_cartesian"
