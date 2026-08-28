# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Replay the UR5e sweep checkpoint and mirror its raw OSC arm action over UDP.

This is a dedicated, fail-closed variant of :mod:`play`.  It accepts only the
``Isaac-Sweep-Object-UR5e-Random-v2`` task, runs one environment for one
episode in real time, and sends only the six-dimensional ``pose_rel`` arm
action.  The gripper action remains inside the simulation and is never placed
on the wire.

The UDP datagrams implement ``osc_pose_rel_v1``.  They are intentionally small
JSON objects so the independent ROS 2 bridge can reject malformed, stale, or
out-of-order commands without depending on Isaac Lab.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


TASK_ID = "Isaac-Sweep-Object-UR5e-Random-v2"
PROTOCOL = "osc_pose_rel_v1"
EXPECTED_POLICY_OBS_DIM = 35
EXPECTED_POLICY_ACTION_DIM = 7
EXPECTED_ARM_ACTION_DIM = 6
EXPECTED_STEP_DT = 0.02
DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[4]
    / "logs"
    / "01. Models"
    / "0813"
    / "2026-08-13_21-45-25 - OSC+pose_rel+s200+d1"
    / "model_89999.pt"
)


# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play the UR5e OSC checkpoint and mirror its 6D pose_rel arm action over UDP."
)
parser.add_argument("--video", action="store_true", default=False, help="Record a video of the simulated episode.")
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video in steps (the action stream still stops only at episode end).",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of environments. This dedicated sender accepts only 1.",
)
parser.add_argument(
    "--task",
    type=str,
    default=TASK_ID,
    help=f"Task name. This dedicated sender accepts only {TASK_ID}.",
)
parser.add_argument(
    "--agent",
    type=str,
    default="rsl_rl_cfg_entry_point",
    help="Name of the RL agent configuration entry point.",
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Unsupported here; use --checkpoint so the sim-to-real checkpoint is explicit.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=True,
    help="Compatibility flag. Real-time pacing is always enabled by this script.",
)
parser.add_argument(
    "--udp-host",
    type=str,
    default="127.0.0.1",
    help="Destination host for osc_pose_rel_v1 datagrams.",
)
parser.add_argument(
    "--udp-port",
    type=int,
    default=5005,
    help="Destination UDP port for osc_pose_rel_v1 datagrams.",
)
parser.add_argument(
    "--session-id",
    type=str,
    default=None,
    help="Optional non-empty stream session ID. The default is a random UUID.",
)
parser.add_argument(
    "--osc-log",
    type=str,
    default=None,
    help="CSV output path. Defaults to <checkpoint-dir>/sim2real_logs/<session-id>.csv.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
parser.set_defaults(checkpoint=str(DEFAULT_CHECKPOINT))
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# Fail before starting Isaac Sim when a command-line choice violates the wire contract.
if args_cli.task is None or args_cli.task.split(":")[-1] != TASK_ID:
    parser.error(f"--task must resolve to exactly {TASK_ID!r}.")
if args_cli.num_envs != 1:
    parser.error("--num_envs must be 1; vectorized actions cannot be mirrored to one real robot.")
if args_cli.use_pretrained_checkpoint:
    parser.error("--use_pretrained_checkpoint is not supported; pass the archived checkpoint with --checkpoint.")
if not 1 <= args_cli.udp_port <= 65535:
    parser.error("--udp-port must be in the inclusive range 1..65535.")
if not args_cli.udp_host.strip():
    parser.error("--udp-host must not be empty.")
if args_cli.session_id is not None and not args_cli.session_id.strip():
    parser.error("--session-id must not be empty when provided.")
if args_cli.session_id is not None and len(args_cli.session_id.strip()) > 128:
    parser.error("--session-id must contain at most 128 characters.")

# Real-time and single-environment operation are safety invariants, not preferences.
args_cli.num_envs = 1
args_cli.real_time = True

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Everything else follows after Isaac Sim is running."""

import csv
import json
import math
import os
import socket
import time
import uuid
from collections.abc import Sequence

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
    handle_deprecated_rsl_rl_cfg,
    handle_deprecated_rsl_rl_checkpoint,
)

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# PLACEHOLDER: Extension template (do not remove this comment)
try:
    import sweep_rl  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "sweep_rl":
        raise

try:
    import sweeping_policy  # noqa: F401
except ModuleNotFoundError as exc:
    if exc.name != "sweeping_policy":
        raise


class OscPoseRelUdpSender:
    """Send sequenced OSC actions and persist the exact datagrams to CSV."""

    _fieldnames = [
        "protocol",
        "event",
        "session_id",
        "sequence",
        "monotonic_ns",
        "send_wall_time_ns",
        "action_dx",
        "action_dy",
        "action_dz",
        "action_rx",
        "action_ry",
        "action_rz",
        "sim_tcp_x",
        "sim_tcp_y",
        "sim_tcp_z",
        "sim_tcp_qx",
        "sim_tcp_qy",
        "sim_tcp_qz",
        "sim_tcp_qw",
    ]

    def __init__(self, host: str, port: int, session_id: str, csv_path: str):
        self.host = host
        self.port = port
        self.session_id = session_id
        self.csv_path = csv_path
        self.sequence = 0
        self.action_count = 0
        self._stopped = False
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

        csv_dir = os.path.dirname(csv_path)
        if csv_dir:
            os.makedirs(csv_dir, exist_ok=True)
        # This handle intentionally lives for the sender's whole lifetime and
        # is closed together with the UDP socket in close().
        self._csv_file = open(csv_path, "x", encoding="utf-8", newline="")  # noqa: SIM115
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
        self._csv_writer.writeheader()
        self._csv_file.flush()

    @staticmethod
    def _finite_vector(values: Sequence[float], length: int, name: str) -> list[float]:
        result = [float(value) for value in values]
        if len(result) != length:
            raise ValueError(f"{name} must contain exactly {length} values; received {len(result)}.")
        if not all(math.isfinite(value) for value in result):
            raise ValueError(f"{name} contains a non-finite value: {result}.")
        return result

    @staticmethod
    def _encode(packet: dict) -> bytes:
        return json.dumps(packet, separators=(",", ":"), allow_nan=False).encode("utf-8")

    def _write_csv(
        self,
        packet: dict,
        send_wall_time_ns: int,
        action: Sequence[float] | None = None,
        sim_tcp_pose: Sequence[float] | None = None,
    ) -> None:
        row = {
            "protocol": packet["protocol"],
            "event": packet["event"],
            "session_id": packet["session_id"],
            "sequence": packet["sequence"],
            "monotonic_ns": packet["monotonic_ns"],
            "send_wall_time_ns": send_wall_time_ns,
        }
        if action is not None:
            for key, value in zip(self._fieldnames[6:12], action, strict=True):
                row[key] = value
        if sim_tcp_pose is not None:
            for key, value in zip(self._fieldnames[12:19], sim_tcp_pose, strict=True):
                row[key] = value
        self._csv_writer.writerow(row)
        self._csv_file.flush()

    def send_action(self, action: Sequence[float], sim_tcp_pose: Sequence[float]) -> None:
        """Send one raw arm action; no gripper field is accepted or serialized."""
        if self._stopped:
            raise RuntimeError("Cannot send an action after the stop event.")

        action_values = self._finite_vector(action, EXPECTED_ARM_ACTION_DIM, "action")
        sim_pose_values = self._finite_vector(sim_tcp_pose, 7, "sim_tcp_pose")
        monotonic_ns = time.monotonic_ns()
        packet = {
            "protocol": PROTOCOL,
            "event": "action",
            "session_id": self.session_id,
            "sequence": self.sequence,
            "monotonic_ns": monotonic_ns,
            "action": action_values,
            # ROS-facing quaternion order is x, y, z, w.
            "sim_tcp_pose": sim_pose_values,
        }
        self._socket.sendto(self._encode(packet), (self.host, self.port))
        self.sequence += 1
        self.action_count += 1
        self._write_csv(packet, time.time_ns(), action_values, sim_pose_values)

    def stop(self) -> None:
        """Send the terminal event once. Cleanup never hides an earlier exception."""
        if self._stopped:
            return
        self._stopped = True
        packet = {
            "protocol": PROTOCOL,
            "event": "stop",
            "session_id": self.session_id,
            "sequence": self.sequence,
            "monotonic_ns": time.monotonic_ns(),
        }
        try:
            self._socket.sendto(self._encode(packet), (self.host, self.port))
            self.sequence += 1
            self._write_csv(packet, time.time_ns())
        except (OSError, ValueError) as exc:
            print(f"[WARN] Failed to send or log the OSC stop packet: {exc}")

    def close(self) -> None:
        self.stop()
        self._socket.close()
        self._csv_file.close()


def _apply_archived_checkpoint_contract(env_cfg: ManagerBasedRLEnvCfg) -> None:
    """Override the mutable task config to the archived s200/d1 training contract."""
    try:
        arm_cfg = env_cfg.actions.arm_action
        controller_cfg = arm_cfg.controller_cfg
    except AttributeError as exc:
        raise ValueError(f"{TASK_ID} must expose actions.arm_action.controller_cfg.") from exc

    env_cfg.scene.num_envs = 1
    env_cfg.sim.dt = 0.01
    env_cfg.decimation = 2
    env_cfg.episode_length_s = 10.0

    # These values are recorded in the checkpoint's params/env.yaml.  Setting
    # them here avoids silently using a later source-tree baseline.
    controller_cfg.target_types = ["pose_rel"]
    controller_cfg.impedance_mode = "fixed"
    controller_cfg.motion_control_axes_task = (1, 1, 1, 1, 1, 1)
    controller_cfg.contact_wrench_control_axes_task = (0, 0, 0, 0, 0, 0)
    controller_cfg.inertial_dynamics_decoupling = True
    controller_cfg.partial_inertial_dynamics_decoupling = False
    controller_cfg.gravity_compensation = True
    controller_cfg.motion_stiffness_task = (200.0,) * 6
    controller_cfg.motion_damping_ratio_task = (1.0,) * 6
    controller_cfg.nullspace_control = "none"
    arm_cfg.position_scale = 1.0
    arm_cfg.orientation_scale = 1.0
    arm_cfg.clip = None
    arm_cfg.nullspace_joint_pos_target = "none"

    # OSC writes arm effort targets, so implicit arm drives must stay disabled.
    env_cfg.scene.robot.actuators["arm"].stiffness = 0.0
    env_cfg.scene.robot.actuators["arm"].damping = 0.0


def _validate_pre_make_contract(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg) -> None:
    """Reject configuration drift that would make the real command differ from simulation."""
    arm_cfg = env_cfg.actions.arm_action
    controller_cfg = arm_cfg.controller_cfg
    errors: list[str] = []

    if list(controller_cfg.target_types) != ["pose_rel"]:
        errors.append(f"target_types={controller_cfg.target_types!r}")
    if controller_cfg.impedance_mode != "fixed":
        errors.append(f"impedance_mode={controller_cfg.impedance_mode!r}")
    if tuple(controller_cfg.motion_control_axes_task) != (1, 1, 1, 1, 1, 1):
        errors.append(f"motion_control_axes_task={controller_cfg.motion_control_axes_task!r}")
    if tuple(controller_cfg.contact_wrench_control_axes_task) != (0, 0, 0, 0, 0, 0):
        errors.append(f"contact_wrench_control_axes_task={controller_cfg.contact_wrench_control_axes_task!r}")
    if controller_cfg.inertial_dynamics_decoupling is not True:
        errors.append(f"inertial_dynamics_decoupling={controller_cfg.inertial_dynamics_decoupling!r}")
    if controller_cfg.partial_inertial_dynamics_decoupling is not False:
        errors.append(f"partial_inertial_dynamics_decoupling={controller_cfg.partial_inertial_dynamics_decoupling!r}")
    if controller_cfg.gravity_compensation is not True:
        errors.append(f"gravity_compensation={controller_cfg.gravity_compensation!r}")
    if tuple(controller_cfg.motion_stiffness_task) != (200.0,) * 6:
        errors.append(f"motion_stiffness_task={controller_cfg.motion_stiffness_task!r}")
    if tuple(controller_cfg.motion_damping_ratio_task) != (1.0,) * 6:
        errors.append(f"motion_damping_ratio_task={controller_cfg.motion_damping_ratio_task!r}")
    if arm_cfg.position_scale != 1.0 or arm_cfg.orientation_scale != 1.0:
        errors.append(f"action scales position={arm_cfg.position_scale!r}, orientation={arm_cfg.orientation_scale!r}")
    if arm_cfg.clip is not None:
        errors.append(f"arm clip={arm_cfg.clip!r}")
    if controller_cfg.nullspace_control != "none" or arm_cfg.nullspace_joint_pos_target != "none":
        errors.append(
            f"nullspace control={controller_cfg.nullspace_control!r}, "
            f"joint target={arm_cfg.nullspace_joint_pos_target!r}"
        )
    if arm_cfg.body_name != "robotiq_base_link":
        errors.append(f"body_name={arm_cfg.body_name!r}")
    if arm_cfg.body_offset is None or tuple(arm_cfg.body_offset.pos) != (0.13, 0.0, 0.0):
        errors.append(f"body_offset={arm_cfg.body_offset!r}")
    elif tuple(arm_cfg.body_offset.rot) != (1.0, 0.0, 0.0, 0.0):
        errors.append(f"body_offset rotation={arm_cfg.body_offset.rot!r}")
    if arm_cfg.task_frame_rel_path is not None:
        errors.append(f"task_frame_rel_path={arm_cfg.task_frame_rel_path!r}")
    if agent_cfg.clip_actions is not None:
        errors.append(f"agent clip_actions={agent_cfg.clip_actions!r}")
    if not math.isclose(env_cfg.sim.dt * env_cfg.decimation, EXPECTED_STEP_DT, rel_tol=0.0, abs_tol=1e-12):
        errors.append(f"step_dt={env_cfg.sim.dt * env_cfg.decimation!r}")

    if errors:
        raise ValueError("Archived checkpoint contract mismatch: " + "; ".join(errors))


def _validate_runtime_contract(env: RslRlVecEnvWrapper) -> slice:
    """Validate the instantiated action layout and return the arm slice."""
    action_manager = env.unwrapped.action_manager
    term_names = list(action_manager.active_terms)
    term_dims = list(action_manager.action_term_dim)
    if term_names != ["arm_action", "gripper_action"] or term_dims != [6, 1]:
        raise ValueError(
            "Expected action terms [('arm_action', 6), ('gripper_action', 1)], "
            f"received {list(zip(term_names, term_dims, strict=True))}."
        )
    if action_manager.total_action_dim != EXPECTED_POLICY_ACTION_DIM:
        raise ValueError(
            f"Expected total policy action dimension {EXPECTED_POLICY_ACTION_DIM}, "
            f"received {action_manager.total_action_dim}."
        )
    if env.clip_actions is not None:
        raise ValueError(f"RSL-RL action clipping must be disabled, received {env.clip_actions!r}.")
    if not math.isclose(env.unwrapped.step_dt, EXPECTED_STEP_DT, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"Expected step_dt={EXPECTED_STEP_DT}, received {env.unwrapped.step_dt}.")
    return slice(0, EXPECTED_ARM_ACTION_DIM)


def _validate_policy_observation(obs) -> None:
    try:
        policy_obs = obs["policy"]
    except (KeyError, TypeError) as exc:
        raise ValueError("The checkpoint requires a TensorDict observation group named 'policy'.") from exc
    if policy_obs.ndim != 2 or tuple(policy_obs.shape) != (1, EXPECTED_POLICY_OBS_DIM):
        raise ValueError(
            f"Expected policy observation shape (1, {EXPECTED_POLICY_OBS_DIM}), received {tuple(policy_obs.shape)}."
        )
    if not torch.isfinite(policy_obs).all().item():
        raise ValueError("The initial policy observation contains a non-finite value.")


def _sim_tcp_pose_xyzw(base_env) -> list[float]:
    """Return current controlled TCP pose in robot-root frame as x,y,z,qx,qy,qz,qw."""
    robot = base_env.scene["robot"]
    ee_frame = base_env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    tcp_pos_b, tcp_quat_b_wxyz = subtract_frame_transforms(
        robot.data.root_pos_w,
        robot.data.root_quat_w,
        ee_pos_w,
        ee_quat_w,
    )
    pose = torch.cat(
        (
            tcp_pos_b[0],
            tcp_quat_b_wxyz[0, 1:4],
            tcp_quat_b_wxyz[0, 0:1],
        )
    )
    if tuple(pose.shape) != (7,) or not torch.isfinite(pose).all().item():
        raise ValueError(f"Invalid simulated TCP pose: shape={tuple(pose.shape)}, value={pose}.")
    return pose.detach().cpu().tolist()


def _resolve_csv_path(resume_path: str, session_id: str) -> str:
    if args_cli.osc_log:
        return os.path.abspath(os.path.expanduser(args_cli.osc_log))
    return os.path.join(os.path.dirname(resume_path), "sim2real_logs", f"osc_actions_{session_id}.csv")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    agent_cfg: RslRlBaseRunnerCfg,
):
    """Play one simulated episode while streaming its exact raw arm action."""
    task_name = args_cli.task.split(":")[-1]
    if task_name != TASK_ID:
        raise ValueError(f"This sender only supports {TASK_ID}; received {task_name}.")
    if not isinstance(env_cfg, ManagerBasedRLEnvCfg):
        raise TypeError(f"{TASK_ID} must use ManagerBasedRLEnvCfg; received {type(env_cfg).__name__}.")

    # Apply non-Hydra CLI values first, then pin the archived environment contract.
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    agent_cfg = handle_deprecated_rsl_rl_cfg(agent_cfg, installed_version)
    _apply_archived_checkpoint_contract(env_cfg)

    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    _validate_pre_make_contract(env_cfg, agent_cfg)

    if not args_cli.checkpoint:
        raise ValueError("A checkpoint is required.")
    resume_path = retrieve_file_path(args_cli.checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Loading model checkpoint from: {resume_path}")

    env = None
    sender = None
    try:
        env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
        if isinstance(env.unwrapped, DirectMARLEnv):
            raise TypeError(f"{TASK_ID} unexpectedly created a DirectMARLEnv.")

        if args_cli.video:
            video_kwargs = {
                "video_folder": os.path.join(log_dir, "videos", "play_osc_sim2real"),
                "step_trigger": lambda step: step == 0,
                "video_length": args_cli.video_length,
                "disable_logger": True,
            }
            print("[INFO] Recording the simulated episode.")
            print_dict(video_kwargs, nesting=4)
            env = gym.wrappers.RecordVideo(env, **video_kwargs)

        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        arm_slice = _validate_runtime_contract(env)

        if agent_cfg.class_name == "OnPolicyRunner":
            runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        elif agent_cfg.class_name == "DistillationRunner":
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
        else:
            raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")

        resume_path = handle_deprecated_rsl_rl_checkpoint(resume_path, installed_version)
        runner.load(resume_path)
        policy = runner.get_inference_policy(device=env.unwrapped.device)

        # Preserve play.py's model export behavior.
        export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
        if version.parse(installed_version) >= version.parse("4.0.0"):
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
            policy_nn = None
        else:
            if version.parse(installed_version) >= version.parse("2.3.0"):
                policy_nn = runner.alg.policy
            else:
                policy_nn = runner.alg.actor_critic

            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")

        obs = env.get_observations()
        _validate_policy_observation(obs)

        session_id = args_cli.session_id.strip() if args_cli.session_id else uuid.uuid4().hex
        csv_path = _resolve_csv_path(resume_path, session_id)
        sender = OscPoseRelUdpSender(args_cli.udp_host, args_cli.udp_port, session_id, csv_path)
        dt = env.unwrapped.step_dt
        print(
            "[INFO] OSC sim-to-real contract: pose_rel=6D, gripper=excluded, "
            "stiffness=200, damping_ratio=1, gravity_compensation=true"
        )
        print(f"[INFO] UDP destination: {args_cli.udp_host}:{args_cli.udp_port}")
        print(f"[INFO] Session ID: {session_id}")
        print(f"[INFO] CSV log: {csv_path}")
        print(f"[INFO] Policy period: {dt:.6f} s ({1.0 / dt:.1f} Hz); stopping after the first episode")

        while simulation_app.is_running():
            step_start = time.monotonic()
            with torch.inference_mode():
                actions = policy(obs)
                if tuple(actions.shape) != (1, EXPECTED_POLICY_ACTION_DIM):
                    raise ValueError(
                        f"Expected policy action shape (1, {EXPECTED_POLICY_ACTION_DIM}), "
                        f"received {tuple(actions.shape)}."
                    )
                if not torch.isfinite(actions).all().item():
                    raise ValueError(f"Policy produced a non-finite action: {actions}.")

                # clip_actions is asserted null. This exact tensor is both sent
                # and handed to env.step; no gripper scalar enters the packet.
                arm_action = actions[0, arm_slice].detach().cpu().tolist()
                sender.send_action(arm_action, _sim_tcp_pose_xyzw(env.unwrapped))
                obs, _, dones, _ = env.step(actions)

                if version.parse(installed_version) >= version.parse("4.0.0"):
                    policy.reset(dones)
                else:
                    policy_nn.reset(dones)

            if bool(dones[0].item()):
                print(f"[INFO] Episode complete after {sender.action_count} streamed actions; sending stop.")
                break

            sleep_time = dt - (time.monotonic() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if sender is not None:
            sender.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
