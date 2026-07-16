# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e inclined-contact and sliding-force demonstration.

This program verifies all directional components of the force acting on a
cylindrical sensor/tool fixed to ``wrist_3_link``.

Main changes from the previous version
--------------------------------------
1. Resolves the exact collision Plane prim below GroundPlane and uses it as the
   ContactSensor filter. Filtering only with the GroundPlane root can leave the
   friction result empty.
2. Assigns explicit high-friction PhysX materials to both the ground and tool.
3. Adds a pressed horizontal sliding phase by moving only shoulder_pan.
4. Logs normal, friction, and total forces in both world frame (``*_w``) and
   cylindrical sensor/tool frame (``*_s``).
5. Logs at 100 Hz by default while physics and sensors update at 200 Hz.

Usage:
    ./isaaclab.sh -p scripts/demos/ur10e_floor_press_demo.py \
        --device cpu --cycles 4 --headless

Recommended visual test:
    ./isaaclab.sh -p scripts/demos/ur10e_floor_press_demo.py \
        --device cpu --cycles 3 --angle_jitter_deg 30 \
        --slide_pan_deg 5 --press_scale 1.15
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="UR10e inclined press and sliding-force logging demo."
)
parser.add_argument("--cycles", type=int, default=3)
parser.add_argument(
    "--press_scale",
    type=float,
    default=1.2,
    help="Increase if the tilted tool does not contact the ground.",
)
parser.add_argument(
    "--depth_jitter_mm",
    type=float,
    default=0.0,
    help="Uniform per-cycle press-depth perturbation in mm (±value).",
)
parser.add_argument(
    "--angle_jitter_deg",
    type=float,
    default=0.0,
    help="Maximum approximate tool-tilt command in degrees.",
)
parser.add_argument(
    "--slide_pan_deg",
    type=float,
    default=0.0,
    help="Maximum shoulder-pan displacement during pressed sliding.",
)
parser.add_argument("--static_friction", type=float, default=1.2)
parser.add_argument("--dynamic_friction", type=float, default=1.0)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--log_hz", type=float, default=100.0)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below executes after Isaac Sim is launched."""

import csv
import math
import os
import random
from dataclasses import fields
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.universal_robots import UR10e_CFG  # isort:skip


# -----------------------------------------------------------------------------
# Simulation and motion parameters
# -----------------------------------------------------------------------------
SIM_DT = 0.005  # 200 Hz

INITIAL_HOLD_TIME = 1.0
DESCEND_TIME = 1.0
PRESS_SETTLE_TIME = 0.35
SLIDE_OUT_TIME = 0.80
SLIDE_BACK_TIME = 0.80
ASCEND_TIME = 1.0
RELEASE_HOLD_TIME = 0.40

CYCLE_TIME = (
    DESCEND_TIME
    + PRESS_SETTLE_TIME
    + SLIDE_OUT_TIME
    + SLIDE_BACK_TIME
    + ASCEND_TIME
    + RELEASE_HOLD_TIME
)
TOTAL_TIME = INITIAL_HOLD_TIME + max(1, args_cli.cycles) * CYCLE_TIME
TOTAL_STEPS = math.ceil(TOTAL_TIME / SIM_DT)

# UR10e joint order:
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
RELEASE_JOINTS = [
    math.pi,
    -1.13651330,
    2.10809615,
    -2.54237918,
    -math.pi / 2.0,
    0.0,
]
PRESS_JOINTS = [
    math.pi,
    -0.92597591,
    2.11699067,
    -2.76181108,
    -math.pi / 2.0,
    0.0,
]

NOMINAL_VERTICAL_TRAVEL_M = 0.125

ROBOT_PATH = "/World/Robot"
WRIST_BODY_PATH = f"{ROBOT_PATH}/wrist_3_link"
TOOL_BODY_PATH = f"{ROBOT_PATH}/contact_tool"
TOOL_JOINT_PATH = f"{ROBOT_PATH}/joints/wrist_3_to_contact_tool"
GROUND_PATH = "/World/GroundPlane"

TOOL_RADIUS = 0.035
TOOL_HEIGHT = 0.018
TOOL_WRIST_GAP = 0.001
TOOL_CENTER_OFFSET = TOOL_WRIST_GAP + 0.5 * TOOL_HEIGHT
TOOL_MASS = 0.025


# -----------------------------------------------------------------------------
# Randomized commands
# -----------------------------------------------------------------------------
def _make_cycle_commands() -> list[dict[str, Any]]:
    rng = random.Random(args_cli.seed)
    commands: list[dict[str, Any]] = []

    depth_limit_m = max(0.0, args_cli.depth_jitter_mm) * 1.0e-3
    tilt_limit_rad = math.radians(max(0.0, args_cli.angle_jitter_deg))
    slide_limit_rad = math.radians(max(0.0, args_cli.slide_pan_deg))

    for cycle_index in range(max(1, args_cli.cycles)):
        depth_delta_m = rng.uniform(-depth_limit_m, depth_limit_m)

        # One two-dimensional tilt command with a bounded magnitude.
        tilt_magnitude = rng.uniform(0.0, tilt_limit_rad)
        tilt_azimuth = rng.uniform(0.0, 2.0 * math.pi)
        wrist_1_delta = tilt_magnitude * math.cos(tilt_azimuth)
        wrist_2_delta = tilt_magnitude * math.sin(tilt_azimuth)

        cycle_press_scale = max(
            0.0,
            args_cli.press_scale + depth_delta_m / NOMINAL_VERTICAL_TRAVEL_M,
        )

        press_target = [
            release + cycle_press_scale * (press - release)
            for release, press in zip(RELEASE_JOINTS, PRESS_JOINTS)
        ]
        press_target[3] += wrist_1_delta
        press_target[4] += wrist_2_delta

        # Changing only shoulder_pan sweeps the entire arm around world Z.
        # Tool height is approximately preserved, making this a deliberate
        # horizontal sliding command under contact.
        slide_sign = -1.0 if rng.random() < 0.5 else 1.0
        slide_fraction = rng.uniform(0.70, 1.00)
        slide_pan_delta = slide_sign * slide_limit_rad * slide_fraction
        slide_target = press_target.copy()
        slide_target[0] += slide_pan_delta

        commands.append(
            {
                "cycle_index": cycle_index,
                "press_joints": press_target,
                "slide_joints": slide_target,
                "press_scale": cycle_press_scale,
                "depth_delta_m": depth_delta_m,
                "tilt_magnitude_rad": tilt_magnitude,
                "tilt_azimuth_rad": tilt_azimuth,
                "wrist_1_delta_rad": wrist_1_delta,
                "wrist_2_delta_rad": wrist_2_delta,
                "slide_pan_delta_rad": slide_pan_delta,
            }
        )

    return commands


CYCLE_COMMANDS = _make_cycle_commands()


# -----------------------------------------------------------------------------
# Math helpers
# -----------------------------------------------------------------------------
def _cosine_blend(alpha: float) -> float:
    alpha = max(0.0, min(1.0, alpha))
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def _interpolate_joints(
    start: list[float], end: list[float], alpha: float
) -> list[float]:
    blend = _cosine_blend(alpha)
    return [s + blend * (e - s) for s, e in zip(start, end)]


def _normalize_quaternion_wxyz(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-12)


def _rotate_local_to_world(
    vector_s: torch.Tensor, quat_wxyz: torch.Tensor
) -> torch.Tensor:
    """Rotate a vector from sensor/tool frame to world frame."""
    quat = _normalize_quaternion_wxyz(quat_wxyz)
    qw = quat[0]
    qv = quat[1:4]
    return (
        vector_s
        + 2.0 * qw * torch.cross(qv, vector_s, dim=0)
        + 2.0 * torch.cross(qv, torch.cross(qv, vector_s, dim=0), dim=0)
    )


def _rotate_world_to_local(
    vector_w: torch.Tensor, quat_wxyz: torch.Tensor
) -> torch.Tensor:
    """Rotate a vector from world frame to sensor/tool frame."""
    quat = _normalize_quaternion_wxyz(quat_wxyz)
    qw = quat[0]
    qv = quat[1:4]
    return (
        vector_w
        - 2.0 * qw * torch.cross(qv, vector_w, dim=0)
        + 2.0 * torch.cross(qv, torch.cross(qv, vector_w, dim=0), dim=0)
    )


def _safe_vector(vector: torch.Tensor) -> torch.Tensor:
    return torch.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)


# -----------------------------------------------------------------------------
# USD / PhysX helpers
# -----------------------------------------------------------------------------
def _find_ground_collider_path(root_path: str) -> str:
    """Return the exact collision Plane prim under GroundPlane."""
    from pxr import Usd, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext is not available.")

    stage = sim_context.stage
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Ground root prim does not exist: '{root_path}'.")

    collision_prims = [
        prim
        for prim in Usd.PrimRange(root_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not collision_prims:
        raise RuntimeError(f"No collision prim exists below '{root_path}'.")

    plane_prims = [prim for prim in collision_prims if prim.GetTypeName() == "Plane"]
    selected = plane_prims[0] if plane_prims else collision_prims[0]

    paths = [prim.GetPath().pathString for prim in collision_prims]
    selected_path = selected.GetPath().pathString
    print(f"[INFO]: Ground collision prims: {paths}")
    print(f"[INFO]: ContactSensor ground filter: {selected_path}")
    return selected_path


def _ensure_contact_reporter(prim_path: str) -> None:
    from pxr import PhysxSchema, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext is not available.")

    prim = sim_context.stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Contact body does not exist: '{prim_path}'.")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Contact body is not a rigid body: '{prim_path}'.")

    reporter = PhysxSchema.PhysxContactReportAPI(prim)
    if not reporter:
        reporter = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    reporter.CreateThresholdAttr().Set(0.0)


def _get_world_pose_from_local_offset(
    parent_path: str, local_offset: tuple[float, float, float]
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    from pxr import Gf, UsdGeom

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext is not available.")

    parent_prim = sim_context.stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise RuntimeError(f"Parent prim does not exist: '{parent_path}'.")

    world_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(parent_prim)
    world_position = world_matrix.Transform(Gf.Vec3d(*local_offset))
    world_quaternion = world_matrix.ExtractRotationQuat()
    imaginary = world_quaternion.GetImaginary()

    position = tuple(float(world_position[i]) for i in range(3))
    orientation_wxyz = (
        float(world_quaternion.GetReal()),
        float(imaginary[0]),
        float(imaginary[1]),
        float(imaginary[2]),
    )
    return position, orientation_wxyz


def _make_contact_material() -> sim_utils.RigidBodyMaterialCfg:
    return sim_utils.RigidBodyMaterialCfg(
        static_friction=max(0.0, args_cli.static_friction),
        dynamic_friction=max(0.0, args_cli.dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
        restitution_combine_mode="min",
    )


def _create_fixed_cylindrical_tool() -> None:
    from pxr import Gf, Sdf, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext is not available.")
    stage = sim_context.stage

    tool_position, tool_orientation = _get_world_pose_from_local_offset(
        WRIST_BODY_PATH,
        (0.0, 0.0, TOOL_CENTER_OFFSET),
    )

    tool_cfg = sim_utils.CylinderCfg(
        radius=TOOL_RADIUS,
        height=TOOL_HEIGHT,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=True,
            linear_damping=0.0,
            angular_damping=0.0,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
            max_depenetration_velocity=1.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=TOOL_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.003,
            rest_offset=0.0,
        ),
        physics_material=_make_contact_material(),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.15, 0.05),
            metallic=0.05,
            roughness=0.75,
        ),
    )
    tool_cfg.func(
        TOOL_BODY_PATH,
        tool_cfg,
        translation=tool_position,
        orientation=tool_orientation,
    )

    tool_prim = stage.GetPrimAtPath(TOOL_BODY_PATH)
    if not tool_prim.IsValid() or not tool_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Failed to create rigid tool at '{TOOL_BODY_PATH}'.")

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(TOOL_JOINT_PATH))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(WRIST_BODY_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(TOOL_BODY_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, TOOL_CENTER_OFFSET))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateCollisionEnabledAttr().Set(False)

    print(
        "[INFO]: Cylindrical sensor/tool created | "
        f"radius={TOOL_RADIUS:.3f} m, height={TOOL_HEIGHT:.3f} m, "
        f"static_friction={max(0.0, args_cli.static_friction):.3f}, "
        f"dynamic_friction={max(0.0, args_cli.dynamic_friction):.3f}"
    )


# -----------------------------------------------------------------------------
# Scene
# -----------------------------------------------------------------------------
def design_scene() -> dict[str, object]:
    ground_cfg = sim_utils.GroundPlaneCfg(
        physics_material=_make_contact_material(),
    )
    ground_cfg.func(GROUND_PATH, ground_cfg)
    ground_collider_path = _find_ground_collider_path(GROUND_PATH)

    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = UR10e_CFG.copy()
    robot_cfg.prim_path = ROBOT_PATH
    robot_cfg.spawn.activate_contact_sensors = True
    robot = Articulation(cfg=robot_cfg)

    _create_fixed_cylindrical_tool()
    _ensure_contact_reporter(TOOL_BODY_PATH)

    contact_sensor_cfg = ContactSensorCfg(
        prim_path=TOOL_BODY_PATH,
        update_period=0.0,
        history_length=8,
        track_pose=True,
        track_contact_points=True,
        track_friction_forces=True,
        track_air_time=True,
        force_threshold=0.1,
        max_contact_data_count_per_prim=64,
        debug_vis=not args_cli.headless,
        # Use the exact Plane collision prim, not the GroundPlane root.
        filter_prim_paths_expr=[ground_collider_path],
    )
    contact_sensor = ContactSensor(cfg=contact_sensor_cfg)

    return {
        "robot": robot,
        "contact_sensor": contact_sensor,
        "ground_collider_path": ground_collider_path,
    }


# -----------------------------------------------------------------------------
# Trajectory
# -----------------------------------------------------------------------------
def _get_motion_command(
    sim_time: float,
) -> tuple[list[float], str, int, dict[str, Any] | None]:
    if sim_time < INITIAL_HOLD_TIME:
        return RELEASE_JOINTS, "INITIAL_RELEASE", 0, None

    elapsed = sim_time - INITIAL_HOLD_TIME
    cycle_index = min(int(elapsed // CYCLE_TIME), len(CYCLE_COMMANDS) - 1)
    cycle_time = elapsed - cycle_index * CYCLE_TIME
    cycle_number = cycle_index + 1
    command = CYCLE_COMMANDS[cycle_index]

    press_joints = command["press_joints"]
    slide_joints = command["slide_joints"]

    if cycle_time < DESCEND_TIME:
        alpha = cycle_time / DESCEND_TIME
        return (
            _interpolate_joints(RELEASE_JOINTS, press_joints, alpha),
            "DESCEND",
            cycle_number,
            command,
        )

    cycle_time -= DESCEND_TIME
    if cycle_time < PRESS_SETTLE_TIME:
        return press_joints, "PRESS_SETTLE", cycle_number, command

    cycle_time -= PRESS_SETTLE_TIME
    if cycle_time < SLIDE_OUT_TIME:
        alpha = cycle_time / SLIDE_OUT_TIME
        return (
            _interpolate_joints(press_joints, slide_joints, alpha),
            "SLIDE_OUT",
            cycle_number,
            command,
        )

    cycle_time -= SLIDE_OUT_TIME
    if cycle_time < SLIDE_BACK_TIME:
        alpha = cycle_time / SLIDE_BACK_TIME
        return (
            _interpolate_joints(slide_joints, press_joints, alpha),
            "SLIDE_BACK",
            cycle_number,
            command,
        )

    cycle_time -= SLIDE_BACK_TIME
    if cycle_time < ASCEND_TIME:
        alpha = cycle_time / ASCEND_TIME
        return (
            _interpolate_joints(press_joints, RELEASE_JOINTS, alpha),
            "ASCEND",
            cycle_number,
            command,
        )

    return RELEASE_JOINTS, "RELEASE_HOLD", cycle_number, command


def _initialize_robot_pose(robot: Articulation, sim: SimulationContext) -> None:
    joint_pos = torch.tensor(
        [RELEASE_JOINTS],
        dtype=torch.float32,
        device=sim.device,
    )
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()


# -----------------------------------------------------------------------------
# Contact extraction
# -----------------------------------------------------------------------------
def _sum_filter_vectors(
    tensor: torch.Tensor | None,
    device: str | torch.device,
) -> torch.Tensor:
    """Sum all filtered vectors for environment 0 and body 0."""
    if tensor is None:
        return torch.zeros(3, dtype=torch.float32, device=device)
    values = tensor[0, 0]
    return _safe_vector(values).reshape(-1, 3).sum(dim=0)


def _read_contact_data(
    contact_sensor: ContactSensor,
    sim_dt: float,
) -> dict[str, torch.Tensor | float | int]:
    data = contact_sensor.data
    device = data.net_forces_w.device

    # Filtered normal force: tool against the exact ground collider only.
    normal_force_w = _sum_filter_vectors(data.force_matrix_w, device)

    # Filtered friction force. Sum all resolved filter shapes.
    friction_force_w = _sum_filter_vectors(data.friction_forces_w, device)

    # Direct PhysX patch-buffer read for diagnostics and as a single-pair
    # fallback. The view contains one sensor body and one ground filter.
    raw_friction_force_w = torch.zeros(3, dtype=torch.float32, device=device)
    friction_patch_count = 0
    try:
        raw_friction, _, buffer_count, _ = (
            contact_sensor.contact_physx_view.get_friction_data(dt=sim_dt)
        )
        if raw_friction.numel() > 0:
            raw_friction_force_w = _safe_vector(raw_friction.reshape(-1, 3)).sum(dim=0)
        friction_patch_count = int(buffer_count.sum().item())
    except Exception:
        raw_friction_force_w = friction_force_w.clone()

    if friction_patch_count > 0:
        friction_force_w = raw_friction_force_w

    total_force_w = normal_force_w + friction_force_w

    tool_position_w = data.pos_w[0, 0]
    tool_quat_wxyz = data.quat_w[0, 0]

    # These values answer: "Which direction does force act relative to the
    # sensor face?" The cylinder body frame is the sensor frame.
    normal_force_s = _rotate_world_to_local(normal_force_w, tool_quat_wxyz)
    friction_force_s = _rotate_world_to_local(friction_force_w, tool_quat_wxyz)
    total_force_s = _rotate_world_to_local(total_force_w, tool_quat_wxyz)

    local_axis_z = torch.tensor(
        [0.0, 0.0, 1.0],
        dtype=tool_position_w.dtype,
        device=device,
    )
    tool_axis_z_w = _rotate_local_to_world(local_axis_z, tool_quat_wxyz)
    cos_tilt = torch.clamp(torch.abs(tool_axis_z_w[2]), 0.0, 1.0)
    tilt_from_normal_deg = float(torch.rad2deg(torch.acos(cos_tilt)).item())

    contact_position_w = torch.full(
        (3,),
        float("nan"),
        dtype=tool_position_w.dtype,
        device=device,
    )
    if data.contact_pos_w is not None:
        positions = data.contact_pos_w[0, 0].reshape(-1, 3)
        valid = ~torch.isnan(positions).any(dim=-1)
        if bool(valid.any()):
            contact_position_w = positions[valid].mean(dim=0)

    return {
        "normal_force_w": normal_force_w,
        "friction_force_w": friction_force_w,
        "total_force_w": total_force_w,
        "normal_force_s": normal_force_s,
        "friction_force_s": friction_force_s,
        "total_force_s": total_force_s,
        "tool_position_w": tool_position_w,
        "tool_quat_wxyz": tool_quat_wxyz,
        "tool_axis_z_w": tool_axis_z_w,
        "tilt_from_normal_deg": tilt_from_normal_deg,
        "contact_position_w": contact_position_w,
        "friction_patch_count": friction_patch_count,
    }


# -----------------------------------------------------------------------------
# Exhaustive ContactSensorData logging
# -----------------------------------------------------------------------------
def _raw_sensor_headers_and_values(data: Any) -> tuple[list[str], list[Any]]:
    """Flatten every ContactSensorData field for environment 0.

    Tensor fields are flattened after removing the leading environment dimension.
    The index suffix preserves the remaining tensor indices, so histories, bodies,
    filters, and XYZ/quaternion components are all retained without aggregation.
    """
    headers: list[str] = []
    values: list[Any] = []

    for field_info in fields(data):
        name = field_info.name
        value = getattr(data, name)

        if value is None:
            headers.append(f"sensor_raw_{name}__none")
            values.append("")
            continue

        if not isinstance(value, torch.Tensor):
            headers.append(f"sensor_raw_{name}")
            values.append(value)
            continue

        tensor = value.detach()
        if tensor.ndim > 0:
            tensor = tensor[0]  # environment 0

        if tensor.ndim == 0:
            headers.append(f"sensor_raw_{name}")
            values.append(float(tensor.item()))
            continue

        # Preserve every remaining index explicitly.
        for flat_index in range(tensor.numel()):
            multi_index = []
            remaining = flat_index
            for size in reversed(tensor.shape):
                multi_index.append(remaining % size)
                remaining //= size
            multi_index.reverse()
            suffix = "__".join(
                f"i{axis}_{index}" for axis, index in enumerate(multi_index)
            )
            headers.append(f"sensor_raw_{name}__{suffix}")
            scalar = tensor.reshape(-1)[flat_index]
            values.append(float(scalar.item()))

    return headers, values


# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
def run_simulator(sim: SimulationContext, entities: dict[str, object]) -> None:
    robot: Articulation = entities["robot"]  # type: ignore[assignment]
    contact_sensor: ContactSensor = entities["contact_sensor"]  # type: ignore[assignment]

    sim_dt = sim.get_physics_dt()
    physics_hz = 1.0 / sim_dt
    requested_log_hz = max(1.0, args_cli.log_hz)
    log_every_steps = max(
        1,
        round(physics_hz / min(requested_log_hz, physics_hz)),
    )
    actual_log_hz = physics_hz / log_every_steps

    log_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    )
    os.makedirs(log_dir, exist_ok=True)
    import datetime

    file_name = (
        f"ur10e_floor_press_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )

    log_path = os.path.join(log_dir, file_name)

    joint_names = [
        "shoulder_pan",
        "shoulder_lift",
        "elbow",
        "wrist_1",
        "wrist_2",
        "wrist_3",
    ]

    csv_header = [
        "time",
        "cycle",
        "phase",
        "cycle_press_scale",
        "cycle_depth_delta_mm",
        "cycle_tilt_command_deg",
        "cycle_tilt_azimuth_deg",
        "cycle_wrist_1_delta_deg",
        "cycle_wrist_2_delta_deg",
        "cycle_slide_pan_deg",
    ]

    for prefix in ["pos", "vel", "effort"]:
        csv_header.extend(f"joint_{prefix}_{joint_name}" for joint_name in joint_names)

    csv_header += [
        "normal_fx_w",
        "normal_fy_w",
        "normal_fz_w",
        "normal_force_norm",
        "friction_fx_w",
        "friction_fy_w",
        "friction_fz_w",
        "friction_force_norm",
        "total_fx_w",
        "total_fy_w",
        "total_fz_w",
        "total_force_norm",
        # Sensor/tool local frame components.
        "normal_fx_s",
        "normal_fy_s",
        "normal_fz_s",
        "friction_fx_s",
        "friction_fy_s",
        "friction_fz_s",
        "total_fx_s",
        "total_fy_s",
        "total_fz_s",
        # Diagnostics.
        "friction_patch_count",
        "contact_pos_x_w",
        "contact_pos_y_w",
        "contact_pos_z_w",
        "tool_pos_x_w",
        "tool_pos_y_w",
        "tool_pos_z_w",
        "tool_quat_w",
        "tool_quat_x",
        "tool_quat_y",
        "tool_quat_z",
        "tool_axis_x_w",
        "tool_axis_y_w",
        "tool_axis_z_w",
        "tool_tilt_from_normal_deg",
        "tool_vel_x_w",
        "tool_vel_y_w",
        "tool_vel_z_w",
        "tool_tangential_speed_w",
    ]

    # Append every field in ContactSensorData, including histories and per-filter data.
    raw_sensor_headers, _ = _raw_sensor_headers_and_values(contact_sensor.data)
    csv_header += raw_sensor_headers

    print(f"[INFO]: Logging to {log_path}")
    print(
        f"[INFO]: physics={physics_hz:.1f} Hz, logging={actual_log_hz:.1f} Hz, "
        f"cycles={max(1, args_cli.cycles)}, total_time={TOTAL_TIME:.2f} s"
    )
    print("[INFO]: '*_w' = world frame, '*_s' = cylindrical sensor frame")

    for index, command in enumerate(CYCLE_COMMANDS, start=1):
        print(
            f"[INFO]: cycle {index:02d} | "
            f"press_scale={float(command['press_scale']):.4f}, "
            f"depth={float(command['depth_delta_m']) * 1.0e3:+.3f} mm, "
            f"tilt_cmd={math.degrees(float(command['tilt_magnitude_rad'])):.3f} deg, "
            f"azimuth={math.degrees(float(command['tilt_azimuth_rad'])):.3f} deg, "
            f"slide_pan={math.degrees(float(command['slide_pan_delta_rad'])):+.3f} deg"
        )

    sim_time = 0.0
    step = 0
    previous_tool_position: torch.Tensor | None = None
    slide_contact_samples = 0
    slide_friction_samples = 0

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

        while simulation_app.is_running() and step < TOTAL_STEPS:
            target_joints, phase_name, cycle_number, cycle_command = (
                _get_motion_command(sim_time)
            )

            target_tensor = torch.tensor(
                [target_joints],
                dtype=torch.float32,
                device=sim.device,
            )
            robot.set_joint_position_target(target_tensor)
            robot.write_data_to_sim()

            sim.step()

            robot.update(sim_dt)
            contact_sensor.update(sim_dt, force_recompute=True)

            sim_time += sim_dt
            step += 1

            contact = _read_contact_data(contact_sensor, sim_dt)

            normal_force_w = contact["normal_force_w"]
            friction_force_w = contact["friction_force_w"]
            total_force_w = contact["total_force_w"]
            normal_force_s = contact["normal_force_s"]
            friction_force_s = contact["friction_force_s"]
            total_force_s = contact["total_force_s"]
            tool_position_w = contact["tool_position_w"]
            tool_quat_wxyz = contact["tool_quat_wxyz"]
            tool_axis_z_w = contact["tool_axis_z_w"]
            contact_position_w = contact["contact_position_w"]

            assert isinstance(normal_force_w, torch.Tensor)
            assert isinstance(friction_force_w, torch.Tensor)
            assert isinstance(total_force_w, torch.Tensor)
            assert isinstance(normal_force_s, torch.Tensor)
            assert isinstance(friction_force_s, torch.Tensor)
            assert isinstance(total_force_s, torch.Tensor)
            assert isinstance(tool_position_w, torch.Tensor)
            assert isinstance(tool_quat_wxyz, torch.Tensor)
            assert isinstance(tool_axis_z_w, torch.Tensor)
            assert isinstance(contact_position_w, torch.Tensor)

            normal_norm = float(torch.linalg.vector_norm(normal_force_w).item())
            friction_norm = float(torch.linalg.vector_norm(friction_force_w).item())
            total_norm = float(torch.linalg.vector_norm(total_force_w).item())

            if previous_tool_position is None:
                tool_velocity_w = torch.zeros_like(tool_position_w)
            else:
                tool_velocity_w = (tool_position_w - previous_tool_position) / sim_dt
            previous_tool_position = tool_position_w.clone()
            tangential_speed = float(
                torch.linalg.vector_norm(tool_velocity_w[:2]).item()
            )

            if phase_name in ("SLIDE_OUT", "SLIDE_BACK") and normal_norm > 1.0:
                slide_contact_samples += 1
                if friction_norm > 1.0e-3:
                    slide_friction_samples += 1

            joint_pos = robot.data.joint_pos[0].tolist()
            joint_vel = robot.data.joint_vel[0].tolist()
            if hasattr(robot.data, "applied_torque"):
                joint_effort = robot.data.applied_torque[0].tolist()
            else:
                joint_effort = [0.0] * len(joint_names)

            if cycle_command is None:
                press_scale = max(0.0, args_cli.press_scale)
                depth_delta_mm = 0.0
                tilt_command_deg = 0.0
                tilt_azimuth_deg = 0.0
                wrist_1_delta_deg = 0.0
                wrist_2_delta_deg = 0.0
                slide_pan_deg = 0.0
            else:
                press_scale = float(cycle_command["press_scale"])
                depth_delta_mm = float(cycle_command["depth_delta_m"]) * 1.0e3
                tilt_command_deg = math.degrees(
                    float(cycle_command["tilt_magnitude_rad"])
                )
                tilt_azimuth_deg = math.degrees(
                    float(cycle_command["tilt_azimuth_rad"])
                )
                wrist_1_delta_deg = math.degrees(
                    float(cycle_command["wrist_1_delta_rad"])
                )
                wrist_2_delta_deg = math.degrees(
                    float(cycle_command["wrist_2_delta_rad"])
                )
                slide_pan_deg = math.degrees(
                    float(cycle_command["slide_pan_delta_rad"])
                )

            if step % log_every_steps == 0:
                row: list[Any] = [
                    sim_time,
                    cycle_number,
                    phase_name,
                    press_scale,
                    depth_delta_mm,
                    tilt_command_deg,
                    tilt_azimuth_deg,
                    wrist_1_delta_deg,
                    wrist_2_delta_deg,
                    slide_pan_deg,
                ]
                row += joint_pos + joint_vel + joint_effort
                row += normal_force_w.tolist() + [normal_norm]
                row += friction_force_w.tolist() + [friction_norm]
                row += total_force_w.tolist() + [total_norm]
                row += normal_force_s.tolist()
                row += friction_force_s.tolist()
                row += total_force_s.tolist()
                row += [int(contact["friction_patch_count"])]
                row += contact_position_w.tolist()
                row += tool_position_w.tolist()
                row += tool_quat_wxyz.tolist()
                row += tool_axis_z_w.tolist()
                row += [float(contact["tilt_from_normal_deg"])]
                row += tool_velocity_w.tolist()
                row += [tangential_speed]

                import inspect

                print(contact_sensor.data)
                print(
                    inspect.getfile(contact_sensor.data.__class__)
                )  # 클래스가 정의된 파일

                # Raw, exhaustive ContactSensorData snapshot for this sample.
                current_raw_headers, raw_sensor_values = _raw_sensor_headers_and_values(
                    contact_sensor.data
                )
                if current_raw_headers != raw_sensor_headers:
                    raise RuntimeError(
                        "ContactSensorData tensor shapes changed during logging; "
                        "CSV columns would no longer align."
                    )
                row += raw_sensor_values

                writer.writerow(
                    [
                        f"{value:.6f}" if isinstance(value, float) else str(value)
                        for value in row
                    ]
                )

            if step % max(1, int(0.25 / sim_dt)) == 0:
                nw = normal_force_w.tolist()
                fw = friction_force_w.tolist()
                ts = total_force_s.tolist()
                print(
                    f"[{sim_time:6.2f}s] cycle={cycle_number:02d} "
                    f"phase={phase_name:13s} | "
                    f"Fn_w=[{nw[0]:8.2f}, {nw[1]:8.2f}, {nw[2]:8.2f}] N | "
                    f"Ff_w=[{fw[0]:8.2f}, {fw[1]:8.2f}, {fw[2]:8.2f}] N | "
                    f"Ftotal_s=[{ts[0]:8.2f}, {ts[1]:8.2f}, {ts[2]:8.2f}] N | "
                    f"tilt={float(contact['tilt_from_normal_deg']):6.2f} deg | "
                    f"v_xy={tangential_speed:7.4f} m/s | "
                    f"patches={int(contact['friction_patch_count'])}"
                )

    print(
        f"\n[INFO]: Simulation complete: {step} physics steps, "
        f"{step // log_every_steps} CSV samples."
    )
    if slide_contact_samples == 0:
        print(
            "[WARN]: No sliding-phase samples had normal force > 1 N. "
            "Increase --press_scale."
        )
    elif slide_friction_samples == 0:
        print(
            "[WARN]: Contact and horizontal motion were detected, but PhysX "
            "reported no friction patches. Check the printed ground collider "
            "filter path and increase --dynamic_friction."
        )
    else:
        ratio = 100.0 * slide_friction_samples / slide_contact_samples
        print(
            f"[INFO]: Friction detected in {slide_friction_samples}/"
            f"{slide_contact_samples} sliding-contact samples ({ratio:.1f}%)."
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    physx_cfg = sim_utils.PhysxCfg(
        enable_external_forces_every_iteration=True,
    )
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args_cli.device,
        physx=physx_cfg,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(
        eye=[1.8, 1.8, 1.4],
        target=[0.55, 0.0, 0.30],
    )

    entities = design_scene()

    sim.reset()

    robot: Articulation = entities["robot"]  # type: ignore[assignment]
    contact_sensor: ContactSensor = entities["contact_sensor"]  # type: ignore[assignment]

    _initialize_robot_pose(robot, sim)

    filter_count = contact_sensor.contact_physx_view.filter_count
    if filter_count < 1:
        raise RuntimeError(
            "ContactSensor resolved zero ground filters. "
            f"Requested collider: {entities['ground_collider_path']}"
        )

    print(
        f"[INFO]: ContactSensor initialized | bodies={contact_sensor.num_bodies}, "
        f"filters={filter_count}"
    )
    print("[INFO]: Starting inclined press and explicit sliding motion.")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
