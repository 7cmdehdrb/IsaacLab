# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e side-push demonstration with a cylindrical virtual sensor and a thin bar.

Structure
---------
    wrist_3_link -- 1 mm gap -- cylindrical sensor -- 1 mm gap -- thin rectangular bar

The bar pushes a heavy cube horizontally on a collision-enabled ground plane.  The
cube and robot are returned to their initial states at the beginning of every cycle.

Important measurement note
--------------------------
Isaac Lab ``ContactSensor`` measures contacts acting directly on the rigid body to
which it is attached.  Since the bar, not the cylindrical sensor, touches the cube,
the ContactSensor is attached to the bar.  The measured bar-cube resultant is then
expressed in the cylindrical sensor frame.  For this fixed, quasi-static assembly,
that resultant is the external load transmitted through the sensor-bar interface.
The moment about the sensor center is estimated as ``r x F`` using the reported
average contact position.

Usage
-----
    ./isaaclab.sh -p scripts/demos/ur10e_side_push_demo.py \
        --device cpu --cycles 4

    ./isaaclab.sh -p scripts/demos/ur10e_side_push_demo.py \
        --device cpu --headless --cycles 10 --log_hz 100
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10e heavy-cube side-push demo.")
parser.add_argument(
    "--cycles", type=int, default=4, help="Number of push/reset cycles."
)
parser.add_argument(
    "--push_scale",
    type=float,
    default=1.0,
    help="Scale the nominal contact-start to push-end joint displacement.",
)
parser.add_argument(
    "--cube_mass", type=float, default=5.0, help="Cube mass in kilograms."
)
parser.add_argument(
    "--log_hz", type=float, default=100.0, help="CSV logging frequency."
)
parser.add_argument("--ground_static_friction", type=float, default=0.9)
parser.add_argument("--ground_dynamic_friction", type=float, default=0.7)
parser.add_argument("--tool_static_friction", type=float, default=1.0)
parser.add_argument("--tool_dynamic_friction", type=float, default=0.8)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below executes after Isaac Sim is launched."""

import csv
import math
import os
from dataclasses import fields
from typing import Any

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim import SimulationContext

from isaaclab_assets.robots.universal_robots import UR10e_CFG  # isort:skip


# -----------------------------------------------------------------------------
# Timing
# -----------------------------------------------------------------------------
SIM_DT = 0.005  # 200 Hz physics and sensor update
RESET_HOLD_TIME = 0.50
APPROACH_TIME = 1.00
PUSH_TIME = 1.50
PUSH_HOLD_TIME = 0.30
RETRACT_TIME = 1.00


# -----------------------------------------------------------------------------
# Robot joint-space trajectory
# -----------------------------------------------------------------------------
# Joint order:
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
#
# These poses maintain an approximately downward tool Z-axis and approximately
# constant wrist height.  They were obtained for the standard Isaac Lab UR10e.
# Runtime calibration below measures the actual bar motion direction before the
# cube is placed, so world-axis sign differences in the USD are handled.
SAFE_JOINTS = [
    math.pi,
    -1.4669384679,
    2.4920619886,
    -2.5959198475,
    -math.pi / 2.0,
    0.0,
]

CONTACT_START_JOINTS = [
    math.pi,
    -1.3738560231,
    2.3708311157,
    -2.5677714194,
    -math.pi / 2.0,
    0.0,
]

NOMINAL_PUSH_END_JOINTS = [
    math.pi,
    -1.0996626641,
    1.9383482305,
    -2.4094818932,
    -math.pi / 2.0,
    0.0,
]

PUSH_END_JOINTS = [
    start + max(0.0, args_cli.push_scale) * (end - start)
    for start, end in zip(CONTACT_START_JOINTS, NOMINAL_PUSH_END_JOINTS)
]


# -----------------------------------------------------------------------------
# Scene paths and dimensions
# -----------------------------------------------------------------------------
ROBOT_PATH = "/World/Robot"
WRIST_BODY_PATH = f"{ROBOT_PATH}/wrist_3_link"
SENSOR_BODY_PATH = f"{ROBOT_PATH}/virtual_sensor"
BAR_BODY_PATH = f"{ROBOT_PATH}/push_bar"
WRIST_SENSOR_JOINT_PATH = f"{ROBOT_PATH}/joints/wrist_3_to_virtual_sensor"
SENSOR_BAR_JOINT_PATH = f"{ROBOT_PATH}/joints/virtual_sensor_to_push_bar"
GROUND_PATH = "/World/GroundPlane"
CUBE_PATH = "/World/HeavyCube"

# Cylinder: virtual sensor.
SENSOR_RADIUS = 0.035
SENSOR_HEIGHT = 0.018
SENSOR_MASS = 0.025
WRIST_SENSOR_GAP = 0.001
WRIST_TO_SENSOR_CENTER = WRIST_SENSOR_GAP + 0.5 * SENSOR_HEIGHT

# Thin vertical pushing bar.  Local Z is vertical/downward with the nominal pose.
# Local Y is the thin pushing direction for the selected trajectory.
BAR_SIZE = (0.080, 0.012, 0.160)  # local X width, local Y thickness, local Z height
BAR_MASS = 0.30
SENSOR_BAR_GAP = 0.001
SENSOR_TO_BAR_CENTER = 0.5 * SENSOR_HEIGHT + SENSOR_BAR_GAP + 0.5 * BAR_SIZE[2]

CUBE_SIZE = 0.120
CUBE_INITIAL_CONTACT_GAP = 0.004
CUBE_FLOOR_CLEARANCE = 0.001
CUBE_INITIAL_FAR_POSITION = (2.5, 2.5, 0.5 * CUBE_SIZE + CUBE_FLOOR_CLEARANCE)


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


def _quat_wxyz_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    """Convert one wxyz quaternion to a 3x3 active rotation matrix."""
    quat = quat / torch.linalg.vector_norm(quat).clamp_min(1.0e-12)
    w, x, y, z = quat.unbind(-1)
    return torch.stack(
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ]
    ).reshape(3, 3)


def _rotate_world_to_local(
    vector_w: torch.Tensor, quat_wxyz: torch.Tensor
) -> torch.Tensor:
    return _quat_wxyz_to_matrix(quat_wxyz).transpose(0, 1) @ vector_w


def _rotate_local_to_world(
    vector_l: torch.Tensor, quat_wxyz: torch.Tensor
) -> torch.Tensor:
    return _quat_wxyz_to_matrix(quat_wxyz) @ vector_l


def _box_support_extent(
    direction_w: torch.Tensor,
    quat_wxyz: torch.Tensor,
    full_size: tuple[float, float, float],
) -> float:
    """Half-extent of an oriented box along a world-frame unit direction."""
    rotation = _quat_wxyz_to_matrix(quat_wxyz)
    half_size = torch.tensor(
        [0.5 * full_size[0], 0.5 * full_size[1], 0.5 * full_size[2]],
        dtype=direction_w.dtype,
        device=direction_w.device,
    )
    projections = torch.abs(rotation.transpose(0, 1) @ direction_w)
    return float(torch.dot(projections, half_size).item())


# -----------------------------------------------------------------------------
# USD / PhysX helpers
# -----------------------------------------------------------------------------
def _get_world_pose_from_local_offset(
    parent_path: str,
    local_offset: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    from pxr import Gf, UsdGeom

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext does not exist.")

    stage = sim_context.stage
    parent_prim = stage.GetPrimAtPath(parent_path)
    if not parent_prim.IsValid():
        raise RuntimeError(f"Parent prim does not exist: {parent_path}")

    world_matrix = UsdGeom.XformCache().GetLocalToWorldTransform(parent_prim)
    world_position = world_matrix.Transform(Gf.Vec3d(*local_offset))
    world_quaternion = world_matrix.ExtractRotationQuat()
    imaginary = world_quaternion.GetImaginary()

    return (
        tuple(float(world_position[i]) for i in range(3)),
        (
            float(world_quaternion.GetReal()),
            float(imaginary[0]),
            float(imaginary[1]),
            float(imaginary[2]),
        ),
    )


def _ensure_contact_reporter(prim_path: str) -> None:
    from pxr import PhysxSchema, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext does not exist.")

    prim = sim_context.stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Contact body does not exist: {prim_path}")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Contact body is not a rigid body: {prim_path}")

    api = PhysxSchema.PhysxContactReportAPI(prim)
    if not api:
        api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    api.CreateThresholdAttr().Set(0.0)


def _find_collision_paths(root_path: str) -> list[str]:
    from pxr import Usd, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext does not exist.")

    root = sim_context.stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Collision root does not exist: {root_path}")

    paths = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(root)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not paths:
        raise RuntimeError(f"No collision prim found below: {root_path}")
    return paths


def _create_sensor_and_bar(tool_material: sim_utils.RigidBodyMaterialCfg) -> None:
    """Create two rigid bodies and connect them sequentially with FixedJoints."""
    from pxr import Gf, Sdf, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError("SimulationContext does not exist.")
    stage = sim_context.stage

    # Cylinder sensor fixed 1 mm from wrist_3_link.
    sensor_position, sensor_orientation = _get_world_pose_from_local_offset(
        WRIST_BODY_PATH,
        (0.0, 0.0, WRIST_TO_SENSOR_CENTER),
    )
    sensor_cfg = sim_utils.CylinderCfg(
        radius=SENSOR_RADIUS,
        height=SENSOR_HEIGHT,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=SENSOR_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.003,
            rest_offset=0.0,
        ),
        physics_material=tool_material,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.85, 0.20, 0.10),
            metallic=0.15,
            roughness=0.45,
        ),
    )
    sensor_cfg.func(
        SENSOR_BODY_PATH,
        sensor_cfg,
        translation=sensor_position,
        orientation=sensor_orientation,
    )

    wrist_sensor_joint = UsdPhysics.FixedJoint.Define(
        stage,
        Sdf.Path(WRIST_SENSOR_JOINT_PATH),
    )
    wrist_sensor_joint.CreateBody0Rel().SetTargets([Sdf.Path(WRIST_BODY_PATH)])
    wrist_sensor_joint.CreateBody1Rel().SetTargets([Sdf.Path(SENSOR_BODY_PATH)])
    wrist_sensor_joint.CreateLocalPos0Attr().Set(
        Gf.Vec3f(0.0, 0.0, WRIST_TO_SENSOR_CENTER)
    )
    wrist_sensor_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    wrist_sensor_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    wrist_sensor_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    wrist_sensor_joint.CreateCollisionEnabledAttr().Set(False)

    # Thin bar fixed 1 mm after the distal sensor face.
    bar_position, bar_orientation = _get_world_pose_from_local_offset(
        SENSOR_BODY_PATH,
        (0.0, 0.0, SENSOR_TO_BAR_CENTER),
    )
    bar_cfg = sim_utils.CuboidCfg(
        size=BAR_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=BAR_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.003,
            rest_offset=0.0,
        ),
        physics_material=tool_material,
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.15, 0.35, 0.85),
            metallic=0.10,
            roughness=0.50,
        ),
    )
    bar_cfg.func(
        BAR_BODY_PATH,
        bar_cfg,
        translation=bar_position,
        orientation=bar_orientation,
    )

    sensor_bar_joint = UsdPhysics.FixedJoint.Define(
        stage,
        Sdf.Path(SENSOR_BAR_JOINT_PATH),
    )
    sensor_bar_joint.CreateBody0Rel().SetTargets([Sdf.Path(SENSOR_BODY_PATH)])
    sensor_bar_joint.CreateBody1Rel().SetTargets([Sdf.Path(BAR_BODY_PATH)])
    sensor_bar_joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, SENSOR_TO_BAR_CENTER))
    sensor_bar_joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    sensor_bar_joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    sensor_bar_joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    sensor_bar_joint.CreateCollisionEnabledAttr().Set(False)

    _ensure_contact_reporter(BAR_BODY_PATH)

    print(
        "[INFO]: Tool chain created | "
        f"wrist-sensor gap={WRIST_SENSOR_GAP * 1e3:.1f} mm, "
        f"sensor-bar gap={SENSOR_BAR_GAP * 1e3:.1f} mm, "
        f"bar size={BAR_SIZE} m"
    )


# -----------------------------------------------------------------------------
# Scene creation
# -----------------------------------------------------------------------------
def design_scene() -> dict[str, object]:
    ground_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=max(0.0, args_cli.ground_static_friction),
        dynamic_friction=max(0.0, args_cli.ground_dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
    )
    tool_material = sim_utils.RigidBodyMaterialCfg(
        static_friction=max(0.0, args_cli.tool_static_friction),
        dynamic_friction=max(0.0, args_cli.tool_dynamic_friction),
        restitution=0.0,
        friction_combine_mode="max",
    )

    ground_cfg = sim_utils.GroundPlaneCfg(physics_material=ground_material)
    ground_cfg.func(GROUND_PATH, ground_cfg)
    print(f"[INFO]: Ground collision prims: {_find_collision_paths(GROUND_PATH)}")

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = UR10e_CFG.copy()
    robot_cfg.prim_path = ROBOT_PATH
    robot_cfg.spawn.activate_contact_sensors = True
    robot = Articulation(cfg=robot_cfg)

    _create_sensor_and_bar(tool_material)

    cube_cfg = RigidObjectCfg(
        prim_path=CUBE_PATH,
        spawn=sim_utils.CuboidCfg(
            size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.02,
                angular_damping=0.05,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=max(0.01, args_cli.cube_mass)),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                contact_offset=0.003,
                rest_offset=0.0,
            ),
            physics_material=ground_material,
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.20, 0.70, 0.25),
                metallic=0.05,
                roughness=0.60,
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_INITIAL_FAR_POSITION),
    )
    cube = RigidObject(cfg=cube_cfg)
    cube_collision_paths = _find_collision_paths(CUBE_PATH)
    print(f"[INFO]: Cube collision prims: {cube_collision_paths}")

    # The bar is the body that physically contacts the cube.  This sensor reports
    # that external contact resultant, which is later transformed into the
    # cylindrical virtual-sensor frame.
    contact_sensor_cfg = ContactSensorCfg(
        prim_path=BAR_BODY_PATH,
        update_period=0.0,
        history_length=8,
        track_pose=True,
        track_contact_points=True,
        track_friction_forces=True,
        track_air_time=True,
        max_contact_data_count_per_prim=64,
        force_threshold=0.1,
        filter_prim_paths_expr=cube_collision_paths,
        debug_vis=not args_cli.headless,
    )
    bar_contact_sensor = ContactSensor(cfg=contact_sensor_cfg)

    return {
        "robot": robot,
        "cube": cube,
        "bar_contact_sensor": bar_contact_sensor,
    }


# -----------------------------------------------------------------------------
# State manipulation and runtime placement calibration
# -----------------------------------------------------------------------------
def _write_robot_state(robot: Articulation, joints: list[float], device: str) -> None:
    joint_pos = torch.tensor([joints], dtype=torch.float32, device=device)
    joint_vel = torch.zeros_like(joint_pos)
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()


def _write_cube_state(cube: RigidObject, pose_wxyz: torch.Tensor) -> None:
    cube.write_root_pose_to_sim(pose_wxyz.reshape(1, 7))
    cube.write_root_velocity_to_sim(
        torch.zeros((1, 6), dtype=pose_wxyz.dtype, device=pose_wxyz.device)
    )


def _step_entities(
    sim: SimulationContext,
    robot: Articulation,
    cube: RigidObject,
    sensor: ContactSensor,
    count: int = 1,
) -> None:
    dt = sim.get_physics_dt()
    for _ in range(count):
        robot.write_data_to_sim()
        sim.step()
        robot.update(dt)
        cube.update(dt)
        sensor.update(dt, force_recompute=True)


def _capture_bar_pose(
    sim: SimulationContext,
    robot: Articulation,
    cube: RigidObject,
    sensor: ContactSensor,
    joints: list[float],
) -> tuple[torch.Tensor, torch.Tensor]:
    _write_robot_state(robot, joints, sim.device)
    _step_entities(sim, robot, cube, sensor, count=4)
    return (
        sensor.data.pos_w[0, 0].detach().clone(),
        sensor.data.quat_w[0, 0].detach().clone(),
    )


def _calibrate_cube_initial_pose(
    sim: SimulationContext,
    robot: Articulation,
    cube: RigidObject,
    sensor: ContactSensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure the actual bar path and place the cube immediately ahead of it."""
    start_pos, start_quat = _capture_bar_pose(
        sim, robot, cube, sensor, CONTACT_START_JOINTS
    )
    end_pos, _ = _capture_bar_pose(sim, robot, cube, sensor, PUSH_END_JOINTS)

    push_vector = end_pos - start_pos
    push_vector[2] = 0.0
    push_distance = torch.linalg.vector_norm(push_vector)
    if float(push_distance.item()) < 0.05:
        raise RuntimeError(
            "The calibrated horizontal push path is too short. "
            "Check the UR10e joint poses or push_scale."
        )
    push_direction = push_vector / push_distance

    bar_extent = _box_support_extent(push_direction, start_quat, BAR_SIZE)
    cube_half_extent = (
        0.5 * CUBE_SIZE * float(torch.sum(torch.abs(push_direction)).item())
    )

    cube_center = start_pos.clone()
    cube_center += push_direction * (
        bar_extent + cube_half_extent + CUBE_INITIAL_CONTACT_GAP
    )
    cube_center[2] = 0.5 * CUBE_SIZE + CUBE_FLOOR_CLEARANCE

    cube_pose = torch.cat(
        [
            cube_center,
            torch.tensor(
                [1.0, 0.0, 0.0, 0.0],
                dtype=cube_center.dtype,
                device=cube_center.device,
            ),
        ]
    )

    print(
        "[INFO]: Runtime push calibration | "
        f"start={start_pos.tolist()}, end={end_pos.tolist()}, "
        f"horizontal distance={float(push_distance.item()):.4f} m, "
        f"direction={push_direction.tolist()}, cube={cube_center.tolist()}"
    )
    return cube_pose, push_direction, start_pos


def _reset_cycle(
    sim: SimulationContext,
    robot: Articulation,
    cube: RigidObject,
    sensor: ContactSensor,
    cube_initial_pose: torch.Tensor,
) -> None:
    _write_robot_state(robot, SAFE_JOINTS, sim.device)
    _write_cube_state(cube, cube_initial_pose)
    robot.reset()
    cube.reset()
    sensor.reset()
    _step_entities(sim, robot, cube, sensor, count=4)


# -----------------------------------------------------------------------------
# Contact measurement
# -----------------------------------------------------------------------------
def _sum_filtered_vectors(
    value: torch.Tensor | None, device: torch.device
) -> torch.Tensor:
    if value is None:
        return torch.zeros(3, dtype=torch.float32, device=device)
    vectors = value[0, 0].reshape(-1, 3)
    vectors = torch.nan_to_num(vectors, nan=0.0, posinf=0.0, neginf=0.0)
    return vectors.sum(dim=0)


def _read_bar_contact(
    sensor: ContactSensor,
    sim_dt: float,
) -> dict[str, torch.Tensor | float | int]:
    data = sensor.data
    if data.net_forces_w is None or data.pos_w is None or data.quat_w is None:
        raise RuntimeError("ContactSensor buffers were not initialized.")

    device = data.net_forces_w.device
    normal_force_w = _sum_filtered_vectors(data.force_matrix_w, device)
    if data.force_matrix_w is None:
        normal_force_w = data.net_forces_w[0, 0].clone()

    friction_force_w = _sum_filtered_vectors(data.friction_forces_w, device)
    friction_patch_count = 0

    # Read the raw PhysX friction patch buffer as a diagnostic fallback.
    try:
        raw_friction, _, buffer_count, _ = sensor.contact_physx_view.get_friction_data(
            dt=sim_dt
        )
        friction_patch_count = int(buffer_count.sum().item())
        if raw_friction.numel() > 0 and friction_patch_count > 0:
            friction_force_w = torch.nan_to_num(
                raw_friction.reshape(-1, 3),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).sum(dim=0)
    except Exception:
        pass

    total_force_w = normal_force_w + friction_force_w

    bar_pos_w = data.pos_w[0, 0]
    bar_quat_w = data.quat_w[0, 0]
    rotation = _quat_wxyz_to_matrix(bar_quat_w)

    # Sensor and bar have identical orientation.  Recover the cylindrical sensor
    # center from the known fixed translation between their body origins.
    sensor_to_bar_local = torch.tensor(
        [0.0, 0.0, SENSOR_TO_BAR_CENTER],
        dtype=bar_pos_w.dtype,
        device=bar_pos_w.device,
    )
    sensor_pos_w = bar_pos_w - rotation @ sensor_to_bar_local
    sensor_quat_w = bar_quat_w

    normal_force_s = _rotate_world_to_local(normal_force_w, sensor_quat_w)
    friction_force_s = _rotate_world_to_local(friction_force_w, sensor_quat_w)
    total_force_s = _rotate_world_to_local(total_force_w, sensor_quat_w)

    contact_pos_w = torch.full(
        (3,), float("nan"), dtype=bar_pos_w.dtype, device=bar_pos_w.device
    )
    if data.contact_pos_w is not None:
        points = data.contact_pos_w[0, 0].reshape(-1, 3)
        valid = ~torch.isnan(points).any(dim=-1)
        if bool(valid.any()):
            contact_pos_w = points[valid].mean(dim=0)

    moment_w = torch.full_like(total_force_w, float("nan"))
    moment_s = torch.full_like(total_force_s, float("nan"))
    if not bool(torch.isnan(contact_pos_w).any()):
        moment_w = torch.linalg.cross(
            contact_pos_w - sensor_pos_w,
            total_force_w,
            dim=0,
        )
        moment_s = _rotate_world_to_local(moment_w, sensor_quat_w)

    return {
        "normal_force_w": normal_force_w,
        "friction_force_w": friction_force_w,
        "total_force_w": total_force_w,
        "normal_force_s": normal_force_s,
        "friction_force_s": friction_force_s,
        "total_force_s": total_force_s,
        "moment_w": moment_w,
        "moment_s": moment_s,
        "bar_pos_w": bar_pos_w,
        "bar_quat_w": bar_quat_w,
        "sensor_pos_w": sensor_pos_w,
        "sensor_quat_w": sensor_quat_w,
        "contact_pos_w": contact_pos_w,
        "friction_patch_count": friction_patch_count,
    }


# -----------------------------------------------------------------------------
# Exhaustive ContactSensorData logging
# -----------------------------------------------------------------------------
def _raw_sensor_headers_and_values(data: Any) -> tuple[list[str], list[Any]]:
    """Flatten every ContactSensorData field for environment zero."""
    headers: list[str] = []
    values: list[Any] = []

    for field_info in fields(data):
        name = field_info.name
        value = getattr(data, name)

        if value is None:
            headers.append(f"bar_sensor_raw_{name}__none")
            values.append("")
            continue

        if not isinstance(value, torch.Tensor):
            headers.append(f"bar_sensor_raw_{name}")
            values.append(value)
            continue

        tensor = value.detach()
        if tensor.ndim > 0:
            tensor = tensor[0]

        if tensor.ndim == 0:
            headers.append(f"bar_sensor_raw_{name}")
            values.append(float(tensor.item()))
            continue

        flat = tensor.reshape(-1)
        for flat_index in range(flat.numel()):
            remaining = flat_index
            multi_index: list[int] = []
            for size in reversed(tensor.shape):
                multi_index.append(remaining % size)
                remaining //= size
            multi_index.reverse()
            suffix = "__".join(
                f"i{axis}_{index}" for axis, index in enumerate(multi_index)
            )
            headers.append(f"bar_sensor_raw_{name}__{suffix}")
            values.append(float(flat[flat_index].item()))

    return headers, values


# -----------------------------------------------------------------------------
# Logging and phase execution
# -----------------------------------------------------------------------------
def _joint_state(robot: Articulation) -> tuple[list[float], list[float], list[float]]:
    positions = robot.data.joint_pos[0].tolist()
    velocities = robot.data.joint_vel[0].tolist()
    if hasattr(robot.data, "applied_torque"):
        efforts = robot.data.applied_torque[0].tolist()
    else:
        efforts = [0.0] * len(positions)
    return positions, velocities, efforts


def run_simulator(
    sim: SimulationContext,
    entities: dict[str, object],
    cube_initial_pose: torch.Tensor,
    push_direction_w: torch.Tensor,
    initial_bar_pos_w: torch.Tensor,
) -> None:
    robot: Articulation = entities["robot"]  # type: ignore[assignment]
    cube: RigidObject = entities["cube"]  # type: ignore[assignment]
    sensor: ContactSensor = entities["bar_contact_sensor"]  # type: ignore[assignment]

    sim_dt = sim.get_physics_dt()
    physics_hz = 1.0 / sim_dt
    requested_hz = max(1.0, min(float(args_cli.log_hz), physics_hz))
    log_every_steps = max(1, round(physics_hz / requested_hz))
    actual_log_hz = physics_hz / log_every_steps

    log_dir = os.path.abspath(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs")
    )
    os.makedirs(log_dir, exist_ok=True)

    import datetime

    file_name = (
        f"ur10e_side_push_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
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
        "phase_time",
        "push_direction_x_w",
        "push_direction_y_w",
        "push_direction_z_w",
        "bar_progress_m",
        "cube_displacement_along_push_m",
    ]
    for prefix in ("pos", "vel", "effort"):
        csv_header.extend(f"joint_{prefix}_{name}" for name in joint_names)

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
        "normal_fx_s",
        "normal_fy_s",
        "normal_fz_s",
        "friction_fx_s",
        "friction_fy_s",
        "friction_fz_s",
        "total_fx_s",
        "total_fy_s",
        "total_fz_s",
        "moment_x_w",
        "moment_y_w",
        "moment_z_w",
        "moment_norm",
        "moment_x_s",
        "moment_y_s",
        "moment_z_s",
        "contact_pos_x_w",
        "contact_pos_y_w",
        "contact_pos_z_w",
        "sensor_pos_x_w",
        "sensor_pos_y_w",
        "sensor_pos_z_w",
        "sensor_quat_w",
        "sensor_quat_x",
        "sensor_quat_y",
        "sensor_quat_z",
        "bar_pos_x_w",
        "bar_pos_y_w",
        "bar_pos_z_w",
        "bar_quat_w",
        "bar_quat_x",
        "bar_quat_y",
        "bar_quat_z",
        "cube_pos_x_w",
        "cube_pos_y_w",
        "cube_pos_z_w",
        "cube_quat_w",
        "cube_quat_x",
        "cube_quat_y",
        "cube_quat_z",
        "cube_lin_vel_x_w",
        "cube_lin_vel_y_w",
        "cube_lin_vel_z_w",
        "cube_ang_vel_x_w",
        "cube_ang_vel_y_w",
        "cube_ang_vel_z_w",
        "friction_patch_count",
        "current_contact_time",
        "current_air_time",
    ]

    raw_headers, _ = _raw_sensor_headers_and_values(sensor.data)
    csv_header += raw_headers

    global_time = 0.0
    global_step = 0
    cube_initial_center = cube_initial_pose[:3].clone()

    phases: list[tuple[str, float, list[float], list[float]]] = [
        ("RESET_HOLD", RESET_HOLD_TIME, SAFE_JOINTS, SAFE_JOINTS),
        ("APPROACH", APPROACH_TIME, SAFE_JOINTS, CONTACT_START_JOINTS),
        ("PUSH", PUSH_TIME, CONTACT_START_JOINTS, PUSH_END_JOINTS),
        ("PUSH_HOLD", PUSH_HOLD_TIME, PUSH_END_JOINTS, PUSH_END_JOINTS),
        ("RETRACT", RETRACT_TIME, PUSH_END_JOINTS, SAFE_JOINTS),
    ]

    print(f"[INFO]: Logging to {log_path}")
    print(
        f"[INFO]: physics={physics_hz:.1f} Hz, logging={actual_log_hz:.1f} Hz, "
        f"cube mass={max(0.01, args_cli.cube_mass):.2f} kg, cycles={max(1, args_cli.cycles)}"
    )

    with open(log_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)

        for cycle_index in range(max(1, args_cli.cycles)):
            if not simulation_app.is_running():
                break

            _reset_cycle(sim, robot, cube, sensor, cube_initial_pose)
            print(f"[INFO]: Cycle {cycle_index + 1} reset.")

            for phase_name, duration, start_joints, end_joints in phases:
                phase_steps = max(1, math.ceil(duration / sim_dt))

                for phase_step in range(phase_steps):
                    if not simulation_app.is_running():
                        break

                    phase_time = phase_step * sim_dt
                    alpha = 1.0 if duration <= 0.0 else phase_time / duration
                    target = _interpolate_joints(start_joints, end_joints, alpha)
                    target_tensor = torch.tensor(
                        [target], dtype=torch.float32, device=sim.device
                    )
                    robot.set_joint_position_target(target_tensor)
                    robot.write_data_to_sim()

                    sim.step()
                    robot.update(sim_dt)
                    cube.update(sim_dt)
                    sensor.update(sim_dt, force_recompute=True)

                    global_time += sim_dt
                    global_step += 1

                    measurement = _read_bar_contact(sensor, sim_dt)
                    cube_state = cube.data.root_state_w[0]
                    cube_pos = cube_state[:3]
                    cube_quat = cube_state[3:7]
                    cube_lin_vel = cube_state[7:10]
                    cube_ang_vel = cube_state[10:13]

                    bar_pos = measurement["bar_pos_w"]
                    assert isinstance(bar_pos, torch.Tensor)
                    bar_progress = float(
                        torch.dot(bar_pos - initial_bar_pos_w, push_direction_w).item()
                    )
                    cube_displacement = float(
                        torch.dot(
                            cube_pos - cube_initial_center, push_direction_w
                        ).item()
                    )

                    if global_step % log_every_steps == 0:
                        joint_pos, joint_vel, joint_effort = _joint_state(robot)

                        normal_w = measurement["normal_force_w"]
                        friction_w = measurement["friction_force_w"]
                        total_w = measurement["total_force_w"]
                        normal_s = measurement["normal_force_s"]
                        friction_s = measurement["friction_force_s"]
                        total_s = measurement["total_force_s"]
                        moment_w = measurement["moment_w"]
                        moment_s = measurement["moment_s"]
                        contact_pos = measurement["contact_pos_w"]
                        sensor_pos = measurement["sensor_pos_w"]
                        sensor_quat = measurement["sensor_quat_w"]
                        bar_quat = measurement["bar_quat_w"]

                        tensor_values = [
                            normal_w,
                            friction_w,
                            total_w,
                            normal_s,
                            friction_s,
                            total_s,
                            moment_w,
                            moment_s,
                            contact_pos,
                            sensor_pos,
                            sensor_quat,
                            bar_pos,
                            bar_quat,
                        ]
                        assert all(
                            isinstance(value, torch.Tensor) for value in tensor_values
                        )

                        current_contact_time = 0.0
                        current_air_time = 0.0
                        if sensor.data.current_contact_time is not None:
                            current_contact_time = float(
                                sensor.data.current_contact_time[0, 0].item()
                            )
                        if sensor.data.current_air_time is not None:
                            current_air_time = float(
                                sensor.data.current_air_time[0, 0].item()
                            )

                        row: list[Any] = [
                            global_time,
                            cycle_index + 1,
                            phase_name,
                            phase_time,
                            *push_direction_w.tolist(),
                            bar_progress,
                            cube_displacement,
                        ]
                        row += joint_pos + joint_vel + joint_effort
                        row += normal_w.tolist() + [
                            float(torch.linalg.vector_norm(normal_w).item())
                        ]
                        row += friction_w.tolist() + [
                            float(torch.linalg.vector_norm(friction_w).item())
                        ]
                        row += total_w.tolist() + [
                            float(torch.linalg.vector_norm(total_w).item())
                        ]
                        row += normal_s.tolist()
                        row += friction_s.tolist()
                        row += total_s.tolist()
                        row += moment_w.tolist() + [
                            float(torch.linalg.vector_norm(moment_w).item())
                        ]
                        row += moment_s.tolist()
                        row += contact_pos.tolist()
                        row += sensor_pos.tolist() + sensor_quat.tolist()
                        row += bar_pos.tolist() + bar_quat.tolist()
                        row += cube_pos.tolist() + cube_quat.tolist()
                        row += cube_lin_vel.tolist() + cube_ang_vel.tolist()
                        row += [
                            int(measurement["friction_patch_count"]),
                            current_contact_time,
                            current_air_time,
                        ]

                        current_raw_headers, raw_values = (
                            _raw_sensor_headers_and_values(sensor.data)
                        )
                        if current_raw_headers != raw_headers:
                            raise RuntimeError(
                                "ContactSensorData shape changed after CSV header creation."
                            )
                        row += raw_values

                        formatted = [
                            f"{value:.8f}" if isinstance(value, float) else str(value)
                            for value in row
                        ]
                        writer.writerow(formatted)

                    if global_step % max(1, int(0.25 / sim_dt)) == 0:
                        total_s = measurement["total_force_s"]
                        total_w = measurement["total_force_w"]
                        assert isinstance(total_s, torch.Tensor)
                        assert isinstance(total_w, torch.Tensor)
                        print(
                            f"[{global_time:6.2f}s] cycle={cycle_index + 1:02d} "
                            f"phase={phase_name:10s} | "
                            f"F_w=[{total_w[0]:8.2f}, {total_w[1]:8.2f}, {total_w[2]:8.2f}] N | "
                            f"F_s=[{total_s[0]:8.2f}, {total_s[1]:8.2f}, {total_s[2]:8.2f}] N | "
                            f"cube_push={cube_displacement:+.4f} m"
                        )

    print(f"[INFO]: Finished. CSV written to {log_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    physx_cfg = sim_utils.PhysxCfg(enable_external_forces_every_iteration=True)
    sim_cfg = sim_utils.SimulationCfg(
        dt=SIM_DT,
        device=args_cli.device,
        physx=physx_cfg,
    )
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.7, 1.6, 1.1], target=[0.55, 0.0, 0.20])

    entities = design_scene()
    sim.reset()

    robot: Articulation = entities["robot"]  # type: ignore[assignment]
    cube: RigidObject = entities["cube"]  # type: ignore[assignment]
    sensor: ContactSensor = entities["bar_contact_sensor"]  # type: ignore[assignment]

    # Initialize all buffers before runtime placement calibration.
    _write_robot_state(robot, SAFE_JOINTS, sim.device)
    robot.reset()
    cube.reset()
    sensor.reset()
    _step_entities(sim, robot, cube, sensor, count=4)

    cube_initial_pose, push_direction_w, initial_bar_pos_w = (
        _calibrate_cube_initial_pose(
            sim,
            robot,
            cube,
            sensor,
        )
    )
    _reset_cycle(sim, robot, cube, sensor, cube_initial_pose)

    print("[INFO]: Setup complete. Starting repeated heavy-cube side pushes.")
    run_simulator(
        sim,
        entities,
        cube_initial_pose,
        push_direction_w,
        initial_bar_pos_w,
    )


if __name__ == "__main__":
    main()
    simulation_app.close()
