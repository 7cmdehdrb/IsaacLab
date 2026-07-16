# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""UR10e floor-press demo with a rigid cylindrical end-effector and contact-force logging.

Structure
---------
1. Spawn a collision-enabled ground plane.
2. Spawn a physical cylinder and attach it rigidly to ``wrist_3_link`` with a USD FixedJoint.
3. Attach an Isaac Lab ContactSensor to the cylinder rigid body.
4. Repeatedly lower the cylinder onto the ground, hold contact, and lift it away.
5. Log joint states, tool pose, and world-frame contact force to CSV.

Usage::

    ./isaaclab.sh -p scripts/demos/ur10e_floor_press_demo.py --device cpu --headless

The trajectory is calibrated for the standard Isaac Lab UR10e asset and the 18 mm
cylindrical contact tool defined below. If the robot asset, base transform, or tool
dimensions change, RELEASE_JOINTS and PRESS_JOINTS must be retuned.
"""

"""Launch Isaac Sim first."""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="UR10e rigid-tool floor contact demo.")
parser.add_argument(
    "--cycles", type=int, default=4, help="Number of press/release cycles."
)
parser.add_argument(
    "--press_scale",
    type=float,
    default=1.0,
    help=(
        "Scale the nominal release-to-press joint displacement. "
        "Use <1.0 for weaker contact and >1.0 for deeper pressing."
    ),
)
parser.add_argument(
    "--depth_jitter_mm",
    type=float,
    default=2.0,
    help="Uniform per-cycle press-depth randomization in millimeters (±value).",
)
parser.add_argument(
    "--angle_jitter_deg",
    type=float,
    default=2.0,
    help=(
        "Uniform per-cycle contact-angle randomization in degrees (±value), "
        "applied independently to wrist_1 and wrist_2."
    ),
)
parser.add_argument(
    "--seed", type=int, default=42, help="Random seed for cycle commands."
)
parser.add_argument(
    "--log_hz",
    type=float,
    default=100.0,
    help="CSV logging frequency in Hz. It cannot exceed the physics frequency.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below executes after Isaac Sim is launched."""

import csv
import math
import os
import random

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
PRESS_HOLD_TIME = 0.5
ASCEND_TIME = 1.0
RELEASE_HOLD_TIME = 0.5
CYCLE_TIME = DESCEND_TIME + PRESS_HOLD_TIME + ASCEND_TIME + RELEASE_HOLD_TIME
TOTAL_TIME = INITIAL_HOLD_TIME + max(1, args_cli.cycles) * CYCLE_TIME
TOTAL_STEPS = math.ceil(TOTAL_TIME / SIM_DT)

# UR10e joint order:
# shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3
#
# The rigid cylinder extends along +Z of wrist_3_link. With these configurations,
# its axis is approximately vertical and its distal tip approaches the z=0 plane.
# RELEASE_JOINTS: tool tip approximately 0.12 m above the floor.
# PRESS_JOINTS: tool tip commands approximately 5 mm beyond the floor plane so that
# the position controller produces a measurable normal contact force.
# These joint configurations are recalibrated for the shortened 18 mm tool.
# The wrist flange is approximately 162 mm lower than in the previous 180 mm-tool
# version, preserving approximately the same tool-tip heights:
#   release tip: about 0.12 m above z=0
#   nominal press tip: about 0.005 m below z=0
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

# The nominal release-to-press motion corresponds to approximately 125 mm of
# vertical tool-tip travel. This is used to convert millimeter depth jitter to
# a small perturbation of the joint-space interpolation scale.
NOMINAL_VERTICAL_TRAVEL_M = 0.125


def _make_cycle_commands() -> list[dict[str, object]]:
    """Sample reproducible press depth and contact angle for every cycle."""
    rng = random.Random(args_cli.seed)
    commands: list[dict[str, object]] = []

    depth_limit_m = max(0.0, args_cli.depth_jitter_mm) * 1.0e-3
    angle_limit_rad = math.radians(max(0.0, args_cli.angle_jitter_deg))

    for cycle_index in range(max(1, args_cli.cycles)):
        depth_delta_m = rng.uniform(-depth_limit_m, depth_limit_m)
        wrist_1_delta = rng.uniform(-angle_limit_rad, angle_limit_rad)
        wrist_2_delta = rng.uniform(-angle_limit_rad, angle_limit_rad)

        cycle_press_scale = max(
            0.0, args_cli.press_scale + depth_delta_m / NOMINAL_VERTICAL_TRAVEL_M
        )
        press_target = [
            release + cycle_press_scale * (press - release)
            for release, press in zip(RELEASE_JOINTS, PRESS_JOINTS)
        ]

        # Apply the angular perturbation only at the contact target. The robot
        # always returns to the same nominal release pose, avoiding target jumps
        # between cycles.
        press_target[3] += wrist_1_delta
        press_target[4] += wrist_2_delta

        commands.append(
            {
                "press_joints": press_target,
                "press_scale": cycle_press_scale,
                "depth_delta_m": depth_delta_m,
                "wrist_1_delta_rad": wrist_1_delta,
                "wrist_2_delta_rad": wrist_2_delta,
                "cycle_index": cycle_index,
            }
        )

    return commands


CYCLE_COMMANDS = _make_cycle_commands()

# USD paths
ROBOT_PATH = "/World/Robot"
WRIST_BODY_PATH = f"{ROBOT_PATH}/wrist_3_link"
TOOL_BODY_PATH = f"{ROBOT_PATH}/contact_tool"
TOOL_JOINT_PATH = f"{ROBOT_PATH}/joints/wrist_3_to_contact_tool"
GROUND_PATH = "/World/GroundPlane"

# Tool geometry. Cylinder axis is local Z.
TOOL_RADIUS = 0.035
TOOL_HEIGHT = 0.018  # 18 mm: one tenth of the previous 180 mm cylinder
TOOL_WRIST_GAP = 0.01
TOOL_CENTER_OFFSET = TOOL_WRIST_GAP + 0.5 * TOOL_HEIGHT
TOOL_MASS = 0.025  # scale mass with tool length to keep density comparable


# -----------------------------------------------------------------------------
# USD / PhysX helpers
# -----------------------------------------------------------------------------
def _assert_collision_geometry(root_path: str) -> None:
    """Verify that at least one collision prim exists below ``root_path``."""
    from pxr import Usd, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError(
            "SimulationContext must exist before checking collision geometry."
        )

    stage = sim_context.stage
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Collision root prim does not exist: '{root_path}'.")

    collision_paths = [
        prim.GetPath().pathString
        for prim in Usd.PrimRange(root_prim)
        if prim.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not collision_paths:
        raise RuntimeError(f"No collision geometry was created below '{root_path}'.")

    print(f"[INFO]: Collision geometry under {root_path}: {collision_paths}")


def _ensure_contact_reporter(prim_path: str) -> None:
    """Ensure that a rigid body has the PhysX contact-report API."""
    from pxr import PhysxSchema, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError(
            "SimulationContext must exist before enabling contact reporting."
        )

    stage = sim_context.stage
    prim = stage.GetPrimAtPath(prim_path)

    if not prim.IsValid():
        raise RuntimeError(f"Contact sensor body does not exist: '{prim_path}'.")
    if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Contact sensor target is not a rigid body: '{prim_path}'.")

    contact_report_api = PhysxSchema.PhysxContactReportAPI(prim)
    if not contact_report_api:
        contact_report_api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
    contact_report_api.CreateThresholdAttr().Set(0.0)


def _get_world_pose_from_local_offset(
    parent_path: str, local_offset: tuple[float, float, float]
):
    """Return world position and wxyz quaternion for a frame offset from a USD prim."""
    from pxr import Gf, UsdGeom

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError(
            "SimulationContext must exist before reading USD transforms."
        )

    stage = sim_context.stage
    parent_prim = stage.GetPrimAtPath(parent_path)
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


def _create_fixed_cylindrical_tool() -> None:
    """Spawn a rigid cylinder and fix it to wrist_3_link without relative motion."""
    from pxr import Gf, Sdf, UsdPhysics

    sim_context = sim_utils.SimulationContext.instance()
    if sim_context is None:
        raise RuntimeError(
            "SimulationContext must exist before creating the fixed tool."
        )
    stage = sim_context.stage

    # Spawn the cylinder at the same pose enforced by the fixed-joint frames.
    # This avoids a large initial snap when PhysX initializes the articulation.
    tool_position, tool_orientation = _get_world_pose_from_local_offset(
        WRIST_BODY_PATH, (0.0, 0.0, TOOL_CENTER_OFFSET)
    )

    tool_cfg = sim_utils.CylinderCfg(
        radius=TOOL_RADIUS,
        height=TOOL_HEIGHT,
        axis="Z",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=TOOL_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.005,
            rest_offset=0.0,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.8, 0.2, 0.1),
            metallic=0.1,
            roughness=0.5,
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
        raise RuntimeError(
            f"Failed to create rigid cylindrical tool at '{TOOL_BODY_PATH}'."
        )

    # Fixed-joint frame on body 0 is placed at the cylinder center.
    # Body 1's local joint frame is its own origin.
    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(TOOL_JOINT_PATH))
    joint.CreateBody0Rel().SetTargets([Sdf.Path(WRIST_BODY_PATH)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(TOOL_BODY_PATH)])
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, TOOL_CENTER_OFFSET))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    # Disable collision only between the two joint-connected bodies. The cylinder
    # remains fully collidable with the ground and all other scene geometry.
    joint.CreateCollisionEnabledAttr().Set(False)

    print(
        "[INFO]: Fixed cylindrical tool created | "
        f"radius={TOOL_RADIUS:.3f} m, height={TOOL_HEIGHT:.3f} m, "
        f"center offset={TOOL_CENTER_OFFSET:.3f} m"
    )


# -----------------------------------------------------------------------------
# Scene creation
# -----------------------------------------------------------------------------
def design_scene() -> dict[str, object]:
    """Create the collision plane, UR10e, rigid tool, and contact sensor."""

    # GroundPlaneCfg creates an actual physics collision plane, not a visual-only mesh.
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func(GROUND_PATH, ground_cfg)
    _assert_collision_geometry(GROUND_PATH)

    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    robot_cfg = UR10e_CFG.copy()
    robot_cfg.prim_path = ROBOT_PATH
    # Keep contact reporting enabled for the original robot bodies as well. The
    # newly created cylinder receives the API explicitly below.
    robot_cfg.spawn.activate_contact_sensors = True
    robot = Articulation(cfg=robot_cfg)

    _create_fixed_cylindrical_tool()
    _ensure_contact_reporter(TOOL_BODY_PATH)

    contact_sensor_cfg = ContactSensorCfg(
        prim_path=TOOL_BODY_PATH,
        update_period=0.0,
        history_length=8,
        track_pose=True,
        track_friction_forces=True,
        force_threshold=0.1,
        debug_vis=not args_cli.headless,
        filter_prim_paths_expr=[GROUND_PATH],
        max_contact_data_count_per_prim=16,
    )
    contact_sensor = ContactSensor(cfg=contact_sensor_cfg)

    return {
        "robot": robot,
        "contact_sensor": contact_sensor,
    }


# -----------------------------------------------------------------------------
# Trajectory helpers
# -----------------------------------------------------------------------------
def _cosine_blend(alpha: float) -> float:
    """Smooth interpolation factor with zero slope at alpha=0 and alpha=1."""
    alpha = max(0.0, min(1.0, alpha))
    return 0.5 - 0.5 * math.cos(math.pi * alpha)


def _interpolate_joints(
    start: list[float], end: list[float], alpha: float
) -> list[float]:
    blend = _cosine_blend(alpha)
    return [s + blend * (e - s) for s, e in zip(start, end)]


def _get_motion_command(
    sim_time: float,
) -> tuple[list[float], str, int, dict[str, object] | None]:
    """Return target joints, phase, cycle number, and sampled cycle command."""
    if sim_time < INITIAL_HOLD_TIME:
        return RELEASE_JOINTS, "INITIAL_RELEASE", 0, None

    cycle_elapsed_total = sim_time - INITIAL_HOLD_TIME
    cycle_index = min(int(cycle_elapsed_total // CYCLE_TIME), len(CYCLE_COMMANDS) - 1)
    cycle_time = cycle_elapsed_total - cycle_index * CYCLE_TIME
    cycle_number = cycle_index + 1
    cycle_command = CYCLE_COMMANDS[cycle_index]
    press_joints = cycle_command["press_joints"]
    assert isinstance(press_joints, list)

    if cycle_time < DESCEND_TIME:
        alpha = cycle_time / DESCEND_TIME
        return (
            _interpolate_joints(RELEASE_JOINTS, press_joints, alpha),
            "DESCEND",
            cycle_number,
            cycle_command,
        )

    cycle_time -= DESCEND_TIME
    if cycle_time < PRESS_HOLD_TIME:
        return press_joints, "PRESS_HOLD", cycle_number, cycle_command

    cycle_time -= PRESS_HOLD_TIME
    if cycle_time < ASCEND_TIME:
        alpha = cycle_time / ASCEND_TIME
        return (
            _interpolate_joints(press_joints, RELEASE_JOINTS, alpha),
            "ASCEND",
            cycle_number,
            cycle_command,
        )

    return RELEASE_JOINTS, "RELEASE_HOLD", cycle_number, cycle_command


# -----------------------------------------------------------------------------
# Simulation loop and logging
# -----------------------------------------------------------------------------
def _initialize_robot_pose(robot: Articulation, sim: SimulationContext) -> None:
    """Place the robot directly in the non-contact release configuration."""
    joint_pos = torch.tensor([RELEASE_JOINTS], dtype=torch.float32, device=sim.device)
    joint_vel = torch.zeros_like(joint_pos)

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.reset()
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()


def run_simulator(sim: SimulationContext, entities: dict[str, object]) -> None:
    robot: Articulation = entities["robot"]  # type: ignore[assignment]
    contact_sensor: ContactSensor = entities["contact_sensor"]  # type: ignore[assignment]

    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    step = 0

    log_dir = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "..", "logs"
    )
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.abspath(os.path.join(log_dir, "ur10e_floor_press_log.csv"))

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
        "cycle_wrist_1_delta_deg",
        "cycle_wrist_2_delta_deg",
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
        "tool_pos_x_w",
        "tool_pos_y_w",
        "tool_pos_z_w",
        "tool_quat_w",
        "tool_quat_x",
        "tool_quat_y",
        "tool_quat_z",
    ]

    physics_hz = 1.0 / sim_dt
    requested_log_hz = max(1.0, args_cli.log_hz)
    log_every_steps = max(1, round(physics_hz / min(requested_log_hz, physics_hz)))
    actual_log_hz = physics_hz / log_every_steps

    print(f"[INFO]: Logging data to {log_path}")
    print(
        f"[INFO]: cycles={max(1, args_cli.cycles)}, "
        f"nominal_press_scale={max(0.0, args_cli.press_scale):.3f}, "
        f"depth_jitter=±{max(0.0, args_cli.depth_jitter_mm):.2f} mm, "
        f"angle_jitter=±{max(0.0, args_cli.angle_jitter_deg):.2f} deg, seed={args_cli.seed}, "
        f"physics={physics_hz:.1f} Hz, logging={actual_log_hz:.1f} Hz, "
        f"total_time={TOTAL_TIME:.2f} s, steps={TOTAL_STEPS}"
    )
    for index, command in enumerate(CYCLE_COMMANDS, start=1):
        print(
            f"[INFO]: cycle {index:02d} randomization | "
            f"depth={float(command['depth_delta_m']) * 1.0e3:+.3f} mm, "
            f"wrist_1={math.degrees(float(command['wrist_1_delta_rad'])):+.3f} deg, "
            f"wrist_2={math.degrees(float(command['wrist_2_delta_rad'])):+.3f} deg, "
            f"press_scale={float(command['press_scale']):.5f}"
        )

    with open(log_path, "w", newline="") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(csv_header)

        while simulation_app.is_running() and step < TOTAL_STEPS:
            target_joints, phase_name, cycle_number, cycle_command = (
                _get_motion_command(sim_time)
            )

            target_tensor = torch.tensor(
                [target_joints], dtype=torch.float32, device=sim.device
            )
            robot.set_joint_position_target(target_tensor)
            robot.write_data_to_sim()

            sim.step()

            robot.update(sim_dt)
            contact_sensor.update(sim_dt, force_recompute=True)

            sim_time += sim_dt
            step += 1

            joint_pos = robot.data.joint_pos[0].tolist()
            joint_vel = robot.data.joint_vel[0].tolist()
            if hasattr(robot.data, "applied_torque"):
                joint_effort = robot.data.applied_torque[0].tolist()
            else:
                joint_effort = [0.0] * len(joint_names)

            normal_force = contact_sensor.data.net_forces_w[0, 0]

            print(contact_sensor.data)

            friction_force = contact_sensor.data.friction_forces_w[0, 0, 0]
            friction_force = torch.nan_to_num(
                friction_force,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            total_force = normal_force + friction_force

            normal_force_list = normal_force.tolist()
            friction_force_list = friction_force.tolist()
            total_force_list = total_force.tolist()

            normal_force_norm = float(torch.linalg.vector_norm(normal_force).item())
            friction_force_norm = float(torch.linalg.vector_norm(friction_force).item())
            total_force_norm = float(torch.linalg.vector_norm(total_force).item())

            tool_position = contact_sensor.data.pos_w[0, 0].tolist()
            tool_orientation = contact_sensor.data.quat_w[0, 0].tolist()

            if cycle_command is None:
                cycle_press_scale = max(0.0, args_cli.press_scale)
                cycle_depth_delta_mm = 0.0
                cycle_wrist_1_delta_deg = 0.0
                cycle_wrist_2_delta_deg = 0.0
            else:
                cycle_press_scale = float(cycle_command["press_scale"])
                cycle_depth_delta_mm = float(cycle_command["depth_delta_m"]) * 1.0e3
                cycle_wrist_1_delta_deg = math.degrees(
                    float(cycle_command["wrist_1_delta_rad"])
                )
                cycle_wrist_2_delta_deg = math.degrees(
                    float(cycle_command["wrist_2_delta_rad"])
                )

            # The physics and sensor buffers are updated at 200 Hz, but CSV output
            # is decimated to approximately args_cli.log_hz (100 Hz by default).
            if step % log_every_steps == 0:
                row = [
                    sim_time,
                    cycle_number,
                    phase_name,
                    cycle_press_scale,
                    cycle_depth_delta_mm,
                    cycle_wrist_1_delta_deg,
                    cycle_wrist_2_delta_deg,
                ]
                row += joint_pos + joint_vel + joint_effort
                row += normal_force_list + [normal_force_norm]
                row += friction_force_list + [friction_force_norm]
                row += total_force_list + [total_force_norm]
                row += tool_position + tool_orientation

                formatted_row = [
                    f"{value:.6f}" if isinstance(value, float) else str(value)
                    for value in row
                ]
                csv_writer.writerow(formatted_row)

            if step % max(1, int(0.25 / sim_dt)) == 0:
                print(
                    f"[{sim_time:6.2f}s] cycle={cycle_number:02d} "
                    f"phase={phase_name:15s} | "
                    f"Fn=[{normal_force_list[0]:8.2f}, "
                    f"{normal_force_list[1]:8.2f}, {normal_force_list[2]:8.2f}] N | "
                    f"Ft=[{friction_force_list[0]:8.2f}, "
                    f"{friction_force_list[1]:8.2f}, {friction_force_list[2]:8.2f}] N | "
                    f"Ftotal=[{total_force_list[0]:8.2f}, "
                    f"{total_force_list[1]:8.2f}, {total_force_list[2]:8.2f}] N | "
                    f"|Ftotal|={total_force_norm:8.2f} N | "
                    f"tool_z={tool_position[2]:7.3f} m"
                )

    logged_samples = step // log_every_steps
    print(
        f"\n[INFO]: Simulation complete. {step} physics steps, "
        f"approximately {logged_samples} CSV samples written to {log_path}"
    )


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
    sim.set_camera_view(eye=[1.8, 1.8, 1.4], target=[0.55, 0.0, 0.35])

    entities = design_scene()

    sim.reset()
    robot: Articulation = entities["robot"]  # type: ignore[assignment]
    _initialize_robot_pose(robot, sim)

    print("[INFO]: Setup complete. Starting repeated floor-contact motion.")
    run_simulator(sim, entities)


if __name__ == "__main__":
    main()
    simulation_app.close()
