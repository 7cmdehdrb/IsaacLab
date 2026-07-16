"""Isaac Lab UR5e + Robotiq 2F-85 virtual F/T joint example.

This script keeps the original USD assets, inserts a fixed mount joint between
the UR5e tool body and the Robotiq base body, and reads the resulting virtual
force/torque signal through Isaac Lab's tensor-backed articulation data.

The important part for RL is ``get_virtual_ft_wrench_b``: it returns a
``torch.Tensor`` directly from the PhysX tensor articulation view.  The printed
Python lists are only for human-readable logging.

Usage:
    ./isaaclab.sh -p scripts/lab/ur5e_2f85_ft_test.py --device cuda:0 --num_envs 16 --headless
"""

from __future__ import annotations

import argparse
import math
import re

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="UR5e + Robotiq 2F-85 virtual F/T joint example using Isaac Lab tensors."
)
parser.add_argument(
    "--num_envs", type=int, default=1, help="Number of parallel environments."
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=2.0,
    help="Spacing between environments in metres.",
)
parser.add_argument(
    "--steps",
    type=int,
    default=0,
    help="Number of simulation steps. 0 means run until closed.",
)
parser.add_argument(
    "--print_every",
    type=int,
    default=30,
    help="Print the wrench every N simulation steps.",
)
parser.add_argument(
    "--contact_force_limit",
    type=float,
    default=60.0,
    help="Force magnitude in N that makes the validation cube back away from the probe.",
)
parser.add_argument(
    "--contact_torque_limit",
    type=float,
    default=12.0,
    help="Torque magnitude in N*m that makes the validation cube back away from the probe.",
)
parser.add_argument(
    "--contact_twist_deg",
    type=float,
    default=30.0,
    help="Peak cube twist angle in degrees during the contact hold phase.",
)
parser.add_argument(
    "--contact_seed",
    type=int,
    default=42,
    help="Random seed for contact axes and twist directions.",
)
parser.add_argument(
    "--disable_ft_visualization",
    action="store_true",
    help="Disable viewport force/torque arrows. Visualization is also skipped in headless mode.",
)
parser.add_argument(
    "--vis_every",
    type=int,
    default=4,
    help="Update viewport F/T arrows every N simulation steps.",
)
parser.add_argument(
    "--debug_gripper_topology",
    action="store_true",
    help="Print Robotiq rigid bodies, joint targets, and runtime body positions for topology diagnosis.",
)
parser.add_argument(
    "--disable_contact_test",
    action="store_true",
    help="Disable the built-in probe/cube contact sequence used to validate the F/T tensor.",
)
parser.add_argument(
    "--keep_invalid_collision_apis",
    action="store_true",
    help="Keep CollisionAPI on non-geometry prims in the referenced USDs. By default these are removed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below executes after Isaac Sim is launched."""

import omni.client
import omni.usd
import torch
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

if not args_cli.headless and not args_cli.disable_ft_visualization:
    import isaacsim.util.debug_draw._debug_draw as omni_debug_draw
else:
    omni_debug_draw = None

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import math as math_utils

# -----------------------------------------------------------------------------
# Asset and mount settings
# -----------------------------------------------------------------------------

UR5E_USD_PATH = "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.0/Isaac/Robots/UniversalRobots/ur5e/ur5e.usd"
ROBOTIQ_2F85_USD_PATH = (
    "omniverse://192.168.0.13/NVIDIA/Assets/Isaac/5.1/Isaac/Robots/Robotiq/2F-85/Robotiq_2F_85_edit.usd"
)

ENV_ROOT_EXPR = "/World/envs/env_.*"
ENV_ROOT_TEMPLATE = "/World/envs/env_{env_id}"
UR_REFERENCE_PATH = f"{ENV_ROOT_EXPR}/UR5e"
ROBOT_PRIM_TEMPLATE = ENV_ROOT_TEMPLATE + "/UR5e"
GRIPPER_PRIM_TEMPLATE = ENV_ROOT_TEMPLATE + "/Robotiq2F85"
FT_SENSOR_BODY_TEMPLATE = ROBOT_PRIM_TEMPLATE + "/VirtualFTSensor"
WRIST_SENSOR_JOINT_TEMPLATE = ENV_ROOT_TEMPLATE + "/UR5e_virtual_ft_parent_joint"
SENSOR_GRIPPER_JOINT_TEMPLATE = ENV_ROOT_TEMPLATE + "/VirtualFTSensor_2F85_child_joint"
PROBE_BODY_TEMPLATE = GRIPPER_PRIM_TEMPLATE + "/FTProbePad"
PROBE_JOINT_TEMPLATE = ENV_ROOT_TEMPLATE + "/Robotiq2F85_probe_pad_joint"
CUBE_PRIM_PATH = f"{ENV_ROOT_EXPR}/TestCube"
CUBE_PRIM_TEMPLATE = ENV_ROOT_TEMPLATE + "/TestCube"

UR_TOOL_FRAME_CANDIDATES = (
    "tool0",
    "tool_frame",
    "flange",
    "wrist_3_link",
)

GRIPPER_BASE_BODY_CANDIDATES = (
    "robotiq_arg2f_base_link",
    "robotiq_2f_85_base_link",
    "robotiq_base_link",
    "base_link",
)

PREEXISTING_GRIPPER_MOUNT_JOINT_NAMES = ("robot_gripper_joint",)

# 기존 장착 조인트를 아래의 2단 고정 조인트 체인으로 교체한다.
# UR 손목 -> VirtualFTSensor -> Robotiq 베이스
# Optional mounting correction after automatic frame alignment.
# Translation: metres, Rotation: XYZ Euler degrees.
MOUNT_TRANSLATION_OFFSET = (0.0, 0.0, 0.0)
MOUNT_ROTATION_OFFSET_DEG = (0.0, 90.0, 0.0)

PHYSICS_DT = 1.0 / 120.0

FT_SENSOR_BODY_NAME = "VirtualFTSensor"
FT_SENSOR_SIZE = (0.025, 0.025, 0.025)
FT_SENSOR_MASS = 1.0e-3

PROBE_BODY_NAME = "FTProbePad"
PROBE_SIZE = (0.06, 0.06, 0.06)
PROBE_MASS = 0.02
PROBE_LOCAL_OFFSET = (0.0, 0.0, 0.18)
CONTACT_STIFFNESS = 20_000.0
CONTACT_DAMPING = 100.0
MAX_DEPENETRATION_VELOCITY = 0.5

# A small dynamic cube is kept from the original example so that the gripper can
# be pushed against something during interactive tests.
CUBE_SIZE = 0.08
CUBE_MASS = 1.0
CUBE_POSITION = (0.45, 0.0, 0.04)
CUBE_FAR_DISTANCE = 0.12
CUBE_CONTACT_PENETRATION = 0.0005
CUBE_CONTACT_DISTANCE = 0.5 * (PROBE_SIZE[0] + CUBE_SIZE) - CUBE_CONTACT_PENETRATION
CONTACT_APPROACH_STEPS = 240
CONTACT_HOLD_STEPS = 180
CONTACT_RELEASE_STEPS = 120
CONTACT_MAX_APPROACH_SPEED = 0.035
CONTACT_MAX_BACKOFF_SPEED = 0.10
CUBE_TILT_SUPPORT_COMPENSATION = 0.6

FORCE_ARROW_METRES_PER_N = 0.003
FORCE_ARROW_MAX_LENGTH = 0.25
TORQUE_ARROW_METRES_PER_NM = 0.025
TORQUE_ARROW_MAX_LENGTH = 0.25
TORQUE_ARROW_OFFSET = 0.035


# -----------------------------------------------------------------------------
# USD helpers
# -----------------------------------------------------------------------------


def check_nucleus_asset(url: str, label: str) -> None:
    """Verify that a Nucleus URL exists and is readable."""
    if not url.startswith("omniverse://"):
        raise ValueError(f"{label} must be an omniverse:// URL. Received: {url}")

    result, _entry = omni.client.stat(url)
    if result != omni.client.Result.OK:
        raise FileNotFoundError(
            f"Cannot access {label} on Nucleus.\n"
            f"URL: {url}\n"
            f"omni.client result: {result}"
        )
    print(f"[INFO]: Found {label}: {url}")


def wait_for_stage_loading(max_updates: int = 300) -> None:
    """Wait until asynchronous USD reference loading completes."""
    context = omni.usd.get_context()
    stable_frames = 0

    for _ in range(max_updates):
        simulation_app.update()
        _message, files_loaded, total_files = context.get_stage_loading_status()
        if int(files_loaded) == 0 and int(total_files) == 0:
            stable_frames += 1
            if stable_frames >= 5:
                return
        else:
            stable_frames = 0

    message, files_loaded, total_files = context.get_stage_loading_status()
    raise TimeoutError(
        "USD stage loading did not finish.\n"
        f"message={message!r}, files_loaded={files_loaded}, total_files={total_files}"
    )


def define_env_roots(
    stage: Usd.Stage, num_envs: int, env_spacing: float
) -> torch.Tensor:
    """Create /World/envs/env_i Xforms and return their origins as a tensor."""
    UsdGeom.Xform.Define(stage, Sdf.Path("/World/envs"))

    cols = max(1, math.ceil(math.sqrt(num_envs)))
    origins = []

    for env_id in range(num_envs):
        row = env_id // cols
        col = env_id % cols
        origin = (col * env_spacing, row * env_spacing, 0.0)
        origins.append(origin)

        env_path = ENV_ROOT_TEMPLATE.format(env_id=env_id)
        xform = UsdGeom.Xform.Define(stage, Sdf.Path(env_path))
        xformable = UsdGeom.Xformable(xform.GetPrim())
        xformable.ClearXformOpOrder()
        xformable.AddTranslateOp().Set(Gf.Vec3d(*origin))

    return torch.tensor(origins, dtype=torch.float32)


def find_named_prim(
    stage: Usd.Stage,
    subtree_path: str,
    candidate_names: tuple[str, ...],
    *,
    rigid_body_only: bool = False,
) -> Usd.Prim:
    """Find a prim by candidate leaf name inside a subtree."""
    root = stage.GetPrimAtPath(subtree_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid subtree: {subtree_path}")

    prims = list(Usd.PrimRange(root))
    for candidate in candidate_names:
        matches = [prim for prim in prims if prim.GetName() == candidate]
        for prim in matches:
            if rigid_body_only and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            return prim

    available = []
    for prim in prims:
        if rigid_body_only and not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        available.append(str(prim.GetPath()))

    raise RuntimeError(
        f"Could not find any candidate {candidate_names} under {subtree_path}.\n"
        "Available relevant prim paths:\n  " + "\n  ".join(available)
    )


def nearest_rigid_body_ancestor(prim: Usd.Prim) -> Usd.Prim:
    current = prim
    while current.IsValid():
        if current.HasAPI(UsdPhysics.RigidBodyAPI):
            return current
        current = current.GetParent()
    raise RuntimeError(f"No rigid-body ancestor for {prim.GetPath()}")


def remove_articulation_root_apis(stage: Usd.Stage, subtree_path: str) -> None:
    """Remove nested articulation roots from the standalone gripper asset."""
    root = stage.GetPrimAtPath(subtree_path)
    removed = []

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
            removed.append(str(prim.GetPath()))

    if removed:
        print(
            f"[INFO]: Removed gripper articulation roots below {subtree_path}: {removed}"
        )


def deinstance_gripper_geometry(
    stage: Usd.Stage, gripper_path: str, env_id: int
) -> None:
    """Materialize link geometry so Fabric updates it with the merged articulation bodies."""
    root = stage.GetPrimAtPath(gripper_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid gripper subtree: {gripper_path}")

    # 병합된 articulation에서는 instance proxy 시각 프림이 Fabric의 링크 자세를
    # 따라가지 않을 수 있다. 원본 USD는 유지하고 현재 stage의 인스턴스만 해제한다.
    instance_roots = [prim for prim in Usd.PrimRange(root) if prim.IsInstance()]
    for prim in instance_roots:
        prim.SetInstanceable(False)

    if env_id == 0:
        print(
            f"[INFO]: De-instanced {len(instance_roots)} Robotiq geometry roots for Fabric synchronization."
        )


def print_gripper_usd_topology(stage: Usd.Stage, gripper_path: str) -> None:
    """Print the authored rigid-body and joint graph for one Robotiq instance."""
    root = stage.GetPrimAtPath(gripper_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid gripper subtree: {gripper_path}")

    print("[DEBUG]: Robotiq rigid bodies:")
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"  BODY  {prim.GetPath()}")

    print("[DEBUG]: Robotiq joints:")
    for prim in Usd.PrimRange(root):
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        body_0 = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body_1 = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        print(
            f"  JOINT {prim.GetTypeName():<14} {prim.GetPath()} enabled={joint.GetJointEnabledAttr().Get()} "
            f"body0={body_0} body1={body_1}"
        )

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render]
    )
    print("[DEBUG]: Robotiq geometry world bounds:")
    geometry_prims = sim_utils.get_all_matching_child_prims(
        gripper_path,
        predicate=lambda prim: prim.IsA(UsdGeom.Gprim),
        stage=stage,
        traverse_instance_prims=True,
    )
    reset_xform_paths = set()
    first_geometry_ancestors = []
    for prim in geometry_prims:
        rigid_body = nearest_rigid_body_ancestor(prim)
        ancestor = prim
        record_ancestor_chain = not first_geometry_ancestors
        while ancestor.IsValid() and ancestor.GetPath().HasPrefix(root.GetPath()):
            if (
                ancestor.IsA(UsdGeom.Xformable)
                and UsdGeom.Xformable(ancestor).GetResetXformStack()
            ):
                reset_xform_paths.add(str(ancestor.GetPath()))
            if record_ancestor_chain:
                ancestor_world = Gf.Transform(
                    world_transform(ancestor)
                ).GetTranslation()
                first_geometry_ancestors.append(
                    (
                        str(ancestor.GetPath()),
                        ancestor.GetTypeName(),
                        ancestor.IsInstance(),
                        ancestor.IsInstanceProxy(),
                        tuple(float(value) for value in ancestor_world),
                    )
                )
            ancestor = ancestor.GetParent()
        world_translation = Gf.Transform(world_transform(prim)).GetTranslation()
        world_bound = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        if world_bound.IsEmpty():
            bound_text = "empty"
        else:
            center = world_bound.GetMidpoint()
            size = world_bound.GetSize()
            bound_text = (
                f"center=({center[0]: .4f}, {center[1]: .4f}, {center[2]: .4f}) "
                f"size=({size[0]: .4f}, {size[1]: .4f}, {size[2]: .4f})"
            )
        print(
            f"  GEOM {prim.GetTypeName():<10} body={rigid_body.GetName():<22} "
            f"xform=({world_translation[0]: .4f}, {world_translation[1]: .4f}, {world_translation[2]: .4f}) "
            f"{bound_text} path={prim.GetPath()}"
        )
    print(
        f"[DEBUG]: Robotiq geometry ancestors with resetXformStack: {sorted(reset_xform_paths)}"
    )
    print(f"[DEBUG]: First geometry ancestor chain: {first_geometry_ancestors}")


def world_transform(prim: Usd.Prim) -> Gf.Matrix4d:
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    return cache.GetLocalToWorldTransform(prim)


def set_world_transform_on_reference_root(
    root_prim: Usd.Prim, matrix: Gf.Matrix4d
) -> None:
    """Set a reference root world transform while preserving its environment parent transform."""
    parent = root_prim.GetParent()
    if not parent.IsValid():
        raise RuntimeError(f"Reference root has no valid parent: {root_prim.GetPath()}")

    local_matrix = matrix * world_transform(parent).GetInverse()
    xformable = UsdGeom.Xformable(root_prim)
    xformable.ClearXformOpOrder()
    op = xformable.AddTransformOp(UsdGeom.XformOp.PrecisionDouble)
    op.Set(local_matrix)
    xformable.SetResetXformStack(False)


def make_offset_matrix(
    translation: tuple[float, float, float],
    rotation_deg: tuple[float, float, float],
) -> Gf.Matrix4d:
    matrix = Gf.Matrix4d(1.0)
    rotation = (
        Gf.Rotation(Gf.Vec3d(1.0, 0.0, 0.0), rotation_deg[0])
        * Gf.Rotation(Gf.Vec3d(0.0, 1.0, 0.0), rotation_deg[1])
        * Gf.Rotation(Gf.Vec3d(0.0, 0.0, 1.0), rotation_deg[2])
    )
    matrix.SetRotate(rotation)
    matrix.SetTranslateOnly(Gf.Vec3d(*translation))
    return matrix


def align_gripper_to_tool(
    gripper_reference_root: Usd.Prim,
    gripper_base_body: Usd.Prim,
    tool_frame: Usd.Prim,
) -> None:
    """Move the gripper so its base rigid body coincides with the UR tool frame."""
    root_world = world_transform(gripper_reference_root)
    base_world = world_transform(gripper_base_body)
    tool_world = world_transform(tool_frame)

    # USD Gf 행렬은 행 벡터 규약이므로 변환 곱셈 순서에 주의한다.
    base_relative_to_root = base_world * root_world.GetInverse()
    desired_base_world = (
        make_offset_matrix(MOUNT_TRANSLATION_OFFSET, MOUNT_ROTATION_OFFSET_DEG)
        * tool_world
    )
    desired_root_world = base_relative_to_root.GetInverse() * desired_base_world

    set_world_transform_on_reference_root(gripper_reference_root, desired_root_world)


def pose_error(actual: Gf.Matrix4d, expected: Gf.Matrix4d) -> tuple[float, float]:
    """Return translation error in metres and quaternion angular error in degrees."""
    actual_transform = Gf.Transform(actual)
    expected_transform = Gf.Transform(expected)
    translation_error = (
        actual_transform.GetTranslation() - expected_transform.GetTranslation()
    ).GetLength()

    actual_quat = actual_transform.GetRotation().GetQuat().GetNormalized()
    expected_quat = expected_transform.GetRotation().GetQuat().GetNormalized()
    actual_imag = actual_quat.GetImaginary()
    expected_imag = expected_quat.GetImaginary()
    quat_dot = abs(
        float(actual_quat.GetReal() * expected_quat.GetReal())
        + float(actual_imag[0] * expected_imag[0])
        + float(actual_imag[1] * expected_imag[1])
        + float(actual_imag[2] * expected_imag[2])
    )
    angular_error_deg = math.degrees(2.0 * math.acos(min(1.0, quat_dot)))
    return float(translation_error), angular_error_deg


def validate_gripper_alignment(
    gripper_base_body: Usd.Prim, tool_frame: Usd.Prim, env_id: int
) -> None:
    """Fail before physics starts if the gripper base is not at the requested tool pose."""
    expected_base_world = make_offset_matrix(
        MOUNT_TRANSLATION_OFFSET, MOUNT_ROTATION_OFFSET_DEG
    ) * world_transform(tool_frame)
    position_error, angular_error = pose_error(
        world_transform(gripper_base_body), expected_base_world
    )
    if position_error > 1.0e-5 or angular_error > 1.0e-3:
        raise RuntimeError(
            f"Gripper alignment failed in env_{env_id}: "
            f"position_error={position_error:.6e} m, angular_error={angular_error:.6e} deg"
        )
    if env_id == 0:
        print(
            f"[INFO]: Gripper alignment error: position={position_error:.3e} m, "
            f"rotation={angular_error:.3e} deg"
        )


def deactivate_preexisting_gripper_mount_joints(
    stage: Usd.Stage, ur_path: str, env_id: int
) -> None:
    """Deactivate stale mount joints because the virtual F/T chain replaces them."""
    ur_root = stage.GetPrimAtPath(ur_path)
    if not ur_root.IsValid():
        raise RuntimeError(f"Invalid UR subtree: {ur_path}")

    disabled = []
    matching_prims = []
    for prim in Usd.PrimRange(ur_root):
        if prim.GetName() not in PREEXISTING_GRIPPER_MOUNT_JOINT_NAMES:
            continue
        joint = UsdPhysics.Joint(prim)
        if not joint:
            continue
        matching_prims.append(prim)

    # 기존 joint를 남기면 새 가상 센서 체인과 중복 구속되어 링크가 튈 수 있다.
    for prim in matching_prims:
        joint = UsdPhysics.Joint(prim)
        body_0_targets = [str(path) for path in joint.GetBody0Rel().GetTargets()]
        body_1_targets = [str(path) for path in joint.GetBody1Rel().GetTargets()]
        joint.GetJointEnabledAttr().Set(False)
        prim.SetActive(False)
        disabled.append((str(prim.GetPath()), body_0_targets, body_1_targets))

    if env_id == 0 and disabled:
        print(f"[INFO]: Deactivated pre-existing gripper mount joints: {disabled}")


def matrix_to_pose(matrix: Gf.Matrix4d) -> tuple[Gf.Vec3f, Gf.Quatf]:
    transform = Gf.Transform(matrix)
    translation = transform.GetTranslation()
    quaternion = transform.GetRotation().GetQuat()
    imag = quaternion.GetImaginary()

    return (
        Gf.Vec3f(float(translation[0]), float(translation[1]), float(translation[2])),
        Gf.Quatf(
            float(quaternion.GetReal()),
            Gf.Vec3f(float(imag[0]), float(imag[1]), float(imag[2])),
        ),
    )


def pose_to_tuples(
    position: Gf.Vec3f,
    orientation: Gf.Quatf,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    imag = orientation.GetImaginary()
    return (
        (float(position[0]), float(position[1]), float(position[2])),
        (float(orientation.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])),
    )


def world_matrix_to_parent_pose(
    stage: Usd.Stage,
    world_matrix: Gf.Matrix4d,
    parent_path: str,
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    parent = stage.GetPrimAtPath(parent_path)
    if not parent.IsValid():
        raise RuntimeError(f"Invalid parent prim: {parent_path}")
    local_matrix = world_matrix * world_transform(parent).GetInverse()
    return pose_to_tuples(*matrix_to_pose(local_matrix))


def sanitize_invalid_collision_apis(stage: Usd.Stage, subtree_path: str) -> None:
    """Remove CollisionAPI from prims that PhysX cannot cook as geometry."""
    root = stage.GetPrimAtPath(subtree_path)
    if not root.IsValid():
        raise RuntimeError(f"Invalid subtree: {subtree_path}")

    valid_geometry_types = (
        UsdGeom.Mesh,
        UsdGeom.Cube,
        UsdGeom.Sphere,
        UsdGeom.Capsule,
        UsdGeom.Cylinder,
        UsdGeom.Cone,
    )
    removed = []
    for prim in Usd.PrimRange(root):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        if any(prim.IsA(geometry_type) for geometry_type in valid_geometry_types):
            continue
        prim.RemoveAPI(UsdPhysics.CollisionAPI)
        removed.append(str(prim.GetPath()))

    if removed:
        print(
            f"[INFO]: Removed invalid CollisionAPI entries below {subtree_path}: {removed}"
        )


def create_fixed_mount_joint(
    stage: Usd.Stage,
    joint_path: str,
    parent_body: Usd.Prim,
    child_body: Usd.Prim,
    joint_frame_world: Gf.Matrix4d,
) -> None:
    """Create a fixed joint while preserving current body poses."""
    parent_world = world_transform(parent_body)
    child_world = world_transform(child_body)

    # 두 body의 현재 world pose를 보존하도록 동일한 joint frame을 각 body의
    # local frame으로 변환한다. 이 계산이 틀리면 PhysX가 시작 시 body를 snap한다.
    local_frame_0 = joint_frame_world * parent_world.GetInverse()
    local_frame_1 = joint_frame_world * child_world.GetInverse()
    local_pos_0, local_rot_0 = matrix_to_pose(local_frame_0)
    local_pos_1, local_rot_1 = matrix_to_pose(local_frame_1)

    joint = UsdPhysics.FixedJoint.Define(stage, Sdf.Path(joint_path))
    joint.CreateBody0Rel().SetTargets([parent_body.GetPath()])
    joint.CreateBody1Rel().SetTargets([child_body.GetPath()])
    joint.CreateLocalPos0Attr().Set(local_pos_0)
    joint.CreateLocalRot0Attr().Set(local_rot_0)
    joint.CreateLocalPos1Attr().Set(local_pos_1)
    joint.CreateLocalRot1Attr().Set(local_rot_1)
    joint.CreateCollisionEnabledAttr().Set(False)


def create_virtual_ft_sensor_body(
    stage: Usd.Stage, env_id: int, sensor_world: Gf.Matrix4d
) -> Usd.Prim:
    """Create the tiny rigid body whose incoming joint wrench is used as the F/T signal."""
    sensor_path = FT_SENSOR_BODY_TEMPLATE.format(env_id=env_id)
    robot_path = ROBOT_PRIM_TEMPLATE.format(env_id=env_id)
    translation, orientation = world_matrix_to_parent_pose(
        stage, sensor_world, robot_path
    )

    # 이 작은 rigid body로 들어오는 joint wrench가 가상 F/T 센서의 측정값이다.
    # 충돌은 끄고 질량은 작게 두어 로봇 동역학에 미치는 영향을 제한한다.
    sensor_cfg = sim_utils.CuboidCfg(
        size=FT_SENSOR_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False),
        mass_props=sim_utils.MassPropertiesCfg(mass=FT_SENSOR_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.55, 0.10)),
    )
    sensor_cfg.func(
        sensor_path, sensor_cfg, translation=translation, orientation=orientation
    )

    sensor_prim = stage.GetPrimAtPath(sensor_path)
    if not sensor_prim.IsValid() or not sensor_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(
            f"Virtual F/T sensor body was not created as a rigid body: {sensor_path}"
        )
    return sensor_prim


def create_probe_pad(stage: Usd.Stage, env_id: int, parent_body: Usd.Prim) -> Usd.Prim:
    """Create a small downstream contact pad for deterministic F/T validation."""
    probe_path = PROBE_BODY_TEMPLATE.format(env_id=env_id)
    gripper_path = GRIPPER_PRIM_TEMPLATE.format(env_id=env_id)
    probe_world = make_offset_matrix(
        PROBE_LOCAL_OFFSET, (0.0, 0.0, 0.0)
    ) * world_transform(parent_body)
    translation, orientation = world_matrix_to_parent_pose(
        stage, probe_world, gripper_path
    )

    probe_cfg = sim_utils.CuboidCfg(
        size=PROBE_SIZE,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=PROBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.003, rest_offset=0.0
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            compliant_contact_stiffness=CONTACT_STIFFNESS,
            compliant_contact_damping=CONTACT_DAMPING,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.10, 0.75, 0.35)),
    )
    probe_cfg.func(
        probe_path, probe_cfg, translation=translation, orientation=orientation
    )

    probe_prim = stage.GetPrimAtPath(probe_path)
    if not probe_prim.IsValid() or not probe_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        raise RuntimeError(f"Probe pad was not created as a rigid body: {probe_path}")

    create_fixed_mount_joint(
        stage=stage,
        joint_path=PROBE_JOINT_TEMPLATE.format(env_id=env_id),
        parent_body=parent_body,
        child_body=probe_prim,
        joint_frame_world=probe_world,
    )
    return probe_prim


def filter_contact_cube_to_probe_only(stage: Usd.Stage, env_id: int) -> None:
    """Filter cube collisions against the robot and gripper links while keeping probe contact."""
    cube_path = CUBE_PRIM_TEMPLATE.format(env_id=env_id)
    cube_prim = stage.GetPrimAtPath(cube_path)
    if not cube_prim.IsValid():
        raise RuntimeError(f"Invalid contact-test cube: {cube_path}")

    # 검증 큐브가 손가락이나 손목을 직접 밀면 articulation이 무너질 수 있으므로
    # 센서 하류의 FTProbePad와만 접촉하게 제한한다.
    filtered_pairs = UsdPhysics.FilteredPairsAPI.Apply(
        cube_prim
    ).CreateFilteredPairsRel()
    filtered_pairs.AddTarget(Sdf.Path(ROBOT_PRIM_TEMPLATE.format(env_id=env_id)))

    gripper_path = GRIPPER_PRIM_TEMPLATE.format(env_id=env_id)
    gripper_root = stage.GetPrimAtPath(gripper_path)
    for prim in Usd.PrimRange(gripper_root):
        if (
            not prim.HasAPI(UsdPhysics.RigidBodyAPI)
            or prim.GetName() == PROBE_BODY_NAME
        ):
            continue
        filtered_pairs.AddTarget(prim.GetPath())


def assemble_env(stage: Usd.Stage, env_id: int) -> str:
    """Attach the Robotiq gripper to the UR5e in one environment.

    Returns:
        The unique virtual F/T body name whose incoming joint wrench should be
        read from the Isaac Lab articulation tensor.
    """
    ur_path = ROBOT_PRIM_TEMPLATE.format(env_id=env_id)
    gripper_path = GRIPPER_PRIM_TEMPLATE.format(env_id=env_id)

    # 물리가 시작되기 전에 시각 프림, 기존 joint, articulation root를 정리한 뒤
    # 하나의 Isaac Lab Articulation으로 묶는다.
    deinstance_gripper_geometry(stage, gripper_path, env_id)
    tool_frame = find_named_prim(stage, ur_path, UR_TOOL_FRAME_CANDIDATES)
    ur_tool_rigid_body = nearest_rigid_body_ancestor(tool_frame)
    gripper_root = stage.GetPrimAtPath(gripper_path)
    gripper_base_body = find_named_prim(
        stage,
        gripper_path,
        GRIPPER_BASE_BODY_CANDIDATES,
        rigid_body_only=True,
    )

    deactivate_preexisting_gripper_mount_joints(stage, ur_path, env_id)
    align_gripper_to_tool(gripper_root, gripper_base_body, tool_frame)
    validate_gripper_alignment(gripper_base_body, tool_frame, env_id)
    remove_articulation_root_apis(stage, gripper_path)
    if env_id == 0 and args_cli.debug_gripper_topology:
        print_gripper_usd_topology(stage, gripper_path)
    if not args_cli.keep_invalid_collision_apis:
        sanitize_invalid_collision_apis(stage, ur_path)
        sanitize_invalid_collision_apis(stage, gripper_path)

    # 실제 센서와 같은 위치에 가상 body를 삽입하고 양쪽을 고정 조인트로 연결한다.
    sensor_world = world_transform(tool_frame)
    virtual_sensor_body = create_virtual_ft_sensor_body(stage, env_id, sensor_world)
    create_fixed_mount_joint(
        stage=stage,
        joint_path=WRIST_SENSOR_JOINT_TEMPLATE.format(env_id=env_id),
        parent_body=ur_tool_rigid_body,
        child_body=virtual_sensor_body,
        joint_frame_world=sensor_world,
    )
    create_fixed_mount_joint(
        stage=stage,
        joint_path=SENSOR_GRIPPER_JOINT_TEMPLATE.format(env_id=env_id),
        parent_body=virtual_sensor_body,
        child_body=gripper_base_body,
        joint_frame_world=sensor_world,
    )
    if not args_cli.disable_contact_test:
        create_probe_pad(stage, env_id, gripper_base_body)
        filter_contact_cube_to_probe_only(stage, env_id)

    if env_id == 0:
        print(f"[INFO]: UR tool frame      : {tool_frame.GetPath()}")
        print(f"[INFO]: UR tool body       : {ur_tool_rigid_body.GetPath()}")
        print(f"[INFO]: F/T sensor body    : {virtual_sensor_body.GetPath()}")
        print(f"[INFO]: Robotiq base body  : {gripper_base_body.GetPath()}")
        print(
            f"[INFO]: Parent F/T joint   : {WRIST_SENSOR_JOINT_TEMPLATE.format(env_id=env_id)}"
        )
        print(
            f"[INFO]: Child F/T joint    : {SENSOR_GRIPPER_JOINT_TEMPLATE.format(env_id=env_id)}"
        )

    return FT_SENSOR_BODY_NAME


# -----------------------------------------------------------------------------
# Isaac Lab setup and tensor observation
# -----------------------------------------------------------------------------


def make_ur5e_spawn_cfg() -> sim_utils.UsdFileCfg:
    return sim_utils.UsdFileCfg(
        usd_path=UR5E_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=1,
        ),
        activate_contact_sensors=False,
    )


def spawn_robots(num_envs: int) -> None:
    """Spawn each UR5e reference at a concrete path to avoid clone-related mesh cooking issues."""
    robot_spawn_cfg = make_ur5e_spawn_cfg()
    for env_id in range(num_envs):
        robot_spawn_cfg.func(ROBOT_PRIM_TEMPLATE.format(env_id=env_id), robot_spawn_cfg)
    wait_for_stage_loading()


def create_robot() -> Articulation:
    """Create the Isaac Lab articulation wrapper around already-spawned UR5e references."""
    robot_cfg = ArticulationCfg(
        prim_path=UR_REFERENCE_PATH,
        spawn=None,
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan_joint": 0.0,
                "shoulder_lift_joint": -1.5708,
                "elbow_joint": 1.5708,
                "wrist_1_joint": -1.5708,
                "wrist_2_joint": -1.5708,
                "wrist_3_joint": 0.0,
            }
        ),
        actuators={
            "all_joints": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=None,
                damping=None,
                effort_limit_sim=None,
                velocity_limit_sim=None,
            )
        },
    )
    return Articulation(cfg=robot_cfg)


def spawn_grippers(num_envs: int) -> None:
    """Spawn each Robotiq reference at a concrete path, then merge it into the UR articulation."""
    gripper_cfg = sim_utils.UsdFileCfg(usd_path=ROBOTIQ_2F85_USD_PATH)
    for env_id in range(num_envs):
        gripper_cfg.func(GRIPPER_PRIM_TEMPLATE.format(env_id=env_id), gripper_cfg)
    wait_for_stage_loading()


def spawn_test_cubes(num_envs: int) -> None:
    """Spawn validation cubes at concrete paths."""
    cube_spawn_cfg = sim_utils.CuboidCfg(
        size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=not args_cli.disable_contact_test,
            kinematic_enabled=not args_cli.disable_contact_test,
            max_depenetration_velocity=MAX_DEPENETRATION_VELOCITY,
        ),
        mass_props=sim_utils.MassPropertiesCfg(mass=CUBE_MASS),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True, contact_offset=0.003, rest_offset=0.0
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,
            dynamic_friction=0.6,
            compliant_contact_stiffness=CONTACT_STIFFNESS,
            compliant_contact_damping=CONTACT_DAMPING,
        ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.55, 0.85)),
    )
    for env_id in range(num_envs):
        cube_spawn_cfg.func(
            CUBE_PRIM_TEMPLATE.format(env_id=env_id),
            cube_spawn_cfg,
            translation=CUBE_POSITION,
        )


def create_test_cubes() -> RigidObject:
    cube_cfg = RigidObjectCfg(
        prim_path=CUBE_PRIM_PATH,
        spawn=None,
        init_state=RigidObjectCfg.InitialStateCfg(pos=CUBE_POSITION),
    )
    return RigidObject(cfg=cube_cfg)


def resolve_ft_body_id(robot: Articulation, body_name: str) -> int:
    """Resolve the Robotiq base body index inside the Isaac Lab Articulation."""
    body_ids, body_names = robot.find_bodies(re.escape(body_name), preserve_order=True)
    if len(body_ids) != 1:
        raise RuntimeError(
            f"Expected exactly one F/T body named '{body_name}', got {body_names}.\n"
            f"Available body names: {robot.body_names}"
        )
    return body_ids[0]


def get_virtual_ft_wrench_b(robot: Articulation, ft_body_id: int) -> torch.Tensor:
    """Return virtual F/T wrench as a tensor with shape ``(num_envs, 6)``.

    The data source is Isaac Lab's ``ArticulationData.body_incoming_joint_wrench_b``,
    which is backed by PhysX ``get_link_incoming_joint_force()``.  No NumPy path
    or post-hoc tensor conversion is used.

    PhysX reports the incoming joint wrench applied from the parent link to the
    child link.  The sign is negated here to preserve the convention used by the
    original standalone script: load transmitted into the gripper base.
    """
    # 반환 시점부터 (num_envs, 6)의 시뮬레이션 장치 텐서이며 CPU/NumPy 변환이 없다.
    return -robot.data.body_incoming_joint_wrench_b[:, ft_body_id, :]


def arrow_line_segments(
    origins_w: torch.Tensor,
    vectors_w: torch.Tensor,
    lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build shaft and arrowhead line segments for a batch of vectors."""
    vector_norm = torch.linalg.norm(vectors_w, dim=-1, keepdim=True)
    directions = vectors_w / torch.clamp(vector_norm, min=1.0e-6)
    tips = origins_w + directions * lengths.unsqueeze(-1)

    reference = torch.zeros_like(directions)
    reference[:, 2] = 1.0
    parallel_to_z = torch.abs(directions[:, 2]) > 0.9
    reference[parallel_to_z, 1] = 1.0
    reference[parallel_to_z, 2] = 0.0
    side = torch.linalg.cross(directions, reference, dim=-1)
    side = side / torch.clamp(torch.linalg.norm(side, dim=-1, keepdim=True), min=1.0e-6)

    head_length = torch.clamp(lengths * 0.30, max=0.025)
    head_width = head_length * 0.45
    head_base = tips - directions * head_length.unsqueeze(-1)
    left = head_base + side * head_width.unsqueeze(-1)
    right = head_base - side * head_width.unsqueeze(-1)

    starts = torch.cat((origins_w, tips, tips), dim=0)
    ends = torch.cat((tips, left, right), dim=0)
    return starts, ends


def visualize_ft_wrench(
    draw_interface,
    robot: Articulation,
    ft_body_id: int,
    wrench_b: torch.Tensor,
) -> None:
    """Visualize body-frame wrench tensors as bounded world-frame arrows."""
    # 센서 body 좌표계의 wrench를 viewport 표시를 위해서만 world 좌표계로 회전한다.
    sensor_pos_w = robot.data.body_pos_w[:, ft_body_id, :]
    sensor_quat_w = robot.data.body_quat_w[:, ft_body_id, :]
    force_w = math_utils.quat_apply(sensor_quat_w, wrench_b[:, :3])
    torque_w = math_utils.quat_apply(sensor_quat_w, wrench_b[:, 3:])

    force_norm = torch.linalg.norm(force_w, dim=-1)
    torque_norm = torch.linalg.norm(torque_w, dim=-1)
    force_length = torch.clamp(
        force_norm * FORCE_ARROW_METRES_PER_N, max=FORCE_ARROW_MAX_LENGTH
    )
    torque_length = torch.clamp(
        torque_norm * TORQUE_ARROW_METRES_PER_NM, max=TORQUE_ARROW_MAX_LENGTH
    )

    force_length = torch.where(
        force_norm > 0.1, torch.clamp(force_length, min=0.01), 0.0
    )
    torque_length = torch.where(
        torque_norm > 0.01, torch.clamp(torque_length, min=0.01), 0.0
    )

    local_z = torch.zeros_like(sensor_pos_w)
    local_z[:, 2] = TORQUE_ARROW_OFFSET
    torque_pos_w = sensor_pos_w + math_utils.quat_apply(sensor_quat_w, local_z)

    force_starts, force_ends = arrow_line_segments(sensor_pos_w, force_w, force_length)
    torque_starts, torque_ends = arrow_line_segments(
        torque_pos_w, torque_w, torque_length
    )
    # debug_draw가 Python list를 요구하므로 시각화 경로에서만 CPU로 복사한다.
    # RL 관측에 사용하는 ft_wrench_b 텐서에는 영향을 주지 않는다.
    starts = torch.cat((force_starts, torque_starts), dim=0).detach().cpu().tolist()
    ends = torch.cat((force_ends, torque_ends), dim=0).detach().cpu().tolist()

    num_force_lines = force_starts.shape[0]
    num_torque_lines = torque_starts.shape[0]
    colors = [[0.95, 0.05, 0.05, 1.0]] * num_force_lines + [
        [0.10, 0.35, 1.00, 1.0]
    ] * num_torque_lines
    thicknesses = [4.0] * (num_force_lines + num_torque_lines)
    draw_interface.clear_lines()
    draw_interface.draw_lines(starts, ends, colors, thicknesses)


def contact_test_phase(step: int) -> str:
    cycle_steps = CONTACT_APPROACH_STEPS + CONTACT_HOLD_STEPS + CONTACT_RELEASE_STEPS
    phase_step = step % cycle_steps
    if phase_step < CONTACT_APPROACH_STEPS:
        return "approach"
    if phase_step < CONTACT_APPROACH_STEPS + CONTACT_HOLD_STEPS:
        return "hold"
    return "release"


def minimum_jerk(alpha: float) -> float:
    """Fifth-order interpolation with zero velocity and acceleration at both ends."""
    alpha = min(max(alpha, 0.0), 1.0)
    return alpha**3 * (10.0 - 15.0 * alpha + 6.0 * alpha**2)


def nominal_contact_distance(step: int) -> float:
    """Return the open-loop cube-to-probe distance for the current validation phase."""
    cycle_steps = CONTACT_APPROACH_STEPS + CONTACT_HOLD_STEPS + CONTACT_RELEASE_STEPS
    phase_step = step % cycle_steps
    if phase_step < CONTACT_APPROACH_STEPS:
        alpha = minimum_jerk(phase_step / max(1, CONTACT_APPROACH_STEPS - 1))
        return CUBE_FAR_DISTANCE + alpha * (CUBE_CONTACT_DISTANCE - CUBE_FAR_DISTANCE)
    if phase_step < CONTACT_APPROACH_STEPS + CONTACT_HOLD_STEPS:
        return CUBE_CONTACT_DISTANCE

    release_step = phase_step - CONTACT_APPROACH_STEPS - CONTACT_HOLD_STEPS
    alpha = minimum_jerk(release_step / max(1, CONTACT_RELEASE_STEPS - 1))
    return CUBE_CONTACT_DISTANCE + alpha * (CUBE_FAR_DISTANCE - CUBE_CONTACT_DISTANCE)


def nominal_contact_twist_angle(step: int) -> float:
    """Return a smooth zero-to-zero twist command during the hold phase."""
    cycle_steps = CONTACT_APPROACH_STEPS + CONTACT_HOLD_STEPS + CONTACT_RELEASE_STEPS
    phase_step = step % cycle_steps
    hold_step = phase_step - CONTACT_APPROACH_STEPS
    if hold_step < 0 or hold_step >= CONTACT_HOLD_STEPS:
        return 0.0
    alpha = minimum_jerk(hold_step / max(1, CONTACT_HOLD_STEPS - 1))
    return math.radians(args_cli.contact_twist_deg) * math.sin(2.0 * math.pi * alpha)


def sample_contact_commands(
    num_envs: int,
    device: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, list[str], torch.Tensor, torch.Tensor, list[str]]:
    """Sample balanced sensor-frame approach axes and independent twist signs."""
    # 환경 수가 3 이상이면 X/Y/Z가 가능한 한 균등하게 포함되도록 먼저 축을 섞는다.
    axis_indices = []
    while len(axis_indices) < num_envs:
        axis_indices.extend(torch.randperm(3, generator=generator).tolist())
    axis_indices_tensor = torch.tensor(axis_indices[:num_envs], dtype=torch.long)
    direction_signs = torch.where(
        torch.randint(0, 2, (num_envs,), generator=generator) == 0,
        -torch.ones(num_envs),
        torch.ones(num_envs),
    )
    twist_signs = torch.where(
        torch.randint(0, 2, (num_envs,), generator=generator) == 0,
        -torch.ones(num_envs),
        torch.ones(num_envs),
    )
    # 회전축은 접촉축과 다른 축을 골라 접선 방향 토크가 발생하도록 한다.
    tangent_offsets = torch.randint(1, 3, (num_envs,), generator=generator)
    tangent_axis_indices = (axis_indices_tensor + tangent_offsets) % 3

    directions_b = torch.zeros((num_envs, 3), dtype=torch.float32)
    directions_b[torch.arange(num_envs), axis_indices_tensor] = direction_signs
    twist_axes_b = torch.zeros_like(directions_b)
    twist_axes_b[torch.arange(num_envs), tangent_axis_indices] = 1.0
    axis_names = ("X", "Y", "Z")
    labels = [
        f"{'+' if direction_signs[index] > 0 else '-'}{axis_names[axis_indices[index]]}"
        for index in range(num_envs)
    ]
    twist_axis_labels = [
        axis_names[tangent_axis_indices[index]] for index in range(num_envs)
    ]
    return (
        directions_b.to(device),
        labels,
        twist_signs.to(device),
        twist_axes_b.to(device),
        twist_axis_labels,
    )


def drive_contact_test_cube(
    cube: RigidObject,
    robot: Articulation,
    ft_body_id: int,
    probe_body_id: int,
    step: int,
    sim_dt: float,
    current_distance: torch.Tensor,
    force_norm: torch.Tensor,
    torque_norm: torch.Tensor,
    direction_b: torch.Tensor,
    twist_sign: torch.Tensor,
    twist_axis_b: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move the cube smoothly and back off automatically when the measured force is excessive."""
    # 직전 step의 측정값이 한계를 넘으면 회전을 풀고 큐브를 즉시 후퇴시킨다.
    unsafe = (force_norm >= args_cli.contact_force_limit) | (
        torque_norm >= args_cli.contact_torque_limit
    )
    twist_angle = (
        torch.full_like(current_distance, nominal_contact_twist_angle(step))
        * twist_sign
    )
    twist_angle = torch.where(unsafe, torch.zeros_like(twist_angle), twist_angle)

    # 큐브를 기울이면 접촉 법선 방향의 지지 길이가 증가한다. 이 증가분을
    # 일부 보상해 목표 침투량 0.5 mm를 유지하고 과도한 접촉력을 방지한다.
    cube_half_extent = 0.5 * CUBE_SIZE
    support_increase = (
        CUBE_TILT_SUPPORT_COMPENSATION
        * cube_half_extent
        * (torch.abs(torch.cos(twist_angle)) + torch.abs(torch.sin(twist_angle)) - 1.0)
    )
    desired_distance = (
        torch.full_like(current_distance, nominal_contact_distance(step))
        + support_increase
    )
    desired_distance = torch.where(
        unsafe, torch.full_like(desired_distance, CUBE_FAR_DISTANCE), desired_distance
    )

    distance_delta = desired_distance - current_distance
    distance_delta = torch.clamp(
        distance_delta,
        min=-CONTACT_MAX_APPROACH_SPEED * sim_dt,
        max=CONTACT_MAX_BACKOFF_SPEED * sim_dt,
    )
    next_distance = current_distance + distance_delta

    probe_pos_w = robot.data.body_pos_w[:, probe_body_id, :]
    sensor_quat_w = robot.data.body_quat_w[:, ft_body_id, :]
    direction_w = math_utils.quat_apply(sensor_quat_w, direction_b)
    twist_axis_w = math_utils.quat_apply(sensor_quat_w, twist_axis_b)

    cube_pos_w = probe_pos_w + direction_w * next_distance.unsqueeze(-1)
    twist_quat_w = math_utils.quat_from_angle_axis(twist_angle, twist_axis_w)
    cube_quat_w = math_utils.quat_mul(twist_quat_w, sensor_quat_w)
    cube_pose_w = torch.cat((cube_pos_w, cube_quat_w), dim=-1)

    # PhysX는 연속 pose target에서 kinematic 속도를 계산한다. 따라서 큐브에는
    # 속도를 직접 쓰지 않고 pose만 기록한다.
    cube.write_root_pose_to_sim(cube_pose_w)
    return next_distance, twist_angle


def print_tensor_summary(
    step: int,
    wrench_b: torch.Tensor,
    *,
    phase: str | None = None,
    contact_axis: str | None = None,
    twist_angle: torch.Tensor | None = None,
    max_force_norm: torch.Tensor | None = None,
    max_torque_norm: torch.Tensor | None = None,
) -> None:
    env0 = wrench_b[0]
    force_norm = torch.linalg.norm(env0[:3])
    torque_norm = torch.linalg.norm(env0[3:])
    values = env0.detach().cpu().tolist()
    phase_text = f" phase={phase}" if phase is not None else ""
    command_text = f" axis={contact_axis}" if contact_axis is not None else ""
    if twist_angle is not None:
        command_text += (
            f" twist={math.degrees(float(twist_angle[0].detach().cpu())):+.2f}deg"
        )
    max_force_text = ""
    if max_force_norm is not None:
        max_force_text = f" max|F|={float(max_force_norm[0].detach().cpu()):.4f}"
    max_torque_text = ""
    if max_torque_norm is not None:
        max_torque_text = f" max|T|={float(max_torque_norm[0].detach().cpu()):.4f}"
    print(
        f"[step {step:07d}]{phase_text}{command_text} tensor={tuple(wrench_b.shape)} device={wrench_b.device} "
        f"F=({values[0]: .4f}, {values[1]: .4f}, {values[2]: .4f}) N |F|={force_norm.item():.4f}{max_force_text} "
        f"T=({values[3]: .4f}, {values[4]: .4f}, {values[5]: .4f}) N*m "
        f"|T|={torque_norm.item():.4f}{max_torque_text}"
    )


def validate_parallel_env_body_layout(
    robot: Articulation, env_origins: torch.Tensor
) -> None:
    """Ensure every environment has the same body layout relative to its origin."""
    if robot.num_instances <= 1:
        return

    origins = env_origins.to(device=robot.device).unsqueeze(1)
    body_pos_relative = robot.data.body_pos_w - origins
    position_error = torch.abs(body_pos_relative - body_pos_relative[0:1])
    body_error = torch.amax(position_error, dim=(0, 2))
    worst_body_id = int(torch.argmax(body_error).item())
    max_error = float(body_error[worst_body_id].item())
    if max_error > 1.0e-4:
        raise RuntimeError(
            "Parallel environment body layout mismatch after the first physics step: "
            f"body='{robot.body_names[worst_body_id]}', max_position_error={max_error:.6e} m"
        )
    print(f"[INFO]: Parallel environment body-layout error: max={max_error:.3e} m")


def print_runtime_body_positions(
    robot: Articulation, env_origin: torch.Tensor, step: int
) -> None:
    """Print env-0 articulation body positions relative to its environment origin."""
    positions = (
        (robot.data.body_pos_w[0] - env_origin.to(robot.device)).detach().cpu().tolist()
    )
    orientations = robot.data.body_quat_w[0].detach().cpu().tolist()
    print(f"[DEBUG]: Runtime body positions at step {step}:")
    for body_name, position, orientation in zip(
        robot.body_names, positions, orientations
    ):
        print(
            f"  {body_name:<36} pos=({position[0]: .5f}, {position[1]: .5f}, {position[2]: .5f}) "
            f"quat=({orientation[0]: .5f}, {orientation[1]: .5f}, {orientation[2]: .5f}, {orientation[3]: .5f})"
        )
    joint_positions = robot.data.joint_pos[0].detach().cpu().tolist()
    print("[DEBUG]: Runtime joint positions:")
    for joint_name, joint_position in zip(robot.joint_names, joint_positions):
        print(f"  {joint_name:<36} q={joint_position: .6f}")


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be at least 1.")
    if args_cli.print_every < 1:
        raise ValueError("--print_every must be at least 1.")
    if args_cli.vis_every < 1:
        raise ValueError("--vis_every must be at least 1.")
    if args_cli.contact_force_limit <= 0.0:
        raise ValueError("--contact_force_limit must be positive.")
    if args_cli.contact_torque_limit <= 0.0:
        raise ValueError("--contact_torque_limit must be positive.")
    if args_cli.contact_twist_deg < 0.0:
        raise ValueError("--contact_twist_deg must be non-negative.")

    check_nucleus_asset(UR5E_USD_PATH, "UR5e USD")
    check_nucleus_asset(ROBOTIQ_2F85_USD_PATH, "Robotiq 2F-85 USD")

    sim_cfg = sim_utils.SimulationCfg(dt=PHYSICS_DT, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[1.6, 1.5, 1.1], target=[0.45, 0.0, 0.25])
    stage = sim.stage

    env_origins = define_env_roots(stage, args_cli.num_envs, args_cli.env_spacing)

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    spawn_robots(args_cli.num_envs)
    spawn_grippers(args_cli.num_envs)
    spawn_test_cubes(args_cli.num_envs)

    ft_body_name = None
    for env_id in range(args_cli.num_envs):
        body_name = assemble_env(stage, env_id)
        if ft_body_name is None:
            ft_body_name = body_name
        elif ft_body_name != body_name:
            raise RuntimeError(
                f"F/T body name mismatch: env0={ft_body_name}, env{env_id}={body_name}"
            )
    if ft_body_name is None:
        raise RuntimeError("F/T body was not resolved.")

    robot = create_robot()
    cube = create_test_cubes()
    ft_draw_interface = None
    if not args_cli.headless and not args_cli.disable_ft_visualization:
        if omni_debug_draw is None:
            raise RuntimeError(
                "F/T visualization was requested but the Isaac Sim debug-draw extension is unavailable."
            )
        ft_draw_interface = omni_debug_draw.acquire_debug_draw_interface()

    sim.reset()

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    robot.set_joint_position_target(joint_pos)
    robot.write_data_to_sim()
    robot.reset()
    cube.reset()

    ft_body_id = resolve_ft_body_id(robot, ft_body_name)
    probe_body_id = None
    if not args_cli.disable_contact_test:
        probe_body_id = resolve_ft_body_id(robot, PROBE_BODY_NAME)
    robot.update(0.0)
    cube.update(0.0)

    print("[INFO]: Setup complete.")
    print(f"[INFO]: env_origins tensor shape={tuple(env_origins.shape)}")
    print(
        f"[INFO]: robot body count={robot.num_bodies}, joint count={robot.num_joints}"
    )
    print(f"[INFO]: F/T body name='{ft_body_name}', body_id={ft_body_id}")
    if probe_body_id is not None:
        print(
            f"[INFO]: Contact-test probe body='{PROBE_BODY_NAME}', body_id={probe_body_id}"
        )
        print(
            f"[INFO]: Contact test: penetration={CUBE_CONTACT_PENETRATION * 1e3:.2f} mm, "
            f"compliance=({CONTACT_STIFFNESS:.0f} N/m, {CONTACT_DAMPING:.0f} N*s/m), "
            f"backoff thresholds=({args_cli.contact_force_limit:.1f} N, {args_cli.contact_torque_limit:.1f} N*m), "
            f"twist={args_cli.contact_twist_deg:.1f} deg, seed={args_cli.contact_seed}"
        )
    print("[INFO]: F/T tensor order: [Fx, Fy, Fz, Tx, Ty, Tz]")
    if ft_draw_interface is not None:
        print(
            "[INFO]: F/T viewport arrows: force=red, torque=blue (arrow lengths are capped)."
        )

    sim_dt = sim.get_physics_dt()
    cycle_steps = CONTACT_APPROACH_STEPS + CONTACT_HOLD_STEPS + CONTACT_RELEASE_STEPS
    cube_distance = torch.full(
        (args_cli.num_envs,),
        CUBE_FAR_DISTANCE,
        dtype=torch.float32,
        device=robot.device,
    )
    measured_force_norm = torch.zeros(
        args_cli.num_envs, dtype=torch.float32, device=robot.device
    )
    measured_torque_norm = torch.zeros_like(measured_force_norm)
    max_force_norm = torch.zeros_like(measured_force_norm)
    max_torque_norm = torch.zeros_like(measured_force_norm)
    current_twist_angle = torch.zeros_like(measured_force_norm)
    contact_direction_b = torch.zeros(
        (args_cli.num_envs, 3), dtype=torch.float32, device=robot.device
    )
    contact_direction_b[:, 0] = 1.0
    contact_twist_sign = torch.ones_like(measured_force_norm)
    contact_twist_axis_b = torch.zeros_like(contact_direction_b)
    contact_twist_axis_b[:, 1] = 1.0
    contact_twist_axis_labels = ["Y"] * args_cli.num_envs
    contact_axis_labels = ["+X"] * args_cli.num_envs
    contact_generator = torch.Generator(device="cpu")
    contact_generator.manual_seed(args_cli.contact_seed)
    step = 0
    while simulation_app.is_running():
        if probe_body_id is not None and step % cycle_steps == 0:
            max_force_norm.zero_()
            max_torque_norm.zero_()
            (
                contact_direction_b,
                contact_axis_labels,
                contact_twist_sign,
                contact_twist_axis_b,
                contact_twist_axis_labels,
            ) = sample_contact_commands(
                args_cli.num_envs, robot.device, contact_generator
            )
            twist_sign_values = contact_twist_sign.detach().cpu().tolist()
            command_count = min(args_cli.num_envs, 8)
            command_summary = ", ".join(
                f"env_{env_id}={contact_axis_labels[env_id]}/rot{contact_twist_axis_labels[env_id]}"
                f"{'+' if twist_sign_values[env_id] > 0 else '-'}"
                for env_id in range(command_count)
            )
            print(f"[INFO]: Contact cycle {step // cycle_steps}: {command_summary}")

        robot.set_joint_position_target(robot.data.default_joint_pos)
        if probe_body_id is not None:
            cube_distance, current_twist_angle = drive_contact_test_cube(
                cube,
                robot,
                ft_body_id,
                probe_body_id,
                step,
                sim_dt,
                cube_distance,
                measured_force_norm,
                measured_torque_norm,
                contact_direction_b,
                contact_twist_sign,
                contact_twist_axis_b,
            )
        robot.write_data_to_sim()
        cube.write_data_to_sim()

        sim.step()
        robot.update(sim_dt)
        cube.update(sim_dt)
        step += 1

        if step == 1:
            validate_parallel_env_body_layout(robot, env_origins)
        if args_cli.debug_gripper_topology and step in (1, 60):
            print_runtime_body_positions(robot, env_origins[0], step)

        # RL 관측으로 바로 사용할 수 있는 (num_envs, 6) 장치 텐서를 매 step 읽는다.
        ft_wrench_b = get_virtual_ft_wrench_b(robot, ft_body_id)
        if not bool(torch.isfinite(ft_wrench_b).all()):
            raise RuntimeError(f"Non-finite F/T wrench detected at step {step}.")
        measured_force_norm = torch.linalg.norm(ft_wrench_b[:, :3], dim=-1)
        measured_torque_norm = torch.linalg.norm(ft_wrench_b[:, 3:], dim=-1)
        if bool(
            torch.any(measured_force_norm > 10.0 * args_cli.contact_force_limit)
            or torch.any(measured_torque_norm > 10.0 * args_cli.contact_torque_limit)
        ):
            raise RuntimeError(
                f"Contact-test safety envelope exceeded at step {step}: "
                f"max|F|={float(torch.max(measured_force_norm)):.3f} N, "
                f"max|T|={float(torch.max(measured_torque_norm)):.3f} N*m"
            )
        max_force_norm = torch.maximum(max_force_norm, measured_force_norm)
        max_torque_norm = torch.maximum(max_torque_norm, measured_torque_norm)
        if ft_draw_interface is not None and step % args_cli.vis_every == 0:
            visualize_ft_wrench(ft_draw_interface, robot, ft_body_id, ft_wrench_b)
        if step % args_cli.print_every == 0:
            phase = contact_test_phase(step - 1) if probe_body_id is not None else None
            contact_axis = contact_axis_labels[0] if probe_body_id is not None else None
            print_tensor_summary(
                step,
                ft_wrench_b,
                phase=phase,
                contact_axis=contact_axis,
                twist_angle=current_twist_angle if probe_body_id is not None else None,
                max_force_norm=max_force_norm,
                max_torque_norm=max_torque_norm,
            )

        if args_cli.steps > 0 and step >= args_cli.steps:
            break


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
