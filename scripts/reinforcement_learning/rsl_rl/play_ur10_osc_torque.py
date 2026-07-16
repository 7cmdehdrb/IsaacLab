# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for trained UR10 torque-level OSC Reach policies."""

from __future__ import annotations

import argparse
import importlib.util
import re
import runpy
import sys
from pathlib import Path


DEFAULT_TASK = "Isaac-Reach-UR10-OSC-Torque-Play-v0"
DEFAULT_EXPERIMENT = "reach_ur10_osc_torque"


def _add_source_paths_if_needed(isaaclab_root: Path) -> None:
    if importlib.util.find_spec("isaaclab") is not None:
        return

    for source_dir in (
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_mimic",
        isaaclab_root / "source" / "isaaclab_rl",
        isaaclab_root / "source" / "isaaclab_tasks",
    ):
        sys.path.insert(0, str(source_dir))


def _checkpoint_sort_key(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _latest_run_dir(experiment_dir: Path) -> Path:
    run_dirs = [path for path in experiment_dir.glob("*") if path.is_dir()]
    if not run_dirs:
        raise FileNotFoundError(f"No training run directory found in: {experiment_dir}")
    return max(run_dirs, key=lambda path: path.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")
    return checkpoints[-1]


def main() -> None:
    script_path = Path(__file__).resolve()
    isaaclab_root = script_path.parents[3]
    default_experiment_dir = isaaclab_root / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT

    parser = argparse.ArgumentParser(
        description="Run a trained UR10 torque-level OSC Reach RSL-RL policy in Isaac Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task ID to play.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to the latest model_*.pt in --run-dir or latest run.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
    args, play_args = parser.parse_known_args()

    run_dir = args.run_dir.expanduser().resolve() if args.run_dir else _latest_run_dir(default_experiment_dir)
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else _latest_checkpoint(run_dir)

    _add_source_paths_if_needed(isaaclab_root)

    play_script = script_path.with_name("play.py")
    sys.path.insert(0, str(play_script.parent))
    sys.argv = [
        str(play_script),
        "--task",
        args.task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        str(checkpoint),
        *play_args,
    ]
    runpy.run_path(str(play_script), run_name="__main__")


if __name__ == "__main__":
    main()
