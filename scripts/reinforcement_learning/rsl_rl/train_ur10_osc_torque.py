# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for UR10 OSC-style Reach training with joint-torque actions."""

from __future__ import annotations

import argparse
import importlib.util
import runpy
import sys
from pathlib import Path


DEFAULT_TASK = "Isaac-Reach-UR10-OSC-Torque-v0"


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


def main() -> None:
    script_path = Path(__file__).resolve()
    isaaclab_root = script_path.parents[3]
    train_script = script_path.with_name("train.py")

    parser = argparse.ArgumentParser(
        description="Train the UR10 Reach task with direct joint-torque actions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task ID to train.")
    parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
    args, train_args = parser.parse_known_args()

    _add_source_paths_if_needed(isaaclab_root)

    sys.path.insert(0, str(train_script.parent))
    sys.argv = [str(train_script), "--task", args.task]
    if args.num_envs is not None:
        sys.argv += ["--num_envs", str(args.num_envs)]
    sys.argv += train_args
    runpy.run_path(str(train_script), run_name="__main__")


if __name__ == "__main__":
    main()
