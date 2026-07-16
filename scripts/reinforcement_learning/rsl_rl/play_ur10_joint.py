# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for trained UR10 joint-space reach policies."""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path


DEFAULT_TASK = "Isaac-Reach-UR10-Play-v0"
DEFAULT_EXPERIMENT = "reach_ur10"
DEFAULT_RUN = "2026-07-11_16-28-05"


def _checkpoint_sort_key(path: Path) -> int:
    match = re.fullmatch(r"model_(\d+)\.pt", path.name)
    return int(match.group(1)) if match else -1


def _latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=_checkpoint_sort_key)
    if not checkpoints:
        raise FileNotFoundError(f"No model_*.pt checkpoint found in: {run_dir}")
    return checkpoints[-1]


def main() -> None:
    script_path = Path(__file__).resolve()
    isaaclab_root = script_path.parents[3]
    default_run_dir = isaaclab_root / "logs" / "rsl_rl" / DEFAULT_EXPERIMENT / DEFAULT_RUN

    parser = argparse.ArgumentParser(
        description="Run a trained UR10 joint-space reach RSL-RL policy in Isaac Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="Task ID to play.")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to the latest model_*.pt in --run-dir.",
    )
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
    args, play_args = parser.parse_known_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else _latest_checkpoint(run_dir)

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

