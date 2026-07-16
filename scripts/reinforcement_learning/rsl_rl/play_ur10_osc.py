# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convenience launcher for the trained UR10 OSC reach policy.

This script delegates to Isaac Lab's standard RSL-RL ``play.py`` while filling in
the task, run directory, and checkpoint for the local UR10 OSC training run.
Any extra arguments are forwarded to ``play.py``.
"""

from __future__ import annotations

import argparse
import re
import runpy
import sys
from pathlib import Path


DEFAULT_TASK = "Isaac-Reach-UR10-OSC-Play-v0"
LEGACY_TASK = "Isaac-Reach-UR10-OSC-Legacy-Play-v0"
DEFAULT_EXPERIMENT = "reach_ur10_osc"
DEFAULT_RUN = "2026-07-11_01-54-00"
LEGACY_RUNS = {"2026-07-11_01-54-00"}


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
        description="Run the trained UR10 OSC reach RSL-RL policy in Isaac Lab.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--task", default=None, help="Task ID to play.")
    parser.add_argument("--run-dir", type=Path, default=default_run_dir, help="Training run directory.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Checkpoint path. Defaults to the latest model_*.pt in --run-dir.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of environments to visualize.",
    )
    args, play_args = parser.parse_known_args()

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else _latest_checkpoint(run_dir)
    task = args.task
    if task is None:
        task = LEGACY_TASK if run_dir.name in LEGACY_RUNS else DEFAULT_TASK

    play_script = script_path.with_name("play.py")
    for source_dir in (
        isaaclab_root / "source" / "isaaclab",
        isaaclab_root / "source" / "isaaclab_assets",
        isaaclab_root / "source" / "isaaclab_mimic",
        isaaclab_root / "source" / "isaaclab_rl",
        isaaclab_root / "source" / "isaaclab_tasks",
    ):
        sys.path.insert(0, str(source_dir))
    sys.path.insert(0, str(play_script.parent))
    sys.argv = [
        str(play_script),
        "--task",
        task,
        "--num_envs",
        str(args.num_envs),
        "--checkpoint",
        str(checkpoint),
        *play_args,
    ]
    runpy.run_path(str(play_script), run_name="__main__")


if __name__ == "__main__":
    main()
