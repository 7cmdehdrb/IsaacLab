# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UR10ReachPPORunnerCfg


@configclass
class UR10OSCReachPPORunnerCfg(UR10ReachPPORunnerCfg):
    experiment_name = "reach_ur10_osc"

