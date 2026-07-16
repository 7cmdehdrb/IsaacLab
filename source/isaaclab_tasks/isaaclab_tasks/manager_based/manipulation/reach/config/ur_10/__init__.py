# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Reach-UR10-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10ReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reach-UR10-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:UR10ReachEnvCfg_PLAY",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UR10ReachPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Reach-UR10-Cartesian-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartesian_reach_env_cfg:UR10CartesianReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.cartesian_reach_env_cfg:UR10CartesianReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-Cartesian-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.cartesian_reach_env_cfg:UR10CartesianReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{__name__}.cartesian_reach_env_cfg:UR10CartesianReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-OSC-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_osc_reach_env_cfg:UR10OSCReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_osc_cfg:UR10OSCReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-OSC-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_osc_reach_env_cfg:UR10OSCReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_osc_cfg:UR10OSCReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-OSC-Legacy-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_osc_reach_legacy_env_cfg:UR10OSCLegacyReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_osc_cfg:UR10OSCReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-OSC-Legacy-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_osc_reach_legacy_env_cfg:UR10OSCLegacyReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_osc_cfg:UR10OSCReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-OSC-Torque-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_osc_torque_reach_env_cfg:UR10OSCTorqueReachEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_osc_torque_cfg:UR10OSCTorqueReachPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Reach-UR10-OSC-Torque-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ur10_osc_torque_reach_env_cfg:UR10OSCTorqueReachEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_osc_torque_cfg:UR10OSCTorqueReachPPORunnerCfg",
    },
)
