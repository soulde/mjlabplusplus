"""RL configuration for Unitree Go2 velocity tasks."""

from mjlabplusplus.tasks.velocity.quadruped_rl_cfgs import (
    quadruped_fastsac_runner_cfg,
    quadruped_ppo_runner_cfg,
    quadruped_tdmpc2_runner_cfg,
)

UNITREE_GO2_FLAT_TASK = "Mjlab-Velocity-Flat-Unitree-Go2"
UNITREE_GO2_ROUGH_TASK = "Mjlab-Velocity-Rough-Unitree-Go2"


def unitree_go2_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_go2_rough", 3_000)


def unitree_go2_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_go2_flat", 1_000)


def unitree_go2_rough_fastsac_runner_cfg():
    return quadruped_fastsac_runner_cfg(UNITREE_GO2_ROUGH_TASK, "unitree_go2_rough")


def unitree_go2_flat_fastsac_runner_cfg():
    return quadruped_fastsac_runner_cfg(UNITREE_GO2_FLAT_TASK, "unitree_go2_flat")


def unitree_go2_rough_tdmpc2_runner_cfg():
    return quadruped_tdmpc2_runner_cfg(UNITREE_GO2_ROUGH_TASK, "unitree_go2_rough")


def unitree_go2_flat_tdmpc2_runner_cfg():
    return quadruped_tdmpc2_runner_cfg(UNITREE_GO2_FLAT_TASK, "unitree_go2_flat")
