"""RL configuration for Unitree A1 velocity tasks."""

from mjlabplusplus.tasks.velocity.quadruped_rl_cfgs import (
    quadruped_fastsac_runner_cfg,
    quadruped_ppo_runner_cfg,
    quadruped_tdmpc2_runner_cfg,
)

UNITREE_A1_FLAT_TASK = "Mjlab-Velocity-Flat-Unitree-A1"
UNITREE_A1_ROUGH_TASK = "Mjlab-Velocity-Rough-Unitree-A1"


def unitree_a1_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_a1_rough", 3_000)


def unitree_a1_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("unitree_a1_flat", 1_000)


def unitree_a1_rough_fastsac_runner_cfg():
    return quadruped_fastsac_runner_cfg(UNITREE_A1_ROUGH_TASK, "unitree_a1_rough")


def unitree_a1_flat_fastsac_runner_cfg():
    return quadruped_fastsac_runner_cfg(UNITREE_A1_FLAT_TASK, "unitree_a1_flat")


def unitree_a1_rough_tdmpc2_runner_cfg():
    return quadruped_tdmpc2_runner_cfg(UNITREE_A1_ROUGH_TASK, "unitree_a1_rough")


def unitree_a1_flat_tdmpc2_runner_cfg():
    return quadruped_tdmpc2_runner_cfg(UNITREE_A1_FLAT_TASK, "unitree_a1_flat")
