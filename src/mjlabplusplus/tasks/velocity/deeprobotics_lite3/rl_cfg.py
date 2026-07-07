"""RL configuration for DeepRobotics Lite3 velocity tasks."""

from mjlabplusplus.tasks.velocity.quadruped_rl_cfgs import (
    quadruped_fastsac_runner_cfg,
    quadruped_ppo_runner_cfg,
    quadruped_tdmpc2_runner_cfg,
)

DEEPROBOTICS_LITE3_FLAT_TASK = "Mjlab-Velocity-Flat-DeepRobotics-Lite3"
DEEPROBOTICS_LITE3_ROUGH_TASK = "Mjlab-Velocity-Rough-DeepRobotics-Lite3"


def deeprobotics_lite3_rough_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("deeprobotics_lite3_rough", 3_000)


def deeprobotics_lite3_flat_ppo_runner_cfg():
    return quadruped_ppo_runner_cfg("deeprobotics_lite3_flat", 1_000)


def deeprobotics_lite3_rough_fastsac_runner_cfg():
    return quadruped_fastsac_runner_cfg(
        DEEPROBOTICS_LITE3_ROUGH_TASK,
        "deeprobotics_lite3_rough",
    )


def deeprobotics_lite3_flat_fastsac_runner_cfg():
    return quadruped_fastsac_runner_cfg(
        DEEPROBOTICS_LITE3_FLAT_TASK,
        "deeprobotics_lite3_flat",
    )


def deeprobotics_lite3_rough_tdmpc2_runner_cfg():
    return quadruped_tdmpc2_runner_cfg(
        DEEPROBOTICS_LITE3_ROUGH_TASK,
        "deeprobotics_lite3_rough",
    )


def deeprobotics_lite3_flat_tdmpc2_runner_cfg():
    return quadruped_tdmpc2_runner_cfg(
        DEEPROBOTICS_LITE3_FLAT_TASK,
        "deeprobotics_lite3_flat",
    )
