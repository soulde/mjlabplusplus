"""DeepRobotics Lite3 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlabplusplus.robots import (
    DEEPROBOTICS_LITE3_ACTION_SCALE,
    DEEPROBOTICS_LITE3_FOOT_BODY_NAMES,
    DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES,
    DEEPROBOTICS_LITE3_FOOT_SITE_NAMES,
    DEEPROBOTICS_LITE3_JOINT_NAMES,
    get_deeprobotics_lite3_robot_cfg,
)
from mjlabplusplus.tasks.velocity.quadruped_env_cfgs import (
    QuadrupedVelocityRobotCfg,
    quadruped_flat_env_cfg,
    quadruped_rough_env_cfg,
)

DEEPROBOTICS_LITE3_VELOCITY_CFG = QuadrupedVelocityRobotCfg(
    robot_cfg_fn=get_deeprobotics_lite3_robot_cfg,
    base_body_name="TORSO",
    joint_names=DEEPROBOTICS_LITE3_JOINT_NAMES,
    foot_site_names=DEEPROBOTICS_LITE3_FOOT_SITE_NAMES,
    foot_body_names=DEEPROBOTICS_LITE3_FOOT_BODY_NAMES,
    foot_geom_names=DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES,
    action_scale=DEEPROBOTICS_LITE3_ACTION_SCALE,
    base_height=0.35,
    feet_height_body_weight=0.0,
    feet_height_body_target=-0.25,
    feet_air_time_weight=0.0,
    feet_air_time_variance_weight=0.0,
    feet_slide_weight=0.0,
    feet_gait_weight=0.0,
    gait_pairs=(("FL_FOOT", "HR_FOOT"), ("FR_FOOT", "HL_FOOT")),
    mirror_joints=[
        ["FL_(HipX|HipY|Knee).*", "HR_(HipX|HipY|Knee).*"],
        ["FR_(HipX|HipY|Knee).*", "HL_(HipX|HipY|Knee).*"],
    ],
)


def deeprobotics_lite3_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create DeepRobotics Lite3 rough terrain velocity configuration."""
    return quadruped_rough_env_cfg(DEEPROBOTICS_LITE3_VELOCITY_CFG, play=play)


def deeprobotics_lite3_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create DeepRobotics Lite3 flat terrain velocity configuration."""
    return quadruped_flat_env_cfg(DEEPROBOTICS_LITE3_VELOCITY_CFG, play=play)
