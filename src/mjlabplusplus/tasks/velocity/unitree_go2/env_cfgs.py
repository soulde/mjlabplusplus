"""Unitree Go2 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlabplusplus.robots import (
    UNITREE_GO2_ACTION_SCALE,
    UNITREE_GO2_FOOT_BODY_NAMES,
    UNITREE_GO2_FOOT_GEOM_NAMES,
    UNITREE_GO2_FOOT_SITE_NAMES,
    UNITREE_GO2_JOINT_NAMES,
    get_unitree_go2_robot_cfg,
)
from mjlabplusplus.tasks.velocity.quadruped_env_cfgs import (
    QuadrupedVelocityRobotCfg,
    quadruped_flat_env_cfg,
    quadruped_rough_env_cfg,
)

UNITREE_GO2_VELOCITY_CFG = QuadrupedVelocityRobotCfg(
    robot_cfg_fn=get_unitree_go2_robot_cfg,
    base_body_name="base",
    joint_names=UNITREE_GO2_JOINT_NAMES,
    foot_site_names=UNITREE_GO2_FOOT_SITE_NAMES,
    foot_body_names=UNITREE_GO2_FOOT_BODY_NAMES,
    foot_geom_names=UNITREE_GO2_FOOT_GEOM_NAMES,
    action_scale=UNITREE_GO2_ACTION_SCALE,
    base_height=0.33,
    feet_height_body_weight=-5.0,
    feet_height_body_target=-0.2,
    feet_air_time_weight=0.1,
    feet_air_time_variance_weight=-1.0,
    feet_slide_weight=-0.1,
    feet_gait_weight=0.5,
    gait_pairs=(
        ("FL_foot_collision", "RR_foot_collision"),
        ("FR_foot_collision", "RL_foot_collision"),
    ),
    mirror_joints=[
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ],
)


def unitree_go2_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 rough terrain velocity configuration."""
    return quadruped_rough_env_cfg(UNITREE_GO2_VELOCITY_CFG, play=play)


def unitree_go2_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree Go2 flat terrain velocity configuration."""
    return quadruped_flat_env_cfg(UNITREE_GO2_VELOCITY_CFG, play=play)
