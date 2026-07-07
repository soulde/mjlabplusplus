"""Unitree A1 velocity environment configurations."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlabplusplus.robots import (
    UNITREE_A1_ACTION_SCALE,
    UNITREE_A1_FOOT_BODY_NAMES,
    UNITREE_A1_FOOT_GEOM_NAMES,
    UNITREE_A1_FOOT_SITE_NAMES,
    UNITREE_A1_JOINT_NAMES,
    get_unitree_a1_robot_cfg,
)
from mjlabplusplus.tasks.velocity.quadruped_env_cfgs import (
    QuadrupedVelocityRobotCfg,
    quadruped_flat_env_cfg,
    quadruped_rough_env_cfg,
)

UNITREE_A1_VELOCITY_CFG = QuadrupedVelocityRobotCfg(
    robot_cfg_fn=get_unitree_a1_robot_cfg,
    base_body_name="base",
    joint_names=UNITREE_A1_JOINT_NAMES,
    foot_site_names=UNITREE_A1_FOOT_SITE_NAMES,
    foot_body_names=UNITREE_A1_FOOT_BODY_NAMES,
    foot_geom_names=UNITREE_A1_FOOT_GEOM_NAMES,
    action_scale=UNITREE_A1_ACTION_SCALE,
    base_height=0.35,
    feet_height_body_weight=-5.0,
    feet_height_body_target=-0.2,
    feet_air_time_weight=0.0,
    feet_air_time_variance_weight=0.0,
    feet_slide_weight=0.0,
    feet_gait_weight=0.0,
    gait_pairs=(("FL_foot", "RR_foot"), ("FR_foot", "RL_foot")),
    mirror_joints=[
        ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
        ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
    ],
)


def unitree_a1_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree A1 rough terrain velocity configuration."""
    return quadruped_rough_env_cfg(UNITREE_A1_VELOCITY_CFG, play=play)


def unitree_a1_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Unitree A1 flat terrain velocity configuration."""
    return quadruped_flat_env_cfg(UNITREE_A1_VELOCITY_CFG, play=play)
