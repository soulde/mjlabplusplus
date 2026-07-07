"""Shared quadruped velocity environment configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

from mjlab.entity import EntityCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers import EventTermCfg, RewardTermCfg, TerminationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import (
    ContactMatch,
    ContactSensorCfg,
    ObjRef,
    RayCastSensorCfg,
    RingPatternCfg,
    TerrainHeightSensorCfg,
)
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlabplusplus.tasks.velocity import rewards as plus_rewards


@dataclass(frozen=True)
class QuadrupedVelocityRobotCfg:
    """Robot-specific parameters needed to align robot_lab velocity tasks."""

    robot_cfg_fn: Callable[[], EntityCfg]
    base_body_name: str
    joint_names: tuple[str, ...]
    foot_site_names: tuple[str, ...]
    foot_body_names: tuple[str, ...]
    foot_geom_names: tuple[str, ...]
    action_scale: dict[str, float]
    base_height: float
    feet_height_body_weight: float
    feet_height_body_target: float
    feet_air_time_weight: float
    feet_air_time_variance_weight: float
    feet_slide_weight: float
    feet_gait_weight: float
    gait_pairs: tuple[tuple[str, str], tuple[str, str]]
    mirror_joints: list[list[str]]


def quadruped_rough_env_cfg(
    robot: QuadrupedVelocityRobotCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create rough terrain velocity configuration for a quadruped."""
    cfg = make_velocity_env_cfg()

    cfg.sim.mujoco.ccd_iterations = 500
    cfg.sim.contact_sensor_maxmatch = 500
    cfg.sim.nconmax = 200

    cfg.scene.entities = {"robot": robot.robot_cfg_fn()}
    cfg.scene.extent = 2.0

    for sensor in cfg.scene.sensors or ():
        if sensor.name == "terrain_scan":
            assert isinstance(sensor, RayCastSensorCfg)
            assert isinstance(sensor.frame, ObjRef)
            sensor.frame.name = robot.base_body_name
        if sensor.name == "foot_height_scan":
            assert isinstance(sensor, TerrainHeightSensorCfg)
            sensor.frame = tuple(
                ObjRef(type="site", name=site_name, entity="robot")
                for site_name in robot.foot_site_names
            )
            sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=4)

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=robot.foot_geom_names,
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
        history_length=4,
    )
    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(
            mode="subtree",
            pattern=robot.base_body_name,
            entity="robot",
        ),
        secondary=ContactMatch(
            mode="subtree",
            pattern=robot.base_body_name,
            entity="robot",
        ),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    nonfoot_ground_cfg = ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=".*_collision.*",
            entity="robot",
            exclude=robot.foot_geom_names,
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
    )
    cfg.scene.sensors = (cfg.scene.sensors or ()) + (
        feet_ground_cfg,
        self_collision_cfg,
        nonfoot_ground_cfg,
    )

    if (
        cfg.scene.terrain is not None
        and cfg.scene.terrain.terrain_generator is not None
    ):
        cfg.scene.terrain.terrain_generator.curriculum = True

    actor_terms = cfg.observations["actor"].terms
    actor_terms.pop("base_lin_vel", None)
    actor_terms.pop("height_scan", None)
    actor_terms["base_ang_vel"].func = envs_mdp.base_ang_vel
    actor_terms["base_ang_vel"].params = {"asset_cfg": SceneEntityCfg("robot")}
    actor_terms["base_ang_vel"].scale = 0.25
    actor_terms["projected_gravity"].func = envs_mdp.projected_gravity
    actor_terms["projected_gravity"].params = {"asset_cfg": SceneEntityCfg("robot")}
    actor_terms["joint_pos"].scale = 1.0
    actor_terms["joint_pos"].params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names)
    }
    actor_terms["joint_vel"].scale = 0.05
    actor_terms["joint_vel"].params = {
        "asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names)
    }
    critic_terms = cfg.observations["critic"].terms
    if "base_lin_vel" in critic_terms:
        critic_terms["base_lin_vel"].func = envs_mdp.base_lin_vel
        critic_terms["base_lin_vel"].params = {"asset_cfg": SceneEntityCfg("robot")}
        critic_terms["base_lin_vel"].scale = 2.0
    critic_terms["base_ang_vel"].func = envs_mdp.base_ang_vel
    critic_terms["base_ang_vel"].params = {"asset_cfg": SceneEntityCfg("robot")}
    critic_terms["base_ang_vel"].scale = 0.25
    critic_terms["projected_gravity"].func = envs_mdp.projected_gravity
    critic_terms["projected_gravity"].params = {"asset_cfg": SceneEntityCfg("robot")}
    critic_terms["joint_pos"].params = actor_terms["joint_pos"].params
    critic_terms["joint_vel"].params = actor_terms["joint_vel"].params

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = robot.action_scale
    joint_pos_action.clip = {".*": (-100.0, 100.0)}

    cfg.viewer.body_name = robot.base_body_name
    cfg.viewer.distance = 1.5
    cfg.viewer.elevation = -10.0

    cfg.events["reset_base"].params = {
        "pose_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (0.0, 0.2),
            "roll": (-3.14, 3.14),
            "pitch": (-3.14, 3.14),
            "yaw": (-3.14, 3.14),
        },
        "velocity_range": {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.5, 0.5),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        },
    }
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = robot.foot_geom_names
    cfg.events["base_com"].params["asset_cfg"].body_names = (robot.base_body_name,)

    _configure_robot_lab_rewards(
        cfg=cfg,
        robot=robot,
        feet_ground_cfg=feet_ground_cfg,
        self_collision_cfg=self_collision_cfg,
        nonfoot_ground_cfg=nonfoot_ground_cfg,
    )

    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(70.0)},
    )
    cfg.terminations.pop("illegal_contact", None)
    cfg.curriculum.pop("command_vel", None)

    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

    if play:
        cfg.episode_length_s = int(1e9)
        cfg.observations["actor"].enable_corruption = False
        cfg.events.pop("push_robot", None)
        cfg.terminations.pop("out_of_terrain_bounds", None)
        cfg.curriculum = {}
        cfg.events["randomize_terrain"] = EventTermCfg(
            func=envs_mdp.randomize_terrain,
            mode="reset",
            params={},
        )
        if cfg.scene.terrain is not None:
            if cfg.scene.terrain.terrain_generator is not None:
                cfg.scene.terrain.terrain_generator.curriculum = False
                cfg.scene.terrain.terrain_generator.num_cols = 5
                cfg.scene.terrain.terrain_generator.num_rows = 5
                cfg.scene.terrain.terrain_generator.border_width = 10.0

    return cfg


def quadruped_flat_env_cfg(
    robot: QuadrupedVelocityRobotCfg,
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create flat terrain velocity configuration for a quadruped."""
    cfg = quadruped_rough_env_cfg(robot=robot, play=play)

    cfg.sim.njmax = 300
    cfg.sim.mujoco.ccd_iterations = 50
    cfg.sim.contact_sensor_maxmatch = 128
    cfg.sim.nconmax = None

    assert cfg.scene.terrain is not None
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    cfg.scene.sensors = tuple(
        sensor for sensor in (cfg.scene.sensors or ()) if sensor.name != "terrain_scan"
    )
    cfg.observations["critic"].terms.pop("height_scan", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.curriculum.pop("terrain_levels", None)

    if play:
        cfg.events.pop("randomize_terrain", None)

    return cfg


def _configure_robot_lab_rewards(
    cfg: ManagerBasedRlEnvCfg,
    robot: QuadrupedVelocityRobotCfg,
    feet_ground_cfg: ContactSensorCfg,
    self_collision_cfg: ContactSensorCfg,
    nonfoot_ground_cfg: ContactSensorCfg,
) -> None:
    cfg.rewards.pop("angular_momentum", None)
    cfg.rewards.pop("pose", None)
    cfg.rewards.pop("upright", None)
    cfg.rewards.pop("foot_clearance", None)
    cfg.rewards.pop("foot_swing_height", None)
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["lin_vel_z_l2"] = RewardTermCfg(
        func=plus_rewards.lin_vel_z_l2,
        weight=-2.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["track_linear_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].weight = 1.5
    cfg.rewards["body_ang_vel"].weight = -0.05
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = (robot.base_body_name,)
    cfg.rewards["dof_pos_limits"].weight = -5.0
    cfg.rewards["action_rate_l2"].weight = -0.01
    cfg.rewards["air_time"].weight = robot.feet_air_time_weight
    cfg.rewards["air_time"].params["threshold_max"] = 0.5

    cfg.rewards["stand_still"] = RewardTermCfg(
        func=plus_rewards.stand_still,
        weight=-2.0,
        params={
            "command_name": "twist",
            "command_threshold": 0.1,
            "asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names),
        },
    )
    cfg.rewards["joint_pos_penalty"] = RewardTermCfg(
        func=plus_rewards.joint_pos_penalty,
        weight=-1.0,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names),
            "stand_still_scale": 5.0,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
        },
    )
    cfg.rewards["joint_mirror"] = RewardTermCfg(
        func=plus_rewards.joint_mirror,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names),
            "mirror_joints": robot.mirror_joints,
        },
    )
    cfg.rewards["feet_contact_without_cmd"] = RewardTermCfg(
        func=plus_rewards.feet_contact_without_cmd,
        weight=0.1,
        params={"command_name": "twist", "sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["feet_height_body"] = RewardTermCfg(
        func=plus_rewards.feet_height_body,
        weight=robot.feet_height_body_weight,
        params={
            "command_name": "twist",
            "asset_cfg": SceneEntityCfg("robot", body_names=robot.foot_body_names),
            "target_height": robot.feet_height_body_target,
            "tanh_mult": 2.0,
        },
    )
    cfg.rewards["upward"] = RewardTermCfg(
        func=plus_rewards.upward,
        weight=1.0,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    cfg.rewards["feet_air_time_variance"] = RewardTermCfg(
        func=plus_rewards.feet_air_time_variance,
        weight=robot.feet_air_time_variance_weight,
        params={"sensor_name": feet_ground_cfg.name},
    )
    cfg.rewards["feet_gait"] = RewardTermCfg(
        func=plus_rewards.feet_gait,
        weight=robot.feet_gait_weight,
        params={
            "std": math.sqrt(0.5),
            "command_name": "twist",
            "max_err": 0.2,
            "velocity_threshold": 0.5,
            "command_threshold": 0.1,
            "synced_feet_pair_names": robot.gait_pairs,
            "sensor_name": feet_ground_cfg.name,
        },
    )
    cfg.rewards["foot_slip"].weight = robot.feet_slide_weight
    cfg.rewards["foot_slip"].params["asset_cfg"].site_names = robot.foot_site_names
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
    )
    cfg.rewards["undesired_contacts"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-1.0,
        params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 1.0},
    )
    cfg.rewards["contact_forces"] = RewardTermCfg(
        func=plus_rewards.contact_forces,
        weight=-1.5e-4,
        params={"sensor_name": feet_ground_cfg.name, "threshold": 100.0},
    )
    cfg.rewards["joint_torques_l2"] = RewardTermCfg(
        func=mdp.joint_torques_l2,
        weight=-2.5e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names)},
    )
    cfg.rewards["joint_acc_l2"] = RewardTermCfg(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names)},
    )
    cfg.rewards["joint_power"] = RewardTermCfg(
        func=mdp.electrical_power_cost,
        weight=-2.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=robot.joint_names)},
    )

    for name in tuple(cfg.rewards):
        if cfg.rewards[name].weight == 0:
            cfg.rewards.pop(name)
