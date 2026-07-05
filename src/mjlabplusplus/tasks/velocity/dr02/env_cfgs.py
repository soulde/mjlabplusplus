"""DR02 velocity environment configurations."""

import math

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
from mjlabplusplus.robots import DR02_ACTION_SCALE, get_dr02_robot_cfg


def dr02_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create DR02 rough terrain velocity configuration."""
  cfg = make_velocity_env_cfg()

  cfg.sim.mujoco.ccd_iterations = 500
  cfg.sim.contact_sensor_maxmatch = 500
  cfg.sim.nconmax = 200

  cfg.scene.entities = {"robot": get_dr02_robot_cfg()}
  cfg.scene.extent = 2.5

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      assert isinstance(sensor.frame, ObjRef)
      sensor.frame.name = "base_link"

  cfg.observations["actor"].terms.pop("base_lin_vel", None)
  cfg.observations["actor"].terms["base_ang_vel"].scale = 0.25
  cfg.observations["actor"].terms[
    "projected_gravity"
  ].func = mdp.projected_gravity_from_sensor
  cfg.observations["actor"].terms["projected_gravity"].params = {
    "sensor_name": "robot/imu_upvector"
  }
  cfg.observations["actor"].terms["joint_pos"].scale = 1.0
  cfg.observations["actor"].terms["joint_vel"].scale = 0.05
  del cfg.observations["actor"].terms["height_scan"]

  site_names = ("left_foot", "right_foot")
  foot_geom_names = ("left_foot_collision", "right_foot_collision")

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "foot_height_scan":
      assert isinstance(sensor, TerrainHeightSensorCfg)
      sensor.frame = tuple(
        ObjRef(type="site", name=site_name, entity="robot") for site_name in site_names
      )
      sensor.pattern = RingPatternCfg.single_ring(radius=0.04, num_samples=6)

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=foot_geom_names, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )
  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  nonfoot_ground_cfg = ContactSensorCfg(
    name="nonfoot_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=".*_collision",
      entity="robot",
      exclude=foot_geom_names,
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

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = DR02_ACTION_SCALE
  joint_pos_action.clip = {".*": (-100.0, 100.0)}

  cfg.viewer.body_name = "body"
  cfg.viewer.distance = 3.0
  cfg.viewer.elevation = -10.0

  cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_geom_names
  cfg.events["base_com"].params["asset_cfg"].body_names = ("body",)

  cfg.rewards["pose"].params["std_standing"] = {".*": 0.05}
  cfg.rewards["pose"].params["std_walking"] = {
    r".*hip_y_joint": 0.3,
    r".*hip_x_joint": 0.15,
    r".*hip_z_joint": 0.15,
    r".*knee_joint": 0.35,
    r".*ankle_y_joint": 0.25,
    r".*ankle_x_joint": 0.1,
    r"waist_z_joint": 0.2,
    r".*shoulder_y_joint": 0.2,
    r".*shoulder_x_joint": 0.2,
    r".*shoulder_z_joint": 0.15,
    r".*elbow_joint": 0.2,
  }
  cfg.rewards["pose"].params["std_running"] = {
    r".*hip_y_joint": 0.5,
    r".*hip_x_joint": 0.2,
    r".*hip_z_joint": 0.2,
    r".*knee_joint": 0.6,
    r".*ankle_y_joint": 0.35,
    r".*ankle_x_joint": 0.15,
    r"waist_z_joint": 0.3,
    r".*shoulder_y_joint": 0.4,
    r".*shoulder_x_joint": 0.25,
    r".*shoulder_z_joint": 0.2,
    r".*elbow_joint": 0.35,
  }

  cfg.rewards["upright"].params["asset_cfg"].body_names = ("body",)
  cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("body",)
  cfg.rewards["track_linear_velocity"].weight = 3.0
  cfg.rewards["track_angular_velocity"].weight = 3.0
  cfg.rewards["body_ang_vel"].weight = -0.1
  cfg.rewards.pop("angular_momentum", None)
  cfg.rewards["dof_pos_limits"].weight = -0.5
  cfg.rewards["action_rate_l2"].weight = -0.005
  cfg.rewards["air_time"].weight = 1.0
  cfg.rewards["air_time"].params["threshold_max"] = 0.6
  cfg.rewards["foot_slip"].weight = -0.2

  for reward_name in ["foot_clearance", "foot_slip"]:
    cfg.rewards[reward_name].params["asset_cfg"].site_names = site_names

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
  cfg.rewards["joint_torques_l2"] = RewardTermCfg(
    func=mdp.joint_torques_l2,
    weight=-1.5e-7,
    params={"asset_cfg": SceneEntityCfg("robot")},
  )
  cfg.rewards["joint_acc_l2"] = RewardTermCfg(
    func=mdp.joint_acc_l2,
    weight=-1.25e-7,
    params={
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*_hip_.*", ".*_knee_joint"))
    },
  )
  cfg.rewards["joint_power"] = RewardTermCfg(
    func=mdp.electrical_power_cost,
    weight=-2.0e-5,
    params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
  )
  cfg.rewards["flat_orientation"] = RewardTermCfg(
    func=mdp.flat_orientation_l2,
    weight=-0.2,
  )

  cfg.terminations["fell_over"] = TerminationTermCfg(
    func=mdp.bad_orientation,
    params={"limit_angle": math.radians(70.0)},
  )
  cfg.terminations["illegal_contact"] = TerminationTermCfg(
    func=mdp.illegal_contact,
    params={"sensor_name": nonfoot_ground_cfg.name, "force_threshold": 1.0},
  )

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.1
  twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
  twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
  twist_cmd.ranges.ang_vel_z = (-1.0, 1.0)

  if play:
    cfg.episode_length_s = int(1e9)
    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.terminations.pop("out_of_terrain_bounds", None)
    cfg.terminations.pop("illegal_contact", None)
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
    twist_cmd.ranges.lin_vel_x = (-1.5, 2.0)
    twist_cmd.ranges.ang_vel_z = (-0.7, 0.7)

  return cfg


def dr02_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create DR02 flat terrain velocity configuration."""
  cfg = dr02_rough_env_cfg(play=play)

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
