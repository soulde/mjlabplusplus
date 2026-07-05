"""DeepRobotics DR02 constants."""

from importlib.resources import files
from pathlib import Path

import mujoco

from mjlab.actuator import XmlActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF, URDF, and assets.
##

_ASSET_ROOT = files("mjlabplusplus") / "assets" / "robots" / "dr02"
DR02_XML: Path = Path(str(_ASSET_ROOT / "mjcf" / "dr02.xml"))
DR02_URDF: Path = Path(str(_ASSET_ROOT / "urdf" / "dr02_std.urdf"))


def get_spec() -> mujoco.MjSpec:
  return mujoco.MjSpec.from_file(str(DR02_XML))


##
# Actuator config.
##

DR02_XML_ACTUATORS = XmlActuatorCfg(
  target_names_expr=(".*_joint",),
  command_field="position",
)

##
# Keyframe config.
##

STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.904),
  joint_pos={".*": 0.0},
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_FOOT_REGEX = r"^(left|right)_foot_collision$"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_FOOT_REGEX: 3, ".*_collision": 1},
  priority={_FOOT_REGEX: 1},
  friction={_FOOT_REGEX: (0.6,)},
)

##
# Final config.
##

DR02_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(DR02_XML_ACTUATORS,),
  soft_joint_pos_limit_factor=0.9,
)


def get_dr02_robot_cfg() -> EntityCfg:
  """Get a fresh DR02 robot configuration instance."""
  return EntityCfg(
    init_state=STANDING_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=DR02_ARTICULATION,
  )


DR02_ACTION_SCALE = 0.25
