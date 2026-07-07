"""Unitree A1 constants."""

from importlib.resources import files
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ASSET_ROOT = files("mjlabplusplus") / "assets" / "robots" / "unitree_a1"
UNITREE_A1_XML: Path = Path(str(_ASSET_ROOT / "mjcf" / "a1.xml"))
UNITREE_A1_URDF: Path = Path(str(_ASSET_ROOT / "urdf" / "a1.urdf"))

UNITREE_A1_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
UNITREE_A1_FOOT_SITE_NAMES = ("FR_foot", "FL_foot", "RR_foot", "RL_foot")
UNITREE_A1_FOOT_BODY_NAMES = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")
UNITREE_A1_FOOT_GEOM_NAMES = tuple(
    f"{site}_collision" for site in UNITREE_A1_FOOT_SITE_NAMES
)
UNITREE_A1_ACTION_SCALE = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}


def _name_collision_geoms(spec: mujoco.MjSpec) -> None:
    for body in spec.bodies:
        for index, geom in enumerate(body.geoms):
            if not geom.name:
                geom.name = f"{body.name}_collision{index}"
    for body_name, site_name in zip(
        UNITREE_A1_FOOT_BODY_NAMES, UNITREE_A1_FOOT_SITE_NAMES, strict=True
    ):
        body = spec.body(body_name)
        body.geoms[-1].name = f"{site_name}_collision"
        body.add_site(name=site_name, pos=[0.0, 0.0, -0.2], size=[0.01])


def get_spec() -> mujoco.MjSpec:
    """Load Unitree A1 MJCF and add training helper names/sites."""
    spec = mujoco.MjSpec.from_file(str(UNITREE_A1_XML))
    _name_collision_geoms(spec)
    return spec


STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.38),
    joint_pos={
        ".*L_hip_joint": 0.0,
        ".*R_hip_joint": 0.0,
        "F.*_thigh_joint": 0.8,
        "R.*_thigh_joint": 0.8,
        ".*_calf_joint": -1.5,
    },
    joint_vel={".*": 0.0},
)

FOOT_COLLISION = CollisionCfg(
    geom_names_expr=UNITREE_A1_FOOT_GEOM_NAMES,
    condim={".*": 3},
    priority={".*": 1},
    friction={".*": (0.6,)},
)

UNITREE_A1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_joint",),
            stiffness=20.0,
            damping=0.5,
            effort_limit=33.5,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_unitree_a1_robot_cfg() -> EntityCfg:
    """Get a fresh Unitree A1 robot configuration instance."""
    return EntityCfg(
        init_state=STANDING_KEYFRAME,
        collisions=(FOOT_COLLISION,),
        spec_fn=get_spec,
        articulation=UNITREE_A1_ARTICULATION,
    )
