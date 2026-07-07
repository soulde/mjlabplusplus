"""DeepRobotics Lite3 constants."""

from importlib.resources import files
from pathlib import Path

import mujoco

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

_ASSET_ROOT = files("mjlabplusplus") / "assets" / "robots" / "deeprobotics_lite3"
DEEPROBOTICS_LITE3_XML: Path = Path(str(_ASSET_ROOT / "mjcf" / "lite3.xml"))
DEEPROBOTICS_LITE3_URDF: Path = Path(str(_ASSET_ROOT / "urdf" / "lite3.urdf"))

DEEPROBOTICS_LITE3_JOINT_NAMES = (
    "FL_HipX_joint",
    "FL_HipY_joint",
    "FL_Knee_joint",
    "FR_HipX_joint",
    "FR_HipY_joint",
    "FR_Knee_joint",
    "HL_HipX_joint",
    "HL_HipY_joint",
    "HL_Knee_joint",
    "HR_HipX_joint",
    "HR_HipY_joint",
    "HR_Knee_joint",
)
DEEPROBOTICS_LITE3_FOOT_SITE_NAMES = (
    "FL_FOOT",
    "FR_FOOT",
    "HL_FOOT",
    "HR_FOOT",
)
DEEPROBOTICS_LITE3_FOOT_BODY_NAMES = (
    "FL_SHANK",
    "FR_SHANK",
    "HL_SHANK",
    "HR_SHANK",
)
DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES = tuple(
    f"{site}_collision" for site in DEEPROBOTICS_LITE3_FOOT_SITE_NAMES
)
DEEPROBOTICS_LITE3_ACTION_SCALE = {
    ".*_HipX_joint": 0.125,
    "^(?!.*_HipX_joint).*": 0.25,
}


def _name_collision_geoms(spec: mujoco.MjSpec) -> None:
    for body in spec.bodies:
        for index, geom in enumerate(body.geoms):
            if not geom.name:
                geom.name = f"{body.name}_collision{index}"
    for body_name, site_name in zip(
        DEEPROBOTICS_LITE3_FOOT_BODY_NAMES,
        DEEPROBOTICS_LITE3_FOOT_SITE_NAMES,
        strict=True,
    ):
        body = spec.body(body_name)
        body.geoms[-2].name = f"{site_name}_collision"
        body.add_site(name=site_name, pos=[0.0, 0.0, -0.21012], size=[0.01])


def get_spec() -> mujoco.MjSpec:
    """Load DeepRobotics Lite3 MJCF and add training helper names/sites."""
    spec = mujoco.MjSpec.from_file(str(DEEPROBOTICS_LITE3_XML))
    _name_collision_geoms(spec)
    return spec


STANDING_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.35),
    joint_pos={
        ".*HipX_joint": 0.0,
        ".*HipY_joint": -0.8,
        ".*Knee_joint": 1.6,
    },
    joint_vel={".*": 0.0},
)

FOOT_COLLISION = CollisionCfg(
    geom_names_expr=DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES,
    condim={".*": 3},
    priority={".*": 1},
    friction={".*": (0.6,)},
)

DEEPROBOTICS_LITE3_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_Hip[X,Y]_joint",),
            stiffness=30.0,
            damping=0.5,
            effort_limit=24.0,
        ),
        BuiltinPositionActuatorCfg(
            target_names_expr=(".*_Knee_joint",),
            stiffness=30.0,
            damping=0.5,
            effort_limit=36.0,
        ),
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_deeprobotics_lite3_robot_cfg() -> EntityCfg:
    """Get a fresh DeepRobotics Lite3 robot configuration instance."""
    return EntityCfg(
        init_state=STANDING_KEYFRAME,
        collisions=(FOOT_COLLISION,),
        spec_fn=get_spec,
        articulation=DEEPROBOTICS_LITE3_ARTICULATION,
    )
