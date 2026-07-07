"""Quadruped asset tests."""

import mujoco

from mjlab.entity import Entity
from mjlabplusplus.robots import (
    DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES,
    DEEPROBOTICS_LITE3_FOOT_SITE_NAMES,
    UNITREE_A1_FOOT_GEOM_NAMES,
    UNITREE_A1_FOOT_SITE_NAMES,
    UNITREE_GO2_FOOT_GEOM_NAMES,
    UNITREE_GO2_FOOT_SITE_NAMES,
    get_deeprobotics_lite3_robot_cfg,
    get_unitree_a1_robot_cfg,
    get_unitree_go2_robot_cfg,
)


def test_unitree_a1_model_dimensions_and_foot_names() -> None:
    model = Entity(get_unitree_a1_robot_cfg()).compile()

    assert isinstance(model, mujoco.MjModel)
    assert model.nq == 19
    assert model.nv == 18
    assert model.nu == 12
    assert {model.site(i).name for i in range(model.nsite)} == set(
        UNITREE_A1_FOOT_SITE_NAMES
    )
    assert set(_matching_geom_names(model, "foot_collision")) == set(
        UNITREE_A1_FOOT_GEOM_NAMES
    )


def test_unitree_go2_model_dimensions_and_foot_names() -> None:
    model = Entity(get_unitree_go2_robot_cfg()).compile()

    assert isinstance(model, mujoco.MjModel)
    assert model.nq == 19
    assert model.nv == 18
    assert model.nu == 12
    assert {model.site(i).name for i in range(model.nsite)} == set(
        UNITREE_GO2_FOOT_SITE_NAMES
    )
    assert set(_matching_geom_names(model, "foot_collision")) == set(
        UNITREE_GO2_FOOT_GEOM_NAMES
    )


def test_deeprobotics_lite3_model_dimensions_and_foot_names() -> None:
    model = Entity(get_deeprobotics_lite3_robot_cfg()).compile()

    assert isinstance(model, mujoco.MjModel)
    assert model.nq == 19
    assert model.nv == 18
    assert model.nu == 12
    assert {model.site(i).name for i in range(model.nsite)} == set(
        DEEPROBOTICS_LITE3_FOOT_SITE_NAMES
    )
    assert set(_matching_geom_names(model, "FOOT_collision")) == set(
        DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES
    )


def _matching_geom_names(model: mujoco.MjModel, pattern: str) -> list[str]:
    return [
        model.geom(i).name for i in range(model.ngeom) if pattern in model.geom(i).name
    ]
