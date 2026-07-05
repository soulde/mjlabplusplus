import re

import mujoco

from mjlab.entity import Entity
from mjlabplusplus.robots import get_dr02_robot_cfg


def test_dr02_model_dimensions_and_foot_collisions() -> None:
    """DR02 should expose the expected PPO action space and foot collisions."""
    model = Entity(get_dr02_robot_cfg()).compile()

    assert isinstance(model, mujoco.MjModel)
    assert model.nq == 28
    assert model.nv == 27
    assert model.nu == 21

    foot_pattern = r"^(left|right)_foot_collision$"
    foot_geoms = [
        model.geom(i)
        for i in range(model.ngeom)
        if re.match(foot_pattern, model.geom(i).name)
    ]
    assert len(foot_geoms) == 2
    for geom in foot_geoms:
        assert geom.condim == 3
        assert geom.priority == 1
        assert geom.friction[0] == 0.6
