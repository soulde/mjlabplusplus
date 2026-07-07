"""Quadruped velocity task configuration tests."""

import mjlabplusplus  # noqa: F401

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab_algo.registry import load_fastsac_cfg, load_tdmpc2_cfg
from mjlabplusplus.robots import (
    DEEPROBOTICS_LITE3_ACTION_SCALE,
    DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES,
    UNITREE_A1_ACTION_SCALE,
    UNITREE_A1_FOOT_GEOM_NAMES,
    UNITREE_GO2_ACTION_SCALE,
    UNITREE_GO2_FOOT_GEOM_NAMES,
)
from mjlabplusplus.tasks.velocity import rewards as plus_rewards

A1_FLAT = "Mjlab-Velocity-Flat-Unitree-A1"
A1_ROUGH = "Mjlab-Velocity-Rough-Unitree-A1"
GO2_FLAT = "Mjlab-Velocity-Flat-Unitree-Go2"
GO2_ROUGH = "Mjlab-Velocity-Rough-Unitree-Go2"
LITE3_FLAT = "Mjlab-Velocity-Flat-DeepRobotics-Lite3"
LITE3_ROUGH = "Mjlab-Velocity-Rough-DeepRobotics-Lite3"


def test_quadruped_tasks_register_when_package_is_imported() -> None:
    task_ids = list_tasks()

    for task in (A1_FLAT, A1_ROUGH, GO2_FLAT, GO2_ROUGH, LITE3_FLAT, LITE3_ROUGH):
        assert task in task_ids


def test_quadruped_policy_observations_match_robot_lab_setup() -> None:
    cfg = load_env_cfg(GO2_FLAT)
    terms = cfg.observations["actor"].terms

    assert tuple(terms) == (
        "base_ang_vel",
        "projected_gravity",
        "joint_pos",
        "joint_vel",
        "actions",
        "command",
    )
    assert terms["base_ang_vel"].scale == 0.25
    assert terms["joint_pos"].scale == 1.0
    assert terms["joint_vel"].scale == 0.05


def test_quadruped_actions_match_robot_lab_scales() -> None:
    expected = {
        A1_FLAT: UNITREE_A1_ACTION_SCALE,
        GO2_FLAT: UNITREE_GO2_ACTION_SCALE,
        LITE3_FLAT: DEEPROBOTICS_LITE3_ACTION_SCALE,
    }

    for task, scale in expected.items():
        action_cfg = load_env_cfg(task).actions["joint_pos"]
        assert isinstance(action_cfg, JointPositionActionCfg)
        assert action_cfg.scale == scale
        assert action_cfg.clip == {".*": (-100.0, 100.0)}


def test_quadruped_contact_sensors_match_foot_geom_names() -> None:
    expected = {
        A1_ROUGH: UNITREE_A1_FOOT_GEOM_NAMES,
        GO2_ROUGH: UNITREE_GO2_FOOT_GEOM_NAMES,
        LITE3_ROUGH: DEEPROBOTICS_LITE3_FOOT_GEOM_NAMES,
    }

    for task, foot_geoms in expected.items():
        sensors = {
            sensor.name: sensor for sensor in load_env_cfg(task).scene.sensors or ()
        }
        feet_sensor = sensors["feet_ground_contact"]
        nonfoot_sensor = sensors["nonfoot_ground_contact"]
        assert isinstance(feet_sensor, ContactSensorCfg)
        assert feet_sensor.primary.mode == "geom"
        assert feet_sensor.primary.pattern == foot_geoms
        assert feet_sensor.track_air_time
        assert isinstance(nonfoot_sensor, ContactSensorCfg)
        assert nonfoot_sensor.primary.exclude == foot_geoms


def test_quadruped_rewards_match_robot_lab_key_weights() -> None:
    a1 = load_env_cfg(A1_ROUGH)
    go2 = load_env_cfg(GO2_ROUGH)
    lite3 = load_env_cfg(LITE3_ROUGH)

    for cfg in (a1, go2, lite3):
        assert "angular_momentum" not in cfg.rewards
        assert "pose" not in cfg.rewards
        assert cfg.rewards["lin_vel_z_l2"].func is plus_rewards.lin_vel_z_l2
        assert cfg.rewards["lin_vel_z_l2"].weight == -2.0
        assert cfg.rewards["track_linear_velocity"].weight == 3.0
        assert cfg.rewards["track_angular_velocity"].weight == 1.5
        assert cfg.rewards["body_ang_vel"].weight == -0.05
        assert cfg.rewards["dof_pos_limits"].weight == -5.0
        assert cfg.rewards["action_rate_l2"].weight == -0.01
        assert cfg.rewards["joint_mirror"].func is plus_rewards.joint_mirror
        assert cfg.rewards["joint_mirror"].weight == -0.05
        assert cfg.rewards["stand_still"].weight == -2.0
        assert cfg.rewards["joint_pos_penalty"].weight == -1.0
        assert cfg.rewards["feet_contact_without_cmd"].weight == 0.1
        assert cfg.rewards["upward"].func is plus_rewards.upward
        assert cfg.rewards["upward"].weight == 1.0
        assert cfg.rewards["undesired_contacts"].weight == -1.0
        assert cfg.rewards["contact_forces"].weight == -1.5e-4
        assert cfg.rewards["joint_torques_l2"].weight == -2.5e-5
        assert cfg.rewards["joint_acc_l2"].weight == -2.5e-7
        assert cfg.rewards["joint_power"].weight == -2.0e-5

    assert "feet_air_time_variance" not in a1.rewards
    assert "feet_gait" not in a1.rewards
    assert "foot_slip" not in a1.rewards
    assert go2.rewards["air_time"].weight == 0.1
    assert go2.rewards["feet_air_time_variance"].weight == -1.0
    assert go2.rewards["feet_gait"].func is plus_rewards.feet_gait
    assert go2.rewards["feet_gait"].weight == 0.5
    assert go2.rewards["foot_slip"].weight == -0.1
    assert "feet_height_body" not in lite3.rewards


def test_quadruped_flat_and_rough_terrain_differences() -> None:
    flat_cfg = load_env_cfg(A1_FLAT)
    rough_cfg = load_env_cfg(A1_ROUGH)

    assert flat_cfg.scene.terrain is not None
    assert rough_cfg.scene.terrain is not None
    assert flat_cfg.scene.terrain.terrain_type == "plane"
    assert flat_cfg.scene.terrain.terrain_generator is None
    assert rough_cfg.scene.terrain.terrain_type == "generator"
    assert rough_cfg.scene.terrain.terrain_generator is not None
    assert "height_scan" not in flat_cfg.observations["critic"].terms
    assert "height_scan" in rough_cfg.observations["critic"].terms


def test_quadruped_rl_configs_are_registered_by_task() -> None:
    assert load_rl_cfg(GO2_FLAT).experiment_name == "unitree_go2_flat"
    assert load_rl_cfg(GO2_ROUGH).max_iterations == 3_000
    assert load_fastsac_cfg(GO2_FLAT).task == GO2_FLAT
    assert load_tdmpc2_cfg(GO2_ROUGH).task == GO2_ROUGH
