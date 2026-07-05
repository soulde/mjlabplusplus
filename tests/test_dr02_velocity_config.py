"""DR02 velocity task configuration tests."""

import mjlabplusplus  # noqa: F401

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import projected_gravity_from_sensor
from mjlabplusplus.robots import DR02_ACTION_SCALE

DR02_FLAT_TASK = "Mjlab-Velocity-Flat-DR02"
DR02_ROUGH_TASK = "Mjlab-Velocity-Rough-DR02"
DR02_POLICY_TERMS = (
    "base_ang_vel",
    "projected_gravity",
    "joint_pos",
    "joint_vel",
    "actions",
    "command",
)


def test_dr02_tasks_register_when_package_is_imported() -> None:
    """Importing mjlabplusplus should register DR02 tasks with mjlab."""
    task_ids = list_tasks()

    assert DR02_FLAT_TASK in task_ids
    assert DR02_ROUGH_TASK in task_ids


def test_dr02_policy_observations_match_real_imu_training_setup() -> None:
    """DR02 policy obs should match the robot_lab proprioceptive setup."""
    cfg = load_env_cfg(DR02_FLAT_TASK)
    policy_terms = cfg.observations["actor"].terms

    assert tuple(policy_terms) == DR02_POLICY_TERMS
    assert policy_terms["base_ang_vel"].scale == 0.25
    assert policy_terms["projected_gravity"].func is projected_gravity_from_sensor
    assert policy_terms["projected_gravity"].params == {
        "sensor_name": "robot/imu_upvector"
    }
    assert policy_terms["joint_pos"].scale == 1.0
    assert policy_terms["joint_vel"].scale == 0.05


def test_dr02_actions_match_robot_lab_scale_and_clip() -> None:
    """DR02 PPO action scale and clip should follow robot_lab's DR02 config."""
    cfg = load_env_cfg(DR02_FLAT_TASK)
    action_cfg = cfg.actions["joint_pos"]

    assert isinstance(action_cfg, JointPositionActionCfg)
    assert action_cfg.scale == DR02_ACTION_SCALE
    assert DR02_ACTION_SCALE == 0.25
    assert action_cfg.clip == {".*": (-100.0, 100.0)}


def test_dr02_rewards_match_robot_lab_key_weights() -> None:
    """DR02 reward weights should stay aligned with the robot_lab DR02 task."""
    cfg = load_env_cfg(DR02_ROUGH_TASK)

    assert "angular_momentum" not in cfg.rewards
    assert cfg.rewards["track_linear_velocity"].weight == 3.0
    assert cfg.rewards["track_angular_velocity"].weight == 3.0
    assert cfg.rewards["body_ang_vel"].weight == -0.1
    assert cfg.rewards["flat_orientation"].weight == -0.2
    assert cfg.rewards["dof_pos_limits"].weight == -0.5
    assert cfg.rewards["action_rate_l2"].weight == -0.005
    assert cfg.rewards["air_time"].weight == 1.0
    assert cfg.rewards["air_time"].params["threshold_max"] == 0.6
    assert cfg.rewards["foot_slip"].weight == -0.2
    assert cfg.rewards["undesired_contacts"].weight == -1.0
    assert cfg.rewards["joint_torques_l2"].weight == -1.5e-7
    assert cfg.rewards["joint_acc_l2"].weight == -1.25e-7
    assert cfg.rewards["joint_power"].weight == -2.0e-5


def test_dr02_flat_and_rough_terrain_differences() -> None:
    """Flat task should remove terrain scan while rough keeps it for the critic."""
    flat_cfg = load_env_cfg(DR02_FLAT_TASK)
    rough_cfg = load_env_cfg(DR02_ROUGH_TASK)

    assert flat_cfg.scene.terrain is not None
    assert rough_cfg.scene.terrain is not None
    assert flat_cfg.scene.terrain.terrain_type == "plane"
    assert flat_cfg.scene.terrain.terrain_generator is None
    assert rough_cfg.scene.terrain.terrain_type == "generator"
    assert rough_cfg.scene.terrain.terrain_generator is not None
    assert "height_scan" not in flat_cfg.observations["critic"].terms
    assert "height_scan" in rough_cfg.observations["critic"].terms


def test_dr02_ppo_runner_configs_match_robot_lab_iteration_counts() -> None:
    """Flat and rough DR02 runners should use the robot_lab training horizons."""
    flat_rl_cfg = load_rl_cfg(DR02_FLAT_TASK)
    rough_rl_cfg = load_rl_cfg(DR02_ROUGH_TASK)

    assert flat_rl_cfg.max_iterations == 1_000
    assert flat_rl_cfg.experiment_name == "deeprobotics_dr02_standard_flat"
    assert rough_rl_cfg.max_iterations == 3_000
    assert rough_rl_cfg.experiment_name == "deeprobotics_dr02_standard_rough"
