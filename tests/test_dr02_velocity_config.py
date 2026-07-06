"""DR02 velocity task configuration tests."""

from types import SimpleNamespace

import mjlabplusplus  # noqa: F401
import torch

from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensorCfg
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.velocity.mdp import projected_gravity_from_sensor
from mjlab_algo.registry import load_fastsac_cfg, load_tdmpc2_cfg
from mjlabplusplus.robots import DR02_ACTION_SCALE
from mjlabplusplus.tasks.velocity import rewards as plus_rewards

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
    assert cfg.rewards["lin_vel_z_l2"].func is plus_rewards.lin_vel_z_l2
    assert cfg.rewards["lin_vel_z_l2"].weight == -2.0
    assert cfg.rewards["track_linear_velocity"].weight == 3.0
    assert cfg.rewards["track_angular_velocity"].weight == 3.0
    assert cfg.rewards["body_ang_vel"].weight == -0.1
    assert cfg.rewards["flat_orientation"].weight == -0.2
    assert cfg.rewards["joint_deviation_hip_l1"].func is plus_rewards.joint_deviation_l1
    assert cfg.rewards["joint_deviation_hip_l1"].weight == -0.2
    assert cfg.rewards["joint_deviation_hip_l1"].params["asset_cfg"].joint_names == (
        ".*hip_[xz].*",
    )
    assert (
        cfg.rewards["joint_deviation_arms_l1"].func is plus_rewards.joint_deviation_l1
    )
    assert cfg.rewards["joint_deviation_arms_l1"].weight == -0.2
    assert cfg.rewards["joint_deviation_arms_l1"].params["asset_cfg"].joint_names == (
        ".*shoulder_.*",
        ".*elbow.*",
    )
    assert (
        cfg.rewards["joint_deviation_torso_l1"].func is plus_rewards.joint_deviation_l1
    )
    assert cfg.rewards["joint_deviation_torso_l1"].weight == -0.1
    assert cfg.rewards["joint_deviation_torso_l1"].params["asset_cfg"].joint_names == (
        "waist_z_joint",
    )
    assert cfg.rewards["dof_pos_limits"].weight == -0.5
    assert cfg.rewards["stand_still"].func is plus_rewards.stand_still
    assert cfg.rewards["stand_still"].weight == -2.0
    assert cfg.rewards["stand_still"].params["command_name"] == "twist"
    assert cfg.rewards["stand_still"].params["command_threshold"] == 0.1
    assert cfg.rewards["joint_pos_penalty"].func is plus_rewards.joint_pos_penalty
    assert cfg.rewards["joint_pos_penalty"].weight == -1.0
    assert cfg.rewards["joint_pos_penalty"].params["stand_still_scale"] == 5.0
    assert cfg.rewards["joint_pos_penalty"].params["velocity_threshold"] == 0.5
    assert cfg.rewards["joint_pos_penalty"].params["command_threshold"] == 0.1
    assert cfg.rewards["action_rate_l2"].weight == -0.005
    assert cfg.rewards["air_time"].weight == 1.0
    assert cfg.rewards["air_time"].params["threshold_max"] == 0.6
    assert (
        cfg.rewards["feet_air_time_variance"].func
        is plus_rewards.feet_air_time_variance
    )
    assert cfg.rewards["feet_air_time_variance"].weight == -1.0
    assert (
        cfg.rewards["feet_contact_without_cmd"].func
        is plus_rewards.feet_contact_without_cmd
    )
    assert cfg.rewards["feet_contact_without_cmd"].weight == 0.1
    assert cfg.rewards["foot_slip"].weight == -0.2
    assert cfg.rewards["feet_height_body"].func is plus_rewards.feet_height_body
    assert cfg.rewards["feet_height_body"].weight == -5.0
    assert cfg.rewards["feet_height_body"].params["target_height"] == -0.2
    assert cfg.rewards["feet_height_body"].params["tanh_mult"] == 2.0
    assert cfg.rewards["upward"].func is plus_rewards.upward
    assert cfg.rewards["upward"].weight == 1.0
    assert cfg.rewards["undesired_contacts"].weight == -1.0
    assert cfg.rewards["contact_forces"].func is plus_rewards.contact_forces
    assert cfg.rewards["contact_forces"].weight == -1.5e-4
    assert cfg.rewards["contact_forces"].params["sensor_name"] == "feet_ground_contact"
    assert cfg.rewards["contact_forces"].params["threshold"] == 100.0
    assert cfg.rewards["joint_torques_l2"].weight == -1.5e-7
    assert cfg.rewards["joint_acc_l2"].weight == -1.25e-7
    assert cfg.rewards["joint_power"].weight == -2.0e-5


def test_dr02_robot_lab_contact_sensor_matching() -> None:
    """DR02 reward sensors should match robot_lab's foot/non-foot split."""
    cfg = load_env_cfg(DR02_ROUGH_TASK)
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors or ()}
    feet_sensor = sensors["feet_ground_contact"]
    nonfoot_sensor = sensors["nonfoot_ground_contact"]

    assert isinstance(feet_sensor, ContactSensorCfg)
    assert feet_sensor.primary.mode == "geom"
    assert feet_sensor.primary.pattern == (
        "left_foot_collision",
        "right_foot_collision",
    )
    assert feet_sensor.secondary is not None
    assert feet_sensor.secondary.pattern == "terrain"
    assert feet_sensor.reduce == "netforce"
    assert feet_sensor.track_air_time
    assert feet_sensor.history_length == 4

    assert isinstance(nonfoot_sensor, ContactSensorCfg)
    assert nonfoot_sensor.primary.pattern == ".*_collision"
    assert nonfoot_sensor.primary.exclude == (
        "left_foot_collision",
        "right_foot_collision",
    )


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


def test_dr02_fastsac_and_tdmpc2_configs_are_registered_by_task() -> None:
    """DR02 off-policy algorithm defaults should live with the task cfg."""
    flat_fastsac = load_fastsac_cfg(DR02_FLAT_TASK)
    rough_fastsac = load_fastsac_cfg(DR02_ROUGH_TASK)
    flat_tdmpc2 = load_tdmpc2_cfg(DR02_FLAT_TASK)
    rough_tdmpc2 = load_tdmpc2_cfg(DR02_ROUGH_TASK)

    assert flat_fastsac.task == DR02_FLAT_TASK
    assert flat_fastsac.exp_name == "deeprobotics_dr02_standard_flat"
    assert flat_fastsac.num_envs == 16
    assert flat_fastsac.total_steps == 2_000_000
    assert flat_fastsac.learning_starts == 10_000
    assert rough_fastsac.task == DR02_ROUGH_TASK
    assert rough_fastsac.exp_name == "deeprobotics_dr02_standard_rough"

    assert flat_tdmpc2.task == DR02_FLAT_TASK
    assert flat_tdmpc2.exp_name == "deeprobotics_dr02_standard_flat"
    assert flat_tdmpc2.steps == 1_000_000
    assert flat_tdmpc2.model_size == 5
    assert flat_tdmpc2.batch_size == 256
    assert flat_tdmpc2.compile
    assert rough_tdmpc2.task == DR02_ROUGH_TASK
    assert rough_tdmpc2.exp_name == "deeprobotics_dr02_standard_rough"


def test_feet_height_body_handles_multiple_feet_per_env() -> None:
    """Body-frame foot-height reward should expand root quat across feet."""

    class _CommandManager:
        def get_command(self, _name: str) -> torch.Tensor:
            return torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])

    asset = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=torch.tensor(
                [
                    [[0.0, 0.0, -0.2], [0.0, 0.0, -0.1]],
                    [[0.0, 0.0, -0.3], [0.0, 0.0, -0.2]],
                ]
            ),
            body_link_lin_vel_w=torch.ones(2, 2, 3),
            root_link_pos_w=torch.zeros(2, 3),
            root_link_lin_vel_w=torch.zeros(2, 3),
            root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
            projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0], [0.0, 0.0, -1.0]]),
        )
    )
    env = SimpleNamespace(scene={"robot": asset}, command_manager=_CommandManager())

    reward = plus_rewards.feet_height_body(
        env,
        command_name="twist",
        asset_cfg=SceneEntityCfg("robot", body_ids=(0, 1)),
        target_height=-0.2,
        tanh_mult=2.0,
    )

    assert reward.shape == (2,)
    assert torch.all(torch.isfinite(reward))
