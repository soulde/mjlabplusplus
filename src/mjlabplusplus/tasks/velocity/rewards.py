"""Reward terms used by mjlabplusplus velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _upright_scale(env: ManagerBasedRlEnv) -> torch.Tensor:
    """Scale reward terms down when the base is far from upright."""
    asset: Entity = env.scene["robot"]
    return torch.clamp(-asset.data.projected_gravity_b[:, 2], 0.0, 0.7) / 0.7


def lin_vel_z_l2(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize z-axis base linear velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_link_lin_vel_b[:, 2])
    return reward * _upright_scale(env)


def joint_deviation_l1(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Penalize absolute joint position deviation from the default pose."""
    asset: Entity = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None
    joint_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_joint_pos[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.abs(joint_error), dim=1) * _upright_scale(env)


def stand_still(
    env: ManagerBasedRlEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Penalize joint deviation from default pose when command is near zero."""
    command = env.command_manager.get_command(command_name)
    assert command is not None
    reward = joint_deviation_l1(env, asset_cfg)
    return reward * (torch.linalg.norm(command, dim=1) < command_threshold).float()


def joint_pos_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
) -> torch.Tensor:
    """Penalize joint position deviation, scaled higher while standing still."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    default_joint_pos = asset.data.default_joint_pos
    assert default_joint_pos is not None

    command_norm = torch.linalg.norm(command, dim=1)
    body_vel = torch.linalg.norm(asset.data.root_link_lin_vel_b[:, :2], dim=1)
    joint_error = (
        asset.data.joint_pos[:, asset_cfg.joint_ids]
        - default_joint_pos[:, asset_cfg.joint_ids]
    )
    running_reward = torch.linalg.norm(joint_error, dim=1)
    standing = torch.logical_and(
        command_norm <= command_threshold,
        body_vel <= velocity_threshold,
    )
    reward = torch.where(standing, stand_still_scale * running_reward, running_reward)
    return reward * _upright_scale(env)


def joint_mirror(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    mirror_joints: list[list[str]],
) -> torch.Tensor:
    """Penalize mirrored joint pairs that deviate from each other."""
    asset: Entity = env.scene[asset_cfg.name]
    cache_name = "_mjlabplusplus_joint_mirror_cache"
    if not hasattr(env, cache_name):
        setattr(
            env,
            cache_name,
            [
                [asset.find_joints(joint_name) for joint_name in joint_pair]
                for joint_pair in mirror_joints
            ],
        )
    reward = torch.zeros(env.num_envs, device=env.device)
    joint_pair_cache = getattr(env, cache_name)
    for joint_pair in joint_pair_cache:
        diff = torch.sum(
            torch.square(
                asset.data.joint_pos[:, joint_pair[0][0]]
                - asset.data.joint_pos[:, joint_pair[1][0]]
            ),
            dim=-1,
        )
        reward += diff
    if mirror_joints:
        reward *= 1 / len(mirror_joints)
    return reward * _upright_scale(env)


def contact_forces(
    env: ManagerBasedRlEnv,
    sensor_name: str,
    threshold: float,
) -> torch.Tensor:
    """Penalize foot contact forces above a threshold."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    if data.force_history is not None:
        force_norm = torch.linalg.norm(data.force_history, dim=-1).max(dim=2).values
    else:
        assert data.force is not None
        force_norm = torch.linalg.norm(data.force, dim=-1)
    return torch.sum(torch.clamp(force_norm - threshold, min=0.0), dim=1) * (
        _upright_scale(env)
    )


def feet_air_time_variance(
    env: ManagerBasedRlEnv,
    sensor_name: str,
) -> torch.Tensor:
    """Penalize variance between the feet's recent air/contact phase durations."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    data = contact_sensor.data
    assert data.last_air_time is not None
    assert data.last_contact_time is not None
    reward = torch.var(torch.clip(data.last_air_time, max=0.5), dim=1)
    reward += torch.var(torch.clip(data.last_contact_time, max=0.5), dim=1)
    return reward * _upright_scale(env)


def feet_contact_without_cmd(
    env: ManagerBasedRlEnv,
    command_name: str,
    sensor_name: str,
) -> torch.Tensor:
    """Reward first foot contacts when commanded velocity is near zero."""
    contact_sensor: ContactSensor = env.scene[sensor_name]
    command = env.command_manager.get_command(command_name)
    assert command is not None
    first_contact = contact_sensor.compute_first_contact(env.step_dt)
    reward = torch.sum(first_contact.float(), dim=1)
    reward *= (torch.linalg.norm(command, dim=1) < 0.1).float()
    return reward * _upright_scale(env)


class feet_gait:
    """Quadruped gait timing reward matching robot_lab's GaitReward."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene[cfg.params["sensor_name"]]
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("feet_gait only supports two synchronized foot pairs.")
        self.synced_feet_pairs = [
            self._resolve_pair(synced_feet_pair_names[0]),
            self._resolve_pair(synced_feet_pair_names[1]),
        ]

    def __call__(
        self,
        env: ManagerBasedRlEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names: tuple[tuple[str, str], tuple[str, str]],
        sensor_name: str,
    ) -> torch.Tensor:
        del std
        del command_name
        del max_err
        del velocity_threshold
        del command_threshold
        del synced_feet_pair_names
        del sensor_name
        sync_reward_0 = self._sync_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1]
        )
        sync_reward_1 = self._sync_reward_func(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1]
        )
        sync_reward = sync_reward_0 * sync_reward_1
        async_reward_0 = self._async_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0]
        )
        async_reward_1 = self._async_reward_func(
            self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1]
        )
        async_reward_2 = self._async_reward_func(
            self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1]
        )
        async_reward_3 = self._async_reward_func(
            self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1]
        )
        async_reward = async_reward_0 * async_reward_1 * async_reward_2
        async_reward *= async_reward_3

        command = env.command_manager.get_command(self.command_name)
        assert command is not None
        asset: Entity = env.scene["robot"]
        cmd = torch.linalg.norm(command, dim=1)
        body_vel = torch.linalg.norm(asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(
                cmd > self.command_threshold,
                body_vel > self.velocity_threshold,
            ),
            sync_reward * async_reward,
            0.0,
        )
        return reward * _upright_scale(env)

    def _resolve_pair(self, pair: tuple[str, str]) -> tuple[int, int]:
        primary_names = self.contact_sensor.primary_names
        return primary_names.index(pair[0]), primary_names.index(pair[1])

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        assert air_time is not None
        assert contact_time is not None
        se_air = torch.clip(
            torch.square(air_time[:, foot_0] - air_time[:, foot_1]),
            max=self.max_err**2,
        )
        se_contact = torch.clip(
            torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        assert air_time is not None
        assert contact_time is not None
        se_act_0 = torch.clip(
            torch.square(air_time[:, foot_0] - contact_time[:, foot_1]),
            max=self.max_err**2,
        )
        se_act_1 = torch.clip(
            torch.square(contact_time[:, foot_0] - air_time[:, foot_1]),
            max=self.max_err**2,
        )
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def feet_height_body(
    env: ManagerBasedRlEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Penalize body-frame swing foot height error, weighted by foot velocity."""
    asset: Entity = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    assert command is not None

    foot_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
    foot_vel = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
    rel_pos = foot_pos - asset.data.root_link_pos_w[:, None, :]
    rel_vel = foot_vel - asset.data.root_link_lin_vel_w[:, None, :]
    root_quat_w = asset.data.root_link_quat_w[:, None, :].expand(*rel_pos.shape[:-1], 4)
    foot_pos_b = quat_apply_inverse(root_quat_w, rel_pos)
    foot_vel_b = quat_apply_inverse(root_quat_w, rel_vel)

    height_error = torch.square(foot_pos_b[:, :, 2] - target_height)
    velocity_scale = torch.tanh(
        tanh_mult * torch.linalg.norm(foot_vel_b[:, :, :2], dim=2)
    )
    reward = torch.sum(height_error * velocity_scale, dim=1)
    reward *= (torch.linalg.norm(command, dim=1) > 0.1).float()
    return reward * _upright_scale(env)


def upward(
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
    """Reward term matching robot_lab's projected-gravity upward expression."""
    asset: Entity = env.scene[asset_cfg.name]
    return torch.square(1.0 - asset.data.projected_gravity_b[:, 2])
