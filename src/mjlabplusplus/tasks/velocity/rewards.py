"""Reward terms used by mjlabplusplus velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
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
    foot_pos_b = quat_apply_inverse(asset.data.root_link_quat_w[:, None, :], rel_pos)
    foot_vel_b = quat_apply_inverse(asset.data.root_link_quat_w[:, None, :], rel_vel)

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
