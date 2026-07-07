"""RL configuration for DR02 velocity tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)

if TYPE_CHECKING:
    from mjlab_algo.fastsac import FastSACConfig
    from mjlab_algo.tdmpc2 import TDMPC2Config

DR02_FLAT_TASK = "Mjlab-Velocity-Flat-DR02"
DR02_ROUGH_TASK = "Mjlab-Velocity-Rough-DR02"


def dr02_rough_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create PPO runner configuration for DR02 rough walking."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=False,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.01,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name="deeprobotics_dr02_standard_rough",
        save_interval=100,
        num_steps_per_env=24,
        max_iterations=3_000,
    )


def dr02_flat_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create PPO runner configuration for DR02 flat walking."""
    cfg = dr02_rough_ppo_runner_cfg()
    cfg.experiment_name = "deeprobotics_dr02_standard_flat"
    cfg.max_iterations = 1_000
    return cfg


def dr02_rough_fastsac_runner_cfg() -> FastSACConfig:
    """Create FastSAC runner configuration for DR02 rough walking."""
    from mjlab_algo.fastsac import FastSACConfig

    return FastSACConfig(
        task=DR02_ROUGH_TASK,
        total_steps=2_000_000,
        num_envs=16,
        batch_size=256,
        buffer_size=1_000_000,
        learning_starts=10_000,
        train_every=1,
        gradient_steps=1,
        gamma=0.99,
        tau=0.005,
        actor_lr=3.0e-4,
        critic_lr=3.0e-4,
        alpha_lr=3.0e-4,
        hidden_dims=(512, 256, 128),
        log_root="logs/fastsac",
        exp_name="deeprobotics_dr02_standard_rough",
        save_interval=100_000,
        log_interval=1_000,
    )


def dr02_flat_fastsac_runner_cfg() -> FastSACConfig:
    """Create FastSAC runner configuration for DR02 flat walking."""
    cfg = dr02_rough_fastsac_runner_cfg()
    cfg.task = DR02_FLAT_TASK
    cfg.exp_name = "deeprobotics_dr02_standard_flat"
    return cfg


def dr02_rough_tdmpc2_runner_cfg() -> TDMPC2Config:
    """Create TD-MPC2 runner configuration for DR02 rough walking."""
    from mjlab_algo.tdmpc2 import make_tdmpc2_config

    cfg = make_tdmpc2_config(
        model_size=5,
        task=DR02_ROUGH_TASK,
        steps=1_000_000,
        batch_size=256,
        buffer_size=1_000_000,
        lr=3.0e-4,
        mpc=True,
        enable_wandb=True,
        save_video=False,
        episodic=False,
        log_root="logs/tdmpc2",
        exp_name="deeprobotics_dr02_standard_rough",
    )
    cfg.model_size = 5
    return cfg


def dr02_flat_tdmpc2_runner_cfg() -> TDMPC2Config:
    """Create TD-MPC2 runner configuration for DR02 flat walking."""
    cfg = dr02_rough_tdmpc2_runner_cfg()
    cfg.task = DR02_FLAT_TASK
    cfg.exp_name = "deeprobotics_dr02_standard_flat"
    return cfg
