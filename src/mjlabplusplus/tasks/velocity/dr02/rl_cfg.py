"""RL configuration for DR02 velocity tasks."""

from mjlab.rl import (
    RslRlModelCfg,
    RslRlOnPolicyRunnerCfg,
    RslRlPpoAlgorithmCfg,
)


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
