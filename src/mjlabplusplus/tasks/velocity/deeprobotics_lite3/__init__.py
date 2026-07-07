"""DeepRobotics Lite3 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlabplusplus.tasks.velocity.deeprobotics_lite3.env_cfgs import (
    deeprobotics_lite3_flat_env_cfg,
    deeprobotics_lite3_rough_env_cfg,
)
from mjlabplusplus.tasks.velocity.deeprobotics_lite3.rl_cfg import (
    DEEPROBOTICS_LITE3_FLAT_TASK,
    DEEPROBOTICS_LITE3_ROUGH_TASK,
    deeprobotics_lite3_flat_fastsac_runner_cfg,
    deeprobotics_lite3_flat_ppo_runner_cfg,
    deeprobotics_lite3_flat_tdmpc2_runner_cfg,
    deeprobotics_lite3_rough_fastsac_runner_cfg,
    deeprobotics_lite3_rough_ppo_runner_cfg,
    deeprobotics_lite3_rough_tdmpc2_runner_cfg,
)

register_mjlab_task(
    task_id=DEEPROBOTICS_LITE3_ROUGH_TASK,
    env_cfg=deeprobotics_lite3_rough_env_cfg(),
    play_env_cfg=deeprobotics_lite3_rough_env_cfg(play=True),
    rl_cfg=deeprobotics_lite3_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id=DEEPROBOTICS_LITE3_FLAT_TASK,
    env_cfg=deeprobotics_lite3_flat_env_cfg(),
    play_env_cfg=deeprobotics_lite3_flat_env_cfg(play=True),
    rl_cfg=deeprobotics_lite3_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

try:
    from mjlab_algo.registry import register_fastsac_cfg, register_tdmpc2_cfg
except ImportError:
    pass
else:
    register_fastsac_cfg(
        DEEPROBOTICS_LITE3_ROUGH_TASK,
        deeprobotics_lite3_rough_fastsac_runner_cfg(),
    )
    register_fastsac_cfg(
        DEEPROBOTICS_LITE3_FLAT_TASK,
        deeprobotics_lite3_flat_fastsac_runner_cfg(),
    )
    register_tdmpc2_cfg(
        DEEPROBOTICS_LITE3_ROUGH_TASK,
        deeprobotics_lite3_rough_tdmpc2_runner_cfg(),
    )
    register_tdmpc2_cfg(
        DEEPROBOTICS_LITE3_FLAT_TASK,
        deeprobotics_lite3_flat_tdmpc2_runner_cfg(),
    )
