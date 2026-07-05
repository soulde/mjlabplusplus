"""DR02 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlabplusplus.tasks.velocity.dr02.env_cfgs import (
    dr02_flat_env_cfg,
    dr02_rough_env_cfg,
)
from mjlabplusplus.tasks.velocity.dr02.rl_cfg import (
    dr02_flat_fastsac_runner_cfg,
    dr02_flat_ppo_runner_cfg,
    dr02_flat_tdmpc2_runner_cfg,
    dr02_rough_fastsac_runner_cfg,
    dr02_rough_ppo_runner_cfg,
    dr02_rough_tdmpc2_runner_cfg,
)

DR02_ROUGH_TASK = "Mjlab-Velocity-Rough-DR02"
DR02_FLAT_TASK = "Mjlab-Velocity-Flat-DR02"

register_mjlab_task(
    task_id=DR02_ROUGH_TASK,
    env_cfg=dr02_rough_env_cfg(),
    play_env_cfg=dr02_rough_env_cfg(play=True),
    rl_cfg=dr02_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id=DR02_FLAT_TASK,
    env_cfg=dr02_flat_env_cfg(),
    play_env_cfg=dr02_flat_env_cfg(play=True),
    rl_cfg=dr02_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

try:
    from mjlab_algo.registry import register_fastsac_cfg, register_tdmpc2_cfg
except ImportError:
    pass
else:
    register_fastsac_cfg(DR02_ROUGH_TASK, dr02_rough_fastsac_runner_cfg())
    register_fastsac_cfg(DR02_FLAT_TASK, dr02_flat_fastsac_runner_cfg())
    register_tdmpc2_cfg(DR02_ROUGH_TASK, dr02_rough_tdmpc2_runner_cfg())
    register_tdmpc2_cfg(DR02_FLAT_TASK, dr02_flat_tdmpc2_runner_cfg())
