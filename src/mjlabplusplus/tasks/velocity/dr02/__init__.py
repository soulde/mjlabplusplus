"""DR02 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

from mjlabplusplus.tasks.velocity.dr02.env_cfgs import (
    dr02_flat_env_cfg,
    dr02_rough_env_cfg,
)
from mjlabplusplus.tasks.velocity.dr02.rl_cfg import (
    dr02_flat_ppo_runner_cfg,
    dr02_rough_ppo_runner_cfg,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Rough-DR02",
    env_cfg=dr02_rough_env_cfg(),
    play_env_cfg=dr02_rough_env_cfg(play=True),
    rl_cfg=dr02_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-DR02",
    env_cfg=dr02_flat_env_cfg(),
    play_env_cfg=dr02_flat_env_cfg(play=True),
    rl_cfg=dr02_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
