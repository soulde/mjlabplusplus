"""Unitree Go2 velocity task registrations."""

from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner
from mjlabplusplus.tasks.velocity.unitree_go2.env_cfgs import (
    unitree_go2_flat_env_cfg,
    unitree_go2_rough_env_cfg,
)
from mjlabplusplus.tasks.velocity.unitree_go2.rl_cfg import (
    UNITREE_GO2_FLAT_TASK,
    UNITREE_GO2_ROUGH_TASK,
    unitree_go2_flat_fastsac_runner_cfg,
    unitree_go2_flat_ppo_runner_cfg,
    unitree_go2_flat_tdmpc2_runner_cfg,
    unitree_go2_rough_fastsac_runner_cfg,
    unitree_go2_rough_ppo_runner_cfg,
    unitree_go2_rough_tdmpc2_runner_cfg,
)

register_mjlab_task(
    task_id=UNITREE_GO2_ROUGH_TASK,
    env_cfg=unitree_go2_rough_env_cfg(),
    play_env_cfg=unitree_go2_rough_env_cfg(play=True),
    rl_cfg=unitree_go2_rough_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)
register_mjlab_task(
    task_id=UNITREE_GO2_FLAT_TASK,
    env_cfg=unitree_go2_flat_env_cfg(),
    play_env_cfg=unitree_go2_flat_env_cfg(play=True),
    rl_cfg=unitree_go2_flat_ppo_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
)

try:
    from mjlab_algo.registry import register_fastsac_cfg, register_tdmpc2_cfg
except ImportError:
    pass
else:
    register_fastsac_cfg(UNITREE_GO2_ROUGH_TASK, unitree_go2_rough_fastsac_runner_cfg())
    register_fastsac_cfg(UNITREE_GO2_FLAT_TASK, unitree_go2_flat_fastsac_runner_cfg())
    register_tdmpc2_cfg(UNITREE_GO2_ROUGH_TASK, unitree_go2_rough_tdmpc2_runner_cfg())
    register_tdmpc2_cfg(UNITREE_GO2_FLAT_TASK, unitree_go2_flat_tdmpc2_runner_cfg())
