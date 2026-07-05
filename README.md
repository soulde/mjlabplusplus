# mjlabplusplus

Additional robot assets and tasks for [MJLab](https://github.com/mujocolab/mjlab).

`mjlabplusplus` is an extension package: install it into the same Python
environment as `mjlab`, and its tasks are registered through the `mjlab.tasks`
entry point. The upstream `mjlab` repository does not need to be modified.

## Contents

- DR02 robot assets:
  - `assets/robots/dr02/urdf/dr02_std.urdf`
  - `assets/robots/dr02/mjcf/dr02.xml`
  - `assets/robots/dr02/assets/*.STL`
- DR02 velocity tracking tasks:
  - `Mjlab-Velocity-Flat-DR02`
  - `Mjlab-Velocity-Rough-DR02`
- DR02 PPO/RSL-RL configuration.
- DR02 FastSAC and TD-MPC2 algorithm defaults for `mjlab-algo`.
- DR02 reward terms aligned with the `robot_lab` velocity training setup.

## Requirements

- Python version compatible with your `mjlab` checkout.
- A working `mjlab` environment.
- `uv` for local editable installation and command execution.

This package intentionally does not vendor `mjlab`; install `mjlab` separately
and then install this extension package into the same environment.

## Install

From a workspace that contains both `mjlab` and this repository:

```sh
uv pip install --python .venv/bin/python -e ./mjlabplusplus --no-build-isolation
```

From GitHub:

```sh
uv pip install --python .venv/bin/python \
  git+https://github.com/soulde/mjlabplusplus.git
```

## Verify Registration

```sh
uv run list-envs --keyword DR02
```

Expected tasks:

```text
Mjlab-Velocity-Flat-DR02
Mjlab-Velocity-Rough-DR02
```

## PPO Training

Train flat walking:

```sh
uv run train Mjlab-Velocity-Flat-DR02
```

Train rough terrain walking:

```sh
uv run train Mjlab-Velocity-Rough-DR02
```

Select a GPU explicitly:

```sh
CUDA_VISIBLE_DEVICES=0 uv run train Mjlab-Velocity-Flat-DR02
```

Run a short smoke test:

```sh
uv run train Mjlab-Velocity-Flat-DR02 --agent.max-iterations 10
```

## FastSAC and TD-MPC2 Training

Install `mjlab-algo` in the same environment to enable the additional algorithm
entry points. DR02 defaults for these algorithms live in
`tasks/velocity/dr02/rl_cfg.py` and are registered when this package is imported.

FastSAC:

```sh
uv run fastsac-train Mjlab-Velocity-Flat-DR02
uv run fastsac-train Mjlab-Velocity-Rough-DR02
```

TD-MPC2:

```sh
uv run tdmpc2-train Mjlab-Velocity-Flat-DR02
uv run tdmpc2-train Mjlab-Velocity-Rough-DR02
```

CLI flags are still available for temporary overrides:

```sh
uv run fastsac-train Mjlab-Velocity-Flat-DR02 --total-steps 10000
uv run tdmpc2-train Mjlab-Velocity-Flat-DR02 --steps 10000
```

## Play a Checkpoint

```sh
uv run play Mjlab-Velocity-Flat-DR02 \
  --checkpoint logs/rsl_rl/dr02_flat/<run>/model_<iter>.pt
```

Replace `<run>` and `<iter>` with the generated training run directory and
checkpoint iteration.

## Package Layout

```text
src/mjlabplusplus/
  assets/robots/dr02/
    assets/
    mjcf/
    urdf/
  robots/dr02.py
  tasks/velocity/dr02/
    env_cfgs.py
    rl_cfg.py
  tasks/velocity/rewards.py
```

## Development Checks

Run these from the parent `mjlab` workspace:

```sh
uv run ruff check mjlabplusplus/src mjlabplusplus/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest mjlabplusplus/tests
```

If the package is installed editable, changes under `src/mjlabplusplus` are
picked up without reinstalling.
