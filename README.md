# mjlabplusplus

Private MJLab extension package for robots and tasks that should live outside
the upstream `mjlab` repository.

Installing this package in the same environment as `mjlab` registers the DR02
velocity tasks through the `mjlab.tasks` entry point:

```sh
uv add git+ssh://git@github.com/<owner>/mjlabplusplus.git
uv run list-envs --keyword DR02
uv run train Mjlab-Velocity-Flat-DR02
uv run train Mjlab-Velocity-Rough-DR02
```

The package currently provides:

- `Mjlab-Velocity-Flat-DR02`
- `Mjlab-Velocity-Rough-DR02`

`mjlab` must be installed separately in the consuming environment.
