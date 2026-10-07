# Prime Intellect / `verifiers` v1 registration

The package exports the v1 taskset and its custom harness from `temporal_sampling_correlation/__init__.py`.

The v1 taskset follows:

```text
TasksetConfig → TaskData → Task → Taskset
```

The configurable `rho` lives on `TemporalConfig`:

```bash
uv run vf-eval temporal-sampling-correlation \
  --env.taskset.rho 0.50 \
  --env.agent.harness.id temporal-sampling-correlation
```

The default configuration uses the five-condition grid.

## Scientific contract

Each task corresponds to one prompt and one `rho`. The harness generates ten AR(1)-driven trajectories and writes trajectory-level statistics. The primary reward is normalized mean lag-1 CDF correlation; the raw statistic is exposed as `mean_lag1_cdf`.

## Scope note

This release hardens the existing controlled measurement environment rather than changing its scientific design. `rho` remains evaluator-side task data consumed by the custom harness; the environment is therefore not presented as a hidden-`rho` inference benchmark.

A future inference benchmark would need a separate task interface in which an agent receives observations from which `rho` must be estimated and is scored against a held-out condition.
