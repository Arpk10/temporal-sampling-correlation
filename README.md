# Temporal Sampling Correlation

**Controlled temporal structure in stochastic autoregressive decoding, packaged as a Prime Intellect `verifiers` v1 environment.**

[![Tests](https://github.com/arpk10/temporal-sampling-correlation/actions/workflows/tests.yml/badge.svg)](https://github.com/arpk10/temporal-sampling-correlation/actions/workflows/tests.yml)

This repository turns a controlled GPT-2 sampling experiment into a reproducible evaluation environment.

> **If the stochastic source driving autoregressive sampling acquires temporal dependence, does that dependence propagate into the model trajectory even when the marginal sampling mechanism is held fixed?**

The environment is deliberately a **measurement primitive**, not a claim that correlated randomness improves generation or optimization.

## What is measured?

- **20 fixed prompts**
- **5 source-correlation conditions:** `rho = {0, 0.25, 0.50, 0.75, 0.90}`
- **10 seeds** per prompt × condition
- **40 generated tokens** per trajectory
- GPT-2 at **temperature 1.0, top-p 0.90**
- stationary Gaussian **AR(1)** source
- standard-normal CDF transform to uniforms
- inverse-CDF top-p sampling

The default sweep contains **100 condition tasks and 1,000 trajectory-level rows**.

The primary trajectory statistic is the lag-1 correlation between source uniform `u_t` and the next-step sampled CDF position. The diagnostic reward is:

```text
reward = (mean_lag1_cdf + 1) / 2
```

The raw `mean_lag1_cdf` metric is retained for scientific analysis.

## Why package the experiment as an environment?

The scientific experiment asks whether temporal structure propagates into autoregressive trajectories. The environment makes that intervention executable, parameterized, validated, and reproducible through a taskset + harness interface.

`rho` is **evaluator-side task data**. This is therefore **not** a hidden-`rho` inference benchmark.

## Repository layout

```text
.
├── temporal_sampling_correlation/
│   ├── taskset.py
│   ├── harness.py
│   ├── sampler.py
│   ├── metrics.py
│   ├── prompts.py
│   └── runner.py
├── scripts/
├── tests/
├── docs/
├── VERIFIERS_REGISTRATION.md
├── CHANGELOG.md
├── CITATION.cff
├── LICENSE
└── pyproject.toml
```

## Quick start

Install with `uv`:

```bash
uv sync
uv run pytest
uv run ruff check .
uvx ty check ./temporal_sampling_correlation
```

On a CUDA-capable Linux runtime with the Prime `verifiers` v1 stack:

```bash
uv run vf-eval temporal-sampling-correlation \
  --env.taskset.rho 0.50 \
  --env.agent.harness.id temporal-sampling-correlation
```

Omit `--env.taskset.rho` for the default five-condition grid.

## Scientific result and interpretation

The accompanying GPT-2 study reports a positive dose-response between imposed source correlation and the lag-1 sampling-CDF statistic. The reported primary slope was **0.4646 ± 0.0216**, with prompt-clustered inference and all tested primary endpoint families surviving the stated multiple-testing correction.

Those numerical results belong to the accompanying scientific experiment; this repository is the reproducibility/evaluation implementation. The current code is a **protocol reproduction on the present GPT-2/PyTorch stack**, not a byte-level reconstruction of the historical runtime.

## Scope

This repository does **not** claim that correlated randomness improves generation quality, optimization, or reinforcement learning, nor that Prime Intellect infrastructure currently exhibits the measured phenomenon. It provides a controlled environment in which the mechanism can be measured explicitly.

The historical reference artifact available during development contains 950 trajectory rows rather than the planned 1,000, so it is not silently presented as complete ground truth.

## Status

**v0.3.1 — hardened research/evaluation artifact.**

The package validates trajectory results, preserves the 20 × 5 × 10 design, provides model-free CI tests, and keeps the GPT-2 numerical harness separate from unit tests.

See [`docs/SCIENTIFIC_SCOPE.md`](docs/SCIENTIFIC_SCOPE.md) and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md).

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT.
