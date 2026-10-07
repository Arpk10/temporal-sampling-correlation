# Reproducibility

This repository has two distinct reproducibility targets.

## 1. Software-level reproduction

The fixed benchmark contract is:

- 20 fixed prompts
- `rho ∈ {0, 0.25, 0.50, 0.75, 0.90}`
- 10 source seeds (`0..9`)
- 40 generated tokens per trajectory
- temperature `1.0`
- top-p `0.90`
- stationary Gaussian AR(1) source
- standard-normal CDF transform to uniforms
- inverse-CDF top-p sampling

Run:

```bash
uv run pytest
uv run ruff check .
uvx ty check ./temporal_sampling_correlation
```

GPT-2 execution is kept outside the unit-test path because it requires model weights and a CUDA-capable runtime for the numerical harness.

## 2. Scientific numerical reproduction

The current implementation is a **protocol reproduction on the present GPT-2/PyTorch stack**. It is not claimed to be byte-for-byte or numerically identical to the historical experimental runtime.

The historical reference artifact available during development contains 950 trajectory rows rather than the planned 1,000. It is therefore not treated as a complete ground-truth benchmark.

A full numerical reproduction should record Python, PyTorch, Transformers, model identifier, device/backend, prompt list and ordering, random seeds, temperature, top-p, generated-token count, and benchmark configuration.

The evaluator-side `rho` is explicit by design. This is a controlled measurement environment, not a hidden-parameter inference benchmark.
