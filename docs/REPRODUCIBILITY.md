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

## 3. Paired frozen-context control and crossed-bootstrap sensitivity

Run the paired full-vs-frozen experiment on a CUDA-capable machine:

```bash
python scripts/run_paired_sampler_null.py --output results/paired_sampler_null.csv
```

The same source sequence is used for the full autoregressive and frozen-context conditions for each prompt/rho/seed row. Within each rho, the seed-indexed source sequences are reused across prompts; therefore rows sharing a seed are not independent source draws.

After obtaining the CSV, reproduce the crossed prompt/source-seed sensitivity intervals with:

```bash
python scripts/bootstrap_crossed_sampler_null.py \
  --input results/paired_sampler_null.csv \
  --output results/paired_sampler_null_crossed_bootstrap.json \
  --bootstrap 20000 --seed 20261010
```

The bootstrap independently resamples prompt IDs and seed IDs, reusing the selected IDs across rho values. It reports percentile 95% intervals for the full, frozen-context, and paired-excess slopes. These intervals are a sensitivity analysis and should be labelled separately from the prompt-only intervals emitted by the paired experiment script.
