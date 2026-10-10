# Changelog

## 0.3.1

- Hardened trajectory validation to require finite lag-1 and lag-2 CDF statistics.
- Clarified that the environment is a controlled evaluator-side measurement harness, not a hidden-`rho` inference task.
- Added model-free validation tests suitable for CI without GPT-2/model downloads.
- Preserved the existing 20-prompt × 5-rho × 10-seed scientific design.
- Added a paired full-vs-frozen sampler-null run and documented the operational excess-slope interpretation.
- Added a deterministic crossed prompt/source-seed bootstrap script for uncertainty sensitivity analysis.
- Updated README and reproducibility notes with the paired-control slopes and source-sequence reuse caveat.
