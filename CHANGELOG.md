# Changelog

## 0.3.1

- Hardened trajectory validation to require finite lag-1 and lag-2 CDF statistics.
- Clarified that the environment is a controlled evaluator-side measurement harness, not a hidden-`rho` inference task.
- Added model-free validation tests suitable for CI without GPT-2/model downloads.
- Preserved the existing 20-prompt × 5-rho × 10-seed scientific design.
