"""Temporal sampling correlation verifiers v1 environment."""

__all__ = [
    "TemporalSamplingCorrelationTaskset",
    "TemporalSamplingCorrelationHarness",
    "ar1_uniform",
    "sample_top_p",
    "generate_condition",
]


def __getattr__(name):
    if name == "TemporalSamplingCorrelationTaskset":
        from .taskset import TemporalSamplingCorrelationTaskset
        return TemporalSamplingCorrelationTaskset
    if name == "TemporalSamplingCorrelationHarness":
        from .harness import TemporalSamplingCorrelationHarness
        return TemporalSamplingCorrelationHarness
    if name in {"ar1_uniform", "sample_top_p", "generate_condition"}:
        from .sampler import ar1_uniform, sample_top_p, generate_condition
        return {"ar1_uniform": ar1_uniform, "sample_top_p": sample_top_p, "generate_condition": generate_condition}[name]
    raise AttributeError(name)
