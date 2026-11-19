# Canonical sampling protocol v1

This repository uses the same top-p nucleus-set definition as Causal-Structured-stochasticity.

## Nucleus set

Given a normalized probability vector sorted in descending order, p_(1) >= ... >= p_(V), retain the **smallest prefix** whose cumulative probability is greater than or equal to the configured threshold p_top_p:

K = min { k : sum(i=1..k) p_(i) >= p_top_p }.

The retained set is indices 1..K. In particular, if cumulative mass is exactly equal to the threshold at index K, index K is retained and index K+1 is not. This explicit equality rule avoids implementation-dependent boundary behavior.

The retained probabilities are renormalized to sum to one. Sampling uses the inverse CDF of this renormalized distribution. The selected-token CDF metric is measured in this renormalized nucleus distribution; the original model probability of the selected token is recorded separately.

## Parameter constraints

- 0 < top_p <= 1
- Temperature must be positive.
- Probabilities are obtained from softmax after temperature scaling.
- The sorted probability vector must be non-empty.

## Compatibility

This protocol intentionally defines exact-threshold equality as the minimal prefix rule. It is not interchangeable with implementations that retain an additional token when the cumulative mass lands exactly on the threshold. Existing results generated under the previous Temporal Sampling Correlation mask are not silently relabelled as protocol v1. Regenerate affected results before combining them with new runs.
