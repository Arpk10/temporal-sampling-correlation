# Scientific scope

## Question

If the stochastic source used by autoregressive decoding acquires temporal dependence while its one-step marginal remains controlled, does that temporal structure propagate into measurable properties of the generated trajectory?

## Intervention

A stationary Gaussian AR(1) process with correlation parameter `rho` is transformed through the standard-normal CDF to obtain uniforms. Those uniforms drive inverse-CDF top-p sampling from GPT-2.

The controlled variable is the **temporal dependence of the sampling source**, not the language prompt or the marginal sampling mechanism.

## Primary diagnostic

For each trajectory, the primary statistic is the lag-1 correlation between source uniform `u_t` and the CDF position of the token sampled at the next step. The verifier reward exposes the mean after:

`reward = (mean_lag1_cdf + 1) / 2`.

## What this does not establish

The environment does not establish that correlated randomness improves generation quality, optimization, or reinforcement learning. It also does not establish that a deployed Prime Intellect system currently exhibits the measured phenomenon.

The benchmark is a controlled measurement primitive that can support later work on evaluation, rollout dynamics, or training effects.
