"""Paired full-vs-frozen-context sampler null sweep.

Runs the same 20 prompts x 5 rho x 10 seeds in two paired conditions:
1. full autoregressive sampling with the structured u_t source;
2. frozen-context sampling against a greedy reference trajectory.

The exact same u sequence is used in both conditions for every prompt/rho/seed.
Outputs trajectory-level paired differences plus prompt-bootstrap dose-response
CIs. This is a diagnostic/control, not a replacement for the main experiment.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from temporal_sampling_correlation.prompts import (
    PROMPTS, RHO_GRID, SEEDS, N_NEW, TEMPERATURE, TOP_P,
)
from temporal_sampling_correlation.sampler import ar1_uniform, sample_top_p, safe_corr
from transformers import GPT2LMHeadModel, GPT2TokenizerFast


def greedy_reference(model, tok, prompt, device, n_new):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    fixed = []
    for _ in range(n_new):
        fixed.append(logits.detach().clone())
        token = torch.argmax(logits, dim=-1)
        with torch.no_grad():
            out = model(
                input_ids=token[:, None],
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    return fixed


def frozen_metrics(fixed, u, device, temperature, top_p):
    cdf, nrank = [], []
    for t, logits in enumerate(fixed):
        _, _, selected_cdf, _, normalized_rank, _ = sample_top_p(
            logits,
            torch.tensor([u[t]], dtype=torch.float32, device=device),
            temperature,
            top_p,
        )
        cdf.append(float(selected_cdf[0].item()))
        nrank.append(float(normalized_rank[0].item()))
    cdf = np.asarray(cdf)
    nrank = np.asarray(nrank)
    return {
        "lag1_cdf": safe_corr(u[:-1], cdf[1:]),
        "lag1_normrank": safe_corr(u[:-1], nrank[1:]),
    }


def full_metrics(model, prompt, u, device, tok, temperature, top_p):
    ids = tok(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    tokens, cdf, nrank = [], [], []

    for t in range(len(u)):
        (
            next_token, _prob, selected_cdf, _rank, normalized_rank, _entropy
        ) = sample_top_p(
            logits,
            torch.tensor([u[t]], dtype=torch.float32, device=device),
            temperature,
            top_p,
        )
        tokens.append(int(next_token[0].item()))
        cdf.append(float(selected_cdf[0].item()))
        nrank.append(float(normalized_rank[0].item()))

        with torch.no_grad():
            out = model(
                input_ids=next_token[:, None],
                past_key_values=past,
                use_cache=True,
            )
        past = out.past_key_values
        logits = out.logits[:, -1, :]

    cdf = np.asarray(cdf)
    nrank = np.asarray(nrank)
    bigrams = list(zip(tokens[:-1], tokens[1:]))
    distinct2 = len(set(bigrams)) / max(1, len(bigrams))

    # Longest consecutive repeated-token streak as a simple sequence-level
    # feedback-sensitive endpoint.
    longest = 1
    run = 1
    for a, b in zip(tokens[:-1], tokens[1:]):
        if a == b:
            run += 1
            longest = max(longest, run)
        else:
            run = 1

    return {
        "lag1_cdf": safe_corr(u[:-1], cdf[1:]),
        "lag1_normrank": safe_corr(u[:-1], nrank[1:]),
        "distinct2": distinct2,
        "longest_token_streak": longest,
        "token_ids": tokens,
    }


def bootstrap_prompt_slopes(df, value_col, n_boot=5000, seed=0):
    rng = np.random.default_rng(seed)
    prompts = np.array(sorted(df["prompt_id"].unique()))
    slopes = np.empty(n_boot, dtype=float)

    def slope_for(frame):
        g = frame.groupby("rho")[value_col].mean()
        return float(np.polyfit(g.index.to_numpy(float), g.to_numpy(float), 1)[0])

    observed = slope_for(df)
    for i in range(n_boot):
        chosen = rng.choice(prompts, size=len(prompts), replace=True)
        parts = [df[df["prompt_id"] == p] for p in chosen]
        boot = pd.concat(parts, ignore_index=True)
        slopes[i] = slope_for(boot)
    return observed, float(np.quantile(slopes, 0.025)), float(np.quantile(slopes, 0.975))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="results/paired_sampler_null.csv")
    p.add_argument("--n-new", type=int, default=N_NEW)
    p.add_argument("--bootstrap", type=int, default=5000)
    p.add_argument("--seed", type=int, default=1729)
    args = p.parse_args()

    if not torch.cuda.is_available() and args.device.startswith("cuda"):
        raise RuntimeError("CUDA unavailable")

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(args.device).eval()

    rows = []
    total = len(PROMPTS) * len(RHO_GRID) * len(SEEDS)
    done = 0

    for prompt_id, prompt in enumerate(PROMPTS):
        fixed = greedy_reference(model, tok, prompt, args.device, args.n_new)

        for rho in RHO_GRID:
            for seed in SEEDS:
                u = ar1_uniform(rho, args.n_new, seed)
                source_corr = safe_corr(u[:-1], u[1:])

                full = full_metrics(
                    model, prompt, u, args.device, tok, TEMPERATURE, TOP_P
                )
                frozen = frozen_metrics(
                    fixed, u, args.device, TEMPERATURE, TOP_P
                )

                rows.append({
                    "prompt_id": prompt_id,
                    "prompt": prompt,
                    "rho": rho,
                    "seed": seed,
                    "source_corr": source_corr,
                    "full_lag1_cdf": full["lag1_cdf"],
                    "frozen_lag1_cdf": frozen["lag1_cdf"],
                    "excess_lag1_cdf": (
                        full["lag1_cdf"] - frozen["lag1_cdf"]
                    ),
                    "full_lag1_normrank": full["lag1_normrank"],
                    "frozen_lag1_normrank": frozen["lag1_normrank"],
                    "full_distinct2": full["distinct2"],
                    "longest_token_streak": full["longest_token_streak"],
                })

                done += 1
                if done % 10 == 0:
                    print(f"completed {done}/{total}", flush=True)

    df = pd.DataFrame(rows)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    summary = (
        df.groupby("rho")
        .agg(
            source_corr=("source_corr", "mean"),
            full_lag1_cdf=("full_lag1_cdf", "mean"),
            frozen_lag1_cdf=("frozen_lag1_cdf", "mean"),
            excess_lag1_cdf=("excess_lag1_cdf", "mean"),
            full_distinct2=("full_distinct2", "mean"),
            longest_token_streak=("longest_token_streak", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(out.with_name(out.stem + "_summary.csv"), index=False)

    # Prompt-level paired slope: subtract frozen from full before fitting rho.
    slope, lo, hi = bootstrap_prompt_slopes(
        df, "excess_lag1_cdf", n_boot=args.bootstrap, seed=args.seed
    )

    # Also bootstrap the full and frozen slopes for direct comparison.
    full_slope = bootstrap_prompt_slopes(
        df.rename(columns={"full_lag1_cdf": "tmp"}), "tmp",
        n_boot=args.bootstrap, seed=args.seed + 1
    )
    frozen_slope = bootstrap_prompt_slopes(
        df.rename(columns={"frozen_lag1_cdf": "tmp"}), "tmp",
        n_boot=args.bootstrap, seed=args.seed + 2
    )

    rho025 = df[df["rho"] == 0.25]["excess_lag1_cdf"]
    rho025_mean = float(rho025.mean())

    report = {
        "n_rows": int(len(df)),
        "n_prompts": int(df["prompt_id"].nunique()),
        "n_seeds": int(df["seed"].nunique()),
        "paired_excess_slope": slope,
        "paired_excess_slope_ci95": [lo, hi],
        "full_slope": full_slope[0],
        "full_slope_ci95": [full_slope[1], full_slope[2]],
        "frozen_slope": frozen_slope[0],
        "frozen_slope_ci95": [frozen_slope[1], frozen_slope[2]],
        "excess_fraction_of_full_slope": (
            slope / full_slope[0] if full_slope[0] else float("nan")
        ),
        "rho_0.25_excess_mean": rho025_mean,
    }

    import json
    with open(out.with_name(out.stem + "_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n=== SUMMARY ===")
    print(summary.to_string(index=False))
    print("\n=== PAIRED SLOPE ===")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
