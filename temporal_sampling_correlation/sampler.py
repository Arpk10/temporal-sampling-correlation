import math

import numpy as np
import torch

from .prompts import SEEDS, N_NEW, TEMPERATURE, TOP_P


def ar1_uniform(rho: float, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    z = np.empty(n, dtype=np.float64)
    z[0] = rng.normal()
    scale = math.sqrt(max(0.0, 1.0 - rho * rho))
    for t in range(1, n):
        z[t] = rho * z[t - 1] + scale * rng.normal()
    return np.array(
        [0.5 * (1.0 + math.erf(v / math.sqrt(2.0))) for v in z],
        dtype=np.float64,
    )


def safe_corr(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(np.corrcoef(a, b)[0, 1])


def sample_top_p(
    logits: torch.Tensor,
    u: torch.Tensor,
    temperature: float = TEMPERATURE,
    top_p: float = TOP_P,
):
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    sorted_probs, sorted_idx = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)

    keep = cumsum <= top_p
    keep[:, 0] = True

    truncated = sorted_probs * keep
    truncated = truncated / truncated.sum(dim=-1, keepdim=True)
    truncated_cdf = torch.cumsum(truncated, dim=-1)

    tokens = []
    selected_probs = []
    selected_cdf = []
    ranks = []
    entropies = []

    vocab_size = logits.shape[-1]

    for b in range(logits.shape[0]):
        k = torch.searchsorted(truncated_cdf[b], u[b], right=False)
        k = torch.clamp(k, max=vocab_size - 1)

        token = sorted_idx[b, k]
        p = probs[b, token]
        rank = 1 + torch.sum(probs[b] > p)
        entropy = -torch.sum(
            probs[b] * torch.log(torch.clamp(probs[b], min=1e-12))
        )

        tokens.append(token)
        selected_probs.append(p)
        selected_cdf.append(truncated_cdf[b, k])
        ranks.append(rank)
        entropies.append(entropy)

    tokens = torch.stack(tokens)
    selected_probs = torch.stack(selected_probs)
    selected_cdf = torch.stack(selected_cdf)
    ranks = torch.stack(ranks)
    entropies = torch.stack(entropies)

    normalized_rank = (ranks.float() - 1.0) / float(vocab_size - 1)

    return tokens, selected_probs, selected_cdf, ranks, normalized_rank, entropies


def generate_condition(
    prompt: str,
    rho: float,
    seeds=SEEDS,
    n_new=N_NEW,
    device="cuda",
    model_name="gpt2",
    temperature=TEMPERATURE,
    top_p=TOP_P,
    prompt_id=None,
):
    if not torch.cuda.is_available() and str(device).startswith("cuda"):
        raise RuntimeError("CUDA is unavailable.")

    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name).to(device).eval()

    prompt_tokens = tokenizer(
        [prompt] * len(seeds), return_tensors="pt"
    ).input_ids.to(device)

    u_np = np.stack([ar1_uniform(rho, n_new, seed) for seed in seeds], axis=0)
    u_torch = torch.tensor(u_np, dtype=torch.float32, device=device)

    with torch.no_grad():
        outputs = model(input_ids=prompt_tokens, use_cache=True)

    past = outputs.past_key_values
    logits = outputs.logits[:, -1, :]

    token_history = []
    probs_all = []
    cdf_all = []
    normrank_all = []
    entropy_all = []

    for t in range(n_new):
        (
            next_token,
            selected_prob,
            selected_cdf,
            _rank,
            normalized_rank,
            entropy,
        ) = sample_top_p(
            logits,
            u_torch[:, t],
            temperature=temperature,
            top_p=top_p,
        )

        token_history.append(next_token.detach().cpu().numpy())
        probs_all.append(selected_prob.detach().cpu().numpy())
        cdf_all.append(selected_cdf.detach().cpu().numpy())
        normrank_all.append(normalized_rank.detach().cpu().numpy())
        entropy_all.append(entropy.detach().cpu().numpy())

        with torch.no_grad():
            outputs = model(
                input_ids=next_token.unsqueeze(1),
                past_key_values=past,
                use_cache=True,
            )

        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]

    probs_np = np.stack(probs_all, axis=1)
    cdf_np = np.stack(cdf_all, axis=1)
    normrank_np = np.stack(normrank_all, axis=1)
    entropy_np = np.stack(entropy_all, axis=1)

    rows = []
    for b, seed in enumerate(seeds):
        tokens = [int(token_history[t][b]) for t in range(n_new)]
        bigrams = list(zip(tokens[:-1], tokens[1:]))
        distinct2 = len(set(bigrams)) / max(1, len(bigrams))

        rows.append(
            {
                "prompt": prompt,
                "prompt_id": prompt_id,
                "seed": seed,
                "rho": rho,
                "lag1_cdf": safe_corr(u_np[b, :-1], cdf_np[b, 1:]),
                "lag2_cdf": safe_corr(u_np[b, :-2], cdf_np[b, 2:]),
                "lag1_normrank": safe_corr(u_np[b, :-1], normrank_np[b, 1:]),
                "lag2_normrank": safe_corr(u_np[b, :-2], normrank_np[b, 2:]),
                "lag1_prob": safe_corr(u_np[b, :-1], probs_np[b, 1:]),
                "lag2_prob": safe_corr(u_np[b, :-2], probs_np[b, 2:]),
                "entropy": float(np.mean(entropy_np[b])),
                "distinct2": float(distinct2),
                "mean_logp": float(np.log(np.clip(probs_np[b], 1e-12, None)).mean()),
                "token_ids": tokens,
            }
        )

    return rows
