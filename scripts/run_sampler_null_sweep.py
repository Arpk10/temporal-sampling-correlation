"""Small sampler-only sanity sweep: 5 prompts x 5 rho x 10 seeds.

Run on a CUDA machine. The model context is frozen along a deterministic
reference trajectory, so this isolates the inverse-CDF sampler pathway.
"""
import argparse, json, sys, types
from pathlib import Path

# Transformers 5.x imports sklearn.metrics.roc_curve for optional generation
# utilities even though this diagnostic never calls generation helpers. The
# local Windows policy blocks SciPy DLL loading, so provide the tiny optional
# symbol needed during import rather than changing the user environment.
import importlib.machinery
_metrics = types.ModuleType("sklearn.metrics")
_metrics.__spec__ = importlib.machinery.ModuleSpec("sklearn.metrics", loader=None)
_metrics.roc_curve = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("roc_curve is unavailable in sampler-only diagnostic"))
_sklearn = types.ModuleType("sklearn")
_sklearn.__path__ = []
_sklearn.__spec__ = importlib.machinery.ModuleSpec("sklearn", loader=None, is_package=True)
_sklearn.metrics = _metrics
sys.modules["sklearn"] = _sklearn
sys.modules["sklearn.metrics"] = _metrics
import numpy as np
import pandas as pd
import torch
from transformers import GPT2Config, GPT2TokenizerFast
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

from temporal_sampling_correlation.sampler import ar1_uniform, sample_top_p
from temporal_sampling_correlation.prompts import PROMPTS, RHO_GRID

def one(model, tok, prompt, rho, seed, device, n_new=40, temperature=1.0, top_p=.9):
    ids=tok(prompt,return_tensors="pt").input_ids.to(device)
    with torch.no_grad(): out=model(input_ids=ids,use_cache=True)
    past=out.past_key_values; logits=out.logits[:,-1,:]
    fixed=[]
    for _ in range(n_new):
        fixed.append(logits.detach().clone())
        token=torch.argmax(logits,dim=-1)
        with torch.no_grad(): out=model(input_ids=token[:,None],past_key_values=past,use_cache=True)
        past=out.past_key_values; logits=out.logits[:,-1,:]
    u=ar1_uniform(rho,n_new,seed)
    c=[]; rank=[]
    for t in range(n_new):
        _,_,sel,_,nr,_=sample_top_p(fixed[t],torch.tensor([u[t]],device=device),temperature,top_p)
        c.append(float(sel[0])); rank.append(float(nr[0]))
    c=np.asarray(c)
    return {
      "prompt_id": PROMPTS.index(prompt), "rho": rho, "seed": seed,
      "source_corr": float(np.corrcoef(u[:-1],u[1:])[0,1]),
      "lag1_cdf": float(np.corrcoef(u[:-1],c[1:])[0,1]),
      "lag1_normrank": float(np.corrcoef(u[:-1],np.asarray(rank)[1:])[0,1]),
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--device",default="cuda")
    p.add_argument("--output",default="results/sampler_null_sweep.csv")
    p.add_argument("--prompts",type=int,default=5)
    p.add_argument("--seeds",type=int,default=10)
    a=p.parse_args()
    if not torch.cuda.is_available() and a.device.startswith("cuda"): raise RuntimeError("CUDA unavailable")
    tok=GPT2TokenizerFast.from_pretrained("gpt2")
    model=GPT2LMHeadModel.from_pretrained("gpt2").to(a.device).eval()
    rows=[]
    for pid,prompt in enumerate(PROMPTS[:a.prompts]):
        for rho in RHO_GRID:
            for seed in range(a.seeds):
                rows.append(one(model,tok,prompt,rho,seed,a.device))
                print(f"prompt={pid} rho={rho:.2f} seed={seed}",flush=True)
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(out,index=False)
    summary=pd.DataFrame(rows).groupby("rho")[["source_corr","lag1_cdf","lag1_normrank"]].mean().reset_index()
    summary.to_csv(out.with_name(out.stem+"_summary.csv"),index=False)
    print(summary.to_string(index=False))
if __name__=="__main__": main()
