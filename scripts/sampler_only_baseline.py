"""Frozen-context sampler-only baseline.

A reference token trajectory supplies one fixed model distribution per step.
The distributions are never updated in response to the tested u sequence.
Structured and shuffled uniforms are then passed through the same top-p
sampler. Any rho-dependent lag statistic here is therefore attributable to
the sampling interface, not autoregressive feedback.
"""
import argparse, json
from pathlib import Path
import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

from temporal_sampling_correlation.sampler import ar1_uniform, sample_top_p
from temporal_sampling_correlation.nulls import gaussian_copula_uniform_corr

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--prompt",required=True); p.add_argument("--rho",type=float,required=True)
    p.add_argument("--seed",type=int,default=0); p.add_argument("--n-new",type=int,default=40)
    p.add_argument("--model-name",default="gpt2"); p.add_argument("--device",default="cuda")
    p.add_argument("--temperature",type=float,default=1.0); p.add_argument("--top-p",type=float,default=.9)
    p.add_argument("--reference-seed",type=int,default=0)
    a=p.parse_args()
    tok=GPT2TokenizerFast.from_pretrained(a.model_name)
    model=GPT2LMHeadModel.from_pretrained(a.model_name).to(a.device).eval()
    ids=tok(a.prompt,return_tensors="pt").input_ids.to(a.device)
    with torch.no_grad():
        out=model(input_ids=ids,use_cache=True)
    past=out.past_key_values; logits=out.logits[:,-1,:]
    fixed=[]
    for _ in range(a.n_new):
        fixed.append(logits.detach().cpu())
        # Reference context uses greedy decoding, independent of tested u/rho.
        token=torch.argmax(logits,dim=-1)
        with torch.no_grad(): out=model(input_ids=token[:,None],past_key_values=past,use_cache=True)
        past=out.past_key_values; logits=out.logits[:,-1,:]
    u=ar1_uniform(a.rho,a.n_new,a.seed)
    ut=torch.tensor(u,dtype=torch.float32,device=a.device)
    c=[]
    for t,lg in enumerate(fixed):
        _,_,sel,_,_,_=sample_top_p(lg.to(a.device),ut[t:t+1],a.temperature,a.top_p)
        c.append(float(sel[0].item()))
    observed=float(np.corrcoef(u[:-1],np.asarray(c)[1:])[0,1])
    rng=np.random.default_rng(a.reference_seed); ush=u.copy(); rng.shuffle(ush)
    cn=[]
    for t,lg in enumerate(fixed):
        _,_,sel,_,_,_=sample_top_p(lg.to(a.device),torch.tensor([ush[t]],device=a.device),a.temperature,a.top_p)
        cn.append(float(sel[0].item()))
    shuffled=float(np.corrcoef(ush[:-1],np.asarray(cn)[1:])[0,1])
    print(json.dumps({"rho":a.rho,"observed":observed,"shuffled":shuffled,
      "source_uniform_corr":float(np.corrcoef(u[:-1],u[1:])[0,1]),
      "gaussian_copula_prediction":gaussian_copula_uniform_corr(a.rho)},indent=2))
if __name__=="__main__": main()
