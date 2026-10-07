"""Analyze a saved trace for the sampler-only null."""
import argparse, json
from pathlib import Path
import numpy as np
from temporal_sampling_correlation.nulls import decoupled_lag_stat, gaussian_copula_uniform_corr

def main():
    p=argparse.ArgumentParser()
    p.add_argument("trace",type=Path); p.add_argument("--rho",type=float,required=True)
    p.add_argument("--permutations",type=int,default=5000); p.add_argument("--seed",type=int,default=0)
    a=p.parse_args(); d=np.load(a.trace); u=d["u"]; cdf=d["cdf"]
    observed=float(np.nanmean([np.corrcoef(u[i,:-1],cdf[i,1:])[0,1] for i in range(u.shape[0])]))
    source=float(np.nanmean([np.corrcoef(u[i,:-1],u[i,1:])[0,1] for i in range(u.shape[0])]))
    rng=np.random.default_rng(a.seed); null=np.array([decoupled_lag_stat(u,cdf,int(rng.integers(2**31))) for _ in range(a.permutations)])
    print(json.dumps({"rho":a.rho,"observed_mean_lag1_cdf":observed,"source_mean_lag1_uniform":source,
      "gaussian_copula_prediction":gaussian_copula_uniform_corr(a.rho),"null_mean":float(null.mean()),
      "null_sd":float(null.std(ddof=1)),"null_p_two_sided":float((1+np.sum(np.abs(null)>=abs(observed)))/(len(null)+1))},indent=2))
if __name__=="__main__": main()
