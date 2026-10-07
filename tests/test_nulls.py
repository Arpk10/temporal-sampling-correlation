import numpy as np
from temporal_sampling_correlation.nulls import decoupled_uniforms, gaussian_copula_uniform_corr

def test_gaussian_copula_benchmark():
    assert abs(gaussian_copula_uniform_corr(.5)-(6/np.pi)*np.arcsin(.25))<1e-12

def test_decoupled_uniforms_preserves_values():
    u=np.array([[.1,.2,.3,.4],[.5,.6,.7,.8]])
    out=decoupled_uniforms(u,seed=7)
    assert np.allclose(np.sort(out,axis=1),np.sort(u,axis=1))
    assert out.shape==u.shape
