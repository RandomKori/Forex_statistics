import numpy as np
import pymc3 as pm

def cauchyvar(data):
    with pm.Model() as model:
        alfa=pm.Uniform('alfa',lower=-1.0,upper=1.0)
        beta=pm.Uniform('beta',lower=0.0,upper=200.0)
        cauchy=pm.Cauchy('cauchy',alpha=alfa,beta=beta,observed=data)
    return model
