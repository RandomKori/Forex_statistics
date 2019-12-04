import numpy as np
import pymc3 as pm

def cauchyvar(data):
    with pm.Model() as model:
        alfa=pm.Uniform('alfa',lower=-1.0,upper=1.0)
        beta=pm.Uniform('beta',lower=0.0,upper=200.0)
        cauchy=pm.Cauchy('cauchy',alpha=alfa,beta=beta,observed=data)
    return model

def verbar(data):
    v=0
    n=0
    ni=0
    for i in range(len(data)):
        if data[i]==0.0:
            ni=ni+1
        else:
            if data[i]>0.0:
                v=v+1
            else:
                n=n+1
    vr=v/len(data)
    nr=n/len(data)
    nir=ni/len(data)
    return vr,nr,nir

def vert(data):
    g=0
    for i in range(len(data)):
        if data[i]!=0.0:
            g=g+1
    v=g/len(data)
    return v