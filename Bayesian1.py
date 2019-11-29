import pymc3 as pm
import numpy as np
import theano.tensor as T

def mad(data, axis=None):
    return np.median(np.absolute(data - np.median(data, axis)), axis)

def covariance(sigma, rho):
    C = T.alloc(rho, 2, 2)
    C = T.fill_diagonal(C, 1.)
    S = T.diag(sigma)
    return S.dot(C).dot(S)

def analyze_robust1(data):
    with pm.Model() as model:
        # priors might be adapted here to be less flat
        mu = pm.Normal('mu', mu=0., sd=100., shape=2, testval=np.median(data.T, axis=1))
        bound_sigma = pm.Bound(pm.Normal, lower=0.)
        sigma = bound_sigma('sigma', mu=0., sd=100., shape=2, testval=mad(data, axis=0))
        rho = pm.Uniform('r', lower=-1., upper=1., testval=0)
        cov = pm.Deterministic('cov', covariance(sigma, rho))
        bound_nu = pm.Bound(pm.Gamma, lower=1.)
        nu = bound_nu('nu', alpha=2, beta=10)
        mult_t = pm.MvStudentT('mult_t', nu=nu, mu=mu, Sigma=cov, observed=data)
    return model
