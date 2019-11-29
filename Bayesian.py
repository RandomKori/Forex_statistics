from theano.printing import Print
import pymc3 as pm
import numpy as np
import theano.tensor as T


def covariance(sigma, rho):
    C = T.fill_diagonal(T.alloc(rho, 2, 2), 1.)
    S = T.diag(sigma)
    M = S.dot(C).dot(S)
    return M


def analyze_standard(data):
    with pm.Model() as model:
        # priors
        sigma = pm.Uniform('sigma', lower=0, upper=0.001, shape=2,
                           testval=[0.0001, 0.001],  # init with mad
                           transform=None)
        rho = pm.Uniform('r', lower=-1, upper=1,
                         testval=-0.2144021,  # init with Spearman's correlation
                         transform=None)

        # print values for debugging
        rho_p = rho
        sigma_p = sigma

        cov = pm.Deterministic('cov', covariance(sigma_p, rho_p))
        cov_p = cov

        mult_norm = pm.MvNormal('mult_norm', mu=[-100, 100],  # set mu to median
                                cov=cov_p, observed=data.T)

    return model


def analyze_robust(data):
    with pm.Model() as model:
        # priors
        mu = pm.Normal('mu', mu=0., tau=0.000001, shape=2,
                       testval=np.array([-100, 100]))  # set mu to median
        sigma = pm.Uniform('sigma', lower=0, upper=0.001, shape=2,
                           testval=np.array([0.0001, 0.001]),  # init with mad
                           transform=None)
        rho = pm.Uniform('r', lower=-1, upper=1,
                         testval=-0.2144021,  # init with Spearman's correlation
                         transform=None)

        # print values for debugging
        rho_p = rho
        sigma_p = sigma

        cov = pm.Deterministic('cov', covariance(sigma_p, rho_p))
        num = pm.Exponential('nu_minus_one', lam=1. / 29., testval=1)
        nu = pm.Deterministic('nu', num + 1)
        cov_p = cov
        nu_p = nu

        mult_norm = pm.MvStudentT('mult_norm', nu=nu_p, mu=mu,
                                  Sigma=cov_p, observed=data.T)

    return model