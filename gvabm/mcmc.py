import numpy as np
import jax.random as jrand
import pickle
from stat_utils import grenander_pdf, GeneralParams
import jax.random as jrand
from abm_event import conditional_sample
from abm_ll import city_ll
import math

def mcmc(params, proposal_func, P, Q, rng, num_iter=10000, log=False, callback=None):
    """
    params: starting vector of parameters

    proposal_func: a function that accepts a vector of parameters and an rng and generates a single value, as the proposed next state, as well as
    the perturbed parameters

    P: a function that accepts a single value (state) and generates a posterior: prior * likelihood, NOT the log likelihood or it will not work :( 
    but np.exp(a log likelihood) within P should be okay

    Q: a function that accepts two states - the proposed state and the previous accepted state, and generates the pdf under the proposal function

    log: True if want to use log likelihood etc (i think)
    """
    samples = np.array([])
    estimated_likelihoods = np.array([])
    xt = params # xt = current state
    Pxt = P(xt)
    for i in range(num_iter):
        rng, subkey = jrand.split(rng)
        xt, Pxt = mc_update(xt, Pxt, proposal_func, P, Q, subkey, log=log)
        print(f"Accepted: {xt}, P={Pxt}")
        samples = np.append(samples, xt)
        estimated_likelihoods = np.append(estimated_likelihoods, Pxt)
        if callback is not None:
            callback(samples, estimated_likelihoods)
    return samples, estimated_likelihoods

def mc_update(xt, Pxt, proposal_func, P, Q, rng, log=True):
    k1, k2 = jrand.split(rng)
    proposed = proposal_func(k1, xt)
    P_proposed = P(proposed)
    # print(f"{xt} (P=({Pxt})) -> {proposed} (P={P_proposed})")
    if log==True:
        # log_delta = P_proposed - Pxt + Q(xt, proposed) - Q(proposed, xt)
        # print(f"Log delta: {log_delta}")
        acceptance_ratio = min(1, 
                               np.exp( P_proposed - Pxt + Q(xt, proposed) - Q(proposed, xt) ) ) 
    else:
        acceptance_ratio = min(1, P_proposed * Q(xt, proposed) / (Pxt * Q(proposed, xt)))
        # acceptance_ratio = min(1, P_proposed / Pxt)
        # acceptance_ratio = min(1, Pxt / P_proposed)
    # print(f"Acceptance ratio: {acceptance_ratio}")
    p = jrand.uniform(k2)
    if p > acceptance_ratio:
        return xt, Pxt
    return proposed, P_proposed

import scipy.stats
def gaussian_proposal_pdf(x, xt):
    return scipy.stats.norm(loc=xt, scale=0.3).pdf(x) # normal with scale=1, mean=mu, mu=xt (prev value), x=x

def gaussian_proposal(rng, x):
    v = np.random.normal(x, 0.3)
    return v

def proposal_pdf(mu):
    return 1/2


