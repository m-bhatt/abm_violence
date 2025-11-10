import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_event import sample_event, conditional_sample
from gvabm.abm_ll import city_ll, eval_ds
from gvabm.city_data import subsampled_city_data, subsample_weights
from gvabm.stat_utils import logrange
from gvabm.param_distr import ParamDistr, param_config, CityParams, EventOutcome, GeneralParams, MixedGeneralParams

#Sample func recieves inputs normalized to [0,1] range. Outputs log-likelihood
def get_param_density_func_noisy(sample_func, param_distr, log_prior, rngkey, num_samples, cutoff_likelihood=-1100, likelihood_offset=650, logdir=None):
    if logdir is not None:
        logging.basicConfig(filename=f'{logdir}/density_eval.log', level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    else:
        #log to stdout
        logging.basicConfig(level=logging.INFO)
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
    likelihood_offset = 550
    def param_density_func(params):
        nonlocal rngkey
        nonlocal logger
        rngkey, subkey = jrand.split(rngkey)
        params = jnp.array(params)
        logger.info(f"Evaluating params: {params}")
        prior = log_prior(params)
        if prior < -15: #For uniform prior, prior < -15 indicates -inf, so return -1e3
            logger.info(f"Log prior: {prior}")
            return cutoff_likelihood + likelihood_offset, 3
        params = param_distr.transform_sample(params)
        params = param_distr.to_data(np.array(params))
        logger.info(f"Transformed params: {params}, prior: {prior}")
        city_eval = eval_ds(subsampled_city_data, params, 10, subkey, num_samples=num_samples, sample_func=sample_func)
        total_eval = (subsample_weights.reshape((-1, 1))*city_eval).sum(axis=0)[0]
        res = total_eval + prior
        if jnp.isinf(res) or jnp.isnan(res):
            return cutoff_likelihood + likelihood_offset, 3
        logger.info(f"Log posterior: {res}")
        if(res < cutoff_likelihood):
            return cutoff_likelihood + likelihood_offset, 3
        return float(res + likelihood_offset), 1.5 #VBMC expects a tuple of (value, noise)
    return param_density_func

def get_param_density_func(sample_func, param_distr, log_prior, rngkey, num_samples):
    def param_density_func(params):
        nonlocal rngkey
        params = jnp.array(params)
        rngkey, subkey = jrand.split(rngkey)
        params = param_distr.transform_sample(params)
        params = param_distr.to_data(np.array(params))
        city_eval = eval_ds(subsampled_city_data, params, 10, subkey, num_samples=num_samples, sample_func=sample_func)
        total_eval = (subsample_weights.reshape((-1, 1))*city_eval).sum(axis=0)[0]
        res = total_eval + log_prior(params)
        if jnp.isinf(res): 
            #return negative infinity
            return -np.inf
        return float(res)
    return param_density_func