import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_event import sample_event, conditional_sample
from gvabm.abm_ll import city_ll, eval_ds
from gvabm.stat_utils import CityParams, EventOutcome, GeneralParams, logrange
from gvabm.city_data import subsampled_city_data, subsample_weights
from gvabm.param_distr import ParamDistr

import logging
#set log file location and level
# logging.basicConfig(filename='abm_vbmc.log', level=logging.INFO)
#just make logging to console for now
logging.basicConfig(level=logging.INFO)
# logging.getLogger('abm_ll').setLevel(logging.WARNING)

param_config = {
    "propensity_range": {"range": (-9, -5), "logscale": True},
    "walk_radius_range": {"range": (1, 95), "logscale": False},
    "arm_bias_range": {"range": (-3, 1), "logscale": True},
    "arm_weight_range": {"range": (-1, 1), "logscale": True},
    "atrisk_gather_rate_range": {"range": (-8, -4), "logscale": True}
}

def eval_params(rngkey, params, param_distr, num_samples, data_dict=None):
    params = param_distr.transform_sample(params)
    params = GeneralParams(*params)
    city_eval = eval_ds(subsampled_city_data, params, 50, rngkey, num_samples=num_samples)
    if data_dict is not None:
        data_dict[params] = city_eval
    total_eval = (subsample_weights.reshape((-1, 1))*city_eval).sum(axis=0)[0]
    return total_eval

param_distr = ParamDistr(param_config)
# params = param_distr.sample_range(jrand.PRNGKey(1))
params = GeneralParams(propensity=2e-09, walk_radius=40, arm_bias=0.01, arm_weight=1, atrisk_gathering_rate=1.02e-05)
params = jnp.array([params.propensity, params.walk_radius, params.arm_bias, params.arm_weight, params.atrisk_gathering_rate])
params = param_distr.inverse_transform_sample(params)

def get_param_density_func(param_prior, rngkey, num_samples, data_dict=None):
    def param_density_func(params):
        # print("Evaluating params:", params)
        params = jnp.array(params)
        nonlocal rngkey
        # prior = param_prior(params)
        # if prior < 0: #For uniform prior, prior < 0 indicates -inf, so return -2e3
        #     return -1000
        rngkey, subkey = jrand.split(rngkey)
        params = param_distr.transform_sample(params)
        params = GeneralParams(*np.array(params))
        city_eval = eval_ds(subsampled_city_data, params, 50, subkey, num_samples=num_samples)
        if( data_dict is not None):
            data_dict[params] = city_eval
        total_eval = (subsample_weights.reshape((-1, 1))*city_eval).sum(axis=0)[0]
        res = total_eval #+ prior
        if jnp.isinf(res):
            # print("Returning -2e3 for params:", params)
            return -1000
        # print('Outputing log posterior:', res, ' for params:', params, 'with std', 5)
        logging.info(f"Evaluating params: {params}")
        logging.info(f"Log posterior: {res}")
        return max(res, -1000) #VBMC expects a tuple of (value, noise)
    return param_density_func

def get_param_density_func_noisy(param_prior, rngkey, num_samples, data_dict=None):
    def param_density_func(params):
        # print("Evaluating params:", params)
        params = jnp.array(params)
        nonlocal rngkey
        prior = param_prior(params)
        if prior < 0: #For uniform prior, prior < 0 indicates -inf, so return -2e3
            return -1000, 3
        rngkey, subkey = jrand.split(rngkey)
        params = param_distr.transform_sample(params)
        params = GeneralParams(*np.array(params))
        city_eval = eval_ds(subsampled_city_data, params, 50, subkey, num_samples=num_samples)
        if( data_dict is not None):
            data_dict[params] = city_eval
        total_eval = (subsample_weights.reshape((-1, 1))*city_eval).sum(axis=0)[0]
        res = total_eval + prior
        if jnp.isinf(res):
            # print("Returning -2e3 for params:", params)
            return -1000, 3
        # print('Outputing log posterior:', res, ' for params:', params, 'with std', 5)
        logging.info(f"Evaluating params: {params}")
        logging.info(f"Log posterior: {res}")
        if(res < -1000):
            return -1000, 2
        return res, 0.8 #VBMC expects a tuple of (value, noise)
    return param_density_func