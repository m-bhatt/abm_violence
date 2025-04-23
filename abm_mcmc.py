import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from abm_event import sample_event, conditional_sample
from abm_ll import city_ll, eval_ds
from stat_utils import CityParams, EventOutcome, GeneralParams, logrange
from city_data import city_data_base, subsampled_city_data, subsample_weights

param_config = {
    "propensity_range": {"range": (-9, -5), "logscale": True},
    "walk_radius_range": {"range": (1, 95), "logscale": False},
    "arm_bias_range": {"range": (-3, 1), "logscale": True},
    "arm_weight_range": {"range": (-1, 1), "logscale": True},
    "atrisk_gather_rate_range": {"range": (-8, -4), "logscale": True}
}

class ParamDistr:
    def __init__(self, param_config):
        self.param_config = param_config
        self.range_lower = jnp.array([config["range"][0] for config in param_config.values()])
        self.range_upper = jnp.array([config["range"][1] for config in param_config.values()])
        self.range_width = self.range_upper - self.range_lower
        self.logscale = jnp.array([config["logscale"] for config in param_config.values()])
        self.param_names = list(param_config.keys())
        self.param_count = len(param_config)

    def sample_range(self, key):
        return jrand.uniform(key, shape=(self.param_count,)) * (self.range_upper - self.range_lower) + self.range_lower
    
    def transform_sample(self, sample):
        return jnp.where(self.logscale, jnp.exp(sample), sample)
    def inverse_transform_sample(self, sample):
        return jnp.where(self.logscale, jnp.log(sample), sample)
    def get_gaussian_proposal(self):
        return lambda key, params : params + jrand.normal(key, shape=(self.param_count,))*self.range_width*0.05
    
    def get_log_gaussian_proposal_pdf(self):
        def log_gaussian_proposal_pdf(p1, p2):
            return -jnp.sum((p1 - p2)**2 / (2 * (0.05 * self.range_width)**2))
        return log_gaussian_proposal_pdf
    
    def get_uniform_prior(self):
        def uniform_prior(p):
            return jnp.all((p >= self.range_lower) & (p <= self.range_upper))+1e-3
        return uniform_prior
    
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
# eval_ll = eval_params(jrand.PRNGKey(1), params, param_distr, 10000)

def get_param_density_func(param_prior, rngkey, num_samples, data_dict=None):
    def param_density_func(params):
        nonlocal rngkey
        prior = param_prior(params)
        if prior < 0:
            return -1e10
        rngkey, subkey = jrand.split(rngkey)
        params = param_distr.transform_sample(params)
        params = GeneralParams(*np.array(params))
        city_eval = eval_ds(subsampled_city_data, params, 50, subkey, num_samples=num_samples)
        # print(params)
        if( data_dict is not None):
            data_dict[params] = city_eval
        total_eval = (subsample_weights.reshape((-1, 1))*city_eval).sum(axis=0)[0]
        return total_eval + prior
    return param_density_func
data_dict = {}
P = get_param_density_func(param_distr.get_uniform_prior(), jrand.PRNGKey(1), int(1e4), data_dict={})
proposal_func = param_distr.get_gaussian_proposal()
Q = param_distr.get_log_gaussian_proposal_pdf()

def save_data(samples, sample_eval):
    if(len(samples) % 10 != 0):
        return
    with open('data/mcmc_samples.pkl', 'wb') as f:
        pickle.dump((samples, sample_eval, data_dict), f)
    print("Saved data")

from mcmc import mcmc, mc_update #, gaussian_proposal, gaussian_proposal_pdf
samples, sample_eval = mcmc(params, proposal_func, P, Q, 
                            rng=jrand.PRNGKey(0), num_iter=int(1e2), log=True, callback=save_data)