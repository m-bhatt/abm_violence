
import numpy as np
import jax.random as jrand
from scipy.special import gammaln
def poisson_ll(n, mu):
    if(mu == 0):
        if n == 0:
            return 0
        else:
            return -np.inf
    return n*np.log(mu)-mu-gammaln(n+1)

from gvabm.stat_utils import grenander_pdf
from gvabm.abm_event import conditional_sample

def city_ll(city_params, event_list, general_params, timespan, event_distr):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_rate = general_params
    prob_event = 1-event_distr.cdf(5.0)
    pll = poisson_ll(len(event_list), propensity*timespan*population*prob_event)
    eventll = gammaln(len(event_list)+1)
    for event in event_list:
        if(prob_event == 0):
            return -np.inf, -np.inf, -np.inf
        fatalities = event.fatalities
        eventll += (np.log(event_distr.pdf(fatalities)) - np.log(prob_event))
    return pll + eventll, pll, eventll

def event_rate(city_params, general_params, timespan, event_distr):
    population, population_density, arm_count, arm_density = city_params
    propensity, walk_radius, arm_bias, arm_weight, atrisk_rate = general_params
    prob_event = 1-event_distr.cdf(5.0)
    return propensity*timespan*population*prob_event

# cl = city_ll(c_params, c_events, params, 50, city_distr)
def eval_ds(city_data, general_params, timespan, rng_key, num_samples=10000, sample_func=conditional_sample):
    ll_data = []
    for c in city_data:
        rng_key, subkey = jrand.split(rng_key, 2)
        c_params = c['params']
        c_events = c['events']
        event_sim = sample_func(c_params, general_params, subkey, num_samples=num_samples)
        city_distr = grenander_pdf(event_sim+1)
        ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
        ll_data.append([ll, pll, eventll])
    return np.array(ll_data)