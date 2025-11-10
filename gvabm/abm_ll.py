import time
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

from gvabm.stat_utils import grenander_pdf, EmpiricalCDF, hybrid_pdf, grenander_pdf_numba
from gvabm.abm_event import conditional_sample

def city_ll(city_params, event_list, general_params, timespan, event_distr, report_threshold=4.0):
    year, population, population_density, arm_count, arm_density = city_params
    # propensity, walk_radius, arm_bias, arm_weight, atrisk_rate = general_params
    propensity = general_params[0]
    prob_event = (1-event_distr.cdf(report_threshold) + 1e-10)
    pll = poisson_ll(len(event_list), propensity*timespan*population*prob_event)
    eventll = gammaln(len(event_list)+1)
    for event in event_list:
        if(prob_event == 0):
            return -np.inf, -np.inf, -np.inf
        fatalities = event.fatalities
        eventll += (np.log(event_distr.pdf(fatalities)) - np.log(prob_event))
    return pll + eventll, pll, eventll

def event_rate(city_params, general_params, timespan, event_distr):
    year, population, population_density, arm_count, arm_density = city_params
    # propensity, walk_radius, arm_bias, arm_weight, atrisk_rate = general_params
    propensity = general_params[0]
    prob_event = 1-event_distr.cdf(4.0)
    return propensity*timespan*population*prob_event

# import jax.numpy as jnp
# import jax.random as jrand

# def batched_eval_ds(city_data, general_params, timespan, rng_key, num_samples=10000, 
#                     sample_func=conditional_sample, batch_size=int(1e6)):
#     propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate = general_params
#     def params_stream():
#         for population, population_density, arm_count, arm_density in city_data:
#             a = jnp.array([walk_radius, population_density, arm_density*arm_weight+arm_bias, atrisk_gathering_rate])
#             yield jnp.tile(a.reshape((1, -1)), num_samples)
            
#     rng_batch = jrand.split(rng_key, num_samples)
#     event_samples = batch_sample(jnp.full(num_samples, walk_radius), jnp.full(num_samples, population_density), 
#                                  jnp.full(num_samples, arm_density*arm_weight+arm_bias), 
#                                  jnp.full(num_samples, atrisk_gathering_rate), rng_batch)
#     pseudocount_samples = jnp.concatenate([event_samples, jnp.array([10, 100, 1000])])

# def eval_ds_evd(city_data, general_params, timespan, rng_key, nonevent_num_samples=5000, 
#                 event_num_samples=10000, sample_func=conditional_sample):
#     ll_data = []
#     sample_time = 0
#     grenander_time = 0
#     ll_time = 0

#     for c in city_data:
#         rng_key, subkey = jrand.split(rng_key, 2)
#         c_params = c['params']
#         c_events = c['events']
#         if len(c_events) == 0: #special handling for case when no pdf queries are needed
#             t0 = time.time()
#             event_sim = sample_func(c_params, general_params, subkey, num_samples=nonevent_num_samples)
#             sample_time += time.time() - t0
#             t0 = time.time()
#             city_distr = grenander_pdf_numba(event_sim+1)
#             grenander_time += time.time() - t0
#         else:
#             t0 = time.time()
#             event_sim = sample_func(c_params, general_params, subkey, num_samples=event_num_samples)
#             sample_time += time.time() - t0
#             t0 = time.time()
#             city_distr = grenander_pdf_numba(event_sim+1)
#             grenander_time += time.time() - t0
#         t0 = time.time()
#         ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
#         ll_time += time.time() - t0
#         ll_data.append([ll, pll, eventll])
#     print(f"Sampling time: {sample_time}, Grenander time: {grenander_time}, LL time: {ll_time}")
#     return np.array(ll_data)

from gvabm.abm_event import conditional_sample_batch
from gvabm.stat_utils import grenander_pdf_numba
def eval_ds(city_data, general_params, timespan, rng_key, num_samples=10000, sample_func=conditional_sample):
    ll_data = []
    sample_time = 0
    grenander_time = 0
    ll_time = 0
    for c in city_data:
        rng_key, subkey = jrand.split(rng_key, 2)
        c_params = c['params']
        c_events = c['events']
        t0 = time.time()
        event_sim = sample_func(c_params, general_params, subkey, num_samples=num_samples)
        sample_time += time.time() - t0
        t0 = time.time()
        city_distr = grenander_pdf_numba(event_sim+1)
        grenander_time += time.time() - t0
        t0 = time.time()
        ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
        ll_time += time.time() - t0
        ll_data.append([ll, pll, eventll])
    # print(f"Sampling time: {sample_time}, Grenander time: {grenander_time}, LL time: {ll_time}")
    return np.array(ll_data)

def eval_ds_hybrid(city_data, general_params, timespan, rng_key, num_samples=10000, sample_func=conditional_sample):
    ll_data = []
    sample_time = 0
    grenander_time = 0
    ll_time = 0
    for c in city_data:
        rng_key, subkey = jrand.split(rng_key, 2)
        c_params = c['params']
        c_events = c['events']
        t0 = time.time()
        event_sim = sample_func(c_params, general_params, subkey, num_samples=num_samples)
        sample_time += time.time() - t0
        t0 = time.time()
        city_distr = hybrid_pdf(event_sim+1)
        grenander_time += time.time() - t0
        t0 = time.time()
        ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
        ll_time += time.time() - t0
        ll_data.append([ll, pll, eventll])
    print(f"Sampling time: {sample_time}, Grenander time: {grenander_time}, LL time: {ll_time}")
    return np.array(ll_data)

def eval_ds_evd(city_data, general_params, timespan, rng_key, nonevent_num_samples=5000, 
                event_num_samples=10000, sample_func=conditional_sample):
    assert event_num_samples < int(1e6)
    ll_data = []
    num_sample_l = [nonevent_num_samples + (event_num_samples - nonevent_num_samples)*(len(c['events']) > 0) \
                    for c in city_data]
    city_param_l = [c['params'] for c in city_data]
    sample_eval_iter = conditional_sample_batch(city_param_l, general_params, rng_key, num_sample_l, batch_size=int(5e5))
    for c, event_sim in zip(city_data, sample_eval_iter):
        rng_key, subkey = jrand.split(rng_key, 2)
        c_params = c['params']
        c_events = c['events']
        # event_sim = sample_func(c_params, general_params, subkey, num_samples=nonevent_num_samples)
        city_distr = grenander_pdf_numba(event_sim+1)
        ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
        ll_data.append([ll, pll, eventll])
    return np.array(ll_data)

def eval_ds_fast(city_data, general_params, timespan, rng_key, num_samples, sample_func=conditional_sample):
    assert num_samples < int(1e6)
    ll_data = []
    num_sample_l = [num_samples]*len(city_data)
    city_param_l = [c['params'] for c in city_data]
    sample_eval_iter = conditional_sample_batch(city_param_l, general_params, rng_key, num_sample_l, batch_size=int(5e5))
    for c, event_sim in zip(city_data, sample_eval_iter):
        rng_key, subkey = jrand.split(rng_key, 2)
        c_params = c['params']
        c_events = c['events']
        # event_sim = sample_func(c_params, general_params, subkey, num_samples=nonevent_num_samples)
        city_distr = grenander_pdf_numba(event_sim+1)
        ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
        ll_data.append([ll, pll, eventll])
    return np.array(ll_data)