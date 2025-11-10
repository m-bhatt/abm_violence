import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_event import sample_event, conditional_sample, build_conditional_sampler
from gvabm.abm_ll import city_ll, eval_ds, eval_ds_hybrid, eval_ds_evd, eval_ds_fast
from gvabm.stat_utils import logrange
from gvabm.param_distr import ParamDistr, param_config, CityParams, EventOutcome, GeneralParams
from gvabm.city_data import subsampled_city_data, subsample_weights
from gvabm.stat_utils import grenander_pdf, hybrid_pdf
param_distr = ParamDistr(param_config)

import os
import numpy as np
import scipy.stats as scs
param_distr = ParamDistr(param_config)
rng_key = jrand.PRNGKey(1)
param_l = [param_distr.transform_sample(param_distr.sample_range(r)) for r in jrand.split(rng_key, 10)]

from gvabm.abm_ll import eval_ds_evd
import time
# timespan = 50
# rng_key = jrand.PRNGKey(3)
# for param in param_l:
#     general_params = param
#     # general_params = (general_params[0], 4, general_params[2], general_params[3], general_params[4])
#     t = time.time()
#     eval_l = []
#     for _ in range(6):
#         ll_data = eval_ds_evd(subsampled_city_data, general_params, timespan, rng_key, nonevent_num_samples=800, 
#                 event_num_samples=8000, sample_func=conditional_sample)
#         ll_total = (ll_data * subsample_weights.reshape(-1,1)).sum(axis=0)[0]
#         # data_l.append(ll_data)
#         eval_l.append(ll_total)
#     print("Data time:", time.time()-t)
#     print("Standard:", np.mean(eval_l), np.std(eval_l))

import time
timespan = 50
rng_key = jrand.PRNGKey(3)
for param in param_l:
    general_params = param
    general_params = (general_params[0], 4, 1e-3, 1e-3, general_params[4])
    t = time.time()
    eval_l = []
    for _ in range(6):
        rng_key, _ = jrand.split(rng_key, 2)
        ll_data = eval_ds(subsampled_city_data, general_params, timespan, rng_key, num_samples=5000, sample_func=conditional_sample)
        ll_total = (ll_data * subsample_weights.reshape(-1,1)).sum(axis=0)[0]
        eval_l.append(ll_total)
    print("Data time:", time.time()-t)
    print("Standard:", np.mean(eval_l), np.std(eval_l))

# import time
# timespan = 50
# rng_key = jrand.PRNGKey(3)
# for param in param_l:
#     general_params = param
#     general_params = (general_params[0], 4, 1e-3, 1e-3, general_params[4])
#     t = time.time()
#     eval_l = []
#     for _ in range(6):
#         ll_data = eval_ds_fast(subsampled_city_data, general_params, timespan, rng_key, num_samples=500, sample_func=conditional_sample)
#         ll_total = (ll_data * subsample_weights.reshape(-1,1)).sum(axis=0)[0]
#         eval_l.append(ll_total)
#     print("Data time:", time.time()-t)
#     print("Standard:", np.mean(eval_l), np.std(eval_l))