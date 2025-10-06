import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_event import sample_event, conditional_sample
from gvabm.abm_ll import city_ll, eval_ds
from gvabm.stat_utils import CityParams, EventOutcome, GeneralParams, logrange
from gvabm.param_distr import ParamDistr, param_config
param_distr = ParamDistr(param_config)

prior = param_distr.get_uniform_prior()
log_prior = param_distr.get_log_uniform_prior()

from gvabm.abm_mcmc import get_param_density_func_noisy

data_dict = {}
P = get_param_density_func_noisy(param_distr.get_log_uniform_prior(), jrand.PRNGKey(1), int(1e5), data_dict=data_dict)
def save_data(samples, sample_eval):
    with open('proc/evaluation_samples.pkl', 'wb') as f:
        pickle.dump((samples, sample_eval, data_dict), f)
    print("Saved data")

import os
import numpy as np
import scipy.stats as scs
from pyvbmc import VBMC
D = param_distr.param_count
LB = np.full((1, D), -np.inf); UB = np.full((1, D), np.inf)
PLB = param_distr.range_lower.reshape((1, -1)); PUB = param_distr.range_upper.reshape((1, -1))
# x0 = np.array(param_distr.sample_range(jrand.PRNGKey(2)))
x0 = np.array([-7.49712599, 38.49844685,  0.70090904, -1.13903331, -4.42167238])

import os
options = {
    "max_fun_evals": 100 * (D + 2),
    "specify_target_noise": True, 
    # "fun_eval_start": 100,
    "warmup_no_impro_threshold": 20 + 10 * D,
    # "always_refit_vp": True
}
# new_options = {
#     "max_fun_evals": 50 * (D + 2),
#     "specify_target_noise": True
# }
save_file = "proc/abm_vbmc_subsampled.pkl"
# if os.path.exists(save_file):
#     vbmc = VBMC.load(
#         save_file,
#         new_options=new_options,
#         iteration=None,  # the default: start from the last stored iteration.
#         set_random_state=False,  # the default: don't modify the random state
#         # (can be set to True for reproducibility).
#     )
# else:
vbmc = VBMC(P, x0, LB, UB, PLB, PUB, options)
vp, results = vbmc.optimize()
vbmc.save(save_file, overwrite=True)