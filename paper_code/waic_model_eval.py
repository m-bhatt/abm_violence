import argparse
import os
parser = argparse.ArgumentParser(description='Run VBMC with specified GPU and model')
parser.add_argument('--gpu', 
                    type=int,
                    default=0,
                    help='GPU device ID to use (default: 0)')
parser.add_argument('--model',
                    type=str,
                    required=True,
                    choices=['max_select', 'sample_select', 
                             'mixed_choice', 'big_model', 
                             'optstate', 'nooptstate', 'optstate_base'],
                    help='Model to apply.')

args = parser.parse_args()
gpu_id = args.gpu
model_name = args.model
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

from gvabm.abm_event import (conditional_sample, conditional_sample_noselection, conditional_sample_mixed, 
                             sample_event_big, sample_event_optstate, sample_event_optstate_noopt,
                             sample_event_base, 
                             build_conditional_sampler)
from gvabm.param_distr import ParamDistr, param_config, mixed_choice_param_config, big_param_config

model_save_paths = {'max_select': '/home/andrew/abm_violence/models/max_select-v15/vbmc.pkl',
                    'sample_select': '/home/andrew/abm_violence/models/sample_select-v17/vbmc.pkl',
                    'mixed_choice': '/home/andrew/abm_violence/models/mixed_choice-v9/vbmc.pkl',
                    'optstate': "/home/andrew/abm_violence/models/optstate-v10/vbmc.pkl", 
                    'nooptstate': "/home/andrew/abm_violence/models/nooptstate-v3/vbmc.pkl",
                    'optstate_base': "/home/andrew/abm_violence/models/optstate_base/vbmc.pkl"
                    }

model_conditional_samplers = {'max_select': conditional_sample, 
                   'sample_select': conditional_sample_noselection, 
                   'mixed_choice': conditional_sample_mixed, 
                   'big_model': build_conditional_sampler(sample_event_big),
                   'optstate': build_conditional_sampler(sample_event_optstate),
                   'nooptstate': build_conditional_sampler(sample_event_optstate_noopt),
                   'optstate_base': build_conditional_sampler(sample_event_base),
                   }

model_param_configs = {'max_select': param_config, 'sample_select': param_config, 
                    'mixed_choice': mixed_choice_param_config, 
                    'big_model': big_param_config,
                    'optstate': mixed_choice_param_config, 
                    'nooptstate': param_config, 
                    'optstate_base': param_config}


import dill
import jax.random as jrand
import jax.numpy as jnp
import numpy as np
import os

from gvabm.city_data import subsampled_city_data, subsampled_city_data_names
from gvabm.abm_ll import city_ll
from gvabm.stat_utils import grenander_pdf_numba

from tqdm import tqdm
def waic_eval(city_data, param_distr, timespan, rng_key, sample_func, post_sampler, 
              eval_samples=200, abm_samples=10000, repetitions=5):
    out_mat = np.zeros((len(city_data), eval_samples, repetitions))
    for i, c in tqdm(enumerate(city_data), total=len(city_data)):
        posterior_samples, _ = post_sampler.sample(eval_samples)
        for j, sample in enumerate(posterior_samples):
            general_params = param_distr.transform_sample(sample)
            general_params = param_distr.to_data(general_params)
            for rep in range(repetitions):
                rng_key, subkey = jrand.split(rng_key, 2)
                c_params = c['params']
                c_events = c['events']
                event_sim = sample_func(c_params, general_params, subkey, num_samples=abm_samples)
                city_distr = grenander_pdf_numba(event_sim+1)
                ll, pll, eventll = city_ll(c_params, c_events, general_params, timespan, city_distr)
                out_mat[i, j, rep] = ll
    return out_mat

model_path = model_save_paths[model_name]
sample_func = model_conditional_samplers[model_name]
param_config = model_param_configs[model_name]
param_distr = ParamDistr(param_config)
with open(model_path, "rb") as f:
    vbmc_loaded = dill.load(f)
post_sampler = vbmc_loaded.vp
rng_key = jrand.PRNGKey(0)
waic_ll_data = waic_eval(subsampled_city_data, param_distr, timespan=10, rng_key=rng_key,
                         sample_func=sample_func, post_sampler=post_sampler,
                         eval_samples=250, abm_samples=15000, repetitions=5)
with open(f"/home/andrew/abm_violence/paper_code/data/waic_ll_data_{model_name}.pkl", "wb") as f:
    dill.dump(waic_ll_data, f)


