import argparse
import os
parser = argparse.ArgumentParser(description='Run VBMC with specified GPU and model')
# parser.add_argument('--gpu', 
#                     type=int,
#                     default=0,
#                     help='GPU device ID to use (default: 0)')
parser.add_argument('--model',
                    type=str,
                    required=True,
                    choices=['max_select', 'sample_select', 'mixed_choice'],
                    help='Model to apply.')

args = parser.parse_args()
# gpu_id = args.gpu
model_name = args.model
# os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_event import conditional_sample, conditional_sample_noselection, conditional_sample_mixed
from gvabm.abm_ll import city_ll, eval_ds
from gvabm.stat_utils import logrange
from gvabm.param_distr import ParamDistr, param_config, mixed_choice_param_config, CityParams, EventOutcome, GeneralParams, MixedGeneralParams
from gvabm.abm_vbmc import get_param_density_func

param_distr = ParamDistr(mixed_choice_param_config)

def warm_start_sample(param_distr, P, N, save_file=None):
    upper_bounds = param_distr.range_upper
    lower_bounds = param_distr.range_lower
    bound_widths = upper_bounds - lower_bounds
    rngkey = jrand.PRNGKey(42)
    samples = jrand.uniform(rngkey, shape=(N, param_distr.param_count)) * bound_widths + lower_bounds
    samples = np.array(samples)
    sample_eval = []
    for i in range(N):
        if i % 100 == 0:
            print(f"Evaluating sample {i}/{N}")
            if save_file is not None:
                with open(save_file, 'wb') as f:
                    pickle.dump({'samples': samples, 'sample_eval': sample_eval}, f)
        sample_eval.append(P(np.array(samples[i])))
    if save_file is not None:
        with open(save_file, 'wb') as f:
            pickle.dump({'samples': samples, 'sample_eval': sample_eval}, f)
    return samples, sample_eval

if __name__ == "__main__":
    # print(f"Using GPU {gpu_id} for model {model_name}")
    sample_func = {'max_select': conditional_sample, 'sample_select': conditional_sample_noselection, 'mixed_choice': conditional_sample_mixed}[model_name]
    param_config = {'max_select': param_config, 'sample_select': param_config, 'mixed_choice': mixed_choice_param_config}[model_name]
    param_distr = ParamDistr(param_config)
    
    P = get_param_density_func(sample_func, param_distr, jrand.PRNGKey(10), 5000)
    import time
    start_time = time.time()
    samples, sample_eval = warm_start_sample(param_distr, P, 10000,
                                             save_file=f'/home/andrew/abm_violence/data/warm_start_{model_name}.pkl')
    print(f"Sampled {len(samples)} points in {time.time() - start_time} seconds")
    
    #python warm_start_sample.py --gpu 0 --model max_select
    #python warm_start_sample.py --gpu 1 --model sample_select
    #python warm_start_sample.py --gpu 2 --model mixed_choice