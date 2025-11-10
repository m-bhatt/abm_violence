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
from gvabm.abm_vbmc import get_param_density_func_noisy

if __name__ == "__main__":
    # print(f"Using GPU {gpu_id} for model {model_name}")
    sample_func = {'max_select': conditional_sample, 'sample_select': conditional_sample_noselection, 'mixed_choice': conditional_sample_mixed}[model_name]
    param_config = {'max_select': param_config, 'sample_select': param_config, 'mixed_choice': mixed_choice_param_config}[model_name]
    cutoff_likelihood = {'max_select': -1050, 'sample_select': -1050, 'mixed_choice': -1050}[model_name]

    param_distr = ParamDistr(param_config)
    prior = param_distr.get_uniform_prior()

    from pyvbmc import VBMC
    D = param_distr.param_count
    LB = np.full((1, D), -np.inf); UB = np.full((1, D), np.inf)
    PLB = param_distr.range_lower.reshape((1, -1)); PUB = param_distr.range_upper.reshape((1, -1))
    print('Parameter bounds:')
    print('LB:', LB)
    print('UB:', UB)
    print('PLB:', PLB)
    print('PUB:', PUB)
    
    with open(f'/home/andrew/abm_violence/data/warm_start_{model_name}.pkl', 'rb') as f:
        data = pickle.load(f)
    samples = data['samples']
    sample_eval = np.array(data['sample_eval'])
    samples = samples[:len(sample_eval)]

    init_samples = 350
    fsamples = samples[sample_eval > cutoff_likelihood][:init_samples]
    fsample_eval = sample_eval[sample_eval > cutoff_likelihood][:init_samples] + 800
    x0 = fsamples
    print(x0.shape, fsample_eval.shape)

    model_data = {'model_name': model_name, 'param_config': param_config, 'x0': x0.tolist()}
    #Add v-n to model_name if dir already exists

    import os
    options = {
        "max_fun_evals": 150 * (D + 2),
        "specify_target_noise": True, 
        "f_vals": fsample_eval,
        "fun_eval_start": len(fsample_eval),
    }

    model_dir = f'/home/andrew/abm_violence/models/{model_name}/'
    if os.path.exists(model_dir):
        incr = 1
        while os.path.exists(f'/home/andrew/abm_violence/models/{model_name}-v{incr}/'):
            incr += 1
        model_dir = f'/home/andrew/abm_violence/models/{model_name}-v{incr}/'
    os.makedirs(model_dir, exist_ok=True)

    density_func = get_param_density_func_noisy(sample_func, param_distr, jrand.PRNGKey(2), 
                                                num_samples=50000, cutoff_likelihood=cutoff_likelihood, logdir=model_dir)
    P = density_func
    
    logfile = f"{model_dir}/vbmc.log"
    logging.basicConfig(filename=logfile, level=logging.INFO)
    with open(f"{model_dir}/model_data.pkl", "wb") as f:
        pickle.dump(model_data, f)

    import json
    with open(f"{model_dir}/model_desc.txt", "w") as f:
        json.dump({'model_name': model_name, 'x0': x0.tolist()}, f)

    save_file = f"{model_dir}/model.pkl"

    vbmc = VBMC(P, x0, PLB-1, PUB+1, PLB, PUB, options)
    vp, results = vbmc.optimize()

    vbmc.log_joint = None
    vbmc.function_logger = None

    import dill
    with open(f"{model_dir}/vbmc_results.pkl", "wb") as f:
        dill.dump(results, f)
    with open(f"{model_dir}/vp.pkl", "wb") as f:
        dill.dump(vp, f)
    with open(f"{model_dir}/vbmc.pkl", "wb") as f:
        dill.dump(vbmc, f)
    vbmc.save(save_file)
    print(f"Model and results saved to {model_dir}")

    # python run_vbmc.py --model mixed_choice --gpu 2
    # python run_vbmc_warmstart.py --model sample_select --gpu 1
    # python run_vbmc.py --model max_select --gpu 0