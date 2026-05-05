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
                    choices=['max_select', 'sample_select', 'mixed_choice', 'big_model', 'optstate', 'nooptstate', 'optstate_base'],
                    help='Model to apply.')
# Sensitivity sweep flags (nooptstate only)
parser.add_argument('--high-weapon-prob', type=float, default=None,
                    help='Task B: high-weapon encounter probability (baseline: 0.10)')
parser.add_argument('--grid-size',        type=int,   default=None,
                    help='Task C: grid side length in cells (baseline: 30)')
parser.add_argument('--severity-scale',   type=float, default=None,
                    help='Task F: multiplier on low_rate and high_rate_add (baseline: 1.0)')
parser.add_argument('--weibull-shape',    type=float, default=None,
                    help='Task G: Weibull shape for gathering size (baseline: 2.0)')
parser.add_argument('--weibull-scale',    type=float, default=None,
                    help='Task G: Weibull scale for gathering size (baseline: 0.4)')
parser.add_argument('--data-pkl',         type=str,   default=None,
                    help='Task D: path to per-radius subsampled city pkl')

args = parser.parse_args()
gpu_id = args.gpu
model_name = args.model
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

import logging
import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_event import (conditional_sample, conditional_sample_noselection, conditional_sample_mixed,
                             sample_event_big, sample_event_optstate, sample_event_optstate_noopt,
                             sample_event_base,
                             build_conditional_sampler, make_noopt_event)
from gvabm.abm_ll import city_ll, eval_ds
from gvabm.stat_utils import logrange
from gvabm.param_distr import (ParamDistr, param_config, 
                               mixed_choice_param_config, big_param_config,
                               CityParams, EventOutcome, 
                               GeneralParams, MixedGeneralParams, BigGeneralParams)
from gvabm.abm_vbmc import get_param_density_func_noisy

if __name__ == "__main__":
    print(f"Using GPU {gpu_id} for model {model_name}")
    _nooptstate_sampler = build_conditional_sampler(make_noopt_event(
        high_weapon_prob = args.high_weapon_prob if args.high_weapon_prob is not None else 0.10,
        grid_size        = args.grid_size        if args.grid_size        is not None else 30,
        low_rate         = 6.0   * (args.severity_scale if args.severity_scale is not None else 1.0),
        high_rate_add    = 120.0 * (args.severity_scale if args.severity_scale is not None else 1.0),
        weibull_shape    = args.weibull_shape    if args.weibull_shape    is not None else 2.0,
        weibull_scale    = args.weibull_scale    if args.weibull_scale    is not None else 0.4,
    ))
    sample_func = {'max_select': conditional_sample,
                   'sample_select': conditional_sample_noselection,
                   'mixed_choice': conditional_sample_mixed,
                   'big_model': build_conditional_sampler(sample_event_big),
                   'optstate': build_conditional_sampler(sample_event_optstate),
                   'nooptstate': _nooptstate_sampler,
                   'optstate_base': build_conditional_sampler(sample_event_base),
                   }[model_name]
    param_config = {'max_select': param_config, 'sample_select': param_config, 
                    'mixed_choice': mixed_choice_param_config, 
                    'big_model': big_param_config,
                    'optstate': mixed_choice_param_config, 
                    'nooptstate': param_config, 
                    'optstate_base': param_config}[model_name]
    cutoff_likelihood = {'max_select': -1050, 'sample_select': -1050, 'mixed_choice': -1150, 'big_model': -1050, 
                         'optstate': -1300, 'nooptstate': -1300, 'optstate_base': -1250}[model_name]
    likelihood_offset = {'max_select': 700, 'sample_select': 700, 'mixed_choice': 700, 'big_model': 500, 
                         'optstate': 1100, 'nooptstate': 1100, 'optstate_base': 1100}[model_name]

    param_distr = ParamDistr(param_config)
    print(param_distr.param_names)
    print(param_config)
    log_prior = param_distr.get_log_beta_prior(alpha=5.0, beta=5.0)

    from pyvbmc import VBMC
    D = param_distr.param_count
    LB = np.full((1, D), -np.inf); UB = np.full((1, D), np.inf)
    PLB = np.zeros_like(param_distr.range_lower.reshape((1, -1)))
    PUB = np.ones_like(param_distr.range_upper.reshape((1, -1)))
    print('Parameter bounds:')
    print('LB:', LB)
    print('UB:', UB)
    print('PLB:', PLB)
    print('PUB:', PUB)

    big_x0 = "0.38306653 0.6552486  0.73572993 0.9029114  0.80744654 0.9499234 0.19604887 0.34824994"
    big_x0 = np.array([float(x) for x in big_x0.split()])

    x0_options = {'max_select': np.array([0.2, 0.7, 0.5, 0.5, 0.5]),
                  'sample_select': np.array([0.2, 0.7, 0.5, 0.5, 0.5]),
                  'mixed_choice': np.array([[0.2, 0.7, 0.5, 0.5, 0.5, 0.01], 
                                           [0.2, 0.7, 0.5, 0.5, 0.5, 0.99]]),
                  'big_model': big_x0,
                  'optstate': np.array([0.4, 0.267, 0.528, 0.683, 0.833, 0.08]), 
                  'nooptstate': np.array([0.4, 0.267, 0.528, 0.683, 0.833]),
                  'optstate_base': np.array([0.4, 0.267, 0.528, 0.683, 0.833]),
                }
    
    x0 = x0_options[model_name]
    model_data = {'model_name': model_name, 'param_config': param_config, 'x0': x0.tolist()}

    import os
    options = {
        "max_fun_evals": 90 * (D + 2),
        "specify_target_noise": True, 
    }

    # Build a tag from any non-default sensitivity flags so runs don't collide
    tag_parts = []
    if args.high_weapon_prob is not None: tag_parts.append(f'hwp{args.high_weapon_prob}')
    if args.grid_size        is not None: tag_parts.append(f'gs{args.grid_size}')
    if args.severity_scale   is not None: tag_parts.append(f'sev{args.severity_scale}')
    if args.weibull_shape    is not None: tag_parts.append(f'wsh{args.weibull_shape}')
    if args.weibull_scale    is not None: tag_parts.append(f'wsc{args.weibull_scale}')
    if args.data_pkl         is not None:
        # Extract a short token from the filename, e.g. 'r15' from '...ffl_r15.pkl'
        import re
        _m = re.search(r'(r\d+(?:\.\d+)?)', os.path.basename(args.data_pkl))
        tag_parts.append(f'ffl_{_m.group(1)}' if _m else 'ffl_custom')
    tag = ('_' + '_'.join(tag_parts)) if tag_parts else ''

    model_dir = f'/home/andrew/abm_violence/models/{model_name}{tag}/'
    if os.path.exists(model_dir):
        incr = 1
        while os.path.exists(f'/home/andrew/abm_violence/models/{model_name}{tag}-v{incr}/'):
            incr += 1
        model_dir = f'/home/andrew/abm_violence/models/{model_name}{tag}-v{incr}/'
    os.makedirs(model_dir, exist_ok=True)

    # Task D: optionally load a per-radius city pkl instead of the default dataset
    _city_data, _weights = None, None
    if args.data_pkl is not None:
        import pickle as _pickle
        with open(args.data_pkl, 'rb') as _f:
            _d = _pickle.load(_f)
        _city_data = _d['subsampled_city_data']
        _weights   = _d['subsample_weights']
        print(f"Task D: loaded {len(_city_data)} cities from {args.data_pkl}")

    density_func = get_param_density_func_noisy(
        sample_func, param_distr, log_prior, jrand.PRNGKey(2),
        num_samples=15000, cutoff_likelihood=cutoff_likelihood,
        likelihood_offset=likelihood_offset, logdir=model_dir,
        subsampled_city_data=_city_data,
        subsample_weights=_weights,
    )
    P = density_func
    
    logfile = f"{model_dir}/vbmc.log"
    logging.basicConfig(filename=logfile, level=logging.INFO)
    with open(f"{model_dir}/model_data.pkl", "wb") as f:
        pickle.dump(model_data, f)

    import json
    with open(f"{model_dir}/model_desc.txt", "w") as f:
        json.dump({'model_name': model_name, 'x0': x0.tolist()}, f)

    save_file = f"{model_dir}/model.pkl"
    vbmc = VBMC(P, x0, PLB, PUB, PLB+0.1, PUB-0.1, options)
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
    # python run_vbmc.py --model sample_select --gpu 1
    # python run_vbmc.py --model max_select --gpu 0