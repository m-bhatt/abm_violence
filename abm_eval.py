import pickle
import numpy as np
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as jrand

from abm_event import sample_event, conditional_sample
from abm_ll import city_ll, eval_ds
from stat_utils import CityParams, EventOutcome, GeneralParams, logrange
from city_data import subsampled_city_data

from stat_utils import logrange
propensity_range = logrange(2e-9, 5e-5, 21)
walk_radius_range = [40]
arm_bias_range = [0.01]
arm_weight_range = [1]
atrisk_gather_rate_range = logrange(3e-8, 5e-4, 21)

from itertools import product
param_groups = list(product(*[propensity_range, walk_radius_range, arm_bias_range, arm_weight_range, atrisk_gather_rate_range]))
import hashlib
param_hash = hashlib.sha224(str(param_groups).encode('utf-8')).hexdigest()
savepath = f'data/ll_eval_{param_hash[:5]}.pkl'
import os
eval_dict = {}
if(os.path.exists(savepath)):
    with open(savepath, 'rb') as f:
        eval_dict = pickle.load(f)
    print("Loaded existing data")
print(len(eval_dict))
rng_key = jrand.PRNGKey(42)
from tqdm import tqdm
for i, group in tqdm(enumerate(param_groups)):
    if(group in eval_dict):
        # print('skipping')
        continue
    rng_key, subkey = jrand.split(rng_key, 2)
    params = GeneralParams(*group)
    eval_ds_results = eval_ds(subsampled_city_data, params, 50, subkey, num_samples=int(1e5))
    eval_dict[params] = eval_ds_results
    if(i % 10 == 0):
        # print(f"{i / len(param_groups) * 100}%")
        with open(savepath, 'wb') as f:
            pickle.dump(eval_dict, f)
with open(f'data/ll_eval_{param_hash[:5]}.pkl', 'wb') as f:
    pickle.dump(eval_dict, f)