import os
os.environ['CUDA_VISIBLE_DEVICES'] = str(2)

from gvabm.abm_event import conditional_sample, conditional_sample_meta
from gvabm.param_distr import ParamDistr, param_config, mixed_choice_param_config
from gvabm.city_data import subsampled_city_data, subsampled_city_data_names
import numpy as np
import matplotlib.pyplot as plt
import dill


with open("/home/andrew/abm_violence/models/nooptstate-v3/vbmc.pkl", "rb") as f:
    vbmc_loaded = dill.load(f)
vp = vbmc_loaded.vp
param_distr = ParamDistr(param_config)

from gvabm.abm_event import build_conditional_sampler_meta, sample_event_with_meta_optstate_noopt
meta_sampler = build_conditional_sampler_meta(sample_event_with_meta_optstate_noopt)
import jax.random as jrand
import jax.numpy as jnp
import pickle

if(os.path.exists(f"/home/andrew/abm_violence/paper_code/data/city_sample_stats.pkl")):
    with open(f"/home/andrew/abm_violence/paper_code/data/city_sample_stats.pkl", "rb") as f:
        city_stat_dict = pickle.load(f)
else:
    city_stat_dict = {}
from tqdm import tqdm
for i, city_name, city_data in tqdm(zip(range(len(subsampled_city_data)), subsampled_city_data_names, subsampled_city_data), 
                                    desc="Processing cities", total=len(subsampled_city_data)):
    # city_name, city_data = subsampled_city_data_names[10], subsampled_city_data[10]
    print(city_name)
    city_id = '_'.join(city_name)
    if city_id in city_stat_dict:
        print(f"Skipping {city_id}")
        continue
    city_params = city_data['params']
    print(city_id, city_params)
    samples, _ = vp.sample(500)
    rng_key = jrand.PRNGKey(0)
    sample_rows = []
    for sample in samples:
        general_params = param_distr.transform_sample(sample)
        general_params = param_distr.to_data(general_params)
        rng_key, subkey = jrand.split(rng_key)
        sampler_outcomes = meta_sampler(city_params, general_params, subkey)
        event_size, location_count, gathering_size, weapon_access, high_weapon_access, access_count = sampler_outcomes
        propensity = general_params.propensity
        expected_event_size = jnp.mean(event_size)
        p_event = jnp.mean(event_size > 3)
        p_loc = jnp.mean(location_count > 0)
        p_weapon = jnp.mean(weapon_access)
        p_high_weapon = jnp.mean(high_weapon_access)
        access_count = jnp.mean(access_count)
        sample_rows.append(np.array([propensity, expected_event_size, p_event, p_loc, p_weapon, p_high_weapon, access_count]))
    sample_data = np.array(sample_rows) # num_samples x (propensity, expected_event_size, p_event, p_loc, p_weapon)
    print(city_id, sample_data.mean(axis=0))
    city_stat_dict[city_id] = sample_data
    if i % 5 == 0:
        with open(f"/home/andrew/abm_violence/paper_code/data/city_sample_stats.pkl", "wb") as f:
            pickle.dump(city_stat_dict, f)
with open(f"/home/andrew/abm_violence/paper_code/data/city_sample_stats.pkl", "wb") as f:
    pickle.dump(city_stat_dict, f)