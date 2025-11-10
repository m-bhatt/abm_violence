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

from gvabm.param_distr import CityParams
spoof_city = CityParams(year=2020, population=100000, population_density=1000, arm_count=-1, arm_density=0.2)

arm_sweep_values = np.linspace(0.0, 0.6, 100)
sweep_data = []
rng_key = jrand.PRNGKey(0)

from tqdm import tqdm
for i, arm_density in tqdm(enumerate(arm_sweep_values)):
    city_params = spoof_city._replace(arm_density=arm_density)
    samples = vp.sample(200)[0]
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
    print(f"Arm Density {arm_density}: ", sample_data.mean(axis=0))
    sweep_data.append(sample_data)
    if i % 5 == 0:
        with open(f"/home/andrew/abm_violence/paper_code/data/arm_sweep_stats.pkl", "wb") as f:
            pickle.dump(sweep_data, f)
with open(f"/home/andrew/abm_violence/paper_code/data/arm_sweep_stats.pkl", "wb") as f:
    pickle.dump(sweep_data, f)

population_density_sweep_values = np.linspace(10, 10000, 100)
sweep_data = []
rng_key = jrand.PRNGKey(0)

from tqdm import tqdm
for i, population_density in tqdm(enumerate(population_density_sweep_values)):
    city_params = spoof_city._replace(population_density=population_density)
    samples = vp.sample(200)[0]
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
    print(f"Population Density {population_density}: ", sample_data.mean(axis=0))
    sweep_data.append(sample_data)
    if i % 5 == 0:
        with open(f"/home/andrew/abm_violence/paper_code/data/pop_dens_sweep_stats.pkl", "wb") as f:
            pickle.dump(sweep_data, f)
with open(f"/home/andrew/abm_violence/paper_code/data/pop_dens_sweep_stats.pkl", "wb") as f:
    pickle.dump(sweep_data, f)
    
