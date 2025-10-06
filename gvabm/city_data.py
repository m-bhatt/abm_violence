import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

with open('data/county_data.pkl', 'rb') as f:
    city_data = pickle.load(f)
def hasna(p):
    return any(np.isnan(x) for x in p)
city_data = {k: v for k, v in city_data.items() if not hasna(v['params'])}
city_data_l = list(city_data.items())

import random
random.seed(42)
np.random.seed(42)

def softmax(x):
    return np.exp(x) / (1 + np.exp(x))

population_vals = np.array([e[1]['params'][0] for e in city_data_l])
p_weighting = lambda x: 2*(softmax(((x/2e6)))-0.5)
p_weights = np.array([p_weighting(x) for x in population_vals])
had_event = np.array([len(city_data['events']) > 0 for _, city_data in city_data_l])
sum_no_event = p_weights[~had_event].sum()
num_event = had_event.sum()
target_dataset_size = 450
scale_factor = (target_dataset_size - num_event) / sum_no_event
p_sample_probs = p_weights * scale_factor
p_sample_probs[had_event] = 1.0  # Ensure cities with events are always selected
p_sample_probs = np.clip(p_sample_probs, 0, 1)  # Ensure probabilities are between 0 and 1

population_dens_vals = np.array([e[1]['params'][1] for e in city_data_l])
pdens_weighting = lambda x: 2*(softmax(((x/5e5)))-0.5)
pdens_weights = np.array([pdens_weighting(x) for x in population_dens_vals])
sum_no_event = pdens_weights[~had_event].sum()
num_event = had_event.sum()
scale_factor = (target_dataset_size - num_event) / sum_no_event
pden_sample_probs = pdens_weights * scale_factor
pden_sample_probs[had_event] = 1.0  # Ensure cities with events are always selected
pden_sample_probs = np.clip(pden_sample_probs, 0, 1)  # Ensure probabilities are between 0 and 1


gun_dens_vals = np.array([e[1]['params'][2] for e in city_data_l])
gdens_weighting = lambda x: 2*(softmax(((x/1e3)))-0.5) #adjust for gun density
gdens_weights = np.array([gdens_weighting(x) for x in gun_dens_vals])
sum_no_event = gdens_weights[~had_event].sum()
num_event = had_event.sum()
scale_factor = (target_dataset_size - num_event) / sum_no_event
gun_sample_probs = gdens_weights * scale_factor
gun_sample_probs[had_event] = 1.0  # Ensure cities with events are always selected
gun_sample_probs = np.clip(gun_sample_probs, 0, 1)  # Ensure probabilities are between 0 and 1

#add all three sample probability together and divide by 3
sample_probs = (p_sample_probs + gun_sample_probs + gun_sample_probs)/3
subsampled_mask = np.random.uniform(0, 1, sample_probs.shape) < sample_probs
# print(f"Combined sample probabilities sum: {sample_probs.sum()}")

# city_data_l = np.array(city_data_l, dtype=object)

subsampled_city_tuples = [e for e, b in zip(city_data_l, subsampled_mask) if b]
subsampled_city_data_names = np.array([e[0] for e in subsampled_city_tuples])
subsampled_city_data = np.array([e[1] for e in subsampled_city_tuples])
subsample_weights = 1 / sample_probs[subsampled_mask]

# import pickle
# import numpy as np

# with open('data/county_data.pkl', 'rb') as f:
#     city_data = pickle.load(f)
# def hasna(p):
#     return any(np.isnan(x) for x in p)
# city_data = {k: v for k, v in city_data.items() if not hasna(v['params'])}
# city_data_f = [c for c in city_data.values() if c['events']]
# city_data_base = [c for c in city_data.values() if not c['events']]
# city_data_f_names = [c for c, d in city_data.items() if d['events']]
# city_data_base_names = [c for c, d in city_data.items() if not d['events']]
# city_base_ind = np.arange(len(city_data_base))

# np.random.seed(1)
# np.random.shuffle(city_base_ind)
# subsampled_city_data_base = np.array(city_data_base)[city_base_ind[:400]]
# subsampled_city_data_base_names = np.array(city_data_base_names)[city_base_ind[:400]]
# subsampled_city_data = np.concatenate([city_data_f, subsampled_city_data_base])
# subsampled_city_data_names = np.concatenate([city_data_f_names, subsampled_city_data_base_names])
# base_weight = len(city_data_base) / 400
# subsample_weights = np.array([1+base_weight*(len(c['events']) == 0) for c in subsampled_city_data])