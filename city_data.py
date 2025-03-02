import pickle
import numpy as np

with open('data/county_data.pkl', 'rb') as f:
    city_data = pickle.load(f)
def hasna(p):
    return any(np.isnan(x) for x in p)
city_data = {k: v for k, v in city_data.items() if not hasna(v['params'])}
city_data_f = [c for c in city_data.values() if c['events']]
city_data_base = [c for c in city_data.values() if not c['events']]
city_data_f_names = [c for c, d in city_data.items() if d['events']]
city_data_base_names = [c for c, d in city_data.items() if not d['events']]
city_base_ind = np.arange(len(city_data_base))
np.random.seed(1)
np.random.shuffle(city_base_ind)
subsampled_city_data_base = np.array(city_data_base)[city_base_ind[:400]]
subsampled_city_data_base_names = np.array(city_data_base_names)[city_base_ind[:400]]
subsampled_city_data = np.concatenate([city_data_f, subsampled_city_data_base])
subsampled_city_data_names = np.concatenate([city_data_f_names, subsampled_city_data_base_names])
base_weight = len(city_data_base) / 400
subsample_weights = np.array([1+base_weight*(len(c['events']) == 0) for c in subsampled_city_data])