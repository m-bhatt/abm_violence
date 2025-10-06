import numpy as np

# def eval_ds(city_data, general_params, timespan, rng_key, num_samples=10000, sample_func=conditional_sample):
#returns array of shape (num_cities, 3) with columns [ll, pll, event_ll]
#WAIC = llps - 

def estimate_WAIC_correction(y, y_weights, eval_func, proxy_distr, num_samples=200, var_est_iters=10):
    param_samples = proxy_distr.sample(num_samples)
    iter_py_estimates = []
    for iter in range(var_est_iters):
        py_estimates = []
        for i, params in enumerate(param_samples):
            py_est = eval_func(y, params)
            py_estimates.append(py_est)
        iter_py_estimates.append(py_estimates)
    iter_py_estimates = np.array(iter_py_estimates)
    py_estimates = np.mean(iter_py_estimates, axis=0) # shape (num_samples, len(y))
    py_var_estimates = np.var(iter_py_estimates, axis=0) / np.sqrt(var_est_iters) #shape (num_samples, len(y))
    # py_estimates = np.array(py_estimates)
    py_var = np.var(py_estimates, axis=0) - py_var_estimates.sum(axis=0)
    total_var = py_var * y_weights
    return total_var
    
