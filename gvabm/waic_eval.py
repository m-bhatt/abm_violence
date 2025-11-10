import numpy as np

# def eval_ds(city_data, general_params, timespan, rng_key, num_samples=10000, sample_func=conditional_sample):
#returns array of shape (num_cities, 3) with columns [ll, pll, event_ll]
#WAIC = llps - 

# def estimate_WAIC_correction(y, y_weights, eval_func, proxy_distr, num_samples=200, var_est_iters=10):
#     param_samples = proxy_distr.sample(num_samples)
#     iter_py_estimates = []
#     for iter in range(var_est_iters):
#         py_estimates = []
#         for i, params in enumerate(param_samples):
#             py_est = eval_func(y, params)
#             py_estimates.append(py_est)
#         iter_py_estimates.append(py_estimates)
#     iter_py_estimates = np.array(iter_py_estimates)
#     py_estimates = np.mean(iter_py_estimates, axis=0) # shape (num_samples, len(y))
#     py_var_estimates = np.var(iter_py_estimates, axis=0) / np.sqrt(var_est_iters) #shape (num_samples, len(y))
#     # py_estimates = np.array(py_estimates)
#     py_var = np.var(py_estimates, axis=0) - py_var_estimates.sum(axis=0)
#     total_var = py_var * y_weights
#     return total_var
from scipy.special import logsumexp
def estimate_PLLD(ll_eval_mat, y_weights):
    # ll_eval_mat: A 3D array of log-likelihoods
    # ll_eval_mat shape (num_data_points, num_samples, replicates)
    N, S, R = ll_eval_mat.shape
    ll_estimates = np.mean(ll_eval_mat, axis=2) # shape (num_data_points, num_samples)
    log_expected_prob = logsumexp(ll_estimates, axis=1) - np.log(S)  # shape (num_samples,)
    total_lpd = np.sum(log_expected_prob * y_weights)
    return total_lpd, log_expected_prob

def estimate_WAIC_correction(ll_eval_mat, y_weights):
    # ll_eval_mat: A 3D array of log-likelihoods
    # ll_eval_mat shape (num_data_points, num_samples, replicates)
    B, N, R = ll_eval_mat.shape
    ll_estimates = np.mean(ll_eval_mat, axis=2) # shape (num_data_points, num_samples)
    eval_var_estimates = np.var(ll_eval_mat, axis=2) / R #shape (num_samples, len(y))
    # ll_estimates = np.array(ll_estimates) 
    ll_var = np.var(ll_estimates, axis=1) - (eval_var_estimates.sum(axis=1) / (N - 1))
    # print('waic correction correction', (eval_var_estimates.sum(axis=1) / (N - 1)).sum())
    total_var = np.sum(ll_var * y_weights)
    return total_var, ll_var

def weighted_var(x, weights):
    average = np.average(x, weights=weights)
    W = weights.sum()
    variance = np.average((x - average)**2, weights=weights) * weights.size * W / (W**2 - np.sum(weights**2))
    return variance

def WAIC_metric(ll_eval_mat, y_weights):
    total_lpd, log_expected_prob = estimate_PLLD(ll_eval_mat, y_weights)
    total_var, ll_var = estimate_WAIC_correction(ll_eval_mat, y_weights)
    waic_i = log_expected_prob - ll_var
    waic = np.sum(waic_i * y_weights)
    waic_se = (y_weights.size * weighted_var(waic_i, y_weights))**0.5
    return waic, waic_se
