"""Shared utilities for sensitivity sweep scripts (Tasks B–G)."""

import os
import numpy as np
import jax.numpy as jnp
import jax.random as jrand

from gvabm.abm_ll import eval_ds
from gvabm.waic_eval import WAIC_metric
from gvabm.param_distr import ParamDistr
from gvabm.city_data import subsampled_city_data, subsample_weights


def load_posterior_samples(model_dir: str, n_samples: int = 100):
    """Draw n_samples from the variational posterior saved in model_dir/vp.pkl."""
    import dill
    vp_path = os.path.join(model_dir, 'vp.pkl')
    with open(vp_path, 'rb') as f:
        vp = dill.load(f)
    return vp.sample(n_samples)  # shape (n_samples, D) in [0,1] normalized space


def compute_sweep_waic(
    sample_func,
    param_distr: ParamDistr,
    n_posterior_samples: int = 50,
    num_sim_samples: int = 10000,
    rng_seed: int = 42,
    model_dir: str = 'models/nooptstate-v3/',
    city_data=None,
    weights=None,
):
    """
    Evaluate WAIC for sample_func using posterior samples from model_dir.

    Draws n_posterior_samples from the variational posterior, evaluates
    eval_ds for each, and returns (waic, waic_se).

    Parameters
    ----------
    sample_func : callable
        A conditional sampler produced by build_conditional_sampler.
    param_distr : ParamDistr
        Parameter distribution matching the saved posterior.
    n_posterior_samples : int
        Number of posterior draws used for WAIC variance estimate.
    num_sim_samples : int
        ABM samples per city per posterior draw.
    rng_seed : int
        Seed for reproducibility.
    model_dir : str
        Directory containing vp.pkl.
    city_data : array-like or None
        City dataset to evaluate on.  If None, uses the default
        subsampled_city_data loaded from gvabm.city_data.
    weights : array-like or None
        Per-city importance weights matching city_data.  If None, uses
        the default subsample_weights from gvabm.city_data.

    Returns
    -------
    (waic, waic_se) : tuple of float
    """
    if city_data is None:
        city_data = subsampled_city_data
    if weights is None:
        weights = subsample_weights

    rng_key = jrand.PRNGKey(rng_seed)
    posterior_normalized = load_posterior_samples(model_dir, n_samples=n_posterior_samples)

    ll_columns = []
    for i, s in enumerate(posterior_normalized):
        rng_key, subkey = jrand.split(rng_key)
        params = param_distr.transform_sample(jnp.array(s))
        params = param_distr.to_data(np.array(params))
        city_lls = eval_ds(
            city_data, params, timespan=10, rng_key=subkey,
            num_samples=num_sim_samples, sample_func=sample_func,
        )
        ll_columns.append(city_lls[:, 0])  # total ll per city
        if (i + 1) % 10 == 0:
            print(f"    posterior sample {i+1}/{n_posterior_samples}", flush=True)

    # ll_mat shape: (n_cities, n_posterior_samples, 1) — R=1 replicate per draw
    ll_mat = np.stack(ll_columns, axis=1)[:, :, np.newaxis]
    return WAIC_metric(ll_mat, weights)
