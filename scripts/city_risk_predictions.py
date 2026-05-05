"""
Task H — City-level risk predictions (top 500 cities).

Computes posterior predictive risk estimates for each of the 500 subsampled cities
using the fitted nooptstate-v3 posterior.  For each city the script reports:

    - expected mass shootings / decade   (Poisson mean λ = propensity × 10 × pop × P(≥5 killed))
    - P(≥1 event per decade)             (1 − exp(−λ), more intuitive than λ for small cities)
    - expected fatalities / decade       (λ × E[fatalities | fatalities > 4])
    - fatalities / capita / decade

Posterior uncertainty is summarised as mean ± std across n_posterior draws.

Outputs
-------
    models/city_risk_predictions.csv          — full table, sorted by events/decade
    paper_code/images/city_risk_events.png    — top-10 by expected events/decade
    paper_code/images/city_risk_fatalities.png — top-10 by fatalities/capita/decade

Usage
-----
    python scripts/city_risk_predictions.py [--gpu 0] [--n-posterior 100] [--num-sim 5000]
"""

import argparse
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# CLI args and GPU setup (must happen before JAX import)
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--gpu',         type=int,   default=0,
                    help='CUDA device (default: 0)')
parser.add_argument('--n-posterior', type=int,   default=100,
                    help='Posterior draws for risk estimates (default: 100)')
parser.add_argument('--num-sim',     type=int,   default=5000,
                    help='ABM samples per city per posterior draw (default: 5000)')
parser.add_argument('--model-dir',   type=str,   default='models/nooptstate-v3/',
                    help='Directory containing vp.pkl (default: models/nooptstate-v3/)')
parser.add_argument('--timespan',    type=float, default=10.0,
                    help='Prediction horizon in years (default: 10)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import jax.numpy as jnp        # noqa: E402
import jax.random as jrand     # noqa: E402

from gvabm.abm_event import build_conditional_sampler, make_noopt_event
from gvabm.param_distr import ParamDistr, param_config as nooptstate_param_config
from gvabm.city_data import subsampled_city_data, subsampled_city_data_names

sys.path.insert(0, os.path.dirname(__file__))
from sensitivity_utils import load_posterior_samples

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR   = 'paper_code/images'
RESULTS_CSV  = 'models/city_risk_predictions.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

param_distr = ParamDistr(nooptstate_param_config)
sample_func = build_conditional_sampler(make_noopt_event())   # baseline sensitivity params

n_cities = len(subsampled_city_data)

print("=" * 60)
print("Task H: City-level risk predictions")
print(f"  Cities            : {n_cities}")
print(f"  Posterior samples : {args.n_posterior}")
print(f"  ABM samples/city  : {args.num_sim}")
print(f"  Timespan          : {args.timespan:.0f} years")
print(f"  Model dir         : {args.model_dir}")
print("=" * 60)

# ──────────────────────────────────────────────────────────────────────────────
# Load posterior samples from vp.pkl
# ──────────────────────────────────────────────────────────────────────────────
posterior_normalized = load_posterior_samples(args.model_dir, n_samples=args.n_posterior)

# ──────────────────────────────────────────────────────────────────────────────
# Main loop: posterior samples × cities
# ──────────────────────────────────────────────────────────────────────────────
# Accumulators: shape (n_cities, n_posterior)
events_mat     = np.zeros((n_cities, args.n_posterior))   # expected events/decade
fatalities_mat = np.zeros((n_cities, args.n_posterior))   # expected fatalities/decade

rng_key = jrand.PRNGKey(42)
t_start = time.time()

for j, s in enumerate(posterior_normalized):
    rng_key, subkey = jrand.split(rng_key)
    params = param_distr.transform_sample(jnp.array(s))
    params = param_distr.to_data(np.array(params))
    propensity = float(params.propensity)

    for i, city in enumerate(subsampled_city_data):
        rng_key, city_key = jrand.split(rng_key)
        c_params  = city['params']
        population = float(c_params.population)

        # Simulate fatality counts for this (city, parameter) pair
        samples = np.array(sample_func(c_params, params, city_key,
                                       num_samples=args.num_sim))

        # P(mass shooting) = P(fatalities > 4)
        prob_event = float(np.mean(samples > 4.0))

        # Expected events this timespan under the Poisson rate
        exp_events = propensity * args.timespan * population * prob_event

        # Conditional mean fatalities given it is a mass shooting
        mass_shooting_samples = samples[samples > 4.0]
        if len(mass_shooting_samples) > 0:
            cond_mean_fatalities = float(np.mean(mass_shooting_samples))
        else:
            # No qualifying samples — use threshold as conservative floor
            cond_mean_fatalities = 5.0

        events_mat[i, j]     = exp_events
        fatalities_mat[i, j] = exp_events * cond_mean_fatalities

    elapsed   = time.time() - t_start
    remaining = elapsed / (j + 1) * (args.n_posterior - j - 1)
    print(f"  sample {j+1:3d}/{args.n_posterior}  "
          f"{elapsed/60:.1f} min elapsed  ~{remaining/60:.1f} min remaining",
          flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Compute posterior statistics
# ──────────────────────────────────────────────────────────────────────────────
events_mean    = events_mat.mean(axis=1)
events_std     = events_mat.std(axis=1)
fat_mean       = fatalities_mat.mean(axis=1)
fat_std        = fatalities_mat.std(axis=1)

# P(≥1 event / decade) from the posterior-mean Poisson rate
p_any_event    = 1.0 - np.exp(-events_mean)
pop_arr        = np.array([float(city['params'].population) for city in subsampled_city_data])

fat_per_capita      = fat_mean  / pop_arr
fat_per_capita_std  = fat_std   / pop_arr

# ──────────────────────────────────────────────────────────────────────────────
# Build city identifier strings from subsampled_city_names
# Keys are (GISJOIN, decade) tuples; stringify them for CSV/display
# ──────────────────────────────────────────────────────────────────────────────
def _name_to_str(name):
    if isinstance(name, (tuple, list, np.ndarray)):
        return '_'.join(str(x) for x in name)
    return str(name)

city_ids   = [_name_to_str(nm) for nm in subsampled_city_data_names]
gisjoin_list  = [str(nm[0]) if isinstance(nm, (tuple, list, np.ndarray)) else str(nm)
                 for nm in subsampled_city_data_names]
decade_list   = [str(nm[1]) if isinstance(nm, (tuple, list, np.ndarray)) and len(nm) > 1
                 else str(subsampled_city_data[i]['params'].year)
                 for i, nm in enumerate(subsampled_city_data_names)]

observed_events = [len(city['events']) for city in subsampled_city_data]

df = pd.DataFrame({
    'city_id':                   city_ids,
    'gisjoin':                   gisjoin_list,
    'decade':                    decade_list,
    'population':                np.round(pop_arr, 0).astype(int),
    'arm_density':               np.round([float(city['params'].arm_density)
                                           for city in subsampled_city_data], 6),
    'observed_events':           observed_events,
    'events_per_decade':         np.round(events_mean, 4),
    'events_per_decade_std':     np.round(events_std,  4),
    'p_any_event_decade':        np.round(p_any_event, 4),
    'fatalities_per_decade':     np.round(fat_mean, 4),
    'fatalities_per_decade_std': np.round(fat_std,  4),
    'fatalities_per_capita':     np.round(fat_per_capita, 8),
    'fatalities_per_capita_std': np.round(fat_per_capita_std, 8),
})

# ──────────────────────────────────────────────────────────────────────────────
# Save and print tables
# ──────────────────────────────────────────────────────────────────────────────
df_by_events = df.sort_values('events_per_decade', ascending=False).reset_index(drop=True)
df_by_events.to_csv(RESULTS_CSV, index=False)
print(f"\nFull table saved to {RESULTS_CSV}  ({len(df_by_events)} rows)")

def _print_table(header, rows_df, val_col, val_fmt, se_col):
    print(f"\n{header}")
    print(f"{'#':<4} {'City ID':<30} {'Pop':>10} {'Obs':>5}  {val_col}")
    print("-" * 75)
    for rank, (_, row) in enumerate(rows_df.head(10).iterrows(), 1):
        val = row[val_col]
        se  = row[se_col]
        print(f"{rank:<4} {str(row['city_id']):<30} "
              f"{int(row['population']):>10,}  {int(row['observed_events']):>3}  "
              f"{val:{val_fmt}} ± {se:{val_fmt}}")

df_by_fpc = df.sort_values('fatalities_per_capita', ascending=False).reset_index(drop=True)

_print_table(
    "Top 10 by expected events / decade",
    df_by_events, 'events_per_decade', '.4f', 'events_per_decade_std',
)
_print_table(
    "Top 10 by fatalities / capita / decade",
    df_by_fpc, 'fatalities_per_capita', '.3e', 'fatalities_per_capita_std',
)

# ──────────────────────────────────────────────────────────────────────────────
# Figures
# ──────────────────────────────────────────────────────────────────────────────
def _short_label(city_id, max_len=20):
    """Truncate long GISJOIN-style IDs for axis labels."""
    return city_id[:max_len] if len(city_id) <= max_len else city_id[:max_len - 1] + '…'

def _bar_figure(df_top10, val_col, err_col, ylabel, title, color, out_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    x      = np.arange(len(df_top10))
    labels = [_short_label(r['city_id']) for _, r in df_top10.iterrows()]
    ax.bar(x, df_top10[val_col], yerr=df_top10[err_col],
           color=color, capsize=4, alpha=0.85, error_kw={'linewidth': 1.2})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=35, ha='right')
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved: {out_path}")

_bar_figure(
    df_by_events.head(10),
    val_col='events_per_decade',
    err_col='events_per_decade_std',
    ylabel='Expected mass shootings / decade',
    title='Top 10 cities by expected mass shootings per decade\n(nooptstate-v3 posterior mean ± std)',
    color='steelblue',
    out_path=os.path.join(OUTPUT_DIR, 'city_risk_events.png'),
)

_bar_figure(
    df_by_fpc.head(10),
    val_col='fatalities_per_capita',
    err_col='fatalities_per_capita_std',
    ylabel='Expected fatalities / capita / decade',
    title='Top 10 cities by expected fatalities per capita per decade\n(nooptstate-v3 posterior mean ± std)',
    color='firebrick',
    out_path=os.path.join(OUTPUT_DIR, 'city_risk_fatalities.png'),
)

print("\nDone.")
