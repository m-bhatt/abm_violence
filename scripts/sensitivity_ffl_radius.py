"""
Task D — Sensitivity of WAIC to FFL buffer radius.

Requires build_ffl_radius_data.py to have been run first, which produces
one pkl per radius under data/processed/.

Radii tested: 7.5, 15, 25, 30 miles.  The baseline pkl (ffl_radius = 25)
matches the computation in scripts/city_ffl_density.ipynb (cell 5a384585).
Note: reviewer_coding_todo.txt describes the current radius as 15 miles —
verify which pkl corresponds to your baseline before interpreting delta-WAIC.

Output
------
    models/sensitivity_ffl_radius_results.json
    paper_code/images/sensitivity_ffl_radius.png

Usage
-----
    python scripts/sensitivity_ffl_radius.py [--gpu 0] [--n-posterior 50] [--num-sim 10000]
"""

import argparse
import json
import os
import pickle
import sys
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# CLI args and GPU setup (must happen before JAX import)
# ──────────────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='CUDA device (default: 0)')
parser.add_argument('--n-posterior', type=int, default=50,
                    help='Posterior draws for WAIC variance estimate (default: 50)')
parser.add_argument('--num-sim', type=int, default=10000,
                    help='ABM samples per city per posterior draw (default: 10000)')
parser.add_argument('--model-dir', type=str, default='models/nooptstate-v3/',
                    help='Path to fitted VP directory (default: models/nooptstate-v3/)')
parser.add_argument('--data-dir', type=str, default='data',
                    help='Root data directory (default: data)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import jax  # noqa: E402 — must come after CUDA env var

from gvabm.abm_event import build_conditional_sampler, make_noopt_event
from gvabm.param_distr import ParamDistr, param_config as nooptstate_param_config

sys.path.insert(0, os.path.dirname(__file__))
from sensitivity_utils import compute_sweep_waic

# ──────────────────────────────────────────────────────────────────────────────
# Sweep configuration
# ──────────────────────────────────────────────────────────────────────────────
RADII           = [7.5, 15, 25, 30]
BASELINE_RADIUS = 25          # matches city_ffl_density.ipynb cell 5a384585
PROCESSED_DIR   = os.path.join(args.data_dir, 'processed')
OUTPUT_DIR      = 'paper_code/images'
RESULTS_FILE    = 'models/sensitivity_ffl_radius_results.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

param_distr  = ParamDistr(nooptstate_param_config)
sample_func  = build_conditional_sampler(make_noopt_event())  # baseline event model

# ──────────────────────────────────────────────────────────────────────────────
# Run sweep
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Task D: FFL buffer radius sweep")
print(f"  Radii             : {RADII} miles")
print(f"  Baseline radius   : {BASELINE_RADIUS} miles")
print(f"  Posterior samples : {args.n_posterior}")
print(f"  ABM samples/city  : {args.num_sim}")
print(f"  Model dir         : {args.model_dir}")
print(f"  Data dir          : {args.data_dir}")
print("=" * 60)

results = []
for R in RADII:
    pkl_path = os.path.join(PROCESSED_DIR, f'subsampled_city_data_ffl_r{R:.0f}.pkl')
    if not os.path.exists(pkl_path):
        print(f"\nradius = {R} miles  — pkl not found at {pkl_path}, skipping")
        print("  Run build_ffl_radius_data.py first.")
        continue

    t0  = time.time()
    tag = "(baseline)" if R == BASELINE_RADIUS else ""
    print(f"\nradius = {R} miles {tag}", flush=True)

    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    city_data = data['subsampled_city_data']
    weights   = data['subsample_weights']
    print(f"  Loaded {len(city_data)} cities from {pkl_path}", flush=True)

    waic, waic_se = compute_sweep_waic(
        sample_func, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=args.model_dir,
        city_data=city_data,
        weights=weights,
    )
    elapsed = time.time() - t0
    results.append({
        'ffl_radius_miles': float(R),
        'waic': float(waic),
        'waic_se': float(waic_se),
        'elapsed_s': round(elapsed),
    })
    print(f"  WAIC = {waic:.2f} ± {waic_se:.2f}  ({elapsed:.0f}s)", flush=True)

if not results:
    print("\nNo results computed — run build_ffl_radius_data.py first.")
    raise SystemExit(1)

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
radii = [r['ffl_radius_miles'] for r in results]
waics = [r['waic'] for r in results]
ses   = [r['waic_se'] for r in results]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(radii, waics, yerr=ses, marker='^', capsize=4, color='teal',
            linewidth=1.5, label='WAIC ± SE')
ax.axvline(BASELINE_RADIUS, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline ({BASELINE_RADIUS} mi)')
ax.set_xlabel('FFL buffer radius (miles)')
ax.set_ylabel('WAIC')
ax.set_title('Task D: Sensitivity to FFL buffer radius\n(nooptstate-v3 posterior)')
ax.set_xticks(radii)
ax.set_xticklabels([f'{r:.4g} mi' for r in radii])
ax.legend(frameon=False)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'sensitivity_ffl_radius.png')
fig.savefig(out_path, dpi=150)
print(f"Figure saved to {out_path}")
