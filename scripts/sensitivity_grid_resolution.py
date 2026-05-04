"""
Task C — Sensitivity of WAIC to ABM grid resolution.

The focal model (nooptstate-v3) uses a 30×30 grid.  This script tests
[20, 25, 30, 35, 40] while holding all other parameters at the
nooptstate-v3 posterior.

Implementation notes
--------------------
- total_lambda = arm_density * grid_size² (same formula as baseline), so total
  weapon encounters scale with the number of cells.  arm_density is therefore
  interpreted as weapons per cell at each resolution.
- walk_radius is in grid steps, not physical miles.  The same posterior-mean
  step count covers a larger physical fraction of a smaller grid.
- The 9×9 Gaussian smoothing kernel is kept constant across resolutions.

Output
------
    models/sensitivity_grid_resolution_results.json
    paper_code/images/sensitivity_grid_resolution.png

Usage
-----
    python scripts/sensitivity_grid_resolution.py [--gpu 0] [--n-posterior 50] [--num-sim 10000]
"""

import argparse
import json
import os
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
GRID_SIZES    = [20, 25, 30, 35, 40]
BASELINE_GRID = 30
OUTPUT_DIR    = 'paper_code/images'
RESULTS_FILE  = 'models/sensitivity_grid_resolution_results.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

param_distr = ParamDistr(nooptstate_param_config)

# ──────────────────────────────────────────────────────────────────────────────
# Run sweep
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Task C: grid resolution sweep")
print(f"  Grid sizes        : {GRID_SIZES}")
print(f"  Posterior samples : {args.n_posterior}")
print(f"  ABM samples/city  : {args.num_sim}")
print(f"  Model dir         : {args.model_dir}")
print("=" * 60)

results = []
for gs in GRID_SIZES:
    t0 = time.time()
    tag = "(baseline)" if gs == BASELINE_GRID else ""
    print(f"\ngrid_size = {gs}×{gs} {tag}", flush=True)

    sample_func = build_conditional_sampler(make_noopt_event(grid_size=gs))
    waic, waic_se = compute_sweep_waic(
        sample_func, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=args.model_dir,
    )
    elapsed = time.time() - t0
    results.append({
        'grid_size': gs,
        'waic': float(waic),
        'waic_se': float(waic_se),
        'elapsed_s': round(elapsed),
    })
    print(f"  WAIC = {waic:.2f} ± {waic_se:.2f}  ({elapsed:.0f}s)", flush=True)

with open(RESULTS_FILE, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {RESULTS_FILE}")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
grid_sizes = [r['grid_size'] for r in results]
waics      = [r['waic'] for r in results]
ses        = [r['waic_se'] for r in results]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(grid_sizes, waics, yerr=ses, marker='s', capsize=4, color='darkorange',
            linewidth=1.5, label='WAIC ± SE')
ax.axvline(BASELINE_GRID, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline ({BASELINE_GRID}×{BASELINE_GRID})')
ax.set_xlabel('Grid side length N (N×N cells)')
ax.set_ylabel('WAIC')
ax.set_title('Task C: Sensitivity to grid resolution\n(nooptstate-v3 posterior)')
ax.set_xticks(grid_sizes)
ax.set_xticklabels([f'{g}×{g}' for g in grid_sizes])
ax.legend(frameon=False)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'sensitivity_grid_resolution.png')
fig.savefig(out_path, dpi=150)
print(f"Figure saved to {out_path}")
