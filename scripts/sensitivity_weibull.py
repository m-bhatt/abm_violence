"""
Task G — Sensitivity of WAIC to gathering-size Weibull parameters.

The focal model uses weibull_min(shape=2, scale=0.4) for at-risk gathering
sizes.  This script runs two independent 1-D sweeps:

  Sweep 1 (shape): shape in [1.0, 1.5, 2.0, 2.5, 3.0], scale = 0.4 fixed
  Sweep 2 (scale): scale in [0.20, 0.30, 0.40, 0.50, 0.60], shape = 2.0 fixed

All other parameters are held at the nooptstate-v3 posterior.

Output
------
    models/sensitivity_weibull_shape_results.json
    models/sensitivity_weibull_scale_results.json
    paper_code/images/sensitivity_weibull.png

Usage
-----
    python scripts/sensitivity_weibull.py [--gpu 0] [--n-posterior 50] [--num-sim 10000]
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
BASELINE_SHAPE  = 2.0
BASELINE_SCALE  = 0.4
SHAPE_VALUES    = [1.0, 1.5, 2.0, 2.5, 3.0]
SCALE_VALUES    = [0.20, 0.30, 0.40, 0.50, 0.60]
OUTPUT_DIR      = 'paper_code/images'
SHAPE_RESULTS   = 'models/sensitivity_weibull_shape_results.json'
SCALE_RESULTS   = 'models/sensitivity_weibull_scale_results.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

param_distr = ParamDistr(nooptstate_param_config)

# ──────────────────────────────────────────────────────────────────────────────
# Run sweeps
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Task G: Weibull gathering-size sweep")
print(f"  Shape values      : {SHAPE_VALUES}  (scale fixed at {BASELINE_SCALE})")
print(f"  Scale values      : {SCALE_VALUES}  (shape fixed at {BASELINE_SHAPE})")
print(f"  Posterior samples : {args.n_posterior}")
print(f"  ABM samples/city  : {args.num_sim}")
print(f"  Model dir         : {args.model_dir}")
print("=" * 60)

# --- Sweep 1: Weibull shape ---
print("\n--- Sweep 1: Weibull shape (scale fixed at {:.2f}) ---".format(BASELINE_SCALE))
shape_results = []
for sh in SHAPE_VALUES:
    t0 = time.time()
    tag = "(baseline)" if sh == BASELINE_SHAPE else ""
    print(f"\nweibull_shape = {sh:.1f} {tag}", flush=True)

    sample_func = build_conditional_sampler(
        make_noopt_event(weibull_shape=sh, weibull_scale=BASELINE_SCALE)
    )
    waic, waic_se = compute_sweep_waic(
        sample_func, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=args.model_dir,
    )
    elapsed = time.time() - t0
    shape_results.append({
        'weibull_shape': float(sh),
        'weibull_scale': float(BASELINE_SCALE),
        'waic': float(waic),
        'waic_se': float(waic_se),
        'elapsed_s': round(elapsed),
    })
    print(f"  WAIC = {waic:.2f} ± {waic_se:.2f}  ({elapsed:.0f}s)", flush=True)

with open(SHAPE_RESULTS, 'w') as f:
    json.dump(shape_results, f, indent=2)
print(f"\nShape sweep results saved to {SHAPE_RESULTS}")

# --- Sweep 2: Weibull scale ---
print("\n--- Sweep 2: Weibull scale (shape fixed at {:.1f}) ---".format(BASELINE_SHAPE))
scale_results = []
for sc in SCALE_VALUES:
    t0 = time.time()
    tag = "(baseline)" if sc == BASELINE_SCALE else ""
    print(f"\nweibull_scale = {sc:.2f} {tag}", flush=True)

    sample_func = build_conditional_sampler(
        make_noopt_event(weibull_shape=BASELINE_SHAPE, weibull_scale=sc)
    )
    waic, waic_se = compute_sweep_waic(
        sample_func, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=args.model_dir,
    )
    elapsed = time.time() - t0
    scale_results.append({
        'weibull_shape': float(BASELINE_SHAPE),
        'weibull_scale': float(sc),
        'waic': float(waic),
        'waic_se': float(waic_se),
        'elapsed_s': round(elapsed),
    })
    print(f"  WAIC = {waic:.2f} ± {waic_se:.2f}  ({elapsed:.0f}s)", flush=True)

with open(SCALE_RESULTS, 'w') as f:
    json.dump(scale_results, f, indent=2)
print(f"\nScale sweep results saved to {SCALE_RESULTS}")

# ──────────────────────────────────────────────────────────────────────────────
# Plot
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Panel 1: shape sweep
ax = axes[0]
shapes = [r['weibull_shape'] for r in shape_results]
waics  = [r['waic'] for r in shape_results]
ses    = [r['waic_se'] for r in shape_results]
ax.errorbar(shapes, waics, yerr=ses, marker='o', capsize=4, color='mediumseagreen',
            linewidth=1.5, label='WAIC ± SE')
ax.axvline(BASELINE_SHAPE, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline (shape = {BASELINE_SHAPE})')
ax.set_xlabel('Weibull shape parameter')
ax.set_ylabel('WAIC')
ax.set_title('Sweep 1: shape\n(scale = 0.4 fixed)')
ax.set_xticks(shapes)
ax.legend(frameon=False)

# Panel 2: scale sweep
ax = axes[1]
scales = [r['weibull_scale'] for r in scale_results]
waics  = [r['waic'] for r in scale_results]
ses    = [r['waic_se'] for r in scale_results]
ax.errorbar(scales, waics, yerr=ses, marker='s', capsize=4, color='darkorchid',
            linewidth=1.5, label='WAIC ± SE')
ax.axvline(BASELINE_SCALE, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline (scale = {BASELINE_SCALE})')
ax.set_xlabel('Weibull scale parameter')
ax.set_ylabel('WAIC')
ax.set_title('Sweep 2: scale\n(shape = 2.0 fixed)')
ax.set_xticks(scales)
ax.legend(frameon=False)

fig.suptitle('Task G: Sensitivity to gathering-size Weibull\n(nooptstate-v3 posterior)',
             fontsize=11)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'sensitivity_weibull.png')
fig.savefig(out_path, dpi=150)
print(f"\nFigure saved to {out_path}")
