"""
Task B — Sensitivity of WAIC to high-weapon encounter probability.

The focal model (nooptstate-v3) hard-codes a 10% per-encounter probability for
acquiring a high-lethality weapon.  This script varies that value across
[0.05, 0.10, 0.15, 0.20, 0.25, 0.30] and records WAIC at each setting,
holding all other parameters at the nooptstate-v3 posterior.

Output
------
    models/sensitivity_high_weapon_results.json
    paper_code/images/sensitivity_high_weapon.png

Usage
-----
    python scripts/sensitivity_high_weapon.py [--gpu 0] [--n-posterior 50] [--num-sim 10000]
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
SWEEP_VALUES  = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
BASELINE_HWP  = 0.10
OUTPUT_DIR    = 'paper_code/images'
RESULTS_FILE  = 'models/sensitivity_high_weapon_results.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

param_distr = ParamDistr(nooptstate_param_config)

# ──────────────────────────────────────────────────────────────────────────────
# Run sweep
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Task B: high-weapon probability sweep")
print(f"  Values : {SWEEP_VALUES}")
print(f"  Posterior samples : {args.n_posterior}")
print(f"  ABM samples/city  : {args.num_sim}")
print(f"  Model dir         : {args.model_dir}")
print("=" * 60)

results = []
for hwp in SWEEP_VALUES:
    t0 = time.time()
    tag = "(baseline)" if hwp == BASELINE_HWP else ""
    print(f"\nhwp = {hwp:.2f} {tag}", flush=True)

    sample_func = build_conditional_sampler(make_noopt_event(high_weapon_prob=hwp))
    waic, waic_se = compute_sweep_waic(
        sample_func, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=args.model_dir,
    )
    elapsed = time.time() - t0
    results.append({
        'high_weapon_prob': float(hwp),
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
hwps    = [r['high_weapon_prob'] for r in results]
waics   = [r['waic'] for r in results]
ses     = [r['waic_se'] for r in results]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(hwps, waics, yerr=ses, marker='o', capsize=4, color='steelblue',
            linewidth=1.5, label='WAIC ± SE')
ax.axvline(BASELINE_HWP, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline (p = {BASELINE_HWP})')
ax.set_xlabel('High-weapon encounter probability')
ax.set_ylabel('WAIC')
ax.set_title('Task B: Sensitivity to high-weapon probability\n(nooptstate-v3 posterior)')
ax.legend(frameon=False)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'sensitivity_high_weapon.png')
fig.savefig(out_path, dpi=150)
print(f"Figure saved to {out_path}")
