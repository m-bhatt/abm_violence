"""
Task F — Sensitivity of WAIC to severity model parameters.

The focal model uses hard-coded exponential fatality rates:
  - low-lethality  (no high weapon) : expon_lambda = 6
  - high-lethality (high weapon)    : expon_lambda = 6 + 120 = 126

This script varies both rates by a common multiplicative scale across
[0.50, 0.75, 1.00, 1.25, 1.50], holding all other parameters at the
nooptstate-v3 posterior.  Confirms that WAIC model rankings are stable
across severity parametrizations (Reviewer 2, severity mechanisms concern).

Output
------
    models/sensitivity_severity_results.json
    paper_code/images/sensitivity_severity.png

Usage
-----
    python scripts/sensitivity_severity.py [--gpu 0] [--n-posterior 50] [--num-sim 10000]
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
LOW_RATE_BASELINE      = 6.0
HIGH_RATE_ADD_BASELINE = 120.0
SEVERITY_SCALES        = [0.50, 0.75, 1.00, 1.25, 1.50]
BASELINE_SCALE         = 1.00
OUTPUT_DIR             = 'paper_code/images'
RESULTS_FILE           = 'models/sensitivity_severity_results.json'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

param_distr = ParamDistr(nooptstate_param_config)

# ──────────────────────────────────────────────────────────────────────────────
# Run sweep
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("Task F: severity rate sweep")
print(f"  Scales            : {SEVERITY_SCALES}")
print(f"  Baseline low_rate : {LOW_RATE_BASELINE}  high_rate: {LOW_RATE_BASELINE + HIGH_RATE_ADD_BASELINE}")
print(f"  Posterior samples : {args.n_posterior}")
print(f"  ABM samples/city  : {args.num_sim}")
print(f"  Model dir         : {args.model_dir}")
print("=" * 60)

results = []
for scale in SEVERITY_SCALES:
    t0 = time.time()
    tag = "(baseline)" if scale == BASELINE_SCALE else ""
    low   = LOW_RATE_BASELINE * scale
    high  = LOW_RATE_BASELINE * scale + HIGH_RATE_ADD_BASELINE * scale
    print(f"\nseverity_scale = {scale:.2f}  (low={low:.1f}, high={high:.1f}) {tag}", flush=True)

    sample_func = build_conditional_sampler(
        make_noopt_event(low_rate=low, high_rate_add=HIGH_RATE_ADD_BASELINE * scale)
    )
    waic, waic_se = compute_sweep_waic(
        sample_func, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=args.model_dir,
    )
    elapsed = time.time() - t0
    results.append({
        'severity_scale': float(scale),
        'low_rate': float(low),
        'high_rate': float(high),
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
scales = [r['severity_scale'] for r in results]
waics  = [r['waic'] for r in results]
ses    = [r['waic_se'] for r in results]

fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(scales, waics, yerr=ses, marker='D', capsize=4, color='firebrick',
            linewidth=1.5, label='WAIC ± SE')
ax.axvline(BASELINE_SCALE, color='gray', linestyle='--', linewidth=1,
           label=f'Baseline (scale = {BASELINE_SCALE})')
ax.set_xlabel('Severity scale factor (applied to low & high rates)')
ax.set_ylabel('WAIC')
ax.set_title('Task F: Sensitivity to severity rates\n(nooptstate-v3 posterior)')
ax.set_xticks(scales)
ax.set_xticklabels([f'{s:.2f}×' for s in scales])
ax.legend(frameon=False)
fig.tight_layout()
out_path = os.path.join(OUTPUT_DIR, 'sensitivity_severity.png')
fig.savefig(out_path, dpi=150)
print(f"Figure saved to {out_path}")
