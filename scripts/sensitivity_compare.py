"""
Sensitivity comparison table — aggregate WAIC results from all sweep scripts.

Loads JSON result files produced by sensitivity_high_weapon.py,
sensitivity_grid_resolution.py, sensitivity_severity.py, and
sensitivity_weibull.py.  Skips any file that does not yet exist.

Output
------
    models/sensitivity_comparison_table.csv   — machine-readable table
    paper_code/images/sensitivity_comparison.png  — multi-panel WAIC plot

Usage
-----
    python scripts/sensitivity_compare.py
"""

import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Registry of all sensitivity sweeps
# ──────────────────────────────────────────────────────────────────────────────
# Each entry: (sweep_label, json_path, x_key, x_label, baseline_value, plot_color)
SWEEPS = [
    (
        'B: High-weapon prob',
        'models/sensitivity_high_weapon_results.json',
        'high_weapon_prob',
        'High-weapon encounter prob.',
        0.10,
        'steelblue',
    ),
    (
        'C: Grid resolution',
        'models/sensitivity_grid_resolution_results.json',
        'grid_size',
        'Grid side length N',
        30,
        'darkorange',
    ),
    (
        'F: Severity scale',
        'models/sensitivity_severity_results.json',
        'severity_scale',
        'Severity scale factor',
        1.00,
        'firebrick',
    ),
    (
        'G: Weibull shape',
        'models/sensitivity_weibull_shape_results.json',
        'weibull_shape',
        'Weibull shape parameter',
        2.0,
        'mediumseagreen',
    ),
    (
        'G: Weibull scale',
        'models/sensitivity_weibull_scale_results.json',
        'weibull_scale',
        'Weibull scale parameter',
        0.4,
        'darkorchid',
    ),
]

OUTPUT_DIR    = 'paper_code/images'
CSV_OUT       = 'models/sensitivity_comparison_table.csv'
FIGURE_OUT    = os.path.join(OUTPUT_DIR, 'sensitivity_comparison.png')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs('models', exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load results and build comparison table
# ──────────────────────────────────────────────────────────────────────────────
rows = []
loaded_sweeps = []

for sweep_label, json_path, x_key, x_label, baseline_val, color in SWEEPS:
    if not os.path.exists(json_path):
        print(f"  [skip] {sweep_label} — {json_path} not found")
        continue

    with open(json_path) as f:
        data = json.load(f)

    # find baseline WAIC for delta computation
    baseline_waic = None
    for r in data:
        if abs(r[x_key] - baseline_val) < 1e-9:
            baseline_waic = r['waic']
            break

    for r in data:
        is_baseline = abs(r[x_key] - baseline_val) < 1e-9
        delta = (r['waic'] - baseline_waic) if baseline_waic is not None else float('nan')
        rows.append({
            'sweep':       sweep_label,
            'parameter':   x_label,
            'value':       r[x_key],
            'waic':        round(r['waic'], 2),
            'waic_se':     round(r['waic_se'], 2),
            'delta_waic':  round(delta, 2),
            'is_baseline': is_baseline,
        })

    loaded_sweeps.append((sweep_label, data, x_key, x_label, baseline_val, color))
    print(f"  [ok]   {sweep_label} — {len(data)} values loaded from {json_path}")

if not rows:
    print("\nNo result files found.  Run the individual sensitivity scripts first.")
    raise SystemExit(0)

df = pd.DataFrame(rows)
df.to_csv(CSV_OUT, index=False)
print(f"\nTable saved to {CSV_OUT}  ({len(df)} rows)")

# ──────────────────────────────────────────────────────────────────────────────
# Print formatted table
# ──────────────────────────────────────────────────────────────────────────────
COL_W = {'sweep': 24, 'value': 10, 'waic': 10, 'waic_se': 8, 'delta_waic': 12}
header = (f"{'Sweep':<24}  {'Value':>10}  {'WAIC':>10}  {'±SE':>8}  "
          f"{'ΔvBaseline':>12}  {'Baseline'}")
print("\n" + "=" * len(header))
print(header)
print("=" * len(header))

current_sweep = None
for _, row in df.iterrows():
    if row['sweep'] != current_sweep:
        if current_sweep is not None:
            print()
        current_sweep = row['sweep']
    star = " *" if row['is_baseline'] else ""
    print(f"  {row['sweep']:<22}  {row['value']:>10.3g}  {row['waic']:>10.2f}  "
          f"{row['waic_se']:>8.2f}  {row['delta_waic']:>+12.2f}{star}")

print("=" * len(header))
print("* = baseline value\n")

# ──────────────────────────────────────────────────────────────────────────────
# Multi-panel comparison figure
# ──────────────────────────────────────────────────────────────────────────────
n = len(loaded_sweeps)
ncols = min(3, n)
nrows = (n + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

for idx, (sweep_label, data, x_key, x_label, baseline_val, color) in enumerate(loaded_sweeps):
    ax = axes[idx // ncols][idx % ncols]
    xs    = [r[x_key]    for r in data]
    waics = [r['waic']   for r in data]
    ses   = [r['waic_se'] for r in data]

    ax.errorbar(xs, waics, yerr=ses, marker='o', capsize=4, color=color,
                linewidth=1.5, label='WAIC ± SE')
    ax.axvline(baseline_val, color='gray', linestyle='--', linewidth=1,
               label=f'Baseline ({baseline_val})')
    ax.set_xlabel(x_label, fontsize=9)
    ax.set_ylabel('WAIC', fontsize=9)
    ax.set_title(sweep_label, fontsize=10)
    ax.legend(frameon=False, fontsize=8)

# hide unused axes
for idx in range(len(loaded_sweeps), nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.suptitle('Sensitivity analysis: WAIC across parameter sweeps\n(nooptstate-v3 posterior)',
             fontsize=12)
fig.tight_layout()
fig.savefig(FIGURE_OUT, dpi=150)
print(f"Comparison figure saved to {FIGURE_OUT}")
