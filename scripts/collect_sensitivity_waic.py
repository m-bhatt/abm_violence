"""
collect_sensitivity_waic.py — post-hoc WAIC aggregator for sensitivity refits.

After running all 12 sensitivity fits via run_vbmc.py with sensitivity flags,
this script:
  1. Scans models/ for directories whose names encode sensitivity tags
     (e.g. nooptstate_hwp0.05/, nooptstate_gs20/, nooptstate_ffl_r15/)
  2. Calls compute_sweep_waic on each using the refitted vp.pkl
  3. Writes one JSON per sweep task, compatible with sensitivity_compare.py

Output JSON files (written to models/):
    sensitivity_high_weapon_results.json      (Task B)
    sensitivity_grid_resolution_results.json  (Task C)
    sensitivity_ffl_radius_results.json       (Task D)
    sensitivity_severity_results.json         (Task F)
    sensitivity_weibull_shape_results.json    (Task G — shape)
    sensitivity_weibull_scale_results.json    (Task G — scale)

Usage
-----
    python scripts/collect_sensitivity_waic.py [--gpu 0] [--n-posterior 50] [--num-sim 10000]
    python scripts/collect_sensitivity_waic.py [--models-dir /home/andrew/abm_violence/models]
"""

import argparse
import json
import os
import re
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--gpu',         type=int,   default=0,
                    help='CUDA device (default: 0)')
parser.add_argument('--n-posterior', type=int,   default=50,
                    help='Posterior draws for WAIC variance estimate (default: 50)')
parser.add_argument('--num-sim',     type=int,   default=10000,
                    help='ABM samples per city per posterior draw (default: 10000)')
parser.add_argument('--models-dir',  type=str,   default='/home/andrew/abm_violence/models',
                    help='Root models directory (default: /home/andrew/abm_violence/models)')
parser.add_argument('--data-dir',    type=str,   default='/home/andrew/abm_violence/data',
                    help='Root data directory for Task D pkls (default: /home/andrew/abm_violence/data)')
args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)

import numpy as np  # noqa: E402

from gvabm.abm_event import build_conditional_sampler, make_noopt_event  # noqa: E402
from gvabm.param_distr import ParamDistr, param_config as nooptstate_param_config  # noqa: E402
from gvabm.city_data import subsampled_city_data as default_city_data  # noqa: E402
from gvabm.city_data import subsample_weights as default_weights  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from sensitivity_utils import compute_sweep_waic  # noqa: E402

MODELS_DIR = args.models_dir
DATA_DIR   = args.data_dir
OUT_DIR    = os.path.join(MODELS_DIR, '')  # same dir as models/

param_distr = ParamDistr(nooptstate_param_config)

# ──────────────────────────────────────────────────────────────────────────────
# Sweep registry: how to parse the directory tag and build the sampler/data
# Each entry:
#   tag_re   : regex that matches the tag fragment in the dir name
#   json_key : key name in the output JSON
#   parse    : fn(match) → numeric value for the JSON
#   sampler  : fn(value) → (sample_func, city_data, weights)
# ──────────────────────────────────────────────────────────────────────────────

def _sampler_only(make_event_kwargs):
    sf = build_conditional_sampler(make_noopt_event(**make_event_kwargs))
    return sf, None, None


def _task_d_sampler(radius_miles):
    pkl_path = os.path.join(DATA_DIR, 'processed', f'subsampled_city_data_ffl_r{radius_miles:.0f}.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"Task D pkl not found: {pkl_path}\nRun build_ffl_radius_data.py first.")
    import pickle
    with open(pkl_path, 'rb') as f:
        d = pickle.load(f)
    sf = build_conditional_sampler(make_noopt_event())
    return sf, d['subsampled_city_data'], d['subsample_weights']


# (tag_regex, json_key, value_parser, sampler_factory)
SWEEP_SPECS = [
    # Task B — high weapon prob:  nooptstate_hwp0.05/
    (
        re.compile(r'^nooptstate_hwp([\d.]+)(?:-v\d+)?$'),
        'high_weapon_prob',
        float,
        lambda v: _sampler_only({'high_weapon_prob': v}),
    ),
    # Task C — grid size:  nooptstate_gs20/
    (
        re.compile(r'^nooptstate_gs(\d+)(?:-v\d+)?$'),
        'grid_size',
        int,
        lambda v: _sampler_only({'grid_size': v}),
    ),
    # Task D — FFL radius:  nooptstate_ffl_r15/
    (
        re.compile(r'^nooptstate_ffl_r([\d.]+)(?:-v\d+)?$'),
        'ffl_radius_miles',
        float,
        lambda v: _task_d_sampler(v),
    ),
    # Task F — severity scale:  nooptstate_sev0.75/
    (
        re.compile(r'^nooptstate_sev([\d.]+)(?:-v\d+)?$'),
        'severity_scale',
        float,
        lambda v: _sampler_only({'low_rate': 6.0 * v, 'high_rate_add': 120.0 * v}),
    ),
    # Task G — Weibull shape:  nooptstate_wsh1.5/
    (
        re.compile(r'^nooptstate_wsh([\d.]+)(?:-v\d+)?$'),
        'weibull_shape',
        float,
        lambda v: _sampler_only({'weibull_shape': v}),
    ),
    # Task G — Weibull scale:  nooptstate_wsc0.30/
    (
        re.compile(r'^nooptstate_wsc([\d.]+)(?:-v\d+)?$'),
        'weibull_scale',
        float,
        lambda v: _sampler_only({'weibull_scale': v}),
    ),
]

# Map json_key → output file
JSON_FILES = {
    'high_weapon_prob':  os.path.join(MODELS_DIR, 'sensitivity_high_weapon_results.json'),
    'grid_size':         os.path.join(MODELS_DIR, 'sensitivity_grid_resolution_results.json'),
    'ffl_radius_miles':  os.path.join(MODELS_DIR, 'sensitivity_ffl_radius_results.json'),
    'severity_scale':    os.path.join(MODELS_DIR, 'sensitivity_severity_results.json'),
    'weibull_shape':     os.path.join(MODELS_DIR, 'sensitivity_weibull_shape_results.json'),
    'weibull_scale':     os.path.join(MODELS_DIR, 'sensitivity_weibull_scale_results.json'),
}

# Baseline values — used to include the already-fitted nooptstate-v3 result
BASELINES = {
    'high_weapon_prob': (0.10, 'nooptstate-v3'),
    'grid_size':        (30,   'nooptstate-v3'),
    'ffl_radius_miles': (25.0, 'nooptstate-v3'),
    'severity_scale':   (1.00, 'nooptstate-v3'),
    'weibull_shape':    (2.0,  'nooptstate-v3'),
    'weibull_scale':    (0.40, 'nooptstate-v3'),
}

# ──────────────────────────────────────────────────────────────────────────────
# Scan model directories
# ──────────────────────────────────────────────────────────────────────────────
results_by_key = {k: [] for k in JSON_FILES}

if not os.path.isdir(MODELS_DIR):
    print(f"Models directory not found: {MODELS_DIR}")
    raise SystemExit(1)

for dirname in sorted(os.listdir(MODELS_DIR)):
    full_path = os.path.join(MODELS_DIR, dirname)
    if not os.path.isdir(full_path):
        continue
    vp_path = os.path.join(full_path, 'vp.pkl')
    if not os.path.exists(vp_path):
        continue

    for tag_re, json_key, parse_val, sampler_factory in SWEEP_SPECS:
        m = tag_re.match(dirname)
        if m is None:
            continue
        value = parse_val(m.group(1))
        print(f"\n[{json_key}={value}]  dir={dirname}", flush=True)
        try:
            sample_func, city_data, weights = sampler_factory(value)
        except FileNotFoundError as e:
            print(f"  SKIP — {e}")
            break

        waic, waic_se = compute_sweep_waic(
            sample_func, param_distr,
            n_posterior_samples=args.n_posterior,
            num_sim_samples=args.num_sim,
            model_dir=full_path,
            city_data=city_data,
            weights=weights,
        )
        print(f"  WAIC = {waic:.2f} ± {waic_se:.2f}", flush=True)
        results_by_key[json_key].append({json_key: value, 'waic': float(waic), 'waic_se': float(waic_se)})
        break

# ──────────────────────────────────────────────────────────────────────────────
# Add baseline (nooptstate-v3) to every sweep that has at least one result
# ──────────────────────────────────────────────────────────────────────────────
for json_key, sweep_results in results_by_key.items():
    if not sweep_results:
        continue
    baseline_val, baseline_dirname = BASELINES[json_key]
    already_has_baseline = any(
        abs(r[json_key] - baseline_val) < 1e-9 for r in sweep_results
    )
    if already_has_baseline:
        continue

    baseline_dir = os.path.join(MODELS_DIR, baseline_dirname)
    if not os.path.exists(os.path.join(baseline_dir, 'vp.pkl')):
        print(f"\n[{json_key}] baseline dir {baseline_dirname} not found — skipping baseline")
        continue

    print(f"\n[{json_key}=baseline {baseline_val}]  dir={baseline_dirname}", flush=True)
    # baseline always uses default city data and default sampler
    sf = build_conditional_sampler(make_noopt_event())
    city_data_b = None
    weights_b   = None
    if json_key == 'ffl_radius_miles':
        # For Task D baseline use the 25-mile pkl if available, else default
        pkl_path = os.path.join(DATA_DIR, 'processed', 'subsampled_city_data_ffl_r25.pkl')
        if os.path.exists(pkl_path):
            import pickle
            with open(pkl_path, 'rb') as f:
                d = pickle.load(f)
            city_data_b = d['subsampled_city_data']
            weights_b   = d['subsample_weights']

    waic, waic_se = compute_sweep_waic(
        sf, param_distr,
        n_posterior_samples=args.n_posterior,
        num_sim_samples=args.num_sim,
        model_dir=baseline_dir,
        city_data=city_data_b,
        weights=weights_b,
    )
    print(f"  WAIC = {waic:.2f} ± {waic_se:.2f}", flush=True)
    sweep_results.append({json_key: baseline_val, 'waic': float(waic), 'waic_se': float(waic_se)})

# ──────────────────────────────────────────────────────────────────────────────
# Write JSON files
# ──────────────────────────────────────────────────────────────────────────────
written = []
for json_key, sweep_results in results_by_key.items():
    if not sweep_results:
        continue
    sweep_results.sort(key=lambda r: r[json_key])
    out_path = JSON_FILES[json_key]
    with open(out_path, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    written.append(out_path)
    print(f"\nWrote {len(sweep_results)} entries → {out_path}")

if not written:
    print("\nNo refitted sensitivity directories found under", MODELS_DIR)
    print("Run the fits first (see sensitivity_refit_plan.md).")
else:
    print("\nDone. Run scripts/sensitivity_compare.py to generate figures.")
