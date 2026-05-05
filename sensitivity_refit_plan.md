# Sensitivity Sweeps: Refit Plan

## The Problem with the Current Scripts

`sensitivity_high_weapon.py`, `sensitivity_grid_resolution.py`, `sensitivity_severity.py`, and `sensitivity_weibull.py` load the fitted nooptstate-v3 posterior and evaluate WAIC at different parameter values **without refitting**. This is invalid — the five inferred parameters (`propensity`, `arm_bias`, `arm_weight`, etc.) were calibrated assuming the baseline constants. Changing `high_weapon_prob` to 0.20 without refitting shows parameter misspecification, not genuine model sensitivity; the inferred parameters would shift to compensate if the model were re-calibrated. Each sweep value needs its own VBMC run.

Task D (FFL buffer radius) is a data sensitivity — same issue applies. Each radius needs a refit on the corresponding pkl.

---

## Sweep Values (3 per task: low, baseline, high)

| Task | Parameter | Values | Baseline |
|------|-----------|--------|----------|
| B | `high_weapon_prob` | 0.05, **0.10**, 0.20 | 0.10 |
| C | `grid_size` | 20, **30**, 40 | 30 |
| D | FFL buffer radius | 15 mi, **25 mi**, 30 mi | 25 mi |
| F | severity scale | 0.75, **1.00**, 1.25 | 1.00 |
| G | Weibull shape | 1.5, **2.0**, 2.5 | 2.0 |
| G | Weibull scale | 0.30, **0.40**, 0.50 | 0.40 |

Baseline (nooptstate-v3) is already fitted — only 2 new fits per task are needed.

**Total: 12 new fits.** At 2.5 hrs/fit on 3 GPUs in parallel: ~10 hrs wall time (4 rounds of 3).

---

## Code Changes Required

### 1. `scripts/run_vbmc.py`

Add sensitivity flags:

```python
parser.add_argument('--high-weapon-prob', type=float, default=None)
parser.add_argument('--grid-size',        type=int,   default=None)
parser.add_argument('--severity-scale',   type=float, default=None)
parser.add_argument('--weibull-shape',    type=float, default=None)
parser.add_argument('--weibull-scale',    type=float, default=None)
parser.add_argument('--data-pkl',         type=str,   default=None,
                    help='Path to subsampled city pkl for Task D radius sweep.')
```

Replace the `nooptstate` sampler with the factory when any flag is set:

```python
'nooptstate': build_conditional_sampler(make_noopt_event(
    high_weapon_prob = args.high_weapon_prob or 0.10,
    grid_size        = args.grid_size        or 30,
    low_rate         = 6.0   * (args.severity_scale or 1.0),
    high_rate_add    = 120.0 * (args.severity_scale or 1.0),
    weibull_shape    = args.weibull_shape    or 2.0,
    weibull_scale    = args.weibull_scale    or 0.4,
)),
```

Encode the sweep value in the output directory so fits don't collide:

```python
# build a tag string from any non-default flags
tag_parts = []
if args.high_weapon_prob: tag_parts.append(f'hwp{args.high_weapon_prob}')
if args.grid_size:        tag_parts.append(f'gs{args.grid_size}')
if args.severity_scale:   tag_parts.append(f'sev{args.severity_scale}')
if args.weibull_shape:    tag_parts.append(f'wsh{args.weibull_shape}')
if args.weibull_scale:    tag_parts.append(f'wsc{args.weibull_scale}')
if args.data_pkl:         tag_parts.append(f'ffl{args.data_pkl[-6:-4]}')  # e.g. 'r15' from filename
tag = ('_' + '_'.join(tag_parts)) if tag_parts else ''
model_dir = f'/home/andrew/abm_violence/models/{model_name}{tag}/'
```

For Task D, load the per-radius pkl and pass it to the density function:

```python
city_data, weights = None, None
if args.data_pkl:
    import pickle
    with open(args.data_pkl, 'rb') as f:
        d = pickle.load(f)
    city_data = d['subsampled_city_data']
    weights   = d['subsample_weights']

density_func = get_param_density_func_noisy(
    sample_func, param_distr, log_prior, jrand.PRNGKey(2),
    num_samples=15000, cutoff_likelihood=cutoff_likelihood,
    likelihood_offset=likelihood_offset, logdir=model_dir,
    subsampled_city_data=city_data,   # None → uses default FFL dataset
    subsample_weights=weights,
)
```

`get_param_density_func_noisy` in `abm_vbmc.py` already accepts these kwargs — no changes needed there.

### 2. New: `scripts/collect_sensitivity_waic.py`

After all fits complete, this script scans model directories, calls `compute_sweep_waic` on each, and writes JSON files in the format `sensitivity_compare.py` already reads.

---

## Execution Plan

Build the Task D pkl files first (CPU, ~5 min):
```bash
python scripts/build_ffl_radius_data.py --data-dir /path/to/data
```

Then launch all 12 fits in 4 rounds of 3 parallel jobs. The baseline (nooptstate-v3) is already done.

**Round 1**
```bash
python scripts/run_vbmc.py --model=nooptstate --gpu 0 --high-weapon-prob 0.05 &
python scripts/run_vbmc.py --model=nooptstate --gpu 1 --high-weapon-prob 0.20 &
python scripts/run_vbmc.py --model=nooptstate --gpu 2 --grid-size 20 &
wait
```

**Round 2**
```bash
python scripts/run_vbmc.py --model=nooptstate --gpu 0 --grid-size 40 &
python scripts/run_vbmc.py --model=nooptstate --gpu 1 --data-pkl data/processed/subsampled_city_data_ffl_r15.pkl &
python scripts/run_vbmc.py --model=nooptstate --gpu 2 --data-pkl data/processed/subsampled_city_data_ffl_r30.pkl &
wait
```

**Round 3**
```bash
python scripts/run_vbmc.py --model=nooptstate --gpu 0 --severity-scale 0.75 &
python scripts/run_vbmc.py --model=nooptstate --gpu 1 --severity-scale 1.25 &
python scripts/run_vbmc.py --model=nooptstate --gpu 2 --weibull-shape 1.5 &
wait
```

**Round 4**
```bash
python scripts/run_vbmc.py --model=nooptstate --gpu 0 --weibull-shape 2.5 &
python scripts/run_vbmc.py --model=nooptstate --gpu 1 --weibull-scale 0.30 &
python scripts/run_vbmc.py --model=nooptstate --gpu 2 --weibull-scale 0.50 &
wait
```

Total wall time: ~10 hrs.

**Then collect WAIC and generate figures:**
```bash
python scripts/collect_sensitivity_waic.py
python scripts/sensitivity_compare.py
```

---

## What to Do with the Existing Sensitivity Scripts

Keep them — they're useful as a quick sanity check during development (minutes vs. hours). Don't include their output in the paper. After the refits complete, `sensitivity_compare.py` reads the refitted JSONs and the old fixed-posterior JSONs are simply unused.

