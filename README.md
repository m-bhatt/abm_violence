# abm_violence

Bayesian Agent-Based Model for forecasting mass shooting frequencies.
Manuscript under revision at PNAS Nexus (due May 8, 2026).

---

## Setup

```bash
pip install -e .
# GPU required (JAX/CUDA). Set CUDA_VISIBLE_DEVICES via --gpu flag on each script.
```

---

## Inference

Run VBMC on each model variant (~8–12 hrs each, GPU required):

```bash
python scripts/run_vbmc.py --model=nooptstate   # focal model → models/nooptstate-v3/
python scripts/run_vbmc.py --model=sample_select
python scripts/run_vbmc.py --model=max_select
python scripts/run_vbmc.py --model=mixed_choice
python scripts/run_vbmc.py --model=optstate
python scripts/run_vbmc.py --model=optstate_base
python scripts/run_vbmc.py --model=big_model
```

Resume a partial fit:
```bash
python scripts/run_vbmc_warmstart.py
python scripts/warm_start_sample.py   # extract posterior samples post-convergence
```

---

## Sensitivity Sweeps (Revision Tasks B, C, F, G)

Each script is independent; run on any GPU with the fitted `models/nooptstate-v3/` posterior.
Default flags: `--gpu 0 --n-posterior 50 --num-sim 10000`.

```bash
# Task B — high-weapon encounter probability (0.05–0.30)
python scripts/sensitivity_high_weapon.py --gpu 0

# Task C — grid resolution (20×20 to 40×40)
python scripts/sensitivity_grid_resolution.py --gpu 0

# Task F — severity rate scale (0.50×–1.50×)
python scripts/sensitivity_severity.py --gpu 0

# Task G — gathering-size Weibull shape and scale
python scripts/sensitivity_weibull.py --gpu 0
```

After all (or any subset of) the above have finished:

```bash
# Aggregate results → CSV table + comparison figure
python scripts/sensitivity_compare.py
```

Outputs:
- `models/sensitivity_*_results.json` — raw WAIC per sweep value
- `models/sensitivity_comparison_table.csv` — combined table (WAIC ± SE, Δ vs baseline)
- `paper_code/images/sensitivity_*.png` — individual sweep figures
- `paper_code/images/sensitivity_comparison.png` — multi-panel comparison figure

---

## Paper Figures & Analysis

```bash
jupyter notebook paper_code/abm_viz.ipynb
jupyter notebook paper_code/confidence_bounds.ipynb
jupyter notebook paper_code/posterier_distr.ipynb
jupyter notebook paper_code/waic_model_comparison.ipynb
python paper_code/causal_sweep.py
python paper_code/waic_model_eval.py
python paper_code/city_sample_stats.py
```
