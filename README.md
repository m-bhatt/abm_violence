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

## What this produces

| Output file | Script | Description |
|---|---|---|
| `models/nooptstate-v3/` | `run_vbmc.py --model=nooptstate` | Fitted variational posterior (focal model) |
| `models/<variant>/` | `run_vbmc.py --model=<variant>` | Fitted VP for each competing model |
| `models/sensitivity_high_weapon_results.json` | `sensitivity_high_weapon.py` | WAIC vs. high-weapon probability (Task B) |
| `models/sensitivity_grid_resolution_results.json` | `sensitivity_grid_resolution.py` | WAIC vs. grid size (Task C) |
| `models/sensitivity_ffl_radius_results.json` | `sensitivity_ffl_radius.py` | WAIC vs. FFL buffer radius (Task D) |
| `models/sensitivity_severity_results.json` | `sensitivity_severity.py` | WAIC vs. severity rate scale (Task F) |
| `models/sensitivity_weibull_{shape,scale}_results.json` | `sensitivity_weibull.py` | WAIC vs. Weibull params (Task G) |
| `models/sensitivity_comparison_table.csv` | `sensitivity_compare.py` | All sweeps combined: WAIC ± SE, Δ vs baseline |
| `models/city_risk_predictions.csv` | `city_risk_predictions.py` | Expected events + fatalities/capita per city (Task H) |
| `paper_code/images/sensitivity_*.png` | sensitivity scripts | Per-sweep WAIC plots |
| `paper_code/images/sensitivity_comparison.png` | `sensitivity_compare.py` | Multi-panel comparison figure |
| `paper_code/images/city_risk_events.png` | `city_risk_predictions.py` | Top-10 cities by expected events/decade |
| `paper_code/images/city_risk_fatalities.png` | `city_risk_predictions.py` | Top-10 cities by fatalities/capita/decade |

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

## Sensitivity Sweeps (Revision Tasks B, C, D, F, G)

Each sweep is independent; all require the fitted `models/nooptstate-v3/` posterior.
Default flags: `--gpu 0 --n-posterior 50 --num-sim 10000` (~2–4 hrs each).

```bash
# Task B — high-weapon encounter probability (0.05–0.30)
python scripts/sensitivity_high_weapon.py --gpu 0

# Task C — grid resolution (20×20 to 40×40)
python scripts/sensitivity_grid_resolution.py --gpu 0

# Task D — FFL buffer radius (7.5, 15, 25, 30 miles)
#   Step 1: rebuild city pkl files at each radius (CPU, ~5 min)
python scripts/build_ffl_radius_data.py --data-dir /path/to/data
#   Step 2: WAIC sweep using the rebuilt pkls
python scripts/sensitivity_ffl_radius.py --gpu 0 --data-dir /path/to/data

# Task F — severity rate scale (0.50×–1.50×)
python scripts/sensitivity_severity.py --gpu 0

# Task G — gathering-size Weibull shape and scale (two 1-D sweeps)
python scripts/sensitivity_weibull.py --gpu 0
```

Aggregate all completed sweeps into one table and figure:
```bash
python scripts/sensitivity_compare.py
```

---

## City Risk Predictions (Revision Task H)

Requires the fitted `models/nooptstate-v3/` posterior (~2–4 hrs, GPU required).

```bash
python scripts/city_risk_predictions.py --gpu 0   # default: --n-posterior 100 --num-sim 5000
```

Produces a ranked table of all 500 cities with posterior mean ± std for:
- expected mass shootings / decade
- P(≥1 event / decade)
- expected fatalities / decade
- fatalities / capita / decade

---

## Paper Figures & Analysis

```bash
jupyter notebook paper_code/abm_viz.ipynb             # ABM sample visualisations
jupyter notebook paper_code/confidence_bounds.ipynb   # posterior uncertainty bands
jupyter notebook paper_code/posterier_distr.ipynb     # marginal/joint posteriors
jupyter notebook paper_code/waic_model_comparison.ipynb  # WAIC rankings table
jupyter notebook paper_code/city_risk_factors.ipynb   # city-level risk factor plots
jupyter notebook paper_code/arm_density_plots.ipynb   # FFL/arm density visualisation
jupyter notebook paper_code/data_viz.ipynb            # raw data overview
jupyter notebook paper_code/simulation_statistics.ipynb  # ABM simulation diagnostics
python paper_code/causal_sweep.py                     # intervention simulations
python paper_code/waic_model_eval.py                  # likelihood surface heatmaps
python paper_code/city_sample_stats.py                # per-city posterior statistics
```
