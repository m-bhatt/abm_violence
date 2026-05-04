# ABM Gun Violence — Project Context

## What This Is

A Bayesian agent-based model (ABM) for predicting mass shooting frequencies and severity distributions across US cities. The manuscript, "Forecasting the future of American mass shootings with Bayesian Agent-Based Model calibration," is under major revision at **PNAS Nexus** (MS# PNASNEXUS-2025-01683). **Revision due May 8, 2026.**

Calibration uses VBMC (Variational Bayes Monte Carlo via `pyvbmc`) on historical mass shooting data (1982–2024), enabling model comparison via WAIC.

---

## Repository Layout

```
gvabm/              # Core pip-installable package
scripts/            # Inference scripts (run_vbmc.py is the main entry point)
paper_code/         # Analysis and figures for the paper
mira_code/          # Collaborator data-engineering notebooks
data/               # Raw data sources
data/processed/     # Cleaned datasets consumed by the model
models/             # VBMC fit outputs (one subdir per model variant)
```

---

## Core Package: `gvabm/`

| Module | Role |
|--------|------|
| `abm_event.py` | 10 JAX-accelerated event-sampling variants (`sample_select`, `max_select`, `mixed_choice`, `optstate`, etc.) |
| `abm_ll.py` | Log-likelihood: `city_ll()`, `eval_ds()`, `eval_ds_fast()` (batch over 500 cities) |
| `abm_vbmc.py` | VBMC-compatible likelihood wrapper; includes Beta(5,5) prior |
| `param_distr.py` | Parameter bounds, log-scale transforms, normalization (`ParamDistr` class) |
| `stat_utils.py` | Grenander isotonic regression for non-parametric fatality PDFs (Numba-accelerated) |
| `waic_eval.py` | `WAIC_metric()`, `estimate_PLLD()`, `estimate_WAIC_correction()` |
| `city_data.py` | Loads `data/processed/subsampled_city_data.pkl` |
| `mcmc.py` | Legacy Metropolis-Hastings (superseded by VBMC) |
| `abm_mcmc.py` | Legacy MCMC interface wrapper |
| `abm_eval.py` | Grid search over parameter space for preliminary analysis |

---

## ABM Structure

The model simulates a single mass shooting event on a 30×30 mile grid cell. Five generative stages:

1. **Agent emergence** — random walk up to `walk_radius` steps; start location weighted by population density
2. **Firearm acquisition** — access probability from FFL density; `arm_bias + arm_weight * arm_density`; high-lethality weapon: `1 - exp(-0.1 * weapon_rolls)`
3. **Target selection** — at-risk locations ~ Poisson(`population_encounters × atrisk_gathering_rate`); gathering sizes ~ Weibull(shape=2, scale=0.4)
4. **Casualty generation** — fatalities ~ min(Exponential(λ), gathering_size); threshold ≥5 fatalities = reportable event
5. **Event rate** — `propensity × timespan × population × P(fatalities ≥ 5)`

### Parameter Vectors

| Config | Params (D) | Extra params |
|--------|-----------|--------------|
| `GeneralParams` | 5 | `propensity, walk_radius, arm_bias, arm_weight, atrisk_gathering_rate` |
| `MixedGeneralParams` | 6 | + `mix_perc` |
| `BigGeneralParams` | 8 | + `mix_perc, scale_factor, decay_rate` |

---

## Model Variants

Seven competing models; VBMC run independently on each:

| Variant | D | Key feature |
|---------|---|-------------|
| **nooptstate-v3** (focal/submitted) | 5 | No optimization; weapon access varies |
| **sample_select** | 5 | Random target selection (pure baseline) |
| **max_select** | 5 | Perpetrator targets largest gathering |
| **mixed_choice** | 6 | `mix_perc` mixture of random + max |
| **optstate** | 6 | Optimization + high-weapon bias |
| **optstate_base** | 5 | Simpler optimization variant |
| **big_model** | 8 | Flexible Weibull severity parameterization |

Fitted models saved under `models/<variant>/`.

---

## Inference Pipeline

1. **Sampling:** `vmap` over 10,000 RNG keys → batch ABM events in JAX
2. **PDF estimation:** Grenander isotonic regression → monotone decreasing fatality PDF
3. **City likelihood:** `Poisson(count | λ) × ∏ PDF(fatalities_i)` across 500 subsampled cities
4. **VBMC calibration:** ~90×(D+2) likelihood evaluations → posterior mean + covariance
5. **Model comparison:** WAIC computed from posterior samples

### Running VBMC

```bash
python scripts/run_vbmc.py --model=nooptstate
# Variants: max_select, sample_select, mixed_choice, big_model, optstate, nooptstate, optstate_base
python scripts/run_vbmc_warmstart.py  # resume from saved VP state
python scripts/warm_start_sample.py   # extract posterior samples post-convergence
```

GPU required (JAX/CUDA). Single fit takes ~8–12 hours.

---

## Data

### Key Processed Files (what the model actually consumes)

| File | Contents |
|------|----------|
| `data/processed/subsampled_city_data.pkl` | 500 cities with demographics + shooting event lists (weighted subsample) |
| `data/processed/city_data.csv` | City demographics: population, density, gun density |
| `data/processed/event_data.csv` | Parsed shooting events: date, location, fatalities, injuries |

### Raw Sources

| File | Source |
|------|--------|
| `data/gunviolencearchive_allyears.csv` | Gun Violence Archive (1982–2024) |
| `data/mother_jones.csv` | Mother Jones mass shooting database |
| `data/0124-ffl-list-complete.txt` | ATF Federal Firearms License list (~135k dealers) |
| `data/nhgis0001_csv/` | NHGIS county-level time-series demographics |
| Census files | Population, land area by county |

---

## Outstanding Revision Tasks (due May 8, 2026)

From `reviewer_coding_todo.txt` — these are the critical coding items:

- **A (HIGH PRIORITY):** Replace FFL density proxy with firearm share of suicides (1982–2024 historical data). Requires VBMC re-run on all 3 main models.
- **B–G:** Sensitivity sweeps (WAIC heatmaps for each):
  - B) Grid resolution: 20×20, 25×25, 35×35, 40×40 mile cells
  - C) High-weapon encounter probability: 0.05–0.30
  - D) FFL buffer radius: 5, 10, 15, 20, 25 miles
  - E) Severity exponential rates ±50%
  - F) Gathering size Weibull parameters (manual sweep)
  - G) Combined sensitivity
- **H:** City-level risk predictions for top 500 cities
- **J:** Downsampling robustness (3 independent subsamples, compare WAIC)
- **K:** Negative binomial baseline model comparison

See `reviewer_todo.txt` for narrative reviewer responses and `reviewer_coding_todo.txt` for the coding checklist.

---

## Paper Notebooks (`paper_code/`)

| Notebook / Script | Output |
|-------------------|--------|
| `abm_viz.ipynb` | ABM sample visualizations (sanity checks) |
| `confidence_bounds.ipynb` | Posterior uncertainty bands on predictions |
| `posterier_distr.ipynb` | Marginal/joint posterior distributions |
| `waic_model_comparison.ipynb` | WAIC rankings table |
| `causal_sweep.py` | Intervention simulations (vary firearm density, pop. density) |
| `waic_model_eval.py` | Likelihood surface heatmaps |
| `city_sample_stats.py` | Summary statistics by city |

---

## Dependencies

- **JAX + jax-gpu** — vectorized ABM simulation
- **pyvbmc** — VBMC inference
- **numba** — Grenander regression acceleration
- **scipy, numpy, pandas** — standard scientific stack
- **matplotlib** — figures
- **dill** — serializing VP (variational posterior) objects

GPU with CUDA required for reasonable runtimes. Scripts use `CUDA_VISIBLE_DEVICES` for GPU selection.

---

## Key Design Decisions / Non-Obvious Things

- **Grenander vs. parametric:** Fatality distributions are estimated non-parametrically to avoid assuming a family (e.g., Gamma). This is critical for capturing the heavy tail.
- **City subsampling:** 500 cities are weighted-subsampled to balance event cities vs. no-event cities; upsamples high-population and event cities. Robustness to this choice is a pending reviewer concern (task J).
- **FFL as proxy:** Current (2024) FFL density is used as a historical proxy for 1982–2024 firearm access — a known limitation flagged by Reviewer 1. Task A is the fix.
- **Threshold ≥5 fatalities:** Follows FBI definition of a mass shooting. Events below this threshold are simulated but not counted in the likelihood.
- **VBMC over MCMC:** Chosen for sample efficiency (~90D evaluations vs. 1000+D for MCMC) and because it provides WAIC directly from the posterior fit.
