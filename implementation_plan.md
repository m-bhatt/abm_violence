# Implementation Plan: Reviewer Revision Tasks

Tasks A–H from `reviewer_coding_todo.txt`. Each section covers what to change, which files to touch, and the execution plan. Tasks B–G are WAIC sweep sensitivity analyses that follow a common pattern described at the bottom.

---

## Task A: Firearm Share of Suicides as Alternative Proxy

**What it is:** Replace `ffl_density` (current-year ATF FFL counts, static) with firearm share of suicides (CDC WISQARS), which is available annually 1982–2024 and historically grounded.

**Key insight:** `arm_density` in `CityParams` is the only thing that needs to change. The ABM uses it as `arm_density * arm_weight + arm_bias`, so it just needs to be a comparable dimensionless density-like quantity.

### Step 1 — Acquire the data
- Source: CDC WISQARS API or pre-downloaded state-level files. The firearm share of suicides is: `firearm_suicide_count / total_suicide_count` per state per year.
- This gives a state-year matrix. You'll need to merge it to the city/sub-county level (use FIPS state code in `city_data.csv`).
- Fallback option: the Harvard Injury Control Research Center publishes state-level firearm ownership proxies (firearm suicide fraction) historically — this is publicly available and commonly cited.

### Step 2 — Build a new processed dataset
- In a new notebook `scripts/firearm_suicide_proxy.ipynb`:
  1. Load the state-year firearm suicide fractions.
  2. Join to `data/processed/city_data.csv` on state FIPS.
  3. For each city, interpolate across years (1982–2024) to produce a per-year `arm_density_suicide` field.
  4. Recompute `subsampled_city_data.pkl` with this new field in place of the FFL `arm_density`.
- Save as `data/processed/subsampled_city_data_suicide_proxy.pkl`.

### Step 3 — Modify the data loader
- In `gvabm/city_data.py`, add a second loader (don't remove the existing one):
  ```python
  def load_city_data(proxy='ffl'):  # proxy='ffl' or 'suicide'
      fname = 'subsampled_city_data.pkl' if proxy == 'ffl' else 'subsampled_city_data_suicide_proxy.pkl'
      with open(f'{data_dir}/processed/{fname}', 'rb') as f:
          ...
  ```
- Or simply parameterize `run_vbmc.py` with `--proxy ffl|suicide`.

### Step 4 — Run VBMC on the three focal models
```bash
python scripts/run_vbmc.py --model nooptstate --proxy suicide --gpu 0
python scripts/run_vbmc.py --model max_select --proxy suicide --gpu 1
python scripts/run_vbmc.py --model sample_select --proxy suicide --gpu 2
```
- Save outputs to `models/nooptstate-suicide/`, `models/max_select-suicide/`, etc.
- Posterior mean `arm_weight` will need re-interpretation: it now scales firearm suicide share rather than FFL density.

### Step 5 — Compare results
- In `paper_code/waic_model_comparison.ipynb`, add a table column for the suicide-proxy WAIC alongside the FFL-based WAIC.
- Expected deliverable: Table S-X showing WAIC for FFL vs. suicide proxy, demonstrating robustness (or flagging if results diverge materially).

---

## Task B: High-Firearm Availability Sensitivity (WAIC Sweep)

**What it is:** The `0.10` per-encounter probability of a high-lethality weapon is hard-coded. Vary it to 0.05, 0.20. 

**Key file:** `gvabm/abm_event.py`

The constant appears in three event functions:
- `sample_event_optstate_noopt` line 106: `p=1 - jnp.exp(-0.1 * weapon_rolls)`
- `sample_event_optstate` line 40: same
- `sample_event_intervention` line 263: same

### Step 1 — Make the constant a parameter

Create a new family of sample functions that accept `high_weapon_prob` as an argument:

```python
def sample_event_optstate_noopt_parameterized(walk_radius, population_density, arm_density,
                                               atrisk_gathering_rate, high_weapon_prob, rng_key):
    # ... (identical to sample_event_optstate_noopt except:)
    high_weapon_access = jrand.bernoulli(subkey, p=1 - jnp.exp(-high_weapon_prob * weapon_rolls))
    # expon_lambda stays hardcoded at 6 + 120 * high_weapon_access
```

Or more simply, wrap the existing function with `functools.partial`:
```python
import functools

def make_event_func(high_weapon_prob=0.10):
    def sample_event(..., rng_key):
        ...
        high_weapon_access = jrand.bernoulli(subkey, p=1 - jnp.exp(-high_weapon_prob * weapon_rolls))
        ...
    return sample_event
```

### Step 2 — Write the sweep script

New file: `scripts/sensitivity_high_weapon.py`

```python
SWEEP_VALUES = [0.05, 0.20]

# Load posterior mean from nooptstate-v3
posterior_mean = load_posterior_mean('models/nooptstate-v3/')

results = []
for hwp in SWEEP_VALUES:
    sample_func = build_conditional_sampler(make_event_func(hwp))
    city_lls = eval_ds(subsampled_city_data, posterior_mean, timespan=10, rng_key=...,
                       num_samples=15000, sample_func=sample_func)
    # WAIC requires posterior samples, not just posterior mean.
    # For a quick sensitivity sweep, compute total weighted LL as a proxy,
    # OR draw ~100 posterior samples and compute WAIC properly.
    results.append({'hwp': hwp, 'll': city_lls})

# Plot WAIC (or LL) vs. high_weapon_prob
```

**Note on WAIC:** Full WAIC needs posterior samples. For this sweep, using total log-likelihood at the posterior mean is a reasonable and much cheaper proxy unless the reviewer specifically asks for WAIC (they said "compute WAIC at each value"). If full WAIC is needed, draw 50–100 samples from the `nooptstate-v3` VP and evaluate `eval_ds` for each — see `scripts/warm_start_sample.py` for how to extract VP samples.

### Step 3 — Output

Plot: WAIC (y-axis) vs. `high_weapon_prob` (x-axis), with 0.10 marked as the baseline. Save to `paper_code/images/sensitivity_high_weapon.png`.

---

## Task C: Grid Resolution Sensitivity

**What it is:** The 30×30 mile grid is hard-coded everywhere. Test: 20×20, 25×25, 35×35, 40×40.

**Key files:** `gvabm/abm_event.py` (all sample functions use `(30, 30)` hard-coded)

### Step 1 — Understand what "grid size" controls

The grid is `N×N` cells. It controls:
1. How many cells the agent walks across (affects `weapon_rolls`, `population_encounters`)
2. `population_grid.size = N²` → used in `total_lambda = arm_density * N²`
3. `walk_mask = jnp.arange(100) < walk_radius` — `walk_radius` range (5–95 steps) is implicitly relative to N=30. For other grid sizes you'd want to rescale `walk_radius` bounds, or just hold them constant and note that the walk covers more/less of the grid.

The simplest approach: the grid represents a fixed physical area (30×30 miles), so changing the resolution means the cell size changes, and population/arm density per cell scales accordingly.

### Step 2 — Create parameterized sample functions

Add `grid_size` as an argument to `sample_event_optstate_noopt` (or create a factory):

```python
def make_event_func_with_grid(grid_size=30, high_weapon_prob=0.10):
    def sample_event_noopt(walk_radius, population_density, arm_density,
                           atrisk_gathering_rate, rng_key):
        pgrid = (jrand.uniform(rng_key, (grid_size, grid_size)) > 0.95).astype(jnp.float32)
        # ... kernel, convolution same logic ...
        population_grid = pgrid * population_density
        # arm_density scaling: must hold total arms constant as grid changes
        # total_lambda = arm_density * population_grid.size keeps total arms constant
        total_lambda = arm_density * population_grid.size
        ...
    return sample_event_noopt
```

**Important:** `walk_radius` range (5–95) is in grid steps. For a 20×20 grid, a radius of 95 steps wraps the grid ~4.75 times; for 40×40 it wraps ~2.4 times. You may want to hold the physical walk distance constant by rescaling: `walk_radius_scaled = walk_radius * (grid_size / 30)`. Include a note in the paper about how you handled this.

### Step 3 — Preprocessing: recompute city params at each resolution

The `population_density` and `arm_density` in `CityParams` are per-cell quantities. If the grid represents 30×30 miles regardless of N, then:
- Population per cell = (total population / N²) × (30/N)² ... actually this is a dimensionless spatial density issue.
- Simpler: keep `population_density` as the total population in the grid cell (not per-sub-cell), and let the grid just control spatial resolution of the random walk. This is probably the intended interpretation — changing grid resolution tests whether spatial heterogeneity matters.

Check `scripts/city_pop_density.ipynb` to confirm how `population_density` is defined before proceeding.

### Step 4 — Run the sweep

New file: `scripts/sensitivity_grid_resolution.py`

```python
GRID_SIZES = [20, 25, 30, 35, 40]  # 30 is baseline
posterior_mean = load_posterior_mean('models/nooptstate-v3/')

for gs in GRID_SIZES:
    sample_func = build_conditional_sampler(make_event_func_with_grid(grid_size=gs))
    city_lls = eval_ds(subsampled_city_data, posterior_mean, 10, rng_key, 
                       num_samples=15000, sample_func=sample_func)
    ...
```

### Step 5 — Output

Plot WAIC vs. grid size (N). If WAIC is flat, grid choice doesn't matter. Include as supplementary figure.

---

## Task D: FFL Density Buffer Radius Sensitivity

**What it is:** FFL density is currently computed using a 15-mile radius buffer around each sub-county centroid (based on `ffl_within_25_miles` column in city_data.csv — note the 25-mile column exists; the 15-mile claim in the todo may be using a different column or the density normalization). Test: 5, 10, 20, 25 miles.

**Key files:** `scripts/city_ffl_density.ipynb` (FFL density preprocessing), `data/processed/city_data.csv`

### Step 1 — Understand the current computation

Open `scripts/city_ffl_density.ipynb` and find where the spatial join / buffer radius is set. It likely:
1. Loads `data/0124-ffl-list-complete.txt` (FFL locations with lat/lon)
2. For each sub-county centroid, counts FFLs within X miles
3. Normalizes by area → `ffl_density`

The `city_data.csv` already has `ffl_within_25_miles` (count) and `ffl_density`. The current `ffl_density` column uses a specific radius — confirm which one.

### Step 2 — Recompute for each radius

Extend `scripts/city_ffl_density.ipynb` (or create `scripts/sensitivity_ffl_radius.ipynb`):

```python
RADII = [5, 10, 15, 20, 25]  # miles

for radius in RADII:
    # Spatial join: count FFLs within `radius` miles of each centroid
    ffl_counts = count_ffls_within_radius(city_centroids, ffl_locations, radius_miles=radius)
    ffl_density = ffl_counts / (np.pi * radius**2)  # or per-city-area normalization
    city_df[f'ffl_density_{radius}mi'] = ffl_density

city_df.to_csv('data/processed/city_data_ffl_sensitivity.csv')
```

Use `scipy.spatial.cKDTree` for fast nearest-neighbor counting, or a geodesic distance function if precision matters (at these scales, Euclidean in projected coordinates is fine).

### Step 3 — Build city datasets for each radius

For each radius, rebuild `subsampled_city_data_{r}mi.pkl` with the new `arm_density` column, keeping all other city fields identical.

### Step 4 — Run WAIC sweep

New file: `scripts/sensitivity_ffl_radius.py`

```python
RADII = [5, 10, 15, 20, 25]
posterior_mean = load_posterior_mean('models/nooptstate-v3/')

for r in RADII:
    city_data = load_city_data_with_radius(r)
    city_lls = eval_ds(city_data, posterior_mean, 10, rng_key, num_samples=15000,
                       sample_func=nooptstate_sample_func)
    waic = compute_waic(city_lls, subsample_weights)
    results.append({'radius': r, 'waic': waic})
```

### Step 5 — Output

Plot WAIC vs. buffer radius. Also report the posterior mean of `arm_weight` under each radius to show whether the FFL coefficient is stable. If WAIC peaks sharply at one radius, discuss it.

---

## Task E: Simulation Access Radius Sensitivity (Prior Range Sensitivity)

**What it is:** Vary the prior specification on `arm_bias` and `arm_weight` — the parameters that scale how FFL density enters the model — and check that posterior conclusions are robust to the prior range.

**Key file:** `gvabm/param_distr.py` — currently `arm_bias: (-5, 2)`, `arm_weight: (-5, 3)` in log-space.

### Step 1 — Define alternative prior configurations

In `gvabm/param_distr.py`, add new config variants:

```python
# Narrow prior
param_config_narrow_arm = {
    "entry_labels": [...],
    "entry_ranges": {
        ...
        "arm_bias": {"range": (-4, 1), "logscale": True},
        "arm_weight": {"range": (-4, 2), "logscale": True},
        ...
    },
    "data_wrapper": GeneralParams
}

# Wide prior
param_config_wide_arm = {
    "entry_ranges": {
        ...
        "arm_bias": {"range": (-6, 3), "logscale": True},
        "arm_weight": {"range": (-6, 4), "logscale": True},
        ...
    },
    ...
}
```

### Step 2 — Run VBMC under each prior

Add `--prior_spec` flag to `scripts/run_vbmc.py`:

```bash
python scripts/run_vbmc.py --model nooptstate --prior_spec narrow_arm --gpu 0
python scripts/run_vbmc.py --model nooptstate --prior_spec wide_arm --gpu 1
```

Save to `models/nooptstate-prior-narrow/` and `models/nooptstate-prior-wide/`.

### Step 3 — Compare posteriors

Report: posterior mean ± SD for `arm_bias` and `arm_weight` under each prior. If posteriors are data-dominated (similar across priors), conclude robustness. If prior-dependent, flag as a limitation.

**Alternative interpretation:** The todo says "separately vary the scale at which FFLs are placed within the 30×30 grid (alpha, beta prior ranges)." If this refers to the Beta(5,5) prior hyperparameters rather than the parameter bounds, vary alpha and beta: e.g., Beta(2,2) (wider/flatter), Beta(5,5) (current), Beta(10,10) (tighter/more concentrated). This is a single-line change in `run_vbmc.py` line 64: `log_prior = param_distr.get_log_beta_prior(alpha=5.0, beta=5.0)`.

---

## Task F: Severity Model Sensitivity

**What it is:** Hard-coded exponential rates `expon_lambda = 6 + 120 * high_weapon_access` (low=6, high=126). Test ±50%: low ∈ {3, 6, 9}, high_increment ∈ {60, 120, 180}.

**Key file:** `gvabm/abm_event.py` — line 68 (optstate), line 121 (optstate_noopt), line 175 (with_meta_optstate_noopt), line 295 (intervention), line 223 (base).

### Step 1 — Make rates configurable

Add a factory approach:

```python
def make_event_func_with_severity(low_rate=6, high_rate=126, grid_size=30, high_weapon_prob=0.10):
    def sample_event_noopt(walk_radius, population_density, arm_density,
                           atrisk_gathering_rate, rng_key):
        ...
        expon_lambda = low_rate + (high_rate - low_rate) * high_weapon_access
        fatalities = jnp.remainder(jrand.exponential(subkey, shape=()) * expon_lambda, gathering_size)
        ...
    return sample_event_noopt
```

### Step 2 — Define the sweep grid

```python
LOW_RATES = [3, 6, 9]          # ±50% of baseline 6
HIGH_RATES = [63, 126, 189]    # ±50% of baseline 126

# Full 3×3 grid, or just the diagonal (proportional scaling)
SWEEP = [(low, high) for low in LOW_RATES for high in HIGH_RATES]
```

### Step 3 — Run the sweep

New file: `scripts/sensitivity_severity.py`

```python
posterior_mean = load_posterior_mean('models/nooptstate-v3/')

for (low_rate, high_rate) in SWEEP:
    sample_func = build_conditional_sampler(make_event_func_with_severity(low_rate, high_rate))
    city_lls = eval_ds(subsampled_city_data, posterior_mean, 10, rng_key, 15000, sample_func)
    waic = compute_waic(city_lls, subsample_weights)
    results.append({'low_rate': low_rate, 'high_rate': high_rate, 'waic': waic})
```

### Step 4 — Output

Heatmap: WAIC as a function of (low_rate, high_rate). Key question: does the model ranking (nooptstate > max_select > sample_select) change? If WAIC is insensitive to ±50% variation, the assumption is robust.

---

## Task G: Gathering Size Distribution Sensitivity

**What it is:** Weibull(shape=2, scale=0.4) for gathering size is manually tuned. Sweep shape ∈ {1.5, 2.0, 2.5} × scale ∈ {0.2, 0.4, 0.6}.

**Key file:** `gvabm/abm_event.py` — `jrand.weibull_min(subkey, 2, 0.4, ...)` (shape=2, scale=0.4).

Note: JAX's `weibull_min(key, concentration, scale, shape)` — first positional arg after key is `concentration` (= shape parameter k), second is `scale`.

### Step 1 — Parameterize gathering size

```python
def make_event_func_with_gathering(weibull_shape=2.0, weibull_scale=0.4):
    def sample_event_noopt(walk_radius, population_density, arm_density,
                           atrisk_gathering_rate, rng_key):
        ...
        loc_gathering_size = jrand.weibull_min(subkey, weibull_shape, weibull_scale, (max_location+5,))
        ...
    return sample_event_noopt
```

### Step 2 — Sweep

```python
SHAPES = [1.5, 2.0, 2.5]
SCALES = [0.2, 0.4, 0.6]
posterior_mean = load_posterior_mean('models/nooptstate-v3/')

for shape in SHAPES:
    for scale in SCALES:
        sample_func = build_conditional_sampler(make_event_func_with_gathering(shape, scale))
        city_lls = eval_ds(subsampled_city_data, posterior_mean, 10, rng_key, 15000, sample_func)
        waic = compute_waic(city_lls, subsample_weights)
        results.append({'shape': shape, 'scale': scale, 'waic': waic})
```

### Step 3 — Output

Heatmap: WAIC vs. (Weibull shape, scale). Include as supplementary figure. Discuss whether any gathering size parameterization substantially outperforms the manually chosen one — if so, it motivates inferring these as parameters (see `big_model`).

---

## Task H: City-Level Risk Predictions (Top 500 Cities)

**What it is:** Use posterior samples from the focal model (`nooptstate-v3`) to predict (i) event probability per decade and (ii) expected fatalities per capita per decade, for all 500 cities in the dataset (or all cities in `city_data.csv`).

**Key files:**
- `models/nooptstate-v3/vp.pkl` — fitted variational posterior (use `dill.load`)
- `data/processed/city_data.csv` — full city list with demographics
- `data/processed/subsampled_city_data.pkl` — has the event data but only 500 subsampled cities

### Step 1 — Load posterior samples

```python
import dill

with open('models/nooptstate-v3/vp.pkl', 'rb') as f:
    vp = dill.load(f)

# VP has a .sample() method
posterior_samples_normalized = vp.sample(500)  # shape (500, D)

# Transform from [0,1] normalized space to actual parameter space
param_distr = ParamDistr(param_config)
posterior_samples = [param_distr.transform_sample(jnp.array(s)) for s in posterior_samples_normalized]
posterior_samples = [param_distr.to_data(np.array(s)) for s in posterior_samples]
```

### Step 2 — Load all cities (not just the subsampled 500)

```python
city_df = pd.read_csv('data/processed/city_data.csv')

# Build CityParams for each city (use 2020 population and density, year=2024)
all_city_params = [
    CityParams(year=2020, population=row.population_2020,
               population_density=row.CL8AA2020_density,
               arm_count=row.ffl_within_25_miles,
               arm_density=row.ffl_density)
    for _, row in city_df.iterrows()
]
```

Alternatively, use only the 500 subsampled cities if the full `city_data.csv` preprocessing is incomplete.

### Step 3 — Compute predictions

For each city, average over posterior samples:

```python
sample_func = build_conditional_sampler(sample_event_optstate_noopt)
timespan = 10  # decade

city_predictions = []
for city_params in all_city_params:
    event_rates = []
    expected_fatalities = []
    
    for params in posterior_samples[::10]:  # thin to 50 samples for speed
        rng_key, subkey = jrand.split(rng_key)
        event_sim = sample_func(city_params, params, subkey, num_samples=10000)
        
        # P(event) = P(fatalities >= 5)
        prob_event = jnp.mean(event_sim >= 5)
        # Expected events per decade
        expected_events = params.propensity * timespan * city_params.population * prob_event
        # Expected fatalities per event (conditional on event occurring)
        event_mask = event_sim >= 5
        mean_fatalities = jnp.where(event_mask.any(), jnp.mean(event_sim[event_mask]), 0.0)
        
        event_rates.append(float(expected_events))
        expected_fatalities.append(float(expected_events * mean_fatalities))
    
    city_predictions.append({
        'city': city_params,
        'expected_events_per_decade': np.mean(event_rates),
        'expected_events_per_decade_sd': np.std(event_rates),
        'expected_fatalities_per_decade': np.mean(expected_fatalities),
        'expected_fatalities_per_capita_per_decade': np.mean(expected_fatalities) / city_params.population,
    })
```

### Step 4 — Output: ranked table and/or map

```python
pred_df = pd.DataFrame(city_predictions).sort_values('expected_events_per_decade', ascending=False)

# Table: top 20 cities by event probability
print(pred_df[['city_name', 'expected_events_per_decade', 'expected_fatalities_per_capita_per_decade']].head(20))

# Map: choropleth using geopandas or folium
# Color = expected events per decade, filtered to top 500
```

Expected deliverable: a figure showing top-risk cities (map or ranked bar chart), and a supplementary table of all 500 with credible intervals.

**Note on timing:** This task requires no new VBMC runs — just posterior sample evaluation. Estimated runtime: ~2–4 hours depending on how many posterior samples and cities. The 500-city subsampled dataset is the right starting point; use the full `city_data.csv` only if city names are needed for presentation.

---

## Common Infrastructure for Tasks B–G (Sensitivity Sweeps)

All sensitivity sweeps share a common pattern. Consider building a shared utility:

### Shared function: `scripts/sensitivity_utils.py`

```python
import dill
import numpy as np
import jax.random as jrand
from gvabm.abm_ll import eval_ds
from gvabm.waic_eval import WAIC_metric
from gvabm.param_distr import ParamDistr, param_config
from gvabm.city_data import subsampled_city_data, subsample_weights

def load_posterior_mean(model_dir, n_samples=100):
    with open(f'{model_dir}/vp.pkl', 'rb') as f:
        vp = dill.load(f)
    samples = vp.sample(n_samples)  # shape (n_samples, D)
    return samples.mean(axis=0)     # posterior mean in [0,1] space

def load_posterior_samples(model_dir, n_samples=100):
    with open(f'{model_dir}/vp.pkl', 'rb') as f:
        vp = dill.load(f)
    return vp.sample(n_samples)    # for proper WAIC

def compute_sweep_waic(sample_func, params_normalized, param_distr,
                       n_posterior_samples=50, num_sim_samples=10000,
                       rng_key=None, model_dir='models/nooptstate-v3/'):
    """
    Compute WAIC for a given sample_func with posterior samples from model_dir.
    Uses n_posterior_samples posterior draws for WAIC variance estimation.
    """
    if rng_key is None:
        rng_key = jrand.PRNGKey(42)
    
    posterior_samples = load_posterior_samples(model_dir, n_samples=n_posterior_samples)
    
    ll_mat = []  # shape (n_cities, n_posterior_samples)
    for s in posterior_samples:
        rng_key, subkey = jrand.split(rng_key)
        params = param_distr.transform_sample(jnp.array(s))
        params = param_distr.to_data(np.array(params))
        city_lls = eval_ds(subsampled_city_data, params, 10, subkey,
                           num_samples=num_sim_samples, sample_func=sample_func)
        ll_mat.append(city_lls[:, 0])  # total ll per city
    
    ll_mat = np.array(ll_mat).T[:, :, np.newaxis]  # (n_cities, n_samples, 1)
    waic, waic_se = WAIC_metric(ll_mat, subsample_weights)
    return waic, waic_se
```

### Execution priority

Roughly ordered by reviewer importance and implementation complexity:

1. **H** (city predictions) — highest reviewer visibility, no new runs, ~1 day
2. **B** (high-weapon prob) — fast sweep, single parameter, ~1 day  
3. **F** (severity rates) — 9-point grid, same structure as B, ~1–2 days
4. **G** (Weibull gathering) — 9-point grid, ~1–2 days
5. **D** (FFL buffer radius) — requires data preprocessing, ~2 days
6. **C** (grid resolution) — requires ABM restructuring + possible data rescaling, ~2–3 days
7. **E** (prior sensitivity) — quick (just config changes + VBMC runs), ~2 days + GPU time
8. **A** (suicide proxy) — most complex (new data source + 3 full VBMC runs), ~3–5 days + GPU time

### Runtime estimate per sweep point

- Single `eval_ds` call (500 cities × 15k samples): ~3–5 minutes on GPU
- 50 posterior samples × 6 sweep points: ~15–25 hours total per task
- Recommendation: run B, C, F, G sweeps at the **posterior mean only** (1 eval per point, ~18–30 min per task) unless reviewers specifically demand full WAIC. This is defensible as a local sensitivity analysis.
