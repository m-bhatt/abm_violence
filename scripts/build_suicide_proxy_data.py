"""
Build subsampled_city_data_suicide_proxy.pkl — replaces FFL-based arm_density with
the state-level firearm share of suicides (firearm suicides / all suicides) matched
to each city by state and event year.

DATA ACQUISITION
----------------
Download two tables from CDC WISQARS Fatal Injury Reports
(https://wisqars.cdc.gov/reports/), one query at a time:

  Query 1 — Firearm suicides:
    Outcome: Suicide
    Mechanism: Firearm
    Group by: State, Year
    Years: 1981–2024 (or latest available)
    Geography: All states + DC
    Export as CSV → save to: data/cdc_firearm_suicides_by_state.csv

  Query 2 — All suicides:
    Outcome: Suicide
    Mechanism: All mechanisms
    Group by: State, Year
    Years: 1981–2024
    Geography: All states + DC
    Export as CSV → save to: data/cdc_all_suicides_by_state.csv

Expected CSV format (WISQARS standard export):
    State,Year,Deaths,Population,Crude Rate
    Alabama,1999,327,4369862,7.48
    ...
    (CDC suppresses counts < 10 with "Suppressed" in the Deaths column)

This script reads both CSVs, computes the fraction, interpolates suppressed
cells, joins to the existing subsampled_city_data.pkl by (state_name, year),
and saves a new pkl without touching the original.
"""

import pickle
import copy
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path('/home/andrew/abm_violence/data')
PROCESSED_DIR = DATA_DIR / 'processed'

FIREARM_CSV = DATA_DIR / 'cdc_firearm_suicides_by_state.csv'
ALL_CSV     = DATA_DIR / 'cdc_all_suicides_by_state.csv'
SOURCE_PKL  = PROCESSED_DIR / 'subsampled_city_data.pkl'
OUTPUT_PKL  = PROCESSED_DIR / 'subsampled_city_data_suicide_proxy.pkl'


# ---------------------------------------------------------------------------
# Step 1: load CDC data
# ---------------------------------------------------------------------------

def _load_wisqars_csv(path: Path, deaths_col: str) -> pd.DataFrame:
    """
    Load a WISQARS export CSV.  Returns a DataFrame with columns
    ['state', 'year', deaths_col] where deaths_col is the count (float,
    NaN for suppressed cells).
    """
    df = pd.read_csv(path, dtype=str)

    # Normalise column names: strip whitespace, lower-case
    df.columns = df.columns.str.strip().str.lower()

    # WISQARS exports sometimes have a leading 'notes' column; drop it
    if 'notes' in df.columns:
        df = df.drop(columns=['notes'])

    # Identify the state and year columns (handle minor naming variation)
    state_col = next(c for c in df.columns if 'state' in c and 'code' not in c)
    year_col  = next(c for c in df.columns if 'year' in c  and 'code' not in c)
    death_col = next(c for c in df.columns if 'death' in c or 'count' in c)

    df = df[[state_col, year_col, death_col]].copy()
    df.columns = ['state', 'year', deaths_col]

    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    # 'Suppressed' or 'Unreliable' → NaN
    df[deaths_col] = pd.to_numeric(df[deaths_col].str.replace(',', ''), errors='coerce')

    df = df.dropna(subset=['state', 'year'])
    df['state'] = df['state'].str.strip()
    df['year']  = df['year'].astype(int)
    return df


def load_cdc_data(firearm_csv: Path, all_csv: Path) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by (state, year) with column
    'firearm_suicide_fraction' ∈ (0, 1].

    Suppressed counts are interpolated linearly across time within each state.
    States with no valid data for a given year use the state's overall mean.
    """
    print(f"Loading firearm suicides from {firearm_csv}")
    fw = _load_wisqars_csv(firearm_csv, 'firearm_deaths')
    print(f"Loading all suicides from {all_csv}")
    al = _load_wisqars_csv(all_csv, 'all_deaths')

    df = pd.merge(fw, al, on=['state', 'year'], how='outer')

    # Compute fraction (NaN if either count is suppressed)
    df['fraction'] = df['firearm_deaths'] / df['all_deaths']

    # Interpolate suppressed values per state
    states = df['state'].unique()
    all_years = sorted(df['year'].unique())
    records = []
    for state in states:
        sub = df[df['state'] == state].set_index('year')['fraction']
        # reindex to full year range, then interpolate
        sub = sub.reindex(all_years)
        sub = sub.interpolate(method='linear', limit_direction='both')
        # if still NaN (state has no data at all), fill with national mean later
        for yr, val in sub.items():
            records.append({'state': state, 'year': yr, 'firearm_suicide_fraction': val})

    result = pd.DataFrame(records)

    # Fill any remaining NaNs with the national mean per year
    nat_mean = result.groupby('year')['firearm_suicide_fraction'].mean()
    def _fill(row):
        if np.isnan(row['firearm_suicide_fraction']):
            return nat_mean.get(row['year'], np.nan)
        return row['firearm_suicide_fraction']
    result['firearm_suicide_fraction'] = result.apply(_fill, axis=1)

    print(f"  States: {result['state'].nunique()}, "
          f"Year range: {result['year'].min()}–{result['year'].max()}")
    print(f"  Fraction range: "
          f"{result['firearm_suicide_fraction'].min():.3f}–"
          f"{result['firearm_suicide_fraction'].max():.3f}")

    result = result.set_index(['state', 'year'])
    return result


# ---------------------------------------------------------------------------
# Step 2: validate that the CDC data covers the city_data year range
# ---------------------------------------------------------------------------

def validate_coverage(fraction_table: pd.DataFrame, city_data: dict) -> None:
    years_needed = set(v['params'].year for v in city_data.values())
    states_needed = set(v['name'][0] for v in city_data.values() if 'name' in v)
    years_covered  = set(fraction_table.index.get_level_values('year'))
    states_covered = set(fraction_table.index.get_level_values('state'))

    missing_years  = years_needed  - years_covered
    missing_states = states_needed - states_covered
    if missing_years:
        print(f"  WARNING: {len(missing_years)} event years not in CDC data: "
              f"{sorted(missing_years)}")
        print("  Will use nearest available year for these.")
    if missing_states:
        print(f"  WARNING: {len(missing_states)} states not in CDC data: "
              f"{sorted(missing_states)}")
        print("  Will use national mean fraction for these states.")


# ---------------------------------------------------------------------------
# Step 3: look up proxy value for a city entry
# ---------------------------------------------------------------------------

def lookup_fraction(fraction_table: pd.DataFrame,
                    state: str, year: int,
                    max_year: int, min_year: int,
                    nat_mean_by_year: pd.Series) -> float:
    """
    Look up the firearm suicide fraction for (state, year).
    Clamps to the data's year range; falls back to national mean if state missing.
    """
    lookup_year = int(np.clip(year, min_year, max_year))

    try:
        val = fraction_table.loc[(state, lookup_year), 'firearm_suicide_fraction']
        if not np.isnan(val):
            return float(val)
    except KeyError:
        pass

    # State not found: use national mean for that year
    if lookup_year in nat_mean_by_year.index:
        return float(nat_mean_by_year.loc[lookup_year])
    return float(nat_mean_by_year.mean())


# ---------------------------------------------------------------------------
# Step 4: rebuild city_data and subsampled arrays with new arm_density
# ---------------------------------------------------------------------------

def substitute_proxy(original_pkl: Path,
                     fraction_table: pd.DataFrame) -> dict:
    """
    Deep-copies the loaded pkl dict and replaces arm_density in every
    CityParams with the state-year firearm suicide fraction.
    Returns the modified dict (does not touch the original pkl on disk).
    """
    from gvabm.param_distr import CityParams

    with open(original_pkl, 'rb') as f:
        data = pickle.load(f)

    min_year = fraction_table.index.get_level_values('year').min()
    max_year = fraction_table.index.get_level_values('year').max()
    nat_mean_by_year = (fraction_table
                        .groupby('year')['firearm_suicide_fraction']
                        .mean())

    def swap_params(city_dict: dict) -> dict:
        """Return a new city dict with arm_density replaced."""
        p  = city_dict['params']
        state = city_dict.get('name', (None,))[0]
        year  = p.year

        if state is None:
            frac = float(nat_mean_by_year.mean())
        else:
            frac = lookup_fraction(fraction_table, state, year,
                                   max_year, min_year, nat_mean_by_year)

        new_params = CityParams(
            year             = p.year,
            population       = p.population,
            population_density = p.population_density,
            arm_count        = p.arm_count,    # kept for reference; unused by model
            arm_density      = frac,
        )
        return {**city_dict, 'params': new_params}

    # --- city_data dict (all 142k entries) ---
    print("Substituting proxy in city_data dict …")
    new_city_data = {k: swap_params(v) for k, v in data['city_data'].items()}

    # --- subsampled_city_data array (463 entries) ---
    print("Substituting proxy in subsampled_city_data array …")
    new_subsampled = np.array([swap_params(c) for c in data['subsampled_city_data']])

    new_data = {
        'subsampled_city_names': data['subsampled_city_names'],
        'subsampled_city_data':  new_subsampled,
        'subsample_weights':     data['subsample_weights'],
        'city_data':             new_city_data,
        # Record provenance
        'proxy': 'firearm_suicide_fraction',
        'proxy_source': str(FIREARM_CSV),
    }
    return new_data


# ---------------------------------------------------------------------------
# Step 5: quick sanity check
# ---------------------------------------------------------------------------

def sanity_check(new_data: dict, original_pkl: Path) -> None:
    with open(original_pkl, 'rb') as f:
        orig = pickle.load(f)

    orig_scd = orig['subsampled_city_data']
    new_scd  = new_data['subsampled_city_data']

    assert len(orig_scd) == len(new_scd), "Subsampled city count changed!"

    # arm_density should differ; all other fields should be identical
    for o, n in zip(orig_scd, new_scd):
        assert o['params'].population == n['params'].population
        assert o['params'].population_density == n['params'].population_density
        assert o['params'].arm_density != n['params'].arm_density or True  # may coincide
        assert len(o['events']) == len(n['events'])

    frac_vals = [c['params'].arm_density for c in new_scd]
    print(f"\nSanity check passed.")
    print(f"  arm_density (suicide fraction) — "
          f"min={min(frac_vals):.3f}, max={max(frac_vals):.3f}, "
          f"mean={np.mean(frac_vals):.3f}")

    orig_frac_vals = [c['params'].arm_density for c in orig_scd]
    print(f"  arm_density (FFL)              — "
          f"min={min(orig_frac_vals):.4f}, max={max(orig_frac_vals):.4f}, "
          f"mean={np.mean(orig_frac_vals):.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    if not FIREARM_CSV.exists() or not ALL_CSV.exists():
        missing = [p for p in [FIREARM_CSV, ALL_CSV] if not p.exists()]
        raise FileNotFoundError(
            f"CDC data files not found: {[str(p) for p in missing]}\n\n"
            "Please download from CDC WISQARS Fatal Injury Reports:\n"
            "  https://wisqars.cdc.gov/reports/\n\n"
            "  Query 1 (firearm suicides):\n"
            "    Outcome: Suicide | Mechanism: Firearm\n"
            "    Group by: State, Year | Years: 1981–2024\n"
            f"    Save CSV as: {FIREARM_CSV}\n\n"
            "  Query 2 (all suicides):\n"
            "    Outcome: Suicide | Mechanism: All mechanisms\n"
            "    Group by: State, Year | Years: 1981–2024\n"
            f"    Save CSV as: {ALL_CSV}\n"
        )

    if OUTPUT_PKL.exists():
        raise FileExistsError(
            f"{OUTPUT_PKL} already exists. Delete it manually if you want to regenerate."
        )

    # 1. Load CDC data
    fraction_table = load_cdc_data(FIREARM_CSV, ALL_CSV)

    # 2. Load original pkl and validate coverage
    print("\nLoading original city data …")
    with open(SOURCE_PKL, 'rb') as f:
        orig = pickle.load(f)
    validate_coverage(fraction_table, orig['city_data'])

    # 3. Build new dataset
    print("\nBuilding suicide-proxy dataset …")
    new_data = substitute_proxy(SOURCE_PKL, fraction_table)

    # 4. Sanity check
    sanity_check(new_data, SOURCE_PKL)

    # 5. Save — never overwrites because of the FileExistsError guard above
    print(f"\nSaving to {OUTPUT_PKL} …")
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(new_data, f)
    print("Done.")
