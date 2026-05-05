"""
Task D preprocessing — rebuild subsampled city pkl with FFL density recomputed
at alternative buffer radii.

The baseline notebook (scripts/city_ffl_density.ipynb, cell 5a384585) uses
ffl_radius = 25 miles.  This script recomputes FFL density at each value in
RADII, updates arm_count and arm_density in every city's CityParams, and saves
independent pkl files that sensitivity_ffl_radius.py can load directly.

The subsampled city set and events are preserved unchanged; only the FFL-based
columns are swapped.

Input files (all under --data-dir)
-----------------------------------
    ffl_list.txt                   — raw ATF FFL file (tab-separated)
    2013_us_zips.csv               — ZIP code lat/lng  (cols: ZIP, LAT, LNG)
    processed/city_population_density.csv — city grid with GISJOIN, lat, lng
    processed/subsampled_city_data.pkl    — existing baseline pkl

Output
------
    processed/subsampled_city_data_ffl_r{R}.pkl  for each R in RADII

Usage
-----
    python scripts/build_ffl_radius_data.py [--data-dir /path/to/data]
"""

import argparse
import os
import json
import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from gvabm.param_distr import CityParams

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────
RADII           = [7.5, 15, 25, 30]   # miles; 25 = baseline (notebook cell 5a384585)
BASELINE_RADIUS = 25

parser = argparse.ArgumentParser()
parser.add_argument('--data-dir', type=str, default='data',
                    help='Root data directory (default: data)')
args = parser.parse_args()

DATA_DIR      = args.data_dir
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# ──────────────────────────────────────────────────────────────────────────────
# Load and geocode FFL locations (mirrors notebook cell 06dd614d / ffd42607)
# ──────────────────────────────────────────────────────────────────────────────
print("Loading FFL data ...", flush=True)

def clean_string(s):
    return ''.join(['_' if not c.isalnum() else c for c in str(s).strip()]).upper()

with open(os.path.join(DATA_DIR, 'state_acronym.json')) as f:
    state_acr = json.load(f)
state_acr = {k: clean_string(v) for k, v in state_acr.items()}

ffl_df   = pd.read_csv(os.path.join(DATA_DIR, 'ffl_list.txt'), sep='\t')
zips_loc = pd.read_csv(os.path.join(DATA_DIR, '2013_us_zips.csv'))
city_ref = pd.read_csv(os.path.join(DATA_DIR, 'uscities.csv'))

ffl_df['PREMISE_STATE_ACR'] = ffl_df['PREMISE_STATE'].copy()
ffl_df['PREMISE_STATE']     = ffl_df['PREMISE_STATE_ACR'].map(state_acr)
ffl_df['PREMISE_CITY']      = ffl_df['PREMISE_CITY'].apply(clean_string)
city_ref['city_upper']      = city_ref['city_ascii'].apply(clean_string)
city_ref['state_upper']     = city_ref['state_name'].apply(clean_string)

major = ffl_df[ffl_df['LIC_TYPE'].isin([1, 2, 7])].reset_index(drop=True)

premise_loc = pd.merge(major[['PREMISE_ZIP_CODE']], zips_loc,
                       left_on='PREMISE_ZIP_CODE', right_on='ZIP', how='left')
mail_loc    = pd.merge(major[['MAIL_ZIP_CODE']], zips_loc,
                       left_on='MAIL_ZIP_CODE', right_on='ZIP', how='left')
city_loc    = pd.merge(
    major[['PREMISE_CITY', 'PREMISE_STATE']],
    city_ref[['city_upper', 'state_upper', 'lat', 'lng']].drop_duplicates(['city_upper', 'state_upper']),
    left_on=['PREMISE_CITY', 'PREMISE_STATE'],
    right_on=['city_upper', 'state_upper'], how='left'
)

aggr = premise_loc.copy()
aggr.loc[aggr['LAT'].isna(), ['LAT', 'LNG']] = mail_loc.loc[aggr['LAT'].isna(), ['LAT', 'LNG']].values
aggr.loc[aggr['LAT'].isna(), ['LAT', 'LNG']] = city_loc.loc[aggr['LAT'].isna(), ['lat', 'lng']].values
major[['LAT', 'LNG']] = aggr[['LAT', 'LNG']]
major = major.dropna(subset=['LAT', 'LNG'])
print(f"  {len(major)} major FFL licenses geocoded", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Convert to 3-D spherical coordinates (mirrors notebook cell ffd42607)
# ──────────────────────────────────────────────────────────────────────────────
def to_xyz(lat, lon):
    R = 3958.8
    theta = np.where(lon < 180, 180 + (180 - (-lon)), lon)
    phi   = (lat - 90) * -1
    x = R * np.sin(np.deg2rad(phi)) * np.cos(np.deg2rad(theta))
    y = R * np.sin(np.deg2rad(phi)) * np.sin(np.deg2rad(theta))
    z = R * np.cos(np.deg2rad(phi))
    return x, y, z

major['x'], major['y'], major['z'] = to_xyz(major['LAT'].values, major['LNG'].values)
ffl_xyz = major[['x', 'y', 'z']].values

# ──────────────────────────────────────────────────────────────────────────────
# Load city grid — GISJOIN → (lat, lng) mapping
# ──────────────────────────────────────────────────────────────────────────────
print("Loading city grid ...", flush=True)
city_grid = pd.read_csv(os.path.join(PROCESSED_DIR, 'city_population_density.csv'))

city_grid['cx'], city_grid['cy'], city_grid['cz'] = to_xyz(
    city_grid['lat'].values, city_grid['lng'].values
)
# Build BallTree over FFL locations once; radius queries are cheap after this
ffl_tree = BallTree(ffl_xyz, leaf_size=16)

gisjoin_to_xyz = dict(zip(
    city_grid['GISJOIN'],
    zip(city_grid['cx'], city_grid['cy'], city_grid['cz'])
))
print(f"  {len(gisjoin_to_xyz)} city locations loaded", flush=True)

# ──────────────────────────────────────────────────────────────────────────────
# Load existing subsampled pkl
# ──────────────────────────────────────────────────────────────────────────────
baseline_pkl = os.path.join(PROCESSED_DIR, 'subsampled_city_data.pkl')
print(f"\nLoading baseline pkl from {baseline_pkl} ...", flush=True)
with open(baseline_pkl, 'rb') as f:
    base = pickle.load(f)

subsampled_city_data  = base['subsampled_city_data']
subsampled_city_names = base['subsampled_city_names']
subsample_weights     = base['subsample_weights']
city_data_full        = base['city_data']
print(f"  {len(subsampled_city_data)} subsampled cities, "
      f"{len(city_data_full)} total cities", flush=True)

# Extract GISJOIN from city key — keys are (GISJOIN_str, decade_int) tuples
all_keys = list(city_data_full.keys())

def gisjoin_from_key(k):
    return k[0] if isinstance(k, tuple) else k

# ──────────────────────────────────────────────────────────────────────────────
# For each radius: recompute arm_density and save new pkl
# ──────────────────────────────────────────────────────────────────────────────
for R in RADII:
    tag = " (baseline)" if R == BASELINE_RADIUS else ""
    print(f"\nRadius = {R} miles{tag}", flush=True)

    # Batch BallTree query for every city in city_data_full
    gisjoins = [gisjoin_from_key(k) for k in all_keys]
    city_coords = []
    missing = 0
    for gj in gisjoins:
        if gj in gisjoin_to_xyz:
            city_coords.append(gisjoin_to_xyz[gj])
        else:
            city_coords.append((np.nan, np.nan, np.nan))
            missing += 1

    if missing:
        print(f"  WARNING: {missing} cities had no lat/lng match; arm_density unchanged")

    coords_arr = np.array(city_coords, dtype=float)

    # Query BallTree for all cities at once
    valid = ~np.isnan(coords_arr[:, 0])
    counts = np.zeros(len(all_keys))
    if valid.sum() > 0:
        neighbor_lists = ffl_tree.query_radius(coords_arr[valid], r=R)
        counts[valid] = np.array([len(nb) for nb in neighbor_lists])

    arm_densities = counts / (np.pi * R * R)

    # Rebuild city_data with updated CityParams
    new_city_data = {}
    for idx, (k, v) in enumerate(city_data_full.items()):
        old_p = v['params']
        new_p = CityParams(
            year               = old_p.year,
            population         = old_p.population,
            population_density = old_p.population_density,
            arm_count          = float(counts[idx]),
            arm_density        = float(arm_densities[idx]),
        )
        new_city_data[k] = {**v, 'params': new_p}

    # Rebuild subsampled list with matching updated params
    name_to_key = {k: k for k in new_city_data}
    new_subsampled = []
    for entry in subsampled_city_data:
        # CityParams inside entry['params']; match by content
        old_p = entry['params']
        new_entry = dict(entry)
        # Find matching key by (year, population) — a unique fingerprint
        matched = False
        for k, nv in new_city_data.items():
            np_ = nv['params']
            if (np_.year == old_p.year and
                    abs(np_.population - old_p.population) < 1 and
                    abs(np_.population_density - old_p.population_density) < 0.01):
                new_entry = {**entry, 'params': np_}
                matched = True
                break
        new_subsampled.append(new_entry)

    out_path = os.path.join(PROCESSED_DIR, f'subsampled_city_data_ffl_r{R:.0f}.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({
            'subsampled_city_names': subsampled_city_names,
            'subsampled_city_data':  np.array(new_subsampled, dtype=object),
            'subsample_weights':     subsample_weights,
            'city_data':             new_city_data,
        }, f)
    print(f"  Saved → {out_path}  "
          f"(mean arm_density = {arm_densities[valid].mean():.4f})", flush=True)

print("\nDone.")
