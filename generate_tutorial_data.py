"""
generate_tutorial_data.py
─────────────────────────
Pre-generates a small tutorial dataset so students can load pre-computed
results without running the full EnKF cycle during the notebook.

Outputs
-------
tutorial_data.nc   -- NetCDF file with truth, observations, EnKF analysis
                      (full obs and sparse), and metadata attributes

Usage
-----
    python generate_tutorial_data.py

This takes ~5-10 minutes on a laptop. The output is ~5 MB and can be
committed to the repository for quick loading in the tutorial notebook.

Settings match the paper's base case except T=200 (vs T=2000 in the paper)
and nens=20 (vs nens=100) for tractable tutorial runtime.
"""

import numpy as np
import xarray as xr
import sys
import time

from L96 import L96
from Experiment import Experiment

# ── Settings ───────────────────────────────────────────────────────────────
SEED   = 42
N      = 40       # number of L96 variables
F      = 8.0      # forcing
dt     = 0.05     # time step (~6 hours)
T      = 200.0    # simulation length (increase to 2000 for paper results)
nens   = 20       # ensemble size (paper uses 100)
loc    = 5        # localization radius
gamma  = 1.0      # covariance inflation
r_frac = 0.3      # observation noise as fraction of climatological std dev

np.random.seed(SEED)

print("=" * 60)
print("  Generating tutorial dataset")
print("=" * 60)
print(f"  N={N}, F={F}, dt={dt}, T={T}, nens={nens}")
print(f"  Localization={loc}, Inflation={gamma}")
print()

# ── Initial condition ──────────────────────────────────────────────────────
x0 = F * np.ones(N)
x0[0] = F + 0.01   # small perturbation to start off the fixed point

# ── Full-observation EnKF ──────────────────────────────────────────────────
print("Step 1/3: Running full-observation EnKF...")
t0 = time.time()

settings_full = {
    'N': N, 'F': F, 'dt': dt, 'nens': nens,
    'loc': loc, 'gamma': gamma, 'frac': 1.0,
}
exp_full = Experiment(settings=settings_full)
exp_full.ds['xx'] = exp_full.get_true(x0, T)
r = float(exp_full.ds.xx.std()) * r_frac
exp_full.r = r
exp_full.ds['yy'] = exp_full.makeobs(r)
xf0 = exp_full.make_ensemble(x0, r)
exp_full.assimilate(xf0)

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
rmse_full = float(np.sqrt(
    ((exp_full.ds.xaens.mean('ensemble') - exp_full.ds.xx.sel(time=exp_full.ds.xaens.time))**2).mean()
))
print(f"  Full-obs EnKF RMSE: {rmse_full:.4f}")

# ── Sparse-observation EnKF (25% observed) ────────────────────────────────
print("\nStep 2/3: Running sparse-observation EnKF (frac=0.25)...")
t0 = time.time()

settings_sparse = {
    'N': N, 'F': F, 'dt': dt, 'nens': nens,
    'loc': loc, 'gamma': gamma, 'frac': 0.25,
}
exp_sparse = Experiment(settings=settings_sparse)
exp_sparse.ds['xx'] = exp_full.ds.xx    # same truth
exp_sparse.r = r
exp_sparse.ds['yy'] = exp_sparse.makeobs(r)
xf0_sp = exp_sparse.make_ensemble(x0, r)
exp_sparse.assimilate(xf0_sp)

elapsed = time.time() - t0
print(f"  Done in {elapsed:.1f}s")
rmse_sparse = float(np.sqrt(
    ((exp_sparse.ds.xaens.mean('ensemble') - exp_sparse.ds.xx.sel(time=exp_sparse.ds.xaens.time))**2).mean()
))
print(f"  Sparse EnKF RMSE: {rmse_sparse:.4f}")

# ── Build combined NetCDF dataset ─────────────────────────────────────────
print("\nStep 3/3: Saving to tutorial_data/tutorial_data.nc...")

t_da = exp_full.ds.xaens.time.values

ds_out = xr.Dataset(
    {
        # Truth (shared between experiments)
        'xx': exp_full.ds.xx,

        # Full-observation EnKF
        'yy_full':   exp_full.ds.yy,
        'xaens_full': exp_full.ds.xaens,
        'xfens_full': exp_full.ds.xfens,

        # Sparse-observation EnKF
        'yy_sparse':   exp_sparse.ds.yy,
        'xaens_sparse': exp_sparse.ds.xaens,
        'xfens_sparse': exp_sparse.ds.xfens,
    },
    attrs={
        'description': 'Tutorial dataset for ML-augmented DA summer school',
        'N': N,
        'F': F,
        'dt': dt,
        'T': T,
        'nens': nens,
        'localization': loc,
        'inflation': gamma,
        'obs_noise_frac': r_frac,
        'obs_noise_std': float(r),
        'random_seed': SEED,
        'rmse_enkf_full': rmse_full,
        'rmse_enkf_sparse': rmse_sparse,
    }
)

import os; os.makedirs('tutorial_data', exist_ok=True)
ds_out.to_netcdf('tutorial_data/tutorial_data.nc')
print(f"  Saved: tutorial_data/tutorial_data.nc")
print()
print("=" * 60)
print("  Dataset summary")
print("=" * 60)
print(f"  Truth shape:           {exp_full.ds.xx.shape}")
print(f"  Ensemble shape:        {exp_full.ds.xaens.shape}  (space × ensemble × time)")
print(f"  Full-obs EnKF RMSE:    {rmse_full:.4f}")
print(f"  Sparse-obs EnKF RMSE:  {rmse_sparse:.4f}")
print()
print("  Load in notebook with:")
print("    import xarray as xr")
print("    ds = xr.open_dataset('tutorial_data/tutorial_data.nc')")
print("    xx = ds.xx         # truth")
print("    xa = ds.xaens_full # full-obs EnKF analyses")
