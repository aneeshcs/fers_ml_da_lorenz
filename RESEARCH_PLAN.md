# Two-Week Research Plan: Extending ML-Augmented Data Assimilation
## For a Group of 3–4 Graduate Students

---

## Overview

This plan divides the research into four independent but complementary tracks,
each assigned to one student (or one pair). At the end of two weeks, each track
produces a result that can be compared against the same baseline (the paper's
augmented method), making a natural group presentation or short report.

**Baseline to beat:**
- Analysis RMSE of the augmented EnKF+CNN method on sparse observations (25% coverage)
- Rank histogram calibration of the analysis ensemble
- 10-day ensemble forecast accuracy (Figure 5 in the paper)

---

## Git workflow for collaboration

```bash
# Each student creates their own branch from main
git checkout -b track-A-smooth-localization    # Student 1
git checkout -b track-B-architecture          # Student 2
git checkout -b track-C-adaptive-augmentation # Student 3
git checkout -b track-D-robustness            # Student 4

# Push your branch regularly so others can see your work
git push origin track-A-smooth-localization

# At the end of week 2, open a pull request into main for review
gh pr create --base main --title "Track A: Smooth localization results"
```

**Shared resources:**
- Do not modify `L96.py`, `EnKF.py`, `NeuralNet.py`, or `Experiment.py` directly on `main`.
  Instead, copy the class you need to modify into your track's notebook and subclass it,
  or make your changes on your own branch.
- The pre-generated dataset (`tutorial_data.nc`, `tutorial_enkf_full.nc`) is read-only
  shared data — load it, don't overwrite it.
- At the end of Week 1, hold a 30-minute group sync to share your interim RMSE numbers
  so everyone can sanity-check against the same baseline.

---

## Week 1 — Everyone (Days 1–2)

Before splitting into tracks, all students should:

1. Run `tutorial_notebook.ipynb` end-to-end and confirm you get RMSE numbers
   consistent with the tutorial's expected values.
2. Read Sections 3–5 of the tutorial notebook carefully, paying attention to the
   code in `EnKF.py`, `Experiment.py`, and `NeuralNet.py`.
3. Read the paper (Howard, Subramanian & Hoteit, 2024) Sections 2–4 to understand
   the sensitivity experiments and what was already tested.
4. Make sure you can reproduce the baseline augmented RMSE on a short run (T=200)
   and that your numbers match your teammates'.

**Shared baseline script** — all students run this at the start and save the result:
```python
# baseline.py  — run once and share results with the group
from Experiment import Experiment
from NeuralNet import NeuralNet
import numpy as np, xarray as xr

np.random.seed(0)
N, F, dt, T = 40, 8.0, 0.05, 200.0
x0 = F*np.ones(N); x0[0] = F+0.01

# EnKF training run (full obs, best sensitivity case: loc=7)
exp = Experiment(settings={'N':N,'F':F,'dt':dt,'nens':100,'loc':7,'gamma':1,'frac':1})
exp.ds['xx'] = exp.get_true(x0, T)
exp.r = float(exp.ds.xx.std()) * 0.3
exp.ds['yy'] = exp.makeobs(exp.r)
exp.assimilate(exp.make_ensemble(x0, exp.r))

# Train CNN
nn = NeuralNet(nlayers=3, filter_size=3, N=N)
nn.buildmodel()
nn.train(0.7, 20, 'adam', exp.ds)

# Augmented experiment (sparse obs)
exp2 = Experiment(settings={'N':N,'F':F,'dt':dt,'nens':100,'loc':7,'gamma':1,'frac':0.25})
exp2.ds['xx'] = exp.ds.xx
exp2.r = exp.r
exp2.ds['yy'] = exp2.makeobs(exp2.r)
exp2.assimilate(exp2.make_ensemble(x0, exp2.r), nn=nn)

xa = exp2.ds.xaens.mean('ensemble')
xx = exp2.ds.xx.sel(time=xa.time)
rmse = float(np.sqrt(((xa-xx)**2).mean()))
print(f"Baseline augmented RMSE: {rmse:.4f}")  # share this number with the group
```

---

## Track A — Better Classical DA: Smooth Localization and Adaptive Inflation
**Best for:** A student comfortable with linear algebra and statistics,
who wants to understand the classical DA side more deeply.

### Background

The paper uses **hard cutoff localization**: covariance entries are set to exactly
zero beyond distance `loc`. This is crude — in practice, a smooth taper function
that gradually reduces covariance with distance performs better and is used in all
operational weather models.

The paper also uses **fixed inflation** (`gamma` = constant). In reality, the
optimal inflation changes over time. **Adaptive inflation** estimates the correct
`gamma` automatically from the innovation statistics.

### What to implement

#### Part 1: Gaspari-Cohn smooth localization (Days 3–5)

Replace the hard cutoff in `EnKF.localize()` with the Gaspari-Cohn (1999)
fifth-order piecewise rational function, which is the standard in operational systems:

```python
def gaspari_cohn(distance, half_width):
    """
    Gaspari-Cohn (1999) smooth localization taper.
    Returns 1.0 at distance=0, decays smoothly to 0 at distance=2*half_width.
    """
    r = abs(distance) / half_width
    if r >= 2.0:
        return 0.0
    elif r >= 1.0:
        return ( r**5/12 - r**4/2 + 5*r**3/8 + 5*r**2/3 - 5*r + 4 - 2/(3*r) ) / 3  # wait -- this is wrong, use the formula below
    else:
        return 1 - 5*r**2/3 + 5*r**3/8 + r**4/2 - r**5/4
```

> Reference: Gaspari, G. & Cohn, S.E. (1999). Construction of correlation functions
> in two and two dimensions. *Q. J. Roy. Meteor. Soc.* 125, 723–757.
> The correct formula is on p. 740, Eq. 4.10.

Create a subclass of `EnKF` that overrides `localize()`:

```python
class EnKFGC(EnKF):
    """EnKF with Gaspari-Cohn smooth localization."""
    def localize(self, cov):
        for i in range(self.nvars):
            for j in range(self.nvars):
                mid = abs(i - j)
                out = self.nvars - max(i,j) + min(i,j)
                dist = min(mid, out)
                cov[i, j] *= gaspari_cohn(dist, self.loc)
        return cov
```

**Key question:** For the same `loc` value, does GC localization improve RMSE
compared to the hard cutoff? What is the optimal `loc` for GC vs. hard cutoff?

#### Part 2: Adaptive multiplicative inflation (Days 5–7)

After each analysis step, estimate the innovation variance and compare to what
the ensemble predicted. Use the ratio to update `gamma`. This is the
Anderson (2009) spatially-varying adaptive inflation algorithm, simplified:

```
gamma_{t+1} = gamma_t * (observed innovation variance) / (predicted innovation variance)
            = gamma_t * <d_t^2> / (H P^f H^T + R)_diag
```

where `d_t = y_t - H x^f_t` is the innovation. Implement this in a subclass
of `Experiment` that overrides `assimilate()` to update `gamma` at each step.

**Key question:** Does adaptive inflation outperform the paper's best fixed
inflation (`gamma=1.05`, case s6)? What does the time series of `gamma_t` look like?

### Deliverables for Track A

- A notebook `tracks/track_A/track_A_localization_inflation.ipynb` with:
  - RMSE vs. `loc` comparison: hard cutoff vs. Gaspari-Cohn
  - RMSE vs. `gamma` comparison: fixed vs. adaptive
  - Time series of adaptive `gamma_t`
  - Rank histograms for each method

---

## Track B — Better Neural Network Architecture
**Best for:** A student interested in deep learning who wants to explore
how architecture choices affect performance.

### Background

The paper's CNN is deliberately minimal: 3 layers, 5 filters, kernel size 3,
trained only on the ensemble-mean forecast and observation at time `t`. It:
- Has no temporal memory (each time step is treated independently)
- Only predicts the analysis **mean** (not uncertainty)
- Uses a fixed architecture not tuned to the problem

There is substantial room to improve each of these.

### What to implement

#### Part 1: Architecture search (Days 3–4)

Run a systematic comparison of CNN architectures. For each, train on the same
dataset and evaluate on the same test set:

```python
architectures = [
    # (name, nlayers, filter_size, n_filters)
    ('paper_baseline', 3, 3, 5),
    ('wider',          3, 3, 16),
    ('deeper',         5, 3, 5),
    ('larger_kernel',  3, 5, 5),
    ('wider_deeper',   5, 3, 16),
]
```

Note: when changing `nlayers` or `filter_size`, you must update `NeuralNet.__init__`
and rebuild `make_input()` accordingly (the offset formula changes).

For each architecture, record:
- Training and validation RMSE curves
- Analysis RMSE when deployed in the augmented experiment
- Number of trainable parameters

#### Part 2: Adding temporal context — a sliding window input (Days 4–6)

The current CNN only sees the forecast and observation at time `t`. But the
forecast errors at time `t` are correlated with errors at `t-1` and `t-2`.
Giving the network recent history may help.

Modify `NeuralNet.make_input()` to stack the last `k` forecast states as
additional channels:

```
Current input:  (N+2*offset, 2 channels)  ← x^f_t,  y_t - x^f_t
Extended input: (N+2*offset, 2+k channels) ← x^f_t, y_t-x^f_t, x^f_{t-1}, x^f_{t-2}, ...
```

This requires modifying `NeuralNet.train()` to build sliding-window training
pairs. You'll also need to modify `Experiment.assimilate()` to pass the
forecast history to `nn.assimilate()`.

**Key question:** Does adding 1, 2, or 3 previous forecast steps improve the
analysis RMSE? At what point does it stop helping?

#### Part 3: Predicting analysis uncertainty (Days 6–7)

The paper's CNN predicts only the analysis **mean**. A richer model would predict
both the mean and the per-variable **uncertainty** (standard deviation), allowing
it to also adjust ensemble spread rather than just shifting the ensemble mean.

Modify `NeuralNet.buildmodel()` to output 2 channels (mean + log-variance):

```python
def buildmodel_uncertainty(self):
    model = Sequential()
    model.add(Conv1D(16, 3, activation='relu', input_shape=(N + 2*offset, 2)))
    model.add(Conv1D(16, 3, activation='relu'))
    model.add(Conv1D(2, 3, activation=None))  # 2 outputs: mean, log-variance
    self.model = model
```

Train with a **negative log-likelihood loss** instead of MSE:
```
loss = 0.5 * log(sigma^2) + (xa_true - xa_pred)^2 / (2 * sigma^2)
```

Then in `Experiment.assimilate()`, use both the predicted mean and spread:
```python
xa_mean, log_var = nn.assimilate(xf, y)  # two outputs
xa_std = np.exp(0.5 * log_var)
# Shift AND rescale the ensemble
delta = xf.mean(axis=1) - xa_mean
scale = xa_std / xf.std(axis=1)  # rescale spread
for j in range(N):
    xaens[j, :, i] = xa_mean[j] + scale[j] * (xfens[j, :, i] - xf[j])
```

**Key question:** Does a probabilistic CNN produce better-calibrated rank histograms
than the deterministic baseline? Does it improve RMSE as well as spread?

### Deliverables for Track B

- A notebook `tracks/track_B/track_B_architectures.ipynb` with:
  - Architecture comparison table (params, val RMSE, augmented RMSE)
  - Training curves for each architecture
  - Temporal context ablation (k=0,1,2,3)
  - Rank histograms: deterministic vs. probabilistic CNN

---

## Track C — Better Augmentation Strategy
**Best for:** A student interested in the systems / algorithm design side —
how and when to combine the two components.

### Background

The paper's augmentation is simple: alternate EnKF and CNN every other time step
(CNN on odd steps, EnKF on even steps). This is a fixed, rigid schedule that
does not adapt to the current state of the system.

There are two key dimensions to explore:
1. **When** to use CNN vs. EnKF (the schedule)
2. **How** to use the CNN output to update the ensemble (beyond just mean-shifting)

### What to implement

#### Part 1: Adaptive scheduling (Days 3–5)

Instead of a fixed alternation, use the **ensemble spread** to decide when to
apply the CNN. When the ensemble is collapsed (low spread = overconfident),
use EnKF to restore diversity. When spread is healthy, use the cheaper CNN.

Modify `Experiment.assimilate()`:

```python
SPREAD_THRESHOLD = 0.5  # tune this parameter

spread_t = xfens[:, :, i].std(axis=1).mean()  # mean ensemble spread
if spread_t < SPREAD_THRESHOLD:
    # Ensemble is collapsed — use EnKF to re-inflate
    h = self.make_obs()
    xaens[:, :, i] = enkf.ensemble_assim(xfens[:, :, i], y, h, self.r)
else:
    # Spread is healthy — use cheap CNN
    xpmean = nn.assimilate(xfens[:, :, i].mean(axis=1), y.values)
    delta = xfens[:, :, i].mean(axis=1) - xpmean
    for j in range(self.N):
        xaens[j, :, i] = xfens[j, :, i] - delta[j]
```

Also test fixed ratios: CNN every 1st, 2nd, 3rd, 4th, 8th step.

**Key questions:**
- What fraction of steps does the adaptive scheduler assign to CNN vs. EnKF?
- Is adaptive scheduling better than the best fixed ratio?
- How sensitive is the method to the spread threshold?

#### Part 2: Better ensemble updating (Days 5–7)

When the paper applies the CNN, it shifts all ensemble members by the same delta
(mean-shift only). This preserves spread but does not let the CNN influence
where the ensemble members are *relative to the mean*.

Three alternative update rules to test:

**Option 1 — Relaxation to posterior mean (nudging):**
```python
alpha = 0.5  # relaxation coefficient (tune between 0 and 1)
xpmean = nn.assimilate(...)
for j in range(nens):
    xaens[:, j, i] = alpha * xpmean + (1 - alpha) * xfens[:, j, i]
```
This pulls each member toward the CNN mean rather than just shifting.

**Option 2 — Spread scaling:**
```python
target_spread = 0.8 * xfens[:, :, i].std(axis=1)  # slightly reduce spread
xpmean = nn.assimilate(...)
for j in range(nens):
    pert = xfens[:, j, i] - xfens[:, :, i].mean(axis=1)
    xaens[:, j, i] = xpmean + (target_spread / xfens[:, :, i].std(axis=1)) * pert
```

**Option 3 — Per-member CNN application:**
Apply the CNN independently to each ensemble member's forecast (not just the mean):
```python
for j in range(nens):
    xa_member = nn.assimilate(xfens[:, j, i], y.values)
    xaens[:, j, i] = xa_member
```
This is computationally more expensive but may better represent ensemble diversity.

**Key question:** Which update rule produces the best-calibrated ensemble (rank histograms)
while also minimizing RMSE?

### Deliverables for Track C

- A notebook `tracks/track_C/track_C_augmentation.ipynb` with:
  - RMSE vs. CNN frequency (fixed schedules: every 1, 2, 3, 4, 8 steps)
  - Adaptive vs. best fixed schedule comparison
  - Rank histograms and RMSE for the three update rules
  - Time series of ensemble spread for each update rule

---

## Track D — Robustness and Generalization
**Best for:** A student interested in scientific questions rather than
implementation — more experiments, less new code.

### Background

The paper shows the augmented method works for one specific setup (F=8, N=40, T=2000).
Real applications require a method to work across many conditions. This track
asks: **how fragile is the augmented method?**

### What to test

#### Part 1: Model error — what if the forecast model is wrong? (Days 3–5)

In all paper experiments, the *same* L96 model is used for truth and forecasting.
In reality, the forecast model is always imperfect. Introduce **model error** by
training the CNN on a perfect-model run, then testing with a biased forecast model.

```python
F_true     = 8.0   # true forcing (used to generate truth and observations)
F_forecast = 7.5   # biased forcing (used in the ensemble forecast)

# Training run: truth and ensemble both use F=8 (no model error)
exp_train = Experiment(settings={..., 'F': F_true})
# ... (train CNN as usual)

# Test run: truth uses F=8, but forecast uses F=7.5
class BiasedExperiment(Experiment):
    def assimilate(self, xf0, nn=None):
        # Override: integrate ensemble with biased F
        self.model_F = F_forecast  # add this attribute and use it in forecast step
        return super().assimilate(xf0, nn=nn)
```

Test with biases of F ± 0.25, ± 0.5, ± 1.0. At what bias does the augmented
method fail (RMSE worse than the sparse EnKF alone)?

**Key question:** Is the augmented method more or less robust to model error
than the standard EnKF?

#### Part 2: Sparsity generalization (Days 5–6)

The CNN was trained on a fully-observed experiment (`frac=1.0`) and tested at 25%
coverage. How does performance change as coverage varies?

```python
fracs = [0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0]

for frac in fracs:
    # Run augmented experiment at this coverage
    exp_test = Experiment(settings={..., 'frac': frac})
    # ... assimilate with pre-trained nn
    # Record RMSE
```

Also test: what if you train a CNN at 50% coverage and test at 25%?
Is transfer between coverage levels possible?

#### Part 3: Forcing sensitivity — different L96 regimes (Days 6–7)

The paper fixes F=8. How does the augmented method perform for:
- `F=5`: weakly chaotic (Lyapunov exponent ~0.15)
- `F=8`: strongly chaotic (paper default)
- `F=12`: very turbulent

For each F, you need to:
1. Run an EnKF training experiment with that F
2. Train a new CNN on that run
3. Run the augmented experiment at 25% coverage
4. Compare all three F values

**Key question:** Does the augmented method benefit sparse DA more in high-chaos
or low-chaos regimes? Which regime is the CNN easier to train on?

#### Part 4: Non-Gaussian observation operator (Days 7, if time allows)

The paper uses direct point observations (H is a selection matrix). Try
**spatially-averaged observations** (H is a smoothing operator):

```python
def make_superobs(x, frac, window=3):
    """Observe spatial averages over windows of 'window' consecutive grid points."""
    obs = []
    for i in range(0, N, window):
        if np.random.rand() < frac:
            obs.append((i, x[i:i+window].mean()))  # (location, value)
    return obs
```

This mimics satellite footprints or area-averaged station data.

**Key question:** Can the CNN still learn to assimilate averaged observations,
or does it require point observations?

### Deliverables for Track D

- A notebook `tracks/track_D/track_D_robustness.ipynb` with:
  - RMSE vs. model error bias: EnKF alone vs. augmented
  - RMSE vs. observation fraction: augmented vs. sparse EnKF
  - RMSE for F=5, 8, 12: augmented vs. sparse EnKF
  - (Bonus) Results with spatially-averaged observations

---

## Week 2 integration: comparing all tracks

On the last day, each student creates one entry in a shared comparison table:

```python
# shared_comparison.py — each student fills in their best result

results = {
    'Baseline (paper)':          {'rmse': ???, 'method': 'EnKF + CNN, fixed alternation, hard loc'},
    'Track A: GC + adaptive inf':{'rmse': ???, 'method': 'GC localization + adaptive inflation'},
    'Track B: best architecture':{'rmse': ???, 'method': 'wider CNN + temporal context'},
    'Track C: adaptive schedule':{'rmse': ???, 'method': 'spread-adaptive EnKF/CNN switching'},
    'Track D: biased F=7.5':     {'rmse': ???, 'method': 'model error robustness test'},
}

for name, d in results.items():
    pct = (baseline_rmse - d['rmse']) / baseline_rmse * 100
    print(f"{name:35s}: RMSE={d['rmse']:.4f}  ({pct:+.1f}% vs baseline)")
```

Merge all track branches into main and create a final `group_results.ipynb`
that loads each track's saved NetCDF files and produces a single comparison figure.

---

## Suggested reading

Each track has a short reading list (1–2 papers, all freely available):

**Track A:**
- Gaspari & Cohn (1999). Construction of correlation functions in two and two dimensions. *QJRMS* 125, 723–757.
- Anderson (2009). Spatially and temporally varying adaptive covariance inflation. *Tellus A* 61, 72–83.

**Track B:**
- The tutorial notebook, Part 4 (cyclic padding and CNN architecture)
- Lguensat et al. (2017). The Analog Data Assimilation. *Monthly Weather Review* 145, 4093–4107.

**Track C:**
- Bocquet et al. (2020). Bayesian inference of chaotic dynamics by merging data assimilation, machine learning and expectation-maximization. *Foundations of Data Science* 2, 55–80.

**Track D:**
- Lorenz (1996). Predictability — a problem partly solved. (The original L96 paper; freely available online)
- Evensen et al. (2022). *Data Assimilation Fundamentals* (open-access textbook, chapters 1–3)
