# FERS ML + Data Assimilation Tutorial
## Machine Learning-Augmented Data Assimilation with the Lorenz-96 System

> **Who this is for:** Graduate students in atmospheric science, oceanography,
> or related fields. No prior knowledge of machine learning or data assimilation is assumed.
> You should be comfortable with basic Python and have seen a differential equation before.

---

## Table of Contents

1. [What is this project about?](#1-what-is-this-project-about)
2. [The big picture in plain English](#2-the-big-picture-in-plain-english)
3. [What you will learn](#3-what-you-will-learn)
4. [Repository contents](#4-repository-contents)
5. [Getting started: installation](#5-getting-started-installation)
6. [Running the tutorial notebook](#6-running-the-tutorial-notebook)
7. [What to expect as you work through the notebook](#7-what-to-expect-as-you-work-through-the-notebook)
8. [Running longer experiments](#8-running-longer-experiments)
9. [Understanding the source code](#9-understanding-the-source-code)
10. [Troubleshooting](#10-troubleshooting)
11. [Extension ideas](#11-extension-ideas)
12. [Two-week group research plan](#12-two-week-group-research-plan)
13. [Glossary](#13-glossary)
14. [Original paper and citation](#14-original-paper-and-citation)

---

## 1. What is this project about?

Weather forecasting, ocean modeling, and climate science all face the same fundamental problem:
**models are imperfect and observations are incomplete.** A weather model run forward in time
accumulates errors. Observations (from weather stations, satellites, radiosondes) are
sparse and noisy. **Data assimilation (DA)** is the statistical discipline that combines
these two imperfect sources of information to produce the best possible estimate of
the current state of the atmosphere or ocean.

This project explores whether a **neural network** — a machine learning model — can
*augment* (improve upon or work alongside) a classical DA method called the
**Ensemble Kalman Filter (EnKF)**. The experiments are performed on the
**Lorenz-96 (L96) system**, a simple mathematical model that mimics the chaotic
behavior of the atmosphere on a ring of grid points, making it an ideal testbed
for new DA methods.

In short: **Can a neural network learn how to do data assimilation, and if so,
does it do it better than the classical approach, especially when observations are sparse?**

---

## 2. The big picture in plain English

### The weather forecasting problem as an analogy

Imagine you are trying to track the location of a ship at sea. You have two sources
of information:

1. **A physics model:** You know the ship's last position and its approximate speed
   and heading. You can *predict* where it should be now — but the prediction has
   errors that grow over time (ocean currents, wind, human decisions).

2. **GPS pings:** Every few hours you get a noisy GPS reading. The reading is close
   to the true position, but not exact (instrument error, atmospheric delays).

**Data assimilation** is the process of combining the model prediction with the GPS ping
to get a better estimate than either source alone. The *Kalman filter* is the mathematically
optimal way to do this combination when errors are Gaussian.

### Why machine learning?

The classical Ensemble Kalman Filter works by running many copies (an *ensemble*)
of the model simultaneously, slightly perturbed from each other. The spread of the
ensemble tells you how uncertain the forecast is. But running a large ensemble is
expensive, and the method can struggle when observations are sparse.

A neural network, once trained, can map from *(forecast, observations)* directly
to a corrected analysis in a single fast forward pass, without needing to
maintain the full ensemble. The **augmented method** in this project alternates
between using the classical EnKF and the neural network, getting the best of both worlds.

### The Lorenz-96 system: a toy atmosphere

The L96 system is 40 numbers on a ring, evolving according to a simple equation
that produces realistic atmospheric-looking chaos. It is cheap to simulate on a
laptop (seconds to minutes), yet captures the essential difficulty of DA:
errors grow exponentially, just like in the real atmosphere.

```
x₀ ──► x₁ ──► x₂ ──► ... ──► x₃₉ ──► (back to x₀)

Each xᵢ evolves based on its neighbors. Small differences
between two initial states grow until the trajectories
look completely different — this is chaos.
```

---

## 3. What you will learn

By working through the tutorial notebook you will be able to:

- Simulate the Lorenz-96 chaotic dynamical system and visualize its behavior
- Explain what data assimilation is and why it matters for Earth system science
- Implement and run the **Ensemble Kalman Filter** with covariance localization and inflation
- Evaluate a DA system using **RMSE** and **rank histograms**
- Understand what a **1-D convolutional neural network (CNN)** is and how to train one
- Explain the concept of **cyclic padding** and why it matters for periodic domains
- Compare three methods: full-observation EnKF, sparse-observation EnKF, and ML-augmented EnKF
- Identify overfitting in a neural network training curve and know how to address it

---

## 4. Repository contents

```
fers_ml_da_lorenz/
│
├── tutorial_notebook.ipynb      ← START HERE (the main tutorial, ~2 hours)
├── generate_tutorial_data.py    ← Optional: pre-generate a larger dataset
│
├── L96.py                       ← Lorenz-96 model: defines dx/dt and integrates forward
├── EnKF.py                      ← Ensemble Kalman Filter: localization, Kalman gain, analysis
├── Experiment.py                ← Orchestrates the full DA cycle (forecast → observe → analyze)
├── NeuralNet.py                 ← 1-D CNN: cyclic padding, training, inference
│
├── tutorial_data/               ← Pre-generated data the notebook loads or produces
│   ├── enkf_full.nc             ← Results: EnKF with full observations
│   ├── enkf_sparse.nc           ← Results: EnKF with 25% sparse observations
│   ├── augmented.nc             ← Results: ML-augmented method (sparse obs)
│   ├── cnn_weights.weights.h5   ← Pre-trained CNN weights
│   └── tutorial_data.nc         ← (optional) larger pre-generated dataset
│
├── tutorial_figures/            ← Figures generated by the notebook
│   ├── l96_trajectory.png
│   ├── chaos.png
│   └── ... (one file per notebook section)
│
├── tracks/                      ← Student workspace for the two-week research project
│   ├── track_A/                 ← Better localization & inflation
│   ├── track_B/                 ← Better neural network architecture
│   ├── track_C/                 ← Better augmentation strategy
│   └── track_D/                 ← Robustness & generalization
│
└── paper_reproduction/          ← Scripts and data to reproduce original paper figures
    ├── sensitivity/             ← Sensitivity analysis runs
    ├── augmented/               ← Augmented method runs
    ├── plotting/                ← Figure generation scripts
    └── data/                    ← Pre-computed NetCDF files for paper figures
```

> **Files you will interact with:** In the tutorial, you only need `tutorial_notebook.ipynb`.
> The `.py` files in the root directory are imported automatically by the notebook.

---

## 5. Getting started: installation

### Step 1 — Check your Python version

Open a terminal and run:

```bash
python3 --version
```

You need **Python 3.9 or later**. If you see `Python 2.x` or get an error, see
[python.org/downloads](https://www.python.org/downloads/) or ask your sysadmin.

> **Using a university HPC cluster?** Most clusters have Python available via a
> module system. Run `module avail python` and load a Python 3.9+ module.
> Then skip the conda steps below and go straight to Step 3.

### Step 2 — Create an isolated environment (strongly recommended)

An isolated environment keeps this project's packages from conflicting with other
projects on your system. We recommend **conda** (part of Anaconda or Miniconda).

**If you do not have conda:**
Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
(the lightweight version — you do not need the full Anaconda).

**Create and activate the environment:**

```bash
# Create a new environment called "mlda" with Python 3.11
conda create -n mlda python=3.11 -y

# Activate it (you must do this every time you open a new terminal)
conda activate mlda
```

You should now see `(mlda)` at the start of your terminal prompt.

> **Without conda:** You can use Python's built-in `venv` instead:
> ```bash
> python3 -m venv mlda_env
> source mlda_env/bin/activate    # macOS / Linux
> mlda_env\Scripts\activate       # Windows
> ```

### Step 3 — Clone the repository

```bash
# Navigate to where you want to put the project
cd ~/Documents   # or wherever you like

# Clone the repository
git clone https://github.com/aneeshcs/fers_ml_da_lorenz.git

# Enter the project folder
cd fers_ml_da_lorenz
```

> **Don't have git?** Install it from [git-scm.com](https://git-scm.com/) or,
> on macOS, run `xcode-select --install`.

### Step 4 — Install the required packages

With your environment activated, install all dependencies in one command:

```bash
pip install numpy scipy matplotlib xarray tqdm tensorflow jupyter
```

This will download and install:

| Package | What it does in this project |
|---|---|
| `numpy` | Array math (the backbone of everything) |
| `scipy` | ODE solver used to integrate the L96 equations |
| `matplotlib` | Plotting all figures |
| `xarray` | Labeled arrays — stores results with named dimensions (space, time, ensemble) |
| `tqdm` | Progress bars during the EnKF assimilation loop |
| `tensorflow` / `keras` | Building and training the neural network |
| `jupyter` | Running the notebook in your browser |

Installation takes 2–5 minutes depending on your internet speed.

**Verify the installation worked:**

```bash
python3 -c "import numpy, scipy, matplotlib, xarray, tqdm, tensorflow; print('All packages OK!')"
```

You should see `All packages OK!` with no errors. If you get a `ModuleNotFoundError`,
re-run the `pip install` command above and check that your environment is activated.

### Step 5 — Quick sanity check

Before opening the notebook, verify the core model code runs:

```bash
python3 -c "
from L96 import L96
import numpy as np
m = L96(F=8, N=40)
x0 = 8*np.ones(40); x0[0] += 0.01
t = np.linspace(0, 1, 21)
out = m.integrate(x0, t)
print('L96 integration OK. Output shape:', out.shape)
"
```

Expected output: `L96 integration OK. Output shape: (40, 21)`

---

## 6. Running the tutorial notebook

### Launch Jupyter

From inside the `fers_ml_da_lorenz` folder (with your environment activated), run:

```bash
jupyter notebook
```

This opens a browser window (usually at `http://localhost:8888`). You will see a
file browser. Click on **`tutorial_notebook.ipynb`** to open it.

> **On a remote server or HPC?** Use:
> ```bash
> jupyter notebook --no-browser --port=8888
> ```
> Then follow your institution's instructions for port-forwarding to your local browser.
> Many HPC systems also support JupyterHub — ask your sysadmin.

### Running cells

Jupyter notebooks consist of **cells** — blocks of code or text. To run a cell:

- Click inside it, then press **Shift + Enter** to run it and advance to the next cell
- Or click the **▶ Run** button in the toolbar
- **Run all cells at once:** Kernel → Restart & Run All

> **Important:** Always run cells in order from top to bottom. Later cells depend on
> variables created by earlier cells. If you get a `NameError`, it usually means
> you skipped a cell.

### If something goes wrong mid-notebook

Restart the kernel (clears all variables) and re-run from the top:
**Kernel → Restart & Clear Output**, then **Cell → Run All**.

---

## 7. What to expect as you work through the notebook

Here is a section-by-section guide to what you will see and approximately how long
each section takes on a standard laptop.

### Part 1: The Lorenz-96 System (~15 minutes)

**What happens:** You will generate a 50-time-unit (~250 day) trajectory of the
L96 system and visualize it in two ways:

1. A **Hovmöller diagram** — a color plot with space on the vertical axis and time
   on the horizontal axis, like a time-longitude plot in meteorology. You will see
   wave-like structures propagating in time.

2. A **chaos demonstration** — two trajectories that start 0.0001 apart quickly
   diverge. The log-scale divergence plot shows a straight line (exponential growth),
   which is the signature of chaotic dynamics.

**What numbers to expect:**
- State standard deviation ≈ 3.7 (this is the "climatological" variability of L96 with F=8)
- Chaos decorrelation time ≈ 2–3 L96 time units (~10–15 days)

### Part 2: Observations (~10 minutes)

**What happens:** You will add Gaussian noise to the truth to create synthetic
observations, then visualize what "sparse observations" look like — only 10 of the
40 grid points are observed, the rest are missing (NaN).

**What numbers to expect:**
- Observation noise: r ≈ 1.1 (= 30% of the state standard deviation, matching the paper)

### Part 3: The Ensemble Kalman Filter (~15 minutes to read + 10 minutes to run)

**What happens:** The EnKF assimilation loop runs 1,000 analysis steps (one per
observation time). A progress bar shows the speed — expect ~100 iterations/second,
so the full loop takes about 10 seconds.

After it finishes, you will see:
- An **analysis RMSE** value (how far the ensemble-mean analysis is from the truth)
- A **rank histogram** that shows whether the ensemble is well-calibrated

**What numbers to expect with T=50, nens=20:**
- Analysis RMSE ≈ 3.1–3.4 (compared to climatological RMSE ≈ 3.7)
- Spread/RMSE ratio ≈ 0.05–0.1 (much less than 1 — the ensemble is *underdispersive*)

> **Why is the RMSE so close to climatology and the ensemble so underdispersive?**
> With only 20 ensemble members and no inflation (`gamma=1.0`), the EnKF tends to
> collapse — ensemble members cluster together, underestimating uncertainty.
> This is a known limitation at small ensemble sizes, and exploring it is the
> point of Exercise 3. With `gamma=1.05` and a longer run, RMSE drops significantly.

### Part 4: The Neural Network (~15 minutes to read + 2 minutes to run)

**What happens:** A 3-layer 1-D CNN is built and trained for 30 epochs on the
output of the EnKF run. Each epoch takes about 1 second.

**What to watch for — the overfitting signature:**

```
Epoch  1: train RMSE = 5.6,  val RMSE = 4.8
Epoch  5: train RMSE = 3.2,  val RMSE = 4.2
Epoch 12: train RMSE = 0.97, val RMSE = 3.6   ← validation stops improving
Epoch 30: train RMSE = 0.37, val RMSE = 3.67  ← large gap = overfitting
```

The training RMSE drops steadily but the validation RMSE barely moves after epoch 12.
This means the network has **memorised** the training data rather than learning the
underlying pattern. **This is expected** with only ~700 training samples from T=50.
The notebook tells you how to fix it: use T=200+ for a well-generalising model.

### Part 5: The ML-Augmented Method (~10 minutes to run)

**What happens:** Two more experiments run — a sparse-observation EnKF (25%
observed) and an augmented method (alternating EnKF and CNN). A bar chart and
time series compare all three methods.

**What numbers to expect:**

| Method | RMSE |
|--------|------|
| EnKF (full obs, 100%) | ~3.1–3.4 |
| EnKF (sparse obs, 25%) | ~3.1–3.5 |
| Augmented (sparse, 25%) | ~2.4–2.6 |

The augmented method typically shows **15–25% improvement** over the sparse EnKF,
even with an overfitted network. This demonstrates the core result of the paper.

### Parts 6–7: Saving and extensions (~5 minutes)

Results are saved to NetCDF files and a 4-panel summary figure is generated.
Part 7 describes open-ended extension exercises.

---

## 8. Running longer experiments

The tutorial uses `T=50` for speed. For more meaningful results:

### Option A: Change T directly in the notebook

In the **enkf-full-run** cell, change `T_run = 50.0` to `T_run = 200.0` (or `500.0`).

Expected compute times on a standard laptop:

| T | Compute time | Training samples | CNN val RMSE |
|---|---|---|---|
| 50 | ~1 min | ~700 | ~3.7 (overfitting) |
| 200 | ~5 min | ~2,800 | ~1.5–2.0 |
| 500 | ~12 min | ~7,000 | ~0.8–1.2 |
| 2000 (paper) | ~50 min | ~28,000 | ~0.4–0.6 |

### Option B: Pre-generate a dataset

Run the standalone script (this uses T=200, nens=20):

```bash
python3 generate_tutorial_data.py
```

This saves `tutorial_data/tutorial_data.nc`. In the notebook, instead of running the EnKF,
load the pre-generated data:

```python
import xarray as xr
ds = xr.open_dataset('tutorial_data/tutorial_data.nc')
# Assign to exp.ds before training the CNN:
exp.ds['xx']    = ds.xx
exp.ds['xaens'] = ds.xaens_full
exp.ds['xfens'] = ds.xfens_full
exp.ds['yy']    = ds.yy_full
```

---

## 9. Understanding the source code

You do not need to modify these files to complete the tutorial, but understanding
them will help you complete the extension exercises.

### `L96.py` — The Lorenz-96 model

```python
class L96:
    def __init__(self, F, N):   # F = forcing, N = number of variables
    def derivative(self, t, x): # computes dx/dt at a given state x
    def integrate(self, x0, t): # integrates forward using scipy's solve_ivp
```

The core equation for each variable $x_i$ is:
```
dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1}  -  x_i  +  F
           ↑ nonlinear advection             ↑ damping  ↑ forcing
```
Indices are cyclic (wrap around the ring).

### `EnKF.py` — The Ensemble Kalman Filter

```python
class EnKF:
    def obs(self, h, x):              # applies observation operator H
    def getK(self, cov, h, r):        # computes Kalman gain K
    def localize(self, cov):          # zeroes out far-away covariances
    def assimilate(self, xf, y, ...): # single-member analysis update
    def ensemble_assim(self, xf, y, h, r): # full ensemble analysis step
```

The key step is `ensemble_assim`:
1. Compute sample covariance from the ensemble (`np.cov(xf)`)
2. Localize it (set entries to zero where grid-point distance > `loc`)
3. Multiply by `gamma` (inflation)
4. Compute Kalman gain `K`
5. For each ensemble member, add `K * (perturbed_obs - H * forecast)`

### `Experiment.py` — The DA cycle manager

```python
class Experiment:
    # Stores: truth (xx), observations (yy), analyses (xaens), forecasts (xfens)
    # in an xarray.Dataset (exp.ds)
    def get_true(self, x0, tf):     # integrates L96 to make a truth trajectory
    def makeobs(self, std):         # adds Gaussian noise to truth
    def make_ensemble(self, x0, std): # creates perturbed initial ensemble
    def assimilate(self, xf0, nn=None): # runs the full forecast-analyze cycle
```

In `assimilate()`, if `nn=None` the pure EnKF is used. If a `NeuralNet` object
is passed, the augmented method is used (CNN on odd steps, EnKF on even steps).

### `NeuralNet.py` — The 1-D CNN

```python
class NeuralNet:
    def make_input(self, xf, y):    # applies cyclic padding and stacks channels
    def buildmodel(self):           # builds the Keras Sequential model
    def train(self, ..., experiment): # trains on EnKF output
    def assimilate(self, xf, y):    # runs one forward pass (inference)
```

**Why cyclic padding?** The L96 domain is a ring — variable 39 is adjacent to
variable 0. A standard CNN doesn't know this; without padding, edge variables would
see "nothing" on one side. Cyclic padding wraps the ends around so every variable
sees its true neighbours.

**The two input channels:**
- Channel 0: the forecast $x^f$
- Channel 1: the innovation $(y - x^f)$ — the disagreement between observation and forecast

Feeding the innovation instead of the raw observation helps the network learn
*corrections* rather than absolute values.

---

## 10. Troubleshooting

### `ModuleNotFoundError: No module named 'xxx'`

Your environment is not activated, or you forgot to install the packages.

```bash
# Check your environment is active (you should see (mlda) in your prompt)
conda activate mlda   # or: source mlda_env/bin/activate

# Re-install
pip install numpy scipy matplotlib xarray tqdm tensorflow jupyter
```

### `NameError: name 'exp' is not defined`

You skipped a cell. Restart the kernel and run from the top:
**Kernel → Restart & Clear Output → Cell → Run All**

### The EnKF cell hangs or takes forever

The EnKF loop with T=50 should finish in under 30 seconds. If it is much slower,
your scipy version may be slow at the ODE solver. Try:

```bash
pip install --upgrade scipy
```

If still slow, reduce the ensemble size in the notebook (`nens=10` instead of `nens=20`).

### `tensorflow` errors on Apple Silicon (M1/M2/M3 Mac)

Install the Apple-optimized TensorFlow:

```bash
pip uninstall tensorflow -y
pip install tensorflow-macos tensorflow-metal
```

### `ValueError` or shape mismatch in the CNN section

This usually means the EnKF run did not complete or the dataset has unexpected dimensions.
Restart the kernel and re-run all cells in order from the top.

### Jupyter doesn't open in my browser

Copy the URL shown in the terminal (it looks like `http://localhost:8888/?token=abc123...`)
and paste it manually into your browser.

### On Windows: `git` or `python3` not recognized

- Install Git from [git-scm.com](https://git-scm.com/) and tick "Add to PATH" during installation.
- Use `python` instead of `python3` on Windows (they are the same in Anaconda/Miniconda).

---

## 11. Extension ideas

The tutorial notebook includes detailed extension exercises. Here is a quick overview
of what students have explored:

**Difficulty level 1 — Modify and observe:**
- Change the L96 forcing `F` and see how chaos changes
- Try different observation noise levels (`r = 0.1 * std`, `r = 0.5 * std`)
- Tune EnKF localization (`loc`) and inflation (`gamma`) to minimize RMSE
- Compare rank histograms for different ensemble sizes

**Difficulty level 2 — Algorithmic changes:**
- Implement smooth (Gaspari-Cohn) localization instead of the hard cutoff
- Add adaptive inflation that adjusts `gamma` based on innovation statistics
- Change the CNN–EnKF alternation ratio (e.g., CNN every 4th step instead of every other)
- Train the CNN to predict the truth directly rather than the EnKF analysis

**Difficulty level 3 — Research extensions:**
- Apply the method to the two-level (slow + fast) Lorenz-96 system
- Introduce model error: use a slightly wrong `F` in the forecast model
- Add SHAP explainability analysis (see `paper_reproduction/plotting/postprocess_shap.py`)
- Try an LSTM or attention-based architecture instead of the CNN

---

## 12. Two-week group research plan

This section is for a group of 3–4 students who want to go beyond the tutorial
and conduct original research over two weeks. The work is divided into four
independent but complementary tracks, each producing a result comparable against
the same baseline (the paper's augmented method).

**Baseline to beat:**
- Analysis RMSE of the augmented EnKF+CNN method on sparse observations (25% coverage)
- Rank histogram calibration of the analysis ensemble
- 10-day ensemble forecast accuracy (Figure 5 in the paper)

The full research plan, including all code scaffolding, is also available as a
standalone file: [`RESEARCH_PLAN.md`](RESEARCH_PLAN.md)

### Git workflow for collaboration

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

**Ground rules for shared code:**
- Do not modify `L96.py`, `EnKF.py`, `NeuralNet.py`, or `Experiment.py` directly on `main`.
  Instead, subclass the relevant class inside your track's notebook, or work on your own branch.
- The pre-generated datasets (`tutorial_data.nc`, `tutorial_enkf_full.nc`) are read-only
  shared inputs — load them, don't overwrite them.
- At the end of Week 1, hold a 30-minute group sync to share interim RMSE numbers
  so everyone can sanity-check against the same baseline.

### Week 1 — Everyone (Days 1–2)

Before splitting into tracks, all students should:

1. Run `tutorial_notebook.ipynb` end-to-end and confirm your RMSE numbers match
   the expected values in Section 7 of this README.
2. Read Sections 3–5 of the tutorial notebook carefully, paying attention to the
   code in `EnKF.py`, `Experiment.py`, and `NeuralNet.py`.
3. Read the paper (Howard, Subramanian & Hoteit, 2024) Sections 2–4.
4. Run the shared baseline script below and record your number:

```python
# baseline.py — run once and share the result with the group
from Experiment import Experiment
from NeuralNet import NeuralNet
import numpy as np

np.random.seed(0)
N, F, dt, T = 40, 8.0, 0.05, 200.0
x0 = F * np.ones(N); x0[0] = F + 0.01

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
print(f"Baseline augmented RMSE: {float(np.sqrt(((xa-xx)**2).mean())):.4f}")
```

---

### Track A — Better classical DA: smooth localization and adaptive inflation
**Best for:** A student comfortable with linear algebra and statistics who wants
to understand the classical DA side more deeply. No ML required.

The paper uses two crude choices: a **hard-cutoff localization** (covariance
snaps to zero at distance `loc`) and a **fixed inflation factor** (`gamma` = constant).
Both have well-known improvements used in all operational forecast systems.

#### Part 1: Gaspari-Cohn smooth localization (Days 3–5)

Replace `EnKF.localize()` with the Gaspari-Cohn (1999) fifth-order piecewise
rational taper, which decays smoothly to zero rather than cutting off abruptly:

```python
def gaspari_cohn(distance, half_width):
    r = abs(distance) / half_width
    if r >= 2.0:   return 0.0
    elif r >= 1.0: return (r**5/12 - r**4/2 + 5*r**3/8 + 5*r**2/3 - 5*r + 4 - 2/(3*r)) / 3
    else:          return 1 - 5*r**2/3 + 5*r**3/8 + r**4/2 - r**5/4

class EnKFGC(EnKF):
    def localize(self, cov):
        for i in range(self.nvars):
            for j in range(self.nvars):
                mid = abs(i - j)
                out = self.nvars - max(i,j) + min(i,j)
                dist = min(mid, out)
                cov[i, j] *= gaspari_cohn(dist, self.loc)
        return cov
```

> Reference: Gaspari & Cohn (1999). *Q. J. Roy. Meteor. Soc.* 125, 723–757. Eq. 4.10.

**Key question:** For the same `loc`, does GC localization improve RMSE over the
hard cutoff? Does the optimal `loc` value change?

#### Part 2: Adaptive multiplicative inflation (Days 5–7)

After each analysis step, estimate the innovation variance and compare it to what
the ensemble predicted. Use the ratio to update `gamma` automatically:

```
gamma_{t+1} = gamma_t * <(y - H x^f)^2> / diag(H P^f H^T + R)
```

Implement this in a subclass of `Experiment` that overrides `assimilate()` to
recompute `gamma` at each step and log its time series.

**Key question:** Does adaptive inflation outperform the paper's best fixed value
(`gamma=1.05`)? What does the time series of `gamma_t` look like?

#### Deliverables for Track A
A notebook `track_A_localization_inflation.ipynb` with: RMSE vs. `loc` for hard
vs. GC localization; RMSE vs. fixed `gamma`; adaptive `gamma` time series; rank
histograms for each method.

**Reading:** Gaspari & Cohn (1999); Anderson (2009). *Tellus A* 61, 72–83.

---

### Track B — Better neural network architecture
**Best for:** A student interested in deep learning who wants to explore
how architecture choices affect performance.

The paper's CNN is deliberately minimal (3 layers, 5 filters, kernel 3). It has
no temporal memory, only predicts the analysis mean, and uses an architecture that
was not tuned. There is substantial room to improve each of these.

#### Part 1: Architecture search (Days 3–4)

Systematically compare CNN architectures, training each on the same dataset:

```python
architectures = [
    ('paper_baseline', 3, 3, 5),   # (name, nlayers, filter_size, n_filters)
    ('wider',          3, 3, 16),
    ('deeper',         5, 3, 5),
    ('larger_kernel',  3, 5, 5),
    ('wider_deeper',   5, 3, 16),
]
```

Note: changing `nlayers` or `filter_size` changes the padding `offset` — update
`NeuralNet.make_input()` accordingly.

#### Part 2: Adding temporal context (Days 4–6)

The CNN only sees the state at time `t`. Forecast errors at `t` are correlated
with errors at `t-1` and `t-2`. Stack the last `k` forecast states as extra input
channels and modify `NeuralNet.train()` to build sliding-window pairs.

```
Current:  (N+2*offset, 2 channels)    ← x^f_t,  y_t - x^f_t
Extended: (N+2*offset, 2+k channels)  ← x^f_t, y_t-x^f_t, x^f_{t-1}, x^f_{t-2}, ...
```

**Key question:** Does adding 1, 2, or 3 previous steps improve RMSE? At what
point does it stop helping?

#### Part 3: Predicting analysis uncertainty (Days 6–7)

Modify `buildmodel()` to output 2 channels (mean + log-variance) and train with
negative log-likelihood loss. Then use the predicted spread to rescale the ensemble
rather than just shifting it:

```python
xa_mean, log_var = nn.assimilate(xf, y)
xa_std = np.exp(0.5 * log_var)
scale  = xa_std / xf.std(axis=1)
for j in range(N):
    xaens[j, :, i] = xa_mean[j] + scale[j] * (xfens[j, :, i] - xf[j])
```

**Key question:** Does a probabilistic CNN produce better-calibrated rank
histograms? Does it also improve RMSE?

#### Deliverables for Track B
A notebook `track_B_architectures.ipynb` with: architecture comparison table
(params, val RMSE, augmented RMSE); training curves; temporal context ablation
(k=0,1,2,3); rank histograms for deterministic vs. probabilistic CNN.

**Reading:** Tutorial notebook Part 4; Lguensat et al. (2017). *MWR* 145, 4093–4107.

---

### Track C — Better augmentation strategy
**Best for:** A student interested in algorithm design — how and when to
combine the two components.

The paper uses a fixed schedule: CNN on odd steps, EnKF on even steps, regardless
of the system state. There are two dimensions to improve: *when* to use each
method, and *how* to apply the CNN output to the ensemble.

#### Part 1: Adaptive scheduling (Days 3–5)

Use the **ensemble spread** to decide which method to apply at each step. When
spread is low (ensemble collapsed), use EnKF to restore diversity; when spread is
healthy, use the cheaper CNN:

```python
SPREAD_THRESHOLD = 0.5  # tune this

spread_t = xfens[:, :, i].std(axis=1).mean()
if spread_t < SPREAD_THRESHOLD:
    h = self.make_obs()
    xaens[:, :, i] = enkf.ensemble_assim(xfens[:, :, i], y, h, self.r)
else:
    xpmean = nn.assimilate(xfens[:, :, i].mean(axis=1), y.values)
    delta  = xfens[:, :, i].mean(axis=1) - xpmean
    for j in range(self.N):
        xaens[j, :, i] = xfens[j, :, i] - delta[j]
```

Also test fixed ratios: CNN every 1st, 2nd, 3rd, 4th, 8th step.

#### Part 2: Better ensemble updating (Days 5–7)

The paper mean-shifts all members by the same delta. Test three alternatives:

**Nudging (relaxation):** pull each member toward the CNN mean
```python
alpha = 0.5
xaens[:, j, i] = alpha * xpmean + (1 - alpha) * xfens[:, j, i]
```

**Spread scaling:** shift the mean and scale the perturbations
```python
pert = xfens[:, j, i] - xfens[:, :, i].mean(axis=1)
xaens[:, j, i] = xpmean + (target_spread / xfens[:, :, i].std(axis=1)) * pert
```

**Per-member CNN:** apply the CNN independently to each ensemble member's forecast.

**Key question:** Which update rule produces the best-calibrated ensemble while
also minimizing RMSE?

#### Deliverables for Track C
A notebook `track_C_augmentation.ipynb` with: RMSE vs. CNN frequency for fixed
schedules; adaptive vs. best fixed schedule; rank histograms and RMSE for the
three update rules; time series of ensemble spread.

**Reading:** Bocquet et al. (2020). *Foundations of Data Science* 2, 55–80.

---

### Track D — Robustness and generalization
**Best for:** A student interested in scientific questions rather than
implementation — more experiments, less new code.

The paper demonstrates the augmented method under one specific setup (F=8, N=40,
T=2000). Real applications require robustness. This track asks: **how fragile
is the augmented method?**

#### Part 1: Model error (Days 3–5)

Train on a perfect-model run (F=8), then test with a biased forecast model:

```python
F_true     = 8.0   # truth and observations generated with this
F_forecast = 7.5   # ensemble forecasts use this wrong value
```

Test biases of F ± 0.25, ± 0.5, ± 1.0. Implement by subclassing `Experiment`
and changing the `L96` object used in the forecast step of `assimilate()`.

**Key question:** Is the augmented method more or less robust to model bias
than the standard EnKF?

#### Part 2: Sparsity generalization (Days 5–6)

Train the CNN on full observations, then evaluate across coverage fractions:

```python
fracs = [0.1, 0.2, 0.25, 0.33, 0.5, 0.75, 1.0]
```

Also test: train a CNN at 50% coverage and deploy at 25%. Does transfer across
sparsity levels work?

#### Part 3: Forcing sensitivity (Days 6–7)

Repeat the full augmented experiment (train EnKF → train CNN → augmented test)
for `F=5` (weakly chaotic), `F=8` (paper default), and `F=12` (very turbulent).

**Key question:** Does the augmented method help more in high-chaos or low-chaos
regimes? Which regime is the CNN easiest to train on?

#### Part 4 (bonus): Spatially averaged observations

Replace point observations with area averages (mimicking satellite footprints):

```python
def make_superobs(x, frac, window=3):
    obs = []
    for i in range(0, N, window):
        if np.random.rand() < frac:
            obs.append((i, x[i:i+window].mean()))
    return obs
```

#### Deliverables for Track D
A notebook `track_D_robustness.ipynb` with: RMSE vs. model error bias; RMSE vs.
observation fraction; RMSE across F=5, 8, 12; (bonus) spatially-averaged obs results.

**Reading:** Lorenz (1996). *Predictability — a problem partly solved*;
Evensen et al. (2022). *Data Assimilation Fundamentals*, Chapters 1–3 (open access).

---

### Week 2 integration: comparing all tracks

On the final day, each student fills in a shared comparison table and submits
a pull request to `main`:

```python
# shared_comparison.py — each student fills in their best result
baseline_rmse = ???   # from the shared baseline script above

results = {
    'Baseline (paper)':           {'rmse': ???, 'method': 'EnKF+CNN, fixed alternation, hard loc'},
    'Track A: GC + adaptive inf': {'rmse': ???, 'method': 'GC localization + adaptive inflation'},
    'Track B: best architecture': {'rmse': ???, 'method': 'wider CNN + temporal context'},
    'Track C: adaptive schedule': {'rmse': ???, 'method': 'spread-adaptive EnKF/CNN switching'},
    'Track D: model error F±0.5': {'rmse': ???, 'method': 'robustness to biased forecast model'},
}

for name, d in results.items():
    pct = (baseline_rmse - d['rmse']) / baseline_rmse * 100
    print(f"{name:38s}: RMSE={d['rmse']:.4f}  ({pct:+.1f}% vs baseline)")
```

Create a `group_results.ipynb` that loads each track's saved NetCDF files and
produces a single comparison figure for the group presentation.

---

## 13. Glossary

**Analysis** — The updated state estimate produced by data assimilation after incorporating observations. Denoted $\mathbf{x}^a$.

**Chaos / Chaotic system** — A dynamical system whose future state is extremely sensitive to small differences in initial conditions. Nearby trajectories diverge exponentially. This is why weather prediction has a finite time horizon (about 2 weeks).

**Climatological RMSE** — The RMSE you would get by always predicting the long-run average of the system. A useful baseline: your DA system should do much better than this.

**CNN (Convolutional Neural Network)** — A type of neural network that applies a small, shared filter (kernel) repeatedly across an input. On spatial data like L96, this is efficient because the same filter is used at every grid point — the network does not need to learn separate weights for each location.

**Covariance** — A measure of how two variables change together. If $x_i$ tends to be high when $x_j$ is also high, their covariance is positive. In an EnKF, the covariance matrix tells us how errors at one grid point relate to errors elsewhere, which determines how strongly an observation at one location corrects the state at another.

**Covariance inflation** — Multiplying the forecast covariance by a factor $\gamma > 1$ to prevent the ensemble from becoming overconfident (collapsing). A small amount of inflation (e.g., $\gamma = 1.05$) often improves EnKF performance significantly.

**Covariance localization** — Setting the covariance between distant grid points to zero. With a small ensemble, distant correlations are dominated by sampling noise and can mislead the analysis. Localization fixes this.

**Data assimilation (DA)** — The statistical procedure for combining a model forecast (prior) with observations (likelihood) to produce an analysis (posterior). Grounded in Bayes' theorem.

**Ensemble** — A collection of model runs, each started from a slightly different initial condition (or using slightly different model parameters). The spread of the ensemble represents forecast uncertainty.

**EnKF (Ensemble Kalman Filter)** — A DA method that uses an ensemble of model runs to estimate the forecast error covariance, then applies the Kalman update formula to produce an analysis. The standard workhorse of operational weather forecasting.

**Epoch** — One complete pass through the training dataset during neural network training. More epochs = more training, but too many epochs can lead to overfitting.

**Forecast** — The model prediction of the current state, propagated forward from the last analysis. Also called the "prior" or "background". Denoted $\mathbf{x}^f$.

**Hovmöller diagram** — A type of diagram with space on one axis and time on the other, used to visualize wave propagation and patterns in geophysical data.

**Innovation** — The difference between an observation and the forecast at the observation location: $\mathbf{y} - \mathbf{H}\mathbf{x}^f$. Also called "departure". The Kalman update pushes the analysis toward the observations in proportion to the innovation and the Kalman gain.

**Kalman gain** — The matrix $\mathbf{K}$ that determines how much weight to give to the innovation vs. the forecast. Large gain → trust observations more. Computed from the forecast error covariance and the observation error covariance.

**Lorenz-96 (L96)** — A simple mathematical model of atmospheric dynamics on a periodic (ring) domain, introduced by Edward Lorenz in 1996. Used as a cheap, realistic testbed for new DA methods.

**Lyapunov exponent** — The rate at which nearby trajectories diverge in a chaotic system. A positive Lyapunov exponent $\lambda$ means distances grow as $e^{\lambda t}$. For L96 with F=8, $\lambda \approx 0.47$.

**NetCDF** — A file format widely used in geosciences for storing multi-dimensional array data (e.g., temperature as a function of latitude, longitude, and time). Files have the extension `.nc`.

**Observation operator** — The function $\mathbf{H}$ that maps from the model state space to the observation space. In our L96 experiments, it simply selects a subset of grid points.

**Overfitting** — When a machine learning model performs well on the training data but poorly on new (validation or test) data. The model has memorised the training examples rather than learning the underlying pattern. Diagnosed by a large gap between training and validation errors.

**RMSE (Root Mean Square Error)** — The square root of the average squared difference between a prediction and the truth. Lower is better. It has the same units as the quantity being predicted.

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (\hat{x}_i - x_i^{\text{truth}})^2}$$

**Rank histogram (Talagrand diagram)** — A diagnostic plot for ensemble calibration. For each observation time, the rank of the truth among the ensemble members is recorded. A flat histogram means the ensemble is well-calibrated. A U-shape means the ensemble is too wide; an arch (∩) means it is too narrow (common with small ensembles and no inflation).

**Sparse observations** — The case where only a fraction of grid points (or spatial domain) is observed at each time step. Common in practice — weather stations, ships, and satellites do not cover everything. The paper finds the ML-augmented method provides the most benefit in this setting.

**State vector** — The collection of all variables needed to describe the current state of the model. For L96 with N=40, the state vector is a vector of 40 numbers, one per grid point. For a real atmosphere model, the state vector has millions to billions of components.

**Synthetic observations** — Observations created by adding noise to a model-generated "truth". Used in "identical twin" experiments to test DA methods under controlled conditions where the true state is known.

**xarray** — A Python package for labeled multi-dimensional arrays. In this project, it stores results with named dimensions (`space`, `time`, `ensemble`) so you can slice by label (e.g., `ds.xaens.sel(time=5.0)`) rather than by index.

---

## 14. Original paper and citation

This code accompanies the following peer-reviewed paper:

> Howard, L. J., Subramanian, A., & Hoteit, I. (2024).
> **A machine learning augmented data assimilation method for high-resolution observations.**
> *Journal of Advances in Modeling Earth Systems*, 16(1), e2023MS003774.
> https://doi.org/10.1029/2023MS003774

**BibTeX entry:**

```bibtex
@article{howard2024mlda,
  author  = {Howard, Lucas J. and Subramanian, Aneesh and Hoteit, Ibrahim},
  title   = {A Machine Learning Augmented Data Assimilation Method for High-Resolution Observations},
  journal = {Journal of Advances in Modeling Earth Systems},
  year    = {2024},
  volume  = {16},
  number  = {1},
  pages   = {e2023MS003774},
  doi     = {10.1029/2023MS003774},
  url     = {https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2023MS003774}
}
```

**Key result from the paper:** The ML-augmented method reduces analysis RMSE by
up to **37%** compared to an EnKF that ignores high-resolution observations,
and produces ensemble forecasts that remain more accurate at lead times up to 10 days.

Upstream code repository: [https://github.com/climprocpred/machine_learning_DA_part_1](https://github.com/climprocpred/machine_learning_DA_part_1)

---

This tutorial fork (`fers_ml_da_lorenz`) was created for the FERS summer school.
The tutorial notebook, pre-generated datasets, and educational documentation were
added on top of the original research code.

If you use this code in your research, please cite the original paper above.

---

*Questions? Open an issue on this repository or contact your course instructor.*
