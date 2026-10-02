# Guide to Running ACE2 Stratospheric Nudging Experiments

Welcome! This guide is designed to walk you step-by-step through running stratospheric nudging experiments with the **ACE2** AI weather and climate model on `maths-gpu`.

---

## 1. Introduction: What is Nudging?

In weather forecasting, chaotic atmospheric dynamics cause perturbed ensemble members to spread apart over time. In the stratosphere (the upper levels of the atmosphere), large-scale circulation patterns—such as the polar vortex and sudden stratospheric warmings (SSWs)—have a profound influence on surface weather weeks later.

The **SNAPSI** (Stratospheric Nudging And Predictable Surface Impacts) experiment tests this by "nudging" the model's stratosphere towards historical observations (ERA5 reanalysis), while letting the rest of the atmosphere evolve freely.

In ACE2, we can run four types of forecasts:
1. **Free Forecast ($\tau = \infty$):** No nudging. The model runs completely on its own.
2. **Prescribed Nudging ($\tau = 0$):** Strongest nudging. At every 6-hour timestep, the uppermost zonal wind (`eastward_wind_0`, ~1 hPa) is completely replaced with ERA5 observations.
3. **Blended Relaxation Nudging ($\tau > 0$):** Gentle nudging. The wind at the next timestep is calculated as a blend of the model's own prediction and target reanalysis:
   $$u(t + 6\text{h}) = w_{\text{model}} \cdot u_{\text{pred}} + w_{\text{reanalysis}} \cdot u_{\text{target}}$$
   where the weights are calculated from a relaxation timescale $\tau$:
   $$w_{\text{model}} = e^{-6\text{h} / \tau}, \quad w_{\text{reanalysis}} = 1 - w_{\text{model}}$$
   - For a **24-hour timescale ($\tau = 24\,\text{h}$)**: $w_{\text{model}} \approx 0.78$, $w_{\text{reanalysis}} \approx 0.22$.
   - For an **8.7-hour timescale ($\tau \approx 8.7\,\text{h}$)**: $w_{\text{model}} = 0.50$, $w_{\text{reanalysis}} = 0.50$.
4. **SNAPSI Control Forecast (`control-full`):** To isolate the impact of anomalous stratospheric events (such as an SSW) from the seasonal background cycle, SNAPSI nudges the upper stratosphere towards the **daily climatological state** rather than the observed state of the forecast year:
   $$u(t + 6\text{h}) = w_{\text{model}} \cdot u_{\text{pred}} + w_{\text{reanalysis}} \cdot u_{\text{clim}}$$
   where $u_{\text{clim}}$ is a smoothed 40-year daily climatological annual cycle computed from ERA5 (1980–2020).

---

## 2. Setting Up Your Python Environment on `maths-gpu`

Because you will be working with a modified version of the ACE code containing custom stratospheric nudging features, it is best to create a dedicated conda environment for it.

### Step 2.1: Create and Activate a New Conda Environment
Log into `maths-gpu` and create a clean conda environment (e.g. named `ace_nudge`):
```bash
conda create --name ace_nudge python=3.10 -y
conda activate ace_nudge
```

### Step 2.2: Install Required Base Packaging Tools
Before installing ACE, install the required setuptools version:
```bash
pip install setuptools==80.0.0
```

### Step 2.3: Set Essential Environment Variables
To ensure PyTorch manages GPU memory efficiently and dynamically links to the conda environment's C++ NetCDF and CUDA libraries, export these environment variables in your terminal (or add them to your `~/.bashrc`):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
```

---

## 3. Getting the Nudging Code into Your ACE Repository

Since you already have your own local clone and fork of the ACE repository, you do not need to re-clone from scratch! Instead, you can add Will's repository as a git remote, fetch the `stepper_override_nudging` branch, and branch off it.

### Step 3.1: Navigate to Your Existing ACE Repository
```bash
cd ~/ace  # or the path to your existing ACE repository
```

### Step 3.2: Add Will's Repository as a Git Remote
Add Will's GitHub repo as a new remote called `will` (or `wseviour`):
```bash
git remote add will git@github.com:wseviour/ace.git
```
*(If you use HTTPS rather than SSH, you can use: `https://github.com/wseviour/ace.git`)*

### Step 3.3: Fetch Will's Branches
Fetch all branches and commits from Will's repository:
```bash
git fetch will
```

### Step 3.4: Create Your Own Local Branch from Will's Nudging Branch
Create and switch to a new local branch (e.g. `nudging` or `my_nudging`) based on `will/stepper_override_nudging`:
```bash
git checkout -b nudging will/stepper_override_nudging
```

### Step 3.5: Install the Nudging ACE Package in Your Environment
With your new `ace_nudge` conda environment activated, install this modified ACE package in editable mode:
```bash
pip install -e .
```
You now have the full nudging functionality installed and ready in your `ace_nudge` environment!

---

## 4. How to Run an Ensemble Experiment (The Easy Way)

We have created an automated script, **`run_ensemble_experiment.py`**, that handles the entire pipeline for you with a single command!

### What the Script Does Automatically:
1. **Generates Initial Conditions:** Takes the start date from ERA5, creates your ensemble members, and perturbs temperature fields with Gaussian noise (default: $\pm 0.1\,\text{K}$).
2. **Prepares Forcing Data:** Checks if the modified forcing file containing ERA5 upper-level winds exists for that year; if not, it automatically generates it.
3. **Prevents GPU Out-of-Memory (OOM) Errors:** Running 10 ensemble members simultaneously in memory can exceed the available GPU VRAM. The script automatically chunks your ensemble into safe batches (e.g., 5 members per batch).
4. **Runs Inference:** Executes the ACE model for each batch.
5. **Merges Results:** Combines the batch outputs into a single, clean NetCDF file saved directly into:
   ```
   /home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc
   ```

---

## 5. Configuring and Running Your Experiment

You do not need to type long command-line arguments! All experiment settings are set directly inside **`run_ensemble_experiment.py`**.

### Step 5.1: Open `run_ensemble_experiment.py`
Open `run_ensemble_experiment.py` in your favorite editor (e.g., VS Code or nano):
```bash
nano run_ensemble_experiment.py
```

Right near the top of the file, you will find the **`EXPERIMENT CONFIGURATION`** section:

```python
# 1. Forecast Dates and Ensemble Size
START_DATE = "2018-01-25T00:00:00"  # Start date/time (e.g. "2018-01-25" or "2018-01-25T00:00:00")
N_MEMBERS = 10                       # Number of ensemble members (e.g. 10)
BATCH_SIZE = 5                      # Members per batch (keep <= 5 to avoid GPU memory overflow)
N_FORWARD_STEPS = 100               # Number of 6-hour forecast steps (100 steps = 25 days)

# 2. Nudging Mode and Target State
# Choose nudging mode: "blended", "prescribed", "control", or "free"
NUDGING_TYPE = "blended"

# Choose nudging target state: "reanalysis" (forecast year) or "climatology" (40-year mean)
# Note: Selecting NUDGING_TYPE = "control" automatically sets NUDGING_TARGET = "climatology".
NUDGING_TARGET = "reanalysis"

# 3. Blended Nudging Timescale or Weights (used when NUDGING_TYPE is "blended" or "control")
TAU_HOURS = 24.0                    # Timescale in hours (e.g., 24.0 for weak nudging, 8.66 for moderate)
TAU_DAYS = None                     # Timescale in days (e.g., 1.0)
MODEL_WEIGHT = None                 # Optional direct model weight
REANALYSIS_WEIGHT = None            # Optional direct reanalysis weight

# 4. Experiment Naming and Perturbations
EXPERIMENT_NAME = None              # None = auto-generated descriptive name
TEMP_STD_DEV = 0.1                  # Noise added to temperatures in Kelvin (Member 0 is unperturbed)
RANDOM_SEED = 42                    # Random seed for reproducible perturbations

# Optional: reuse exact initial conditions from another experiment for matched paired comparisons:
EXISTING_IC_DIR = None              # None = generate fresh ICs; or pathlib.Path("...")
```

---

### Step 5.2: Common Experiment Configurations

#### Case A: Weak Blended Nudging ($\tau = 24\,\text{hours}$, default)
Nudges towards the actual forecast year's ERA5 reanalysis state:
```python
NUDGING_TYPE = "blended"
TAU_HOURS = 24.0
```

#### Case B: Moderate Blended Nudging ($\tau \approx 8.7\,\text{hours}$ or 50/50 weights)
```python
NUDGING_TYPE = "blended"
TAU_HOURS = 8.66
# Or directly:
# MODEL_WEIGHT = 0.5
# REANALYSIS_WEIGHT = 0.5
```

#### Case C: Prescribed Nudging ($\tau = 0$, complete replacement)
```python
NUDGING_TYPE = "prescribed"
```

#### Case D: Free Forecast ($\tau = \infty$, no nudging)
```python
NUDGING_TYPE = "free"
```

#### Case E: SNAPSI Control Forecast (`control-full`, nudged to climatology)
Nudges the uppermost level towards the 40-year daily smoothed ERA5 climatology rather than the forecast year's observations:
```python
NUDGING_TYPE = "control"
TAU_HOURS = 24.0
```
*(The script will automatically detect that climatological forcing is needed, generate the 40-year climatology if not already created, and build the year-specific forcing file!)*

#### Case F: Paired Comparison (e.g. running Control with the exact same ICs as Nudged)
To ensure Member 0..9 in your Control experiment start with the exact same initial state and perturbations as your Nudged experiment, set `EXISTING_IC_DIR` to point to the `initial_conditions` folder of the first run:
```python
NUDGING_TYPE = "control"
TAU_HOURS = 24.0
EXISTING_IC_DIR = pathlib.Path("/home/links/ws359/ACE/ACE_output/exp_20181213_ens10_blended_tau24h/initial_conditions")
```

---

### Step 5.3: Run the Script
Once you have adjusted the parameters in the file, save it and run:
```bash
python run_ensemble_experiment.py
```
That's it! The script will:
1. Print the experiment summary.
2. Generate the perturbed initial conditions.
3. Prepare the forcing data.
4. Run each batch through ACE2.
5. Merge the results into `/home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc`.

---

---

## 6. Climatological Forcing and Leap Day Handling (SNAPSI Protocol)

When running a SNAPSI control experiment (`NUDGING_TYPE = "control"`), the model nudges towards a smoothed daily climatology rather than the specific weather observations of that year.

This methodology follows **Section 3.1 of Hitchcock et al. (2022)**:
1. **Base Period:** Computed across a 40-year baseline from **1980 to 2020** using 40 annual July-to-June cycles (each cycle is 1,460 6-hour timesteps = 365 days).
2. **Leap Day Treatment:** In leap years, the 365 consecutive days following 1 July are used, omitting 30 June (so 29 February is indexed as 1 March). This ensures that every baseline cycle contains exactly 365 days and places any subtle endpoint discontinuity across 30 June / 1 July—completely outside the winter/spring forecast periods of interest.
3. **Triangular Smoothing:** The raw 365-day multi-year mean is smoothed using a **121-point (30-day) triangular filter** with circular (periodic) boundary conditions.
4. **Calendar Alignment & Leap Forecast Years:** The smoothed July-to-June annual cycle is reordered to a standard calendar year (1 January to 31 December, 1,460 steps). For leap forecast years (such as 2020 with 1,464 steps), 29 February is assigned the 1 March climatological value.

### Dedicated Climatology Script
If needed, you can generate or inspect climatological forcing directly using `create_climatological_forcing.py`:
```bash
# Generate climatological forcing for a specific forecast year:
python create_climatological_forcing.py --year 2018
```
*(Note: When using `run_ensemble_experiment.py`, this is called automatically if the file does not already exist!)*

---

## 7. Understanding the Output Files

All model output files are written to:
```
/home/links/ws359/ACE/ACE_output/<experiment_name>/
```

Inside that folder, you will find:
- **`autoregressive_predictions.nc`**: The complete, merged NetCDF dataset containing all ensemble members across all forecast timesteps and vertical levels.
- **`experiment_summary.json`**: A record of all parameters used (timescale, weights, start date, number of members).
- **`initial_conditions/`**: The chunked initial condition NetCDF files used for the run.
- **`configs/`**: The YAML configuration files generated for each batch.

---

## 8. Analyzing and Plotting Your Results

We have included four helper scripts in the `analysis/` folder:

### 1. Quick Statistical Summary
To print a quick table of the wind speeds and ensemble spreads across all vertical levels:
```bash
python analysis/quick_summary.py /home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc
```

### 2. Plotting Zonal Mean Winds Across Levels
To plot the 60°N zonal mean eastward wind for all 8 levels (comparing your model ensemble against ERA5 reanalysis truth):
```bash
python analysis/plot_zonal_mean.py /home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc
```
This will save an image named `zonal_mean_u_all_levels.png` in your experiment directory.

### 3. Comparing Level 0 Winds with Climatology and Observation
For SNAPSI Case Studies (e.g., 2018 or 2019 SSW), to plot ensemble members, ensemble mean, observed ERA5, and 40-year climatology:
```bash
python analysis/plot_level0_case2_comparison.py \
  --exp-nc /home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc \
  --save-path comparison_level0.png
```

### 4. Comparing Ensemble Spread Between Experiments
To compare the spread $\sigma(U)$ across multiple experiments (Free vs Prescribed vs Blended vs Control):
```bash
python analysis/plot_spread_comparison.py --save-path spread_comparison.png
```

---

## 9. Troubleshooting Common Issues

### Issue 1: "CUDA out of memory" (OOM)
- **Cause:** Too many ensemble members are running simultaneously in the GPU VRAM.
- **Fix:** Lower the `BATCH_SIZE` setting inside `run_ensemble_experiment.py` (e.g., from `BATCH_SIZE = 5` to `BATCH_SIZE = 3` or `2`).

### Issue 2: "version CXXABI_1.3.15 not found"
- **Cause:** The system is picking up an older system library instead of the conda environment's library.
- **Fix:** Make sure you ran:
  ```bash
  export LD_LIBRARY_PATH=/home/links/ws359/miniconda3/envs/ace_nudge/lib:$LD_LIBRARY_PATH
  ```

### Issue 3: "ERA5 file not found for year..."
- **Cause:** The script looks for ERA5 files in `/disco/share/ws359/ERA5_for_ACE/era5_1deg_{year}.nc`.
- **Fix:** Ensure the requested year is between 1980 and 2022.

---

## Summary Checklist
1. `conda activate ace_nudge`
2. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
3. `export LD_LIBRARY_PATH=/home/links/ws359/miniconda3/envs/ace_nudge/lib:$LD_LIBRARY_PATH`
4. Set experiment configuration at top of `run_ensemble_experiment.py` and run:
   ```bash
   python run_ensemble_experiment.py
   ```
5. Check output in `/home/links/ws359/ACE/ACE_output/<experiment_name>/`
6. Plot and analyze using the scripts in `analysis/`
