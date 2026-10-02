# Guide to Running ACE2 Stratospheric Nudging Experiments

Welcome! This guide is designed to walk you step-by-step through running stratospheric nudging experiments with the **ACE2** AI weather and climate model on `maths-gpu`.

---

## 1. Introduction: What is Nudging?

In weather forecasting, chaotic atmospheric dynamics cause perturbed ensemble members to spread apart over time. In the stratosphere (the upper levels of the atmosphere), large-scale circulation patterns—such as the polar vortex and sudden stratospheric warmings (SSWs)—have a profound influence on surface weather weeks later.

The **SNAPSI** (Stratospheric Nudging And Predictable Surface Impacts) experiment tests this by "nudging" the model's stratosphere towards historical observations (ERA5 reanalysis), while letting the rest of the atmosphere evolve freely.

In ACE2, we can run three types of forecasts:
1. **Free Forecast ($\tau = \infty$):** No nudging. The model runs completely on its own.
2. **Prescribed Nudging ($\tau = 0$):** Strongest nudging. At every 6-hour timestep, the uppermost zonal wind (`eastward_wind_0`, ~1 hPa) is completely replaced with ERA5 observations.
3. **Blended Relaxation Nudging ($\tau > 0$):** Gentle nudging. The wind at the next timestep is calculated as a blend of the model's own prediction and ERA5:
   $$u(t + 6\text{h}) = w_{\text{model}} \cdot u_{\text{pred}} + w_{\text{reanalysis}} \cdot u_{\text{reanalysis}}$$
   where the weights are calculated from a relaxation timescale $\tau$:
   $$w_{\text{model}} = e^{-6\text{h} / \tau}, \quad w_{\text{reanalysis}} = 1 - w_{\text{model}}$$
   - For a **24-hour timescale ($\tau = 24\,\text{h}$)**: $w_{\text{model}} \approx 0.78$, $w_{\text{reanalysis}} \approx 0.22$.
   - For an **8.7-hour timescale ($\tau \approx 8.7\,\text{h}$)**: $w_{\text{model}} = 0.50$, $w_{\text{reanalysis}} = 0.50$.

---

## 2. Setting Up Your Environment on `maths-gpu`

Before running any code, you need to configure your Python environment and libraries.

### Step 2.1: Activate the Conda Environment
Log into the server and activate the dedicated environment:
```bash
conda activate ace_nudge
```

### Step 2.2: Set Essential Environment Variables
To ensure PyTorch manages GPU memory efficiently and finds the correct C++ libraries for NetCDF and Matplotlib, export these environment variables in your terminal (or add them to your `~/.bashrc`):
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/links/ws359/miniconda3/envs/ace_nudge/lib:$LD_LIBRARY_PATH
```

---

## 3. Getting the Code

The modified ACE codebase with nudging support lives on GitHub:
- Repository: `git@github.com:wseviour/ace.git`
- Branch: `stepper_override_nudging`

If you are setting it up in your own workspace:
```bash
git clone git@github.com:wseviour/ace.git
cd ace
git checkout stepper_override_nudging
pip install -e .
```

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

# 2. Nudging Mode
# Choose one of: "blended", "prescribed", or "free"
NUDGING_TYPE = "blended"

# 3. Blended Nudging Timescale or Weights (used when NUDGING_TYPE = "blended")
TAU_HOURS = 24.0                    # Timescale in hours (e.g., 24.0 for weak nudging, 8.66 for moderate)
TAU_DAYS = None                     # Timescale in days (e.g., 1.0)
MODEL_WEIGHT = None                 # Optional direct model weight
REANALYSIS_WEIGHT = None            # Optional direct reanalysis weight

# 4. Experiment Naming and Perturbations
EXPERIMENT_NAME = None              # None = auto-generated descriptive name
TEMP_STD_DEV = 0.1                  # Noise added to temperatures in Kelvin (Member 0 is unperturbed)
```

---

### Step 5.2: Common Experiment Configurations

#### Case A: Weak Blended Nudging ($\tau = 24\,\text{hours}$, default)
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

## 6. Understanding the Output Files

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

## 7. Analyzing and Plotting Your Results

We have included three helper scripts in the `analysis/` folder:

### 1. Quick Statistical Summary
To print a quick table of the wind speeds and ensemble spreads across all vertical levels:
```bash
python analysis/quick_summary.py /home/links/ws359/ACE/ACE_output/exp_20180125_blended_tau24h/autoregressive_predictions.nc
```

### 2. Plotting Zonal Mean Winds Across Levels
To plot the 60°N zonal mean eastward wind for all 8 levels (comparing your model ensemble against ERA5 reanalysis truth):
```bash
python analysis/plot_zonal_mean.py /home/links/ws359/ACE/ACE_output/exp_20180125_blended_tau24h/autoregressive_predictions.nc
```
This will save an image named `zonal_mean_u_all_levels.png` in your experiment directory.

### 3. Comparing Ensemble Spread Between Experiments
To compare the spread $\sigma(U)$ across multiple experiments (Free vs Prescribed vs Blended):
```bash
python analysis/plot_spread_comparison.py --save-path spread_comparison.png
```

---

## 8. Troubleshooting Common Issues

### Issue 1: "CUDA out of memory" (OOM)
- **Cause:** Too many ensemble members are running simultaneously in the GPU VRAM.
- **Fix:** Lower the `--batch-size` parameter in `run_ensemble_experiment.py` (e.g., from `--batch-size 5` to `--batch-size 3` or `2`).

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
4. `python run_ensemble_experiment.py --nudging-type blended --tau-hours 24.0 ...`
5. Check output in `/home/links/ws359/ACE/ACE_output/`
6. Plot with `python analysis/plot_zonal_mean.py ...`
