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

## 5. Command-Line Examples

Navigate to the `ACE_nudging_experiments` directory:
```bash
cd /home/links/ws359/ACE/ACE_nudging_experiments
```

### Example A: Run a Weak Nudging Experiment ($\tau = 24\,\text{hours}$)
To run a 10-member ensemble for 25 days (100 six-hour steps) starting on January 25, 2018:
```bash
python run_ensemble_experiment.py \
  --start-date 2018-01-25T00:00:00 \
  --n-members 10 \
  --batch-size 5 \
  --n-forward-steps 100 \
  --nudging-type blended \
  --tau-hours 24.0 \
  --experiment-name exp_20180125_blended_tau24h
```

### Example B: Run a Prescribed Nudging Experiment ($\tau = 0$)
In this experiment, the uppermost level is completely overridden with ERA5 observations:
```bash
python run_ensemble_experiment.py \
  --start-date 2018-01-25T00:00:00 \
  --n-members 10 \
  --batch-size 5 \
  --n-forward-steps 100 \
  --nudging-type prescribed \
  --experiment-name exp_20180125_prescribed
```

### Example C: Run a Free Forecast Experiment (No Nudging)
In this experiment, no nudging is applied:
```bash
python run_ensemble_experiment.py \
  --start-date 2018-01-25T00:00:00 \
  --n-members 10 \
  --batch-size 5 \
  --n-forward-steps 100 \
  --nudging-type free \
  --experiment-name exp_20180125_free
```

### Example D: Quick Test Run (2 members, 1 day)
If you just want to test that everything is working without waiting:
```bash
python run_ensemble_experiment.py \
  --start-date 2018-01-25T00:00:00 \
  --n-members 2 \
  --batch-size 2 \
  --n-forward-steps 4 \
  --nudging-type blended \
  --tau-hours 24.0 \
  --experiment-name test_quick_check
```

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
