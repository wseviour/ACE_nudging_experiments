# ACE2 Stratospheric Nudging Experiments

This repository contains configuration files, automated execution workflows, analysis scripts, and comprehensive documentation for conducting stratospheric nudging experiments with the **ACE2** AI weather and climate model (SFNO foundation model).

---

## Repository Structure

```
ACE_nudging_experiments/
├── README.md                          # Repository overview (this file)
├── ace_nudging_guide.md               # Step-by-step user guide for running experiments
├── nudging_code_documentation.md      # Technical documentation of changes to ACE
├── run_ensemble_experiment.py         # Fully automated ensemble workflow runner
├── configs/                           # Reference YAML configurations for ACE2 inference
│   ├── inference_template_blended_tau24h.yaml
│   ├── inference_template_blended_tau8h.yaml
│   ├── inference_template_prescribed.yaml
│   └── inference_template_free.yaml
└── analysis/                          # Python analysis and visualization scripts
    ├── plot_spread_comparison.py      # Plots ensemble spread across multiple experiments
    ├── plot_zonal_mean.py             # Plots zonal mean eastward wind at 60°N across levels
    └── quick_summary.py               # Prints statistical summary table for any NetCDF run
```

---

## Quick Start

### 1. Environment Setup
```bash
conda activate ace_nudge
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LD_LIBRARY_PATH=/home/links/ws359/miniconda3/envs/ace_nudge/lib:$LD_LIBRARY_PATH
```

### 2. Configure and Run an Ensemble Forecast
Open `run_ensemble_experiment.py` and adjust the parameters in the `EXPERIMENT CONFIGURATION` section at the top of the file (e.g. `START_DATE`, `N_MEMBERS`, `NUDGING_TYPE`, `TAU_HOURS`).

Then simply run:
```bash
python run_ensemble_experiment.py
```

The script will automatically generate the initial conditions, prepare the modified forcing data, batch the ensemble members to prevent GPU out-of-memory errors, execute the forecast, and merge the predictions into:
```
/home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc
```

### 3. Analyze the Results
```bash
python analysis/plot_zonal_mean.py /home/links/ws359/ACE/ACE_output/<experiment_name>/autoregressive_predictions.nc
```

For full details, see [**`ace_nudging_guide.md`**](ace_nudging_guide.md) and [**`nudging_code_documentation.md`**](nudging_code_documentation.md).
