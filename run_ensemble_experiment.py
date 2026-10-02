#!/usr/bin/env python3
"""
run_ensemble_experiment.py

Automated workflow script for running ACE2 ensemble forecast experiments with nudging.

Instructions for Myles:
  1. Open this file and edit the parameters in the 'EXPERIMENT CONFIGURATION' section below.
  2. Run the script from the terminal with:
         python run_ensemble_experiment.py
  3. The script will automatically:
       - Generate initial conditions with temperature perturbations for all members.
       - Prepare the modified forcing file containing ERA5 level-0 wind.
       - Divide the ensemble into safe batches to avoid GPU Out-Of-Memory errors.
       - Run ACE2 inference for each batch.
       - Merge all output chunks into a single NetCDF file in /home/links/ws359/ACE/ACE_output/<experiment_name>.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import pathlib
import subprocess
import sys
import time
from typing import Literal

import numpy as np
import xarray as xr
import yaml


# ==============================================================================
# EXPERIMENT CONFIGURATION
# Edit the parameters below to configure your ensemble simulation run!
# ==============================================================================

# 1. Forecast Dates and Ensemble Size
START_DATE = "2018-12-13T00:00:00"  # Start date/time (SNAPSI Case Study 2 early start: 2018-12-13)
N_MEMBERS = 10                       # Number of ensemble members (e.g. 10)
BATCH_SIZE = 5                      # Members per batch (keep <= 5 to avoid GPU memory overflow)
N_FORWARD_STEPS = 100               # Number of 6-hour forecast steps (100 steps = 25 days)

# 2. Nudging Mode
# Choose one of:
#   - "blended": Newtonian relaxation towards ERA5 using a timescale or weights
#   - "prescribed": Uppermost level completely overwritten with ERA5 (tau = 0)
#   - "free": No nudging applied (unconstrained free forecast)
NUDGING_TYPE: Literal["blended", "prescribed", "free"] = "blended"

# 3. Blended Nudging Timescale or Weights (only used when NUDGING_TYPE = "blended")
# Option A: Set the relaxation timescale (tau) in hours or days
TAU_HOURS = 24.0                    # e.g., 24.0 for weak nudging, 8.66 for moderate nudging
TAU_DAYS = None                     # e.g., 1.0 (if set, overrides TAU_HOURS)

# Option B: Set explicit weights directly (optional, overrides TAU if set)
# u(t+6h) = MODEL_WEIGHT * u_pred + REANALYSIS_WEIGHT * u_era5
MODEL_WEIGHT = None                 # e.g., 0.5 (weight given to model prediction)
REANALYSIS_WEIGHT = None            # e.g., 0.5 (weight given to ERA5 reanalysis)

# 4. Experiment Naming and Perturbations
# Custom experiment name (set to None to auto-generate a descriptive name based on settings):
EXPERIMENT_NAME = None

# Gaussian noise standard deviation added to temperature fields for perturbed members 1..N-1:
TEMP_STD_DEV = 0.1                  # in Kelvin (0.1 K is standard; Member 0 is unperturbed)

# 5. Paths and Environment
OUTPUT_ROOT = pathlib.Path("/home/links/ws359/ACE/ACE_output")
CHECKPOINT_PATH = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/ace2_era5_ckpt.tar")
ERA5_DATA_DIR = pathlib.Path("/disco/share/ws359/ERA5_for_ACE")
BASE_FORCING_DIR = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/forcing_data")
MOD_FORCING_DIR = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/mod_forcing_data")
PYTHON_BIN = "/home/links/ws359/miniconda3/envs/ace_nudge/bin/python"
DRY_RUN = False                     # Set to True to only generate files without executing the model
# ==============================================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Standard prognostic variables required by the ACE2-ERA5 model checkpoint
PROGNOSTIC_VARS = [
    "PRESsfc",
    "surface_temperature",
    "TMP2m",
    "UGRD10m",
    "VGRD10m",
    "Q2m",
    "air_temperature_0",
    "air_temperature_1",
    "air_temperature_2",
    "air_temperature_3",
    "air_temperature_4",
    "air_temperature_5",
    "air_temperature_6",
    "air_temperature_7",
    "eastward_wind_0",
    "eastward_wind_1",
    "eastward_wind_2",
    "eastward_wind_3",
    "eastward_wind_4",
    "eastward_wind_5",
    "eastward_wind_6",
    "eastward_wind_7",
    "northward_wind_0",
    "northward_wind_1",
    "northward_wind_2",
    "northward_wind_3",
    "northward_wind_4",
    "northward_wind_5",
    "northward_wind_6",
    "northward_wind_7",
    "specific_total_water_0",
    "specific_total_water_1",
    "specific_total_water_2",
    "specific_total_water_3",
    "specific_total_water_4",
    "specific_total_water_5",
    "specific_total_water_6",
    "specific_total_water_7",
]

DEFAULT_OUTPUT_NAMES = [
    "TMP2m",
    "VGRD10m",
    "PRATEsfc",
    "air_temperature_0",
    "air_temperature_1",
    "air_temperature_2",
    "air_temperature_3",
    "air_temperature_4",
    "air_temperature_5",
    "air_temperature_6",
    "air_temperature_7",
    "eastward_wind_0",
    "eastward_wind_1",
    "eastward_wind_2",
    "eastward_wind_3",
    "eastward_wind_4",
    "eastward_wind_5",
    "eastward_wind_6",
    "eastward_wind_7",
]


def ensure_forcing_data(
    year: int,
    base_forcing_dir: pathlib.Path,
    era5_data_dir: pathlib.Path,
    mod_forcing_dir: pathlib.Path,
    nudging_var: str = "eastward_wind_0",
) -> pathlib.Path:
    """
    Ensures that forcing data with reanalysis wind exists for the given year.
    If mod_forcing_{year}.nc does not exist in mod_forcing_dir, creates it by injecting
    nudging_var from ERA5 into the base forcing file.
    """
    mod_forcing_dir.mkdir(parents=True, exist_ok=True)
    target_mod_file = mod_forcing_dir / f"mod_forcing_{year}.nc"

    if target_mod_file.exists():
        logging.info("Modified forcing file already exists: %s", target_mod_file)
        return mod_forcing_dir

    base_forcing_file = base_forcing_dir / f"forcing_{year}.nc"
    era5_file = era5_data_dir / f"era5_1deg_{year}.nc"

    if not base_forcing_file.exists():
        raise FileNotFoundError(
            f"Base forcing file not found: {base_forcing_file}. Please check base_forcing_dir."
        )
    if not era5_file.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {era5_file}. Please check era5_data_dir."
        )

    logging.info(
        "Creating modified forcing file %s by combining %s and %s...",
        target_mod_file,
        base_forcing_file,
        era5_file,
    )
    with xr.open_dataset(base_forcing_file) as ds_forcing, xr.open_dataset(era5_file) as ds_era5:
        ds_out = ds_forcing.copy(deep=True)
        ds_out[nudging_var] = ds_era5[nudging_var]
        ds_out.to_netcdf(target_mod_file)

    logging.info("Successfully created %s", target_mod_file)
    return mod_forcing_dir


def create_perturbed_ics(
    start_time: str,
    n_members: int,
    batch_size: int,
    era5_data_dir: pathlib.Path,
    temp_std_dev: float,
    output_ic_dir: pathlib.Path,
) -> list[pathlib.Path]:
    """
    Creates an ensemble of initial conditions, perturbing temperature fields with Gaussian noise.
    Saves the members chunked into files of size <= batch_size.
    Returns list of paths to the chunked NetCDF files.
    """
    output_ic_dir.mkdir(parents=True, exist_ok=True)

    # Determine year from start_time string
    dt = np.datetime64(start_time)
    year = int(str(dt)[:4])

    era5_file = era5_data_dir / f"era5_1deg_{year}.nc"
    if not era5_file.exists():
        raise FileNotFoundError(f"ERA5 file not found for year {year}: {era5_file}")

    logging.info("Extracting initial condition slice for %s from %s...", start_time, era5_file)
    with xr.open_dataset(era5_file) as ds_era:
        # Select target time slice
        ds_start = ds_era.sel(time=start_time, method="nearest")
        # Keep only required prognostic variables
        available_vars = [v for v in PROGNOSTIC_VARS if v in ds_start.data_vars]
        ds_start = ds_start[available_vars]

        temp_vars = [
            v for v in ds_start.data_vars
            if "TMP" in v.upper() or "TEMPERATURE" in v.upper()
        ]
        logging.info("Found %d temperature variables to perturb: %s", len(temp_vars), temp_vars)

        members = []
        logging.info("Generating %d ensemble members (temp noise std: %.3f K)...", n_members, temp_std_dev)
        for i in range(n_members):
            ds_member = ds_start.copy(deep=True)
            if i > 0 and temp_std_dev > 0:
                for var_name in temp_vars:
                    noise = np.random.normal(
                        loc=0.0,
                        scale=temp_std_dev,
                        size=ds_member[var_name].shape,
                    ).astype(ds_member[var_name].dtype)
                    ds_member[var_name] += noise
            members.append(ds_member)

    # Chunk members into batches
    chunk_paths = []
    n_chunks = int(np.ceil(n_members / batch_size))
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * batch_size
        end_idx = min(start_idx + batch_size, n_members)
        chunk_members = members[start_idx:end_idx]

        # Concatenate along the 'time' dimension (repeating the timestamp for each member)
        ds_chunk = xr.concat(chunk_members, dim="time")
        chunk_file = output_ic_dir / f"ic_chunk_{chunk_idx}.nc"
        logging.info(
            "Saving IC chunk %d (members %d..%d) to %s...",
            chunk_idx,
            start_idx,
            end_idx - 1,
            chunk_file,
        )
        ds_chunk.to_netcdf(chunk_file)
        chunk_paths.append(chunk_file)

    return chunk_paths


def build_stepper_override(
    nudging_type: Literal["blended", "prescribed", "free"],
    tau_hours: float | None = None,
    tau_days: float | None = None,
    model_weight: float | None = None,
    reanalysis_weight: float | None = None,
    nudging_var: str = "eastward_wind_0",
) -> dict:
    """
    Builds the stepper_override configuration dictionary.
    """
    override = {"ocean": "keep"}

    if nudging_type == "free":
        return override

    if nudging_type == "prescribed":
        override["prescribed_prognostic_names"] = [nudging_var]
        return override

    # Blended nudging
    nudge_entry: dict = {}
    if tau_days is not None:
        nudge_entry["timescale_days"] = float(tau_days)
    elif tau_hours is not None:
        nudge_entry["timescale_hours"] = float(tau_hours)
    elif model_weight is not None and reanalysis_weight is not None:
        nudge_entry["model_weight"] = float(model_weight)
        nudge_entry["reanalysis_weight"] = float(reanalysis_weight)
    else:
        # Default to 24h if blended selected without explicit timescale
        nudge_entry["timescale_hours"] = 24.0

    override["nudged_prognostics"] = {nudging_var: nudge_entry}
    return override


def run_inference_chunk(
    config_path: pathlib.Path,
    python_bin: str,
) -> None:
    """
    Runs ACE2 inference via python -m fme.ace.inference <config_path>.
    """
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    env_lib = env.get("LD_LIBRARY_PATH", "")
    conda_lib = str(pathlib.Path(python_bin).parent.parent / "lib")
    env["LD_LIBRARY_PATH"] = f"{conda_lib}:{env_lib}" if conda_lib not in env_lib else env_lib

    cmd = [python_bin, "-m", "fme.ace.inference", str(config_path)]
    logging.info("Executing: %s", " ".join(cmd))
    start_time = time.time()
    result = subprocess.run(cmd, env=env, check=False)
    elapsed = time.time() - start_time

    if result.returncode != 0:
        raise RuntimeError(
            f"Inference failed with exit code {result.returncode} for config {config_path}."
        )
    logging.info("Chunk inference completed in %.1f seconds.", elapsed)


def merge_prediction_chunks(
    chunk_dirs: list[pathlib.Path],
    merged_output_file: pathlib.Path,
) -> None:
    """
    Merges autoregressive_predictions.nc across chunks along the 'sample' dimension.
    """
    logging.info("Merging %d prediction chunks into %s...", len(chunk_dirs), merged_output_file)
    datasets = []
    for c_dir in chunk_dirs:
        pred_file = c_dir / "autoregressive_predictions.nc"
        if not pred_file.exists():
            raise FileNotFoundError(f"Expected predictions file not found: {pred_file}")
        ds = xr.open_dataset(pred_file)
        datasets.append(ds)

    merged = xr.concat(datasets, dim="sample")
    merged = merged.assign_coords(sample=np.arange(len(merged["sample"])))
    merged_output_file.parent.mkdir(parents=True, exist_ok=True)
    merged.to_netcdf(merged_output_file)
    logging.info("Merged NetCDF successfully created: %s (shape: %s)", merged_output_file, dict(merged.sizes))


def main() -> None:
    # Read parameters directly from the configuration block at the top of the file
    start_date = START_DATE
    n_members = N_MEMBERS
    batch_size = BATCH_SIZE
    n_forward_steps = N_FORWARD_STEPS
    nudging_type = NUDGING_TYPE
    tau_hours = TAU_HOURS
    tau_days = TAU_DAYS
    model_weight = MODEL_WEIGHT
    reanalysis_weight = REANALYSIS_WEIGHT
    experiment_name = EXPERIMENT_NAME
    temp_std_dev = TEMP_STD_DEV
    output_root = OUTPUT_ROOT
    checkpoint_path = CHECKPOINT_PATH
    era5_data_dir = ERA5_DATA_DIR
    base_forcing_dir = BASE_FORCING_DIR
    mod_forcing_dir = MOD_FORCING_DIR
    python_bin = PYTHON_BIN
    dry_run = DRY_RUN

    # Optional: allow command-line arguments to override in-script settings if provided
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            description="Automated runner for ACE2 ensemble forecast experiments with upper-level nudging."
        )
        parser.add_argument("--start-date", type=str, default=start_date)
        parser.add_argument("--n-members", type=int, default=n_members)
        parser.add_argument("--batch-size", type=int, default=batch_size)
        parser.add_argument("--n-forward-steps", type=int, default=n_forward_steps)
        parser.add_argument("--nudging-type", type=str, choices=["blended", "prescribed", "free"], default=nudging_type)
        parser.add_argument("--tau-hours", type=float, default=tau_hours)
        parser.add_argument("--tau-days", type=float, default=tau_days)
        parser.add_argument("--model-weight", type=float, default=model_weight)
        parser.add_argument("--reanalysis-weight", type=float, default=reanalysis_weight)
        parser.add_argument("--experiment-name", type=str, default=experiment_name)
        parser.add_argument("--temp-std-dev", type=float, default=temp_std_dev)
        parser.add_argument("--dry-run", action="store_true", default=dry_run)
        cli_args = parser.parse_args()

        start_date = cli_args.start_date
        n_members = cli_args.n_members
        batch_size = cli_args.batch_size
        n_forward_steps = cli_args.n_forward_steps
        nudging_type = cli_args.nudging_type
        tau_hours = cli_args.tau_hours
        tau_days = cli_args.tau_days
        model_weight = cli_args.model_weight
        reanalysis_weight = cli_args.reanalysis_weight
        experiment_name = cli_args.experiment_name
        temp_std_dev = cli_args.temp_std_dev
        dry_run = cli_args.dry_run

    # Determine experiment name if not set
    date_str = str(np.datetime64(start_date))[:10].replace("-", "")
    if experiment_name is None:
        if nudging_type == "free":
            experiment_name = f"exp_{date_str}_ens{n_members}_free"
        elif nudging_type == "prescribed":
            experiment_name = f"exp_{date_str}_ens{n_members}_prescribed"
        else:
            if tau_days is not None:
                tag = f"tau{tau_days:.1f}d"
            elif tau_hours is not None:
                tag = f"tau{tau_hours:.0f}h"
            elif model_weight is not None:
                tag = f"mw{model_weight:.2f}_rw{reanalysis_weight:.2f}"
            else:
                tag = "tau24h"
            experiment_name = f"exp_{date_str}_ens{n_members}_blended_{tag}"

    exp_dir = output_root / experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.info("==========================================================")
    logging.info("Starting ACE2 Ensemble Experiment: %s", experiment_name)
    logging.info("Start Date:       %s", start_date)
    logging.info("Ensemble Size:    %d members (batch size: %d)", n_members, batch_size)
    logging.info("Forecast Length:  %d steps (%d hours / %.1f days)", n_forward_steps, n_forward_steps * 6, (n_forward_steps * 6) / 24)
    logging.info("Nudging Mode:     %s", nudging_type)
    if nudging_type == "blended":
        if tau_days:
            logging.info("Timescale (tau):  %.2f days", tau_days)
        elif tau_hours:
            logging.info("Timescale (tau):  %.2f hours", tau_hours)
        elif model_weight is not None:
            logging.info("Weights:          model=%.2f, reanalysis=%.2f", model_weight, reanalysis_weight)
    logging.info("Destination:      %s", exp_dir)
    logging.info("==========================================================")

    # 1. Prepare Forcing Data (if nudging is requested)
    start_dt = np.datetime64(start_date)
    end_dt = start_dt + np.timedelta64(n_forward_steps * 6, "h")
    start_year = int(str(start_dt)[:4])
    end_year = int(str(end_dt)[:4])
    if nudging_type in ("blended", "prescribed"):
        for yr in range(start_year, end_year + 1):
            forcing_dir = ensure_forcing_data(
                year=yr,
                base_forcing_dir=base_forcing_dir,
                era5_data_dir=era5_data_dir,
                mod_forcing_dir=mod_forcing_dir,
                nudging_var="eastward_wind_0",
            )
    else:
        forcing_dir = base_forcing_dir

    # 2. Generate Initial Conditions (chunked into batches)
    ic_dir = exp_dir / "initial_conditions"
    ic_chunks = create_perturbed_ics(
        start_time=start_date,
        n_members=n_members,
        batch_size=batch_size,
        era5_data_dir=era5_data_dir,
        temp_std_dev=temp_std_dev,
        output_ic_dir=ic_dir,
    )

    # 3. Build Stepper Override Configuration
    stepper_override = build_stepper_override(
        nudging_type=nudging_type,
        tau_hours=tau_hours,
        tau_days=tau_days,
        model_weight=model_weight,
        reanalysis_weight=reanalysis_weight,
        nudging_var="eastward_wind_0",
    )

    # 4. Generate YAML configs and execute inference for each chunk
    chunk_output_dirs = []
    config_dir = exp_dir / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    for idx, ic_chunk_path in enumerate(ic_chunks):
        chunk_out_dir = exp_dir / f"chunk_{idx}"
        chunk_output_dirs.append(chunk_out_dir)

        config_dict = {
            "experiment_dir": str(chunk_out_dir),
            "n_forward_steps": n_forward_steps,
            "forward_steps_in_memory": min(10, n_forward_steps),
            "checkpoint_path": str(checkpoint_path),
            "logging": {
                "log_to_screen": True,
                "log_to_wandb": False,
                "log_to_file": True,
                "project": "ace",
            },
            "initial_condition": {
                "path": str(ic_chunk_path),
            },
            "forcing_loader": {
                "dataset": {
                    "data_path": str(forcing_dir),
                },
                "num_data_workers": 4,
            },
            "stepper_override": stepper_override,
            "data_writer": {
                "save_prediction_files": True,
                "save_monthly_files": False,
                "names": DEFAULT_OUTPUT_NAMES,
            },
        }

        chunk_config_path = config_dir / f"config_chunk_{idx}.yaml"
        with open(chunk_config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
        logging.info("Created configuration for chunk %d: %s", idx, chunk_config_path)

        if not dry_run:
            logging.info("--- Starting inference for chunk %d of %d ---", idx + 1, len(ic_chunks))
            run_inference_chunk(chunk_config_path, python_bin=python_bin)

    if dry_run:
        logging.info("Dry-run complete. Configs and initial conditions generated without executing model.")
        return

    # 5. Merge Output Prediction Chunks
    merged_output_file = exp_dir / "autoregressive_predictions.nc"
    merge_prediction_chunks(chunk_output_dirs, merged_output_file)

    # Save summary metadata
    summary = {
        "experiment_name": experiment_name,
        "start_date": start_date,
        "n_members": n_members,
        "batch_size": batch_size,
        "n_forward_steps": n_forward_steps,
        "nudging_type": nudging_type,
        "tau_hours": tau_hours,
        "tau_days": tau_days,
        "model_weight": model_weight,
        "reanalysis_weight": reanalysis_weight,
        "temp_std_dev": temp_std_dev,
        "output_predictions": str(merged_output_file),
    }
    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("==========================================================")
    logging.info("Experiment %s completed successfully!", experiment_name)
    logging.info("Final merged output: %s", merged_output_file)
    logging.info("==========================================================")


if __name__ == "__main__":
    main()
