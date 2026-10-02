#!/usr/bin/env python3
"""
run_ensemble_experiment.py

Automated workflow script for running ACE2 ensemble forecast experiments with nudging.

Features:
  1. Automated initial condition generation from ERA5 with Gaussian temperature perturbations.
  2. Automated forcing dataset preparation (injecting reanalysis level-0 wind into forcing NetCDF).
  3. Automatic batch chunking of ensemble members to prevent GPU out-of-memory (OOM) errors.
  4. Automatic generation of ACE2 inference YAML configuration files for each batch.
  5. Execution of ACE2 inference across batches with appropriate CUDA allocator configurations.
  6. Automatic concatenation of output predictions across chunks into a single unified NetCDF.
  7. Output directed to /home/links/ws359/ACE/ACE_output/<experiment_name>.
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
            f"Base forcing file not found: {base_forcing_file}. Please check --base-forcing-dir."
        )
    if not era5_file.exists():
        raise FileNotFoundError(
            f"ERA5 file not found: {era5_file}. Please check --era5-data-dir."
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
    parser = argparse.ArgumentParser(
        description="Automated runner for ACE2 ensemble forecast experiments with upper-level nudging."
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default="2018-01-25T00:00:00",
        help="Start date/time for forecast (e.g. '2018-01-25' or '2018-01-25T00:00:00').",
    )
    parser.add_argument(
        "--n-members",
        type=int,
        default=10,
        help="Total number of ensemble members to run (default: 10).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5,
        help="Number of members per inference batch to prevent GPU OOM (default: 5).",
    )
    parser.add_argument(
        "--n-forward-steps",
        type=int,
        default=100,
        help="Number of 6-hour forecast steps to run (100 steps = 25 days, default: 100).",
    )
    parser.add_argument(
        "--nudging-type",
        type=str,
        choices=["blended", "prescribed", "free"],
        default="blended",
        help="Type of nudging: 'blended' (relaxation with weights/timescale), 'prescribed' (tau=0), or 'free' (tau=inf).",
    )
    parser.add_argument(
        "--tau-hours",
        type=float,
        default=None,
        help="Effective nudging relaxation timescale tau in hours (e.g. 24.0 or 8.66).",
    )
    parser.add_argument(
        "--tau-days",
        type=float,
        default=None,
        help="Effective nudging relaxation timescale tau in days (e.g. 1.0).",
    )
    parser.add_argument(
        "--model-weight",
        type=float,
        default=None,
        help="Explicit weight for model prediction (0 <= w <= 1).",
    )
    parser.add_argument(
        "--reanalysis-weight",
        type=float,
        default=None,
        help="Explicit weight for reanalysis observation (0 <= w <= 1).",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Custom name for experiment directory. Default: auto-generated based on date and nudging configuration.",
    )
    parser.add_argument(
        "--temp-std-dev",
        type=float,
        default=0.1,
        help="Standard deviation of Gaussian noise added to temperature fields for ensemble members (in K, default: 0.1).",
    )
    parser.add_argument(
        "--output-root",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_output"),
        help="Root directory where experiment outputs will be stored (default: /home/links/ws359/ACE/ACE_output).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/ace2_era5_ckpt.tar"),
        help="Path to ACE2 checkpoint file.",
    )
    parser.add_argument(
        "--era5-data-dir",
        type=pathlib.Path,
        default=pathlib.Path("/disco/share/ws359/ERA5_for_ACE"),
        help="Directory containing era5_1deg_{year}.nc files.",
    )
    parser.add_argument(
        "--base-forcing-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/forcing_data"),
        help="Directory containing base forcing_{year}.nc files.",
    )
    parser.add_argument(
        "--mod-forcing-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/mod_forcing_data"),
        help="Directory for modified forcing files containing reanalysis winds.",
    )
    parser.add_argument(
        "--python-bin",
        type=str,
        default="/home/links/ws359/miniconda3/envs/ace_nudge/bin/python",
        help="Path to python executable with ace_nudge environment.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Prepare ICs, forcing data, and configs without executing model inference.",
    )

    args = parser.parse_args()

    # Determine experiment name if not provided
    date_str = str(np.datetime64(args.start_date))[:10].replace("-", "")
    if args.experiment_name is None:
        if args.nudging_type == "free":
            args.experiment_name = f"exp_{date_str}_ens{args.n_members}_free"
        elif args.nudging_type == "prescribed":
            args.experiment_name = f"exp_{date_str}_ens{args.n_members}_prescribed"
        else:
            if args.tau_days is not None:
                tag = f"tau{args.tau_days:.1f}d"
            elif args.tau_hours is not None:
                tag = f"tau{args.tau_hours:.0f}h"
            elif args.model_weight is not None:
                tag = f"mw{args.model_weight:.2f}_rw{args.reanalysis_weight:.2f}"
            else:
                tag = "tau24h"
            args.experiment_name = f"exp_{date_str}_ens{args.n_members}_blended_{tag}"

    exp_dir = args.output_root / args.experiment_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    logging.info("=== ACE2 Ensemble Experiment: %s ===", args.experiment_name)
    logging.info("Destination directory: %s", exp_dir)

    # 1. Prepare Forcing Data (if nudging is requested)
    year = int(str(np.datetime64(args.start_date))[:4])
    if args.nudging_type in ("blended", "prescribed"):
        forcing_dir = ensure_forcing_data(
            year=year,
            base_forcing_dir=args.base_forcing_dir,
            era5_data_dir=args.era5_data_dir,
            mod_forcing_dir=args.mod_forcing_dir,
            nudging_var="eastward_wind_0",
        )
    else:
        forcing_dir = args.base_forcing_dir

    # 2. Generate Initial Conditions (chunked into batches)
    ic_dir = exp_dir / "initial_conditions"
    ic_chunks = create_perturbed_ics(
        start_time=args.start_date,
        n_members=args.n_members,
        batch_size=args.batch_size,
        era5_data_dir=args.era5_data_dir,
        temp_std_dev=args.temp_std_dev,
        output_ic_dir=ic_dir,
    )

    # 3. Build Stepper Override Configuration
    stepper_override = build_stepper_override(
        nudging_type=args.nudging_type,
        tau_hours=args.tau_hours,
        tau_days=args.tau_days,
        model_weight=args.model_weight,
        reanalysis_weight=args.reanalysis_weight,
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
            "n_forward_steps": args.n_forward_steps,
            "forward_steps_in_memory": min(10, args.n_forward_steps),
            "checkpoint_path": str(args.checkpoint_path),
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

        if not args.dry_run:
            logging.info("--- Starting inference for chunk %d of %d ---", idx + 1, len(ic_chunks))
            run_inference_chunk(chunk_config_path, python_bin=args.python_bin)

    if args.dry_run:
        logging.info("Dry-run complete. Configs and initial conditions generated without executing model.")
        return

    # 5. Merge Output Prediction Chunks
    merged_output_file = exp_dir / "autoregressive_predictions.nc"
    merge_prediction_chunks(chunk_output_dirs, merged_output_file)

    # Save summary metadata
    summary = {
        "experiment_name": args.experiment_name,
        "start_date": args.start_date,
        "n_members": args.n_members,
        "batch_size": args.batch_size,
        "n_forward_steps": args.n_forward_steps,
        "nudging_type": args.nudging_type,
        "tau_hours": args.tau_hours,
        "tau_days": args.tau_days,
        "model_weight": args.model_weight,
        "reanalysis_weight": args.reanalysis_weight,
        "temp_std_dev": args.temp_std_dev,
        "output_predictions": str(merged_output_file),
    }
    with open(exp_dir / "experiment_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logging.info("=== Experiment %s completed successfully! ===", args.experiment_name)
    logging.info("Final merged output: %s", merged_output_file)


if __name__ == "__main__":
    main()
