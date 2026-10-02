#!/usr/bin/env python3
"""
create_climatological_forcing.py

Computes the 40-year daily/sub-daily climatology of uppermost level zonal wind
(eastward_wind_0) from ERA5 reanalysis and generates climatological forcing
files for ACE2 SNAPSI 'control-full' experiments following the protocol in
Section 3.1 of Hitchcock et al. (2022).

Methodology (SNAPSI Section 3.1):
  1. Base Period: 40-year period (1980-07-01 to 2020-06-30).
  2. Leap Days: Handled by using the 365 consecutive days following 1 July,
     omitting 30 June in leap years (so 29 February is treated as 1 March).
     This introduces any endpoint discontinuity across 30 June / 1 July,
     outside forecast periods of interest.
  3. Filtering: The 365-day climatological annual cycle is smoothed with a
     121-point (30-day) triangular filter with circular boundary conditions.
  4. Calendar Mapping: Reordered to standard calendar year (1 Jan - 31 Dec).
     For leap forecast years, 29 Feb receives the 1 March climatology.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import os
import pathlib
import time

import numpy as np
import pandas as pd
from scipy.ndimage import convolve1d
import xarray as xr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

DEFAULT_ERA5_DIR = pathlib.Path("/disco/share/ws359/ERA5_for_ACE")
DEFAULT_BASE_FORCING_DIR = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/forcing_data")
DEFAULT_CLIM_DIR = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/climatology")
DEFAULT_CLIM_FORCING_DIR = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/clim_forcing_data")


def compute_365day_climatology(
    era5_dir: pathlib.Path = DEFAULT_ERA5_DIR,
    output_clim_file: pathlib.Path | None = None,
    start_year: int = 1980,
    end_year: int = 2020,
    nudging_var: str = "eastward_wind_0",
) -> pathlib.Path:
    """
    Computes the 40-year 365-day smoothed climatology for nudging_var
    following Hitchcock et al. (2022) Section 3.1.
    Returns path to the saved master NetCDF file.
    """
    if output_clim_file is None:
        output_clim_file = DEFAULT_CLIM_DIR / f"era5_{nudging_var}_clim_{start_year}_{end_year}.nc"

    output_clim_file.parent.mkdir(parents=True, exist_ok=True)
    if output_clim_file.exists():
        logging.info("Master climatology file already exists: %s", output_clim_file)
        return output_clim_file

    logging.info(
        "Computing %s climatology for %d-%d from %s...",
        nudging_var,
        start_year,
        end_year,
        era5_dir,
    )
    t0 = time.time()

    u_sum = np.zeros((1460, 180, 360), dtype=np.float64)
    n_cycles = 0
    prev_second_half = None
    ref_coords = None

    for y in range(start_year, end_year + 1):
        era_path = era5_dir / f"era5_1deg_{y}.nc"
        if not era_path.exists():
            raise FileNotFoundError(f"Required ERA5 file not found: {era_path}")

        logging.info("Loading %s for year %d...", nudging_var, y)
        with xr.open_dataset(era_path) as ds:
            da_var = ds[nudging_var]
            if ref_coords is None:
                lat_name = "latitude" if "latitude" in da_var.coords else "lat"
                lon_name = "longitude" if "longitude" in da_var.coords else "lon"
                ref_coords = {
                    "lat": da_var[lat_name].values,
                    "lon": da_var[lon_name].values,
                }

            is_leap = (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))
            june_end = f"{y}-06-29 18:00:00" if is_leap else f"{y}-06-30 18:00:00"

            first_half = da_var.sel(time=slice(f"{y}-01-01 00:00:00", june_end)).values
            second_half = da_var.sel(time=slice(f"{y}-07-01 00:00:00", f"{y}-12-31 18:00:00")).values

        if prev_second_half is not None:
            cycle = np.concatenate([prev_second_half, first_half], axis=0)
            if cycle.shape[0] != 1460:
                raise ValueError(
                    f"Unexpected cycle length {cycle.shape[0]} for cycle {y-1}/{y} (expected 1460)."
                )
            u_sum += cycle
            n_cycles += 1
            logging.info("  -> Completed cycle %d/%d (%d of %d)", y - 1, y, n_cycles, end_year - start_year)

        prev_second_half = second_half

    logging.info("Finished reading %d cycles in %.1f s. Computing mean...", n_cycles, time.time() - t0)
    u_mean = (u_sum / n_cycles).astype(np.float32)

    logging.info("Applying 121-point (30-day) triangular filter...")
    half_width = 60
    weights = 1.0 - np.abs(np.arange(-half_width, half_width + 1)) / (half_width + 1)
    weights = (weights / weights.sum()).astype(np.float32)
    u_smoothed_jj = convolve1d(u_mean, weights, axis=0, mode="wrap")

    logging.info("Reordering climatology to calendar year (1 Jan to 31 Dec)...")
    u_calendar = np.concatenate([u_smoothed_jj[736:], u_smoothed_jj[:736]], axis=0)

    doy_steps = np.arange(1460) * 0.25 + 1.0
    ds_clim = xr.Dataset(
        data_vars={
            nudging_var: (("step_in_year", "latitude", "longitude"), u_calendar),
        },
        coords={
            "step_in_year": doy_steps,
            "latitude": ref_coords["lat"],
            "longitude": ref_coords["lon"],
        },
        attrs={
            "title": f"40-year ({start_year}-{end_year}) 365-day ERA5 {nudging_var} climatology",
            "reference": "Hitchcock et al. (2022) SNAPSI Protocol, Section 3.1",
            "filter": "121-point (30-day) triangular filter with periodic wrapping",
            "leap_handling": "29 February treated as 1 March; 30 June omitted in leap years",
            "creation_date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        },
    )
    ds_clim.to_netcdf(output_clim_file)
    logging.info(
        "Successfully saved master climatology: %s (%.1f MB) in %.1f s total.",
        output_clim_file,
        output_clim_file.stat().st_size / 1e6,
        time.time() - t0,
    )
    return output_clim_file


def create_climatological_forcing_for_year(
    year: int,
    base_forcing_dir: pathlib.Path = DEFAULT_BASE_FORCING_DIR,
    clim_master_file: pathlib.Path | None = None,
    output_clim_forcing_dir: pathlib.Path = DEFAULT_CLIM_FORCING_DIR,
    nudging_var: str = "eastward_wind_0",
) -> pathlib.Path:
    """
    Creates a climatological forcing NetCDF file (clim_forcing_{year}.nc) for the given year
    by copying base forcing_{year}.nc and injecting the 365-day climatology for nudging_var.
    Handles leap years according to SNAPSI (29 Feb uses 1 March climatology).
    """
    output_clim_forcing_dir.mkdir(parents=True, exist_ok=True)
    target_file = output_clim_forcing_dir / f"clim_forcing_{year}.nc"

    if target_file.exists():
        logging.info("Climatological forcing file already exists: %s", target_file)
        return target_file

    if clim_master_file is None or not clim_master_file.exists():
        clim_master_file = compute_365day_climatology(
            output_clim_file=DEFAULT_CLIM_DIR / f"era5_{nudging_var}_clim_1980_2020.nc",
            nudging_var=nudging_var,
        )

    base_file = base_forcing_dir / f"forcing_{year}.nc"
    if not base_file.exists():
        raise FileNotFoundError(f"Base forcing file not found: {base_file}")

    logging.info("Generating climatological forcing for %d from master %s...", year, clim_master_file)
    with xr.open_dataset(clim_master_file) as ds_clim, xr.open_dataset(base_file) as ds_base:
        u_cal = ds_clim[nudging_var].values
        n_base_times = len(ds_base["time"])

        is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
        if is_leap and n_base_times == 1464:
            logging.info("Year %d is leap (1464 steps): setting 29 Feb to 1 March climatology...", year)
            u_year = np.concatenate(
                [
                    u_cal[:236],
                    u_cal[236:240],
                    u_cal[236:],
                ],
                axis=0,
            )
        else:
            if n_base_times != 1460:
                logging.warning(
                    "Base forcing has %d steps (expected 1460); slicing or tiling climatology.",
                    n_base_times,
                )
                u_year = u_cal[:n_base_times]
            else:
                u_year = u_cal

        ds_out = ds_base.copy(deep=True)
        ds_out[nudging_var] = (("time", "latitude", "longitude"), u_year)
        ds_out[nudging_var].attrs["long_name"] = f"Climatological {nudging_var} (SNAPSI control)"
        ds_out[nudging_var].attrs["units"] = "m s**-1"
        ds_out.to_netcdf(target_file)

    logging.info("Successfully created climatological forcing: %s (%.1f MB)", target_file, target_file.stat().st_size / 1e6)
    return target_file


def ensure_climatological_forcing(
    year: int,
    base_forcing_dir: pathlib.Path = DEFAULT_BASE_FORCING_DIR,
    era5_dir: pathlib.Path = DEFAULT_ERA5_DIR,
    clim_dir: pathlib.Path = DEFAULT_CLIM_DIR,
    clim_forcing_dir: pathlib.Path = DEFAULT_CLIM_FORCING_DIR,
    nudging_var: str = "eastward_wind_0",
) -> pathlib.Path:
    """
    High-level convenience function: ensures that clim_forcing_{year}.nc exists in clim_forcing_dir.
    Computes master climatology and year-specific forcing if needed.
    Returns path to clim_forcing_dir.
    """
    clim_forcing_dir.mkdir(parents=True, exist_ok=True)
    target_file = clim_forcing_dir / f"clim_forcing_{year}.nc"
    if target_file.exists():
        logging.info("Climatological forcing file already exists: %s", target_file)
        return clim_forcing_dir

    master_file = clim_dir / f"era5_{nudging_var}_clim_1980_2020.nc"
    if not master_file.exists():
        compute_365day_climatology(
            era5_dir=era5_dir,
            output_clim_file=master_file,
            start_year=1980,
            end_year=2020,
            nudging_var=nudging_var,
        )

    create_climatological_forcing_for_year(
        year=year,
        base_forcing_dir=base_forcing_dir,
        clim_master_file=master_file,
        output_clim_forcing_dir=clim_forcing_dir,
        nudging_var=nudging_var,
    )
    return clim_forcing_dir


def main():
    parser = argparse.ArgumentParser(description="Create SNAPSI climatological forcing for ACE2 nudging.")
    parser.add_argument("--year", type=int, default=None, help="Year to generate forcing file for (e.g. 2018).")
    parser.add_argument("--compute-master-only", action="store_true", help="Only compute master 40-year climatology.")
    parser.add_argument("--era5-dir", type=pathlib.Path, default=DEFAULT_ERA5_DIR)
    parser.add_argument("--base-forcing-dir", type=pathlib.Path, default=DEFAULT_BASE_FORCING_DIR)
    parser.add_argument("--clim-dir", type=pathlib.Path, default=DEFAULT_CLIM_DIR)
    parser.add_argument("--clim-forcing-dir", type=pathlib.Path, default=DEFAULT_CLIM_FORCING_DIR)
    parser.add_argument("--start-year", type=int, default=1980)
    parser.add_argument("--end-year", type=int, default=2020)
    args = parser.parse_args()

    master_file = args.clim_dir / f"era5_eastward_wind_0_clim_{args.start_year}_{args.end_year}.nc"
    if not master_file.exists():
        compute_365day_climatology(
            era5_dir=args.era5_dir,
            output_clim_file=master_file,
            start_year=args.start_year,
            end_year=args.end_year,
        )

    if args.compute_master_only:
        logging.info("Master climatology calculation complete.")
        return

    years = [args.year] if args.year is not None else [2018, 2019]
    for yr in years:
        create_climatological_forcing_for_year(
            year=yr,
            base_forcing_dir=args.base_forcing_dir,
            clim_master_file=master_file,
            output_clim_forcing_dir=args.clim_forcing_dir,
        )


if __name__ == "__main__":
    main()
