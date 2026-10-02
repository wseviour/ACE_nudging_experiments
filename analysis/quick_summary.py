#!/usr/bin/env python3
"""
quick_summary.py

Inspects an ACE2 ensemble prediction NetCDF file and prints key summary statistics
for eastward wind levels at 60°N (mean, spread/std, min, max).
"""

import argparse
import pathlib
import numpy as np
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description="Quick statistical summary of an ACE2 ensemble run.")
    parser.add_argument(
        "file_path",
        type=pathlib.Path,
        help="Path to autoregressive_predictions.nc file.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=60.0,
        help="Latitude for zonal mean evaluation (default: 60.0).",
    )
    args = parser.parse_args()

    if not args.file_path.exists():
        print(f"File not found: {args.file_path}")
        return

    ds = xr.open_dataset(args.file_path)
    lat_dim = "lat" if "lat" in ds.coords else "latitude"
    lon_dim = "lon" if "lon" in ds.coords else "longitude"

    n_samples = ds.sizes.get("sample", 1)
    n_times = ds.sizes.get("time", 1)

    print("=" * 65)
    print(f"Dataset: {args.file_path}")
    print(f"Dimensions: {dict(ds.sizes)}")
    print(f"Ensemble Members: {n_samples}, Forecast Timesteps: {n_times}")
    print("=" * 65)

    wind_vars = [f"eastward_wind_{i}" for i in range(8) if f"eastward_wind_{i}" in ds]

    print(f"{'Variable':<20} | {'Mean (m/s)':<12} | {'Mean Spread (m/s)':<18} | {'Max Spread (m/s)'}")
    print("-" * 65)

    for var in wind_vars:
        da = ds[var].sel({lat_dim: args.lat}, method="nearest").mean(dim=lon_dim)
        if "sample" in da.dims and da.sizes["sample"] > 1:
            mean_val = float(da.mean().values)
            std_series = da.std(dim="sample").values
            mean_spread = float(np.mean(std_series))
            max_spread = float(np.max(std_series))
            print(f"{var:<20} | {mean_val:10.3f}   | {mean_spread:16.3f}   | {max_spread:14.3f}")
        else:
            mean_val = float(da.mean().values)
            print(f"{var:<20} | {mean_val:10.3f}   | {'N/A (1 member)':<18} | {'N/A'}")

    print("=" * 65)


if __name__ == "__main__":
    main()
