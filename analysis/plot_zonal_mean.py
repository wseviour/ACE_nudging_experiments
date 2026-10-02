#!/usr/bin/env python3
"""
plot_zonal_mean.py

Plots the timeseries of zonal mean eastward wind at 60°N for all vertical levels (0 to 7)
comparing the ensemble mean against ERA5 reanalysis truth.
"""

import argparse
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def main():
    parser = argparse.ArgumentParser(description="Plot zonal mean zonal wind across levels at 60°N.")
    parser.add_argument(
        "prediction_file",
        type=pathlib.Path,
        help="Path to autoregressive_predictions.nc file.",
    )
    parser.add_argument(
        "--era5-file",
        type=pathlib.Path,
        default=pathlib.Path("/disco/share/ws359/ERA5_for_ACE/era5_1deg_2018.nc"),
        help="Path to ERA5 1-degree file for comparison.",
    )
    parser.add_argument(
        "--save-path",
        type=pathlib.Path,
        default=None,
        help="Path to save the generated figure.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=60.0,
        help="Latitude for zonal mean evaluation (default: 60.0).",
    )
    args = parser.parse_args()

    if not args.prediction_file.exists():
        print(f"Prediction file not found: {args.prediction_file}")
        return

    ds = xr.open_dataset(args.prediction_file)
    lat_dim = "lat" if "lat" in ds.coords else "latitude"
    lon_dim = "lon" if "lon" in ds.coords else "longitude"
    n_steps = ds.sizes.get("time", 100)

    ds_era = xr.open_dataset(args.era5_file) if args.era5_file.exists() else None

    fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(16, 14), sharex=True)
    axes = axes.flatten()

    time_axis = pd.date_range(start="2018-01-25 00:00", periods=n_steps, freq="6h")

    for level in range(8):
        ax = axes[level]
        var_name = f"eastward_wind_{level}"
        if var_name not in ds:
            continue

        u_pred = ds[var_name].sel({lat_dim: args.lat}, method="nearest").mean(dim=lon_dim)
        if "sample" in u_pred.dims:
            # Plot individual faint lines
            for s in range(u_pred.sizes["sample"]):
                ax.plot(time_axis[:len(u_pred.isel(sample=s))], u_pred.isel(sample=s).values, color="tab:blue", alpha=0.2, linewidth=0.8)
            u_mean = u_pred.mean(dim="sample").values
            ax.plot(time_axis[:len(u_mean)], u_mean, color="tab:blue", linewidth=2.0, label="ACE2 Ensemble Mean")
        else:
            ax.plot(time_axis[:len(u_pred)], u_pred.values, color="tab:blue", linewidth=2.0, label="ACE2 Prediction")

        if ds_era is not None and var_name in ds_era:
            era_lat = "latitude" if "latitude" in ds_era.coords else "lat"
            era_lon = "longitude" if "longitude" in ds_era.coords else "lon"
            era_u = ds_era[var_name].sel(time=slice("2018-01-25", None)).isel(time=slice(0, n_steps))
            era_u_60 = era_u.sel({era_lat: args.lat}, method="nearest").mean(dim=era_lon)
            ax.plot(time_axis[:len(era_u_60)], era_u_60.values, color="k", linewidth=2.2, label="ERA5 Truth")

        ax.set_title(f"Level {level} ($U$ at {args.lat}°N)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Wind (m/s)", fontsize=10)
        ax.grid(True, alpha=0.4)
        if level == 0:
            ax.legend(loc="upper right", framealpha=0.9, fontsize=9)

    fig.autofmt_xdate()
    plt.tight_layout()

    save_path = args.save_path or args.prediction_file.parent / "zonal_mean_u_all_levels.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved level profile plot to {save_path}")


if __name__ == "__main__":
    main()
