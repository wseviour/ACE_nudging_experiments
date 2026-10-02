#!/usr/bin/env python3
"""
plot_spread_comparison.py

Plots ensemble spread sigma(U) and individual ensemble member trajectories at 60°N
for Level 0 (uppermost level) and Level 1 (lower stratosphere), comparing different
nudging regimes (e.g., Free, Blended tau=24h, Blended tau=8.7h, Prescribed).
"""

import argparse
import pathlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def get_u_60n(ds: xr.Dataset, var_name: str, n_steps: int = 100, is_era: bool = False) -> xr.DataArray:
    u = ds[var_name].squeeze()
    if is_era:
        u = u.sel(time=slice("2018-01-25", None)).isel(time=slice(0, n_steps))
        lat_dim = "latitude" if "latitude" in u.coords else "lat"
        lon_dim = "longitude" if "longitude" in u.coords else "lon"
        return u.sel({lat_dim: 60}, method="nearest").mean(dim=lon_dim)
    else:
        u = u.isel(time=slice(0, n_steps))
        lat_dim = "lat" if "lat" in u.coords else "latitude"
        lon_dim = "lon" if "lon" in u.coords else "longitude"
        return u.sel({lat_dim: 60}, method="nearest").mean(dim=lon_dim)


def main():
    parser = argparse.ArgumentParser(description="Plot ensemble spread comparison across experiments.")
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_output"),
        help="Root directory where ACE experiments are stored.",
    )
    parser.add_argument(
        "--era5-file",
        type=pathlib.Path,
        default=pathlib.Path("/disco/share/ws359/ERA5_for_ACE/era5_1deg_2018.nc"),
        help="Path to ERA5 1-degree file for truth comparison.",
    )
    parser.add_argument(
        "--save-path",
        type=pathlib.Path,
        default=pathlib.Path(__file__).parent / "ensemble_spread_comparison.png",
        help="File path to save the generated plot.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=100,
        help="Number of 6-hour forecast steps to plot.",
    )
    args = parser.parse_args()

    # Search for available experiments
    exp_candidates = [
        ("Free ($\\tau=\\infty$)", args.output_dir / "output_directory_20180125_ensemble_free" / "autoregressive_predictions.nc", "tab:red", "--"),
        ("Blended ($\\tau=24\\,\\mathrm{h}$)", args.output_dir / "output_directory_20180125_ensemble_blended_tau24h" / "autoregressive_predictions.nc", "tab:cyan", "-"),
        ("Blended ($\\tau=8.7\\,\\mathrm{h}$)", args.output_dir / "output_directory_20180125_ensemble_blended_tau8h" / "autoregressive_predictions.nc", "tab:blue", "-"),
        ("Prescribed ($\\tau=0\\,\\mathrm{h}$)", args.output_dir / "output_directory_20180125_ensemble_nudged" / "autoregressive_predictions.nc", "tab:purple", "-."),
    ]

    # Also check /home/links/ws359/ACE/ACE2-ERA5 fallback for existing runs
    fallback_dir = pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5")
    experiments = []
    for label, path, color, ls in exp_candidates:
        if path.exists():
            experiments.append((label, xr.open_dataset(path), color, ls))
        else:
            fb = fallback_dir / path.parent.name / path.name
            if fb.exists():
                experiments.append((label, xr.open_dataset(fb), color, ls))

    if not experiments:
        print(f"No experiment NetCDF files found in {args.output_dir} or {fallback_dir}.")
        return

    print(f"Loaded {len(experiments)} experiments for comparison.")
    ds_era = xr.open_dataset(args.era5_file) if args.era5_file.exists() else None
    time_axis = pd.date_range(start="2018-01-25 00:00", periods=args.n_steps, freq="6h")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(18, 12), sharex=True)

    for row, (ax_mem, ax_std, level) in enumerate([(axes[0, 0], axes[0, 1], 0), (axes[1, 0], axes[1, 1], 1)]):
        var_name = f"eastward_wind_{level}"

        if ds_era is not None and var_name in ds_era:
            era_u = get_u_60n(ds_era, var_name, n_steps=args.n_steps, is_era=True)
            ax_mem.plot(time_axis, era_u.values, color="k", linewidth=2.5, label="ERA5 Truth")

        for label, ds, color, ls in experiments:
            if var_name not in ds:
                continue
            u = get_u_60n(ds, var_name, n_steps=args.n_steps)
            n_samples = u.shape[0]

            # Individual members (faint)
            for i in range(n_samples):
                ax_mem.plot(time_axis[1:], u[i, :].values[:-1], color=color, alpha=0.18, linewidth=0.8)

            # Ensemble mean
            u_mean = u.mean(dim="sample").values[:-1]
            ax_mem.plot(time_axis[1:], u_mean, color=color, linewidth=2.2, linestyle=ls, label=label)

            # Ensemble spread (standard deviation)
            u_std = u.std(dim="sample").values[:-1]
            ax_std.plot(time_axis[1:], u_std, color=color, linewidth=2.2, linestyle=ls, label=label)

        ax_mem.set_title(f"Level {level} Zonal Mean $U$ at 60°N (Members & Ensemble Mean)", fontsize=13, fontweight="bold")
        ax_mem.set_ylabel("Wind Speed (m/s)", fontsize=11)
        ax_mem.grid(True, alpha=0.4)
        ax_mem.legend(loc="upper right", framealpha=0.9, fontsize=10)

        ax_std.set_title(f"Level {level} Ensemble Spread $\\sigma(U)$ at 60°N", fontsize=13, fontweight="bold")
        ax_std.set_ylabel("Standard Deviation $\\sigma$ (m/s)", fontsize=11)
        ax_std.grid(True, alpha=0.4)
        ax_std.legend(loc="upper left", framealpha=0.9, fontsize=10)

    fig.autofmt_xdate()
    plt.tight_layout()
    args.save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(args.save_path, dpi=150)
    plt.close()
    print(f"Spread comparison plot saved to: {args.save_path}")


if __name__ == "__main__":
    main()
