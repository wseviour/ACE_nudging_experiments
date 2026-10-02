#!/usr/bin/env python3
"""
plot_level0_snapsi_nudged_vs_control.py

Plots Level 0 (uppermost level, ~1 hPa) zonal mean eastward wind at 60°N
for the SNAPSI Case Study 2 (Jan 2019 SSW) 10-member ensemble experiments:
  1. ERA5 Observations (reanalysis truth for 2018/2019)
  2. ERA5 40-year Climatology (1980-2020 smoothed daily cycle)
  3. Blended Nudged Run (tau = 24h towards observed ERA5, ensemble mean + spread)
  4. Control Run (tau = 24h towards 40-yr Climatology, ensemble mean + spread)
"""

from __future__ import annotations

import argparse
import pathlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr


def extract_zonal_mean_60n(
    da: xr.DataArray,
    target_lat: float = 60.0,
) -> xr.DataArray:
    """Extracts zonal mean at target latitude (default 60°N)."""
    lat_name = "latitude" if "latitude" in da.coords else "lat"
    lon_name = "longitude" if "longitude" in da.coords else "lon"

    da_lat = da.sel({lat_name: target_lat}, method="nearest")
    actual_lat = float(da_lat[lat_name].values)
    da_zonal = da_lat.mean(dim=lon_name)
    da_zonal.attrs["latitude_used"] = actual_lat
    return da_zonal


def load_era5_truth(
    start_time: str,
    end_time: str,
    era5_dir: pathlib.Path,
    var_name: str = "eastward_wind_0",
    target_lat: float = 60.0,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Loads observed ERA5 eastward_wind_0 at target_lat across years."""
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    years = list(range(start_dt.year, end_dt.year + 1))

    zonal_series_list = []
    for yr in years:
        era_path = era5_dir / f"era5_1deg_{yr}.nc"
        if not era_path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {era_path}")
        with xr.open_dataset(era_path) as ds_yr:
            da_var = ds_yr[var_name]
            lat_name = "latitude" if "latitude" in da_var.coords else "lat"
            lon_name = "longitude" if "longitude" in da_var.coords else "lon"
            da_lat = da_var.sel({lat_name: target_lat}, method="nearest")
            da_zonal = da_lat.mean(dim=lon_name)
            da_slice = da_zonal.sel(time=slice(start_dt, end_dt))
            zonal_series_list.append(da_slice.load())

    if len(zonal_series_list) == 1:
        da_combined = zonal_series_list[0]
    else:
        da_combined = xr.concat(zonal_series_list, dim="time")

    times = pd.to_datetime(da_combined.time.values)
    values = da_combined.values
    return times, values


def load_clim_forcing(
    start_time: str,
    end_time: str,
    clim_forcing_dir: pathlib.Path,
    var_name: str = "eastward_wind_0",
    target_lat: float = 60.0,
) -> tuple[pd.DatetimeIndex, np.ndarray]:
    """Loads 40-year smoothed climatological forcing at target_lat across years."""
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    years = list(range(start_dt.year, end_dt.year + 1))

    zonal_series_list = []
    for yr in years:
        clim_path = clim_forcing_dir / f"clim_forcing_{yr}.nc"
        if not clim_path.exists():
            raise FileNotFoundError(f"Clim forcing file not found: {clim_path}")
        with xr.open_dataset(clim_path) as ds_yr:
            da_var = ds_yr[var_name]
            lat_name = "latitude" if "latitude" in da_var.coords else "lat"
            lon_name = "longitude" if "longitude" in da_var.coords else "lon"
            da_lat = da_var.sel({lat_name: target_lat}, method="nearest")
            da_zonal = da_lat.mean(dim=lon_name)
            da_slice = da_zonal.sel(time=slice(start_dt, end_dt))
            zonal_series_list.append(da_slice.load())

    if len(zonal_series_list) == 1:
        da_combined = zonal_series_list[0]
    else:
        da_combined = xr.concat(zonal_series_list, dim="time")

    times = pd.to_datetime(da_combined.time.values)
    values = da_combined.values
    return times, values


def extract_ensemble_stats(
    nc_path: pathlib.Path,
    target_lat: float = 60.0,
    start_time_str: str = "2018-12-13T00:00:00",
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extracts time coords, ensemble mean, std, min, max, and full member array.
    """
    ds = xr.open_dataset(nc_path)
    if "time" in ds.coords and np.issubdtype(ds.time.dtype, np.datetime64):
        times = pd.to_datetime(ds.time.values)
    else:
        n_steps = ds.sizes.get("time", 100)
        times = pd.date_range(start=start_time_str, periods=n_steps, freq="6h")

    da_zonal = extract_zonal_mean_60n(ds["eastward_wind_0"], target_lat=target_lat)
    if "sample" in da_zonal.dims:
        n_members = da_zonal.sizes["sample"]
        member_data = da_zonal.values
        if member_data.shape[0] != n_members:
            member_data = member_data.T
        ens_mean = np.mean(member_data, axis=0)
        ens_std = np.std(member_data, axis=0)
        ens_min = np.min(member_data, axis=0)
        ens_max = np.max(member_data, axis=0)
    else:
        ens_mean = da_zonal.values
        member_data = np.expand_dims(ens_mean, axis=0)
        ens_std = np.zeros_like(ens_mean)
        ens_min = ens_mean
        ens_max = ens_mean

    return times, ens_mean, ens_std, ens_min, ens_max, member_data


def plot_nudged_vs_control(
    nudged_nc: pathlib.Path,
    control_nc: pathlib.Path,
    era5_dir: pathlib.Path,
    clim_forcing_dir: pathlib.Path,
    output_path: pathlib.Path,
    target_lat: float = 60.0,
) -> None:
    print(f"Loading Nudged predictions:  {nudged_nc}")
    t_nudge, n_mean, n_std, n_min, n_max, n_members = extract_ensemble_stats(nudged_nc, target_lat=target_lat)

    print(f"Loading Control predictions: {control_nc}")
    t_ctrl, c_mean, c_std, c_min, c_max, c_members = extract_ensemble_stats(control_nc, target_lat=target_lat)

    start_time_str = str(t_nudge[0])
    end_time_str = str(t_nudge[-1])

    print(f"Loading ERA5 observations:   {era5_dir}")
    era_times, era_values = load_era5_truth(start_time_str, end_time_str, era5_dir, target_lat=target_lat)

    print(f"Loading 40-yr Climatology:   {clim_forcing_dir}")
    clim_times, clim_values = load_clim_forcing(start_time_str, end_time_str, clim_forcing_dir, target_lat=target_lat)

    # Reindex ERA5 and Climatology to match forecast times
    era_series = pd.Series(era_values, index=era_times).reindex(t_nudge, method="nearest").values
    clim_series = pd.Series(clim_values, index=clim_times).reindex(t_nudge, method="nearest").values

    # Setup 2-panel figure
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13, 10.5),
        gridspec_kw={"height_ratios": [3.2, 1.3]},
        sharex=True,
    )

    # Reference lines
    ax1.axhline(0, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.8, label="Zero Wind ($U = 0$ m/s, SSW Reversal)")

    ssw_central = pd.Timestamp("2019-01-02 00:00")
    if t_nudge[0] <= ssw_central <= t_nudge[-1]:
        ax1.axvline(
            ssw_central,
            color="darkmagenta",
            linestyle=":",
            linewidth=2.2,
            alpha=0.9,
            label="SSW Central Date (2 Jan 2019)",
        )
        ax1.text(
            ssw_central + pd.Timedelta(hours=14),
            20,
            "Major SSW Onset",
            color="darkmagenta",
            fontweight="bold",
            fontsize=10,
            rotation=90,
            va="top",
        )

    # 1. ERA5 Observation (Truth)
    ax1.plot(
        t_nudge,
        era_series,
        color="black",
        linestyle="-",
        linewidth=2.8,
        label="ERA5 Observed (2018/2019)",
        zorder=6,
    )

    # 2. ERA5 40-year Climatology
    ax1.plot(
        t_nudge,
        clim_series,
        color="#d35400",
        linestyle="--",
        linewidth=2.5,
        label="ERA5 40-yr Climatology (1980–2020)",
        zorder=5,
    )

    # 3. Nudged Ensemble (Blended tau=24h)
    ax1.fill_between(
        t_nudge,
        n_mean - n_std,
        n_mean + n_std,
        color="#2980b9",
        alpha=0.25,
        label=r"Nudged ($\tau=24$h) Spread ($\pm 1\sigma$)",
        zorder=3,
    )
    for m in range(n_members.shape[0]):
        ax1.plot(t_nudge, n_members[m], color="#3498db", linewidth=0.6, alpha=0.35, zorder=2)
    ax1.plot(
        t_nudge,
        n_mean,
        color="#1f618d",
        linestyle="-",
        linewidth=2.4,
        label="Nudged Ensemble Mean (10 members)",
        zorder=4,
    )

    # 4. Control Ensemble (Climatology tau=24h)
    ax1.fill_between(
        t_ctrl,
        c_mean - c_std,
        c_mean + c_std,
        color="#27ae60",
        alpha=0.25,
        label=r"Control (Clim $\tau=24$h) Spread ($\pm 1\sigma$)",
        zorder=3,
    )
    for m in range(c_members.shape[0]):
        ax1.plot(t_ctrl, c_members[m], color="#2ecc71", linewidth=0.6, alpha=0.35, zorder=2)
    ax1.plot(
        t_ctrl,
        c_mean,
        color="#196f3d",
        linestyle="-",
        linewidth=2.4,
        label="Control Ensemble Mean (10 members)",
        zorder=4,
    )

    ax1.set_ylabel("Zonal Mean $U$ (m s$^{-1}$)", fontsize=13, fontweight="bold")
    ax1.set_title(
        "SNAPSI Case Study 2 (Jan 2019 SSW): Upper Stratosphere (~1 hPa) Zonal Mean Wind at 60°N\n"
        "Comparison of 10-Member Nudged Run vs Control Run vs ERA5 Reanalysis & Climatology",
        fontsize=14,
        fontweight="bold",
        pad=14,
    )
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", frameon=True, framealpha=0.92, fontsize=10.5, ncol=2)

    # Panel 2: Ensemble Spread sigma(U)
    ax2.plot(
        t_nudge,
        n_std,
        color="#1f618d",
        linestyle="-",
        linewidth=2.2,
        label=r"Nudged Spread $\sigma(U)$",
    )
    ax2.plot(
        t_ctrl,
        c_std,
        color="#196f3d",
        linestyle="-",
        linewidth=2.2,
        label=r"Control Spread $\sigma(U)$",
    )
    if t_nudge[0] <= ssw_central <= t_nudge[-1]:
        ax2.axvline(ssw_central, color="darkmagenta", linestyle=":", linewidth=2.0, alpha=0.8)

    ax2.set_ylabel(r"Ensemble Spread $\sigma$ (m s$^{-1}$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Forecast Date (UTC)", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper left", frameon=True, framealpha=0.9, fontsize=10.5)

    # Format Date Axis
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    fig.autofmt_xdate(rotation=0, ha="center")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved comparison figure to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot SNAPSI Level 0 Zonal Mean U: Nudged vs Control.")
    parser.add_argument(
        "--nudged-nc",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_output/exp_20181213_ens10_blended_tau24h/autoregressive_predictions.nc"),
    )
    parser.add_argument(
        "--control-nc",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_output/exp_20181213_ens10_control_tau24h/autoregressive_predictions.nc"),
    )
    parser.add_argument(
        "--era5-dir",
        type=pathlib.Path,
        default=pathlib.Path("/disco/share/ws359/ERA5_for_ACE"),
    )
    parser.add_argument(
        "--clim-forcing-dir",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE2-ERA5/clim_forcing_data"),
    )
    parser.add_argument(
        "--save-path",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_nudging_experiments/analysis/case2_level0_nudged_vs_control.png"),
    )
    parser.add_argument("--lat", type=float, default=60.0)
    args = parser.parse_args()

    plot_nudged_vs_control(
        nudged_nc=args.nudged_nc,
        control_nc=args.control_nc,
        era5_dir=args.era5_dir,
        clim_forcing_dir=args.clim_forcing_dir,
        output_path=args.save_path,
        target_lat=args.lat,
    )


if __name__ == "__main__":
    main()
