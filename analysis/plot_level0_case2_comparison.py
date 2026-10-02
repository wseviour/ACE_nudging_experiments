#!/usr/bin/env python3
"""
plot_level0_case2_comparison.py

Plots Level 0 (uppermost level, ~10 hPa) zonal mean eastward wind at 60°N
for the SNAPSI Case Study 2 (Jan 2019 SSW) 10-member ensemble experiment,
comparing the ACE2 forecast against ERA5 reanalysis truth from Disco.
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
    """
    Extracts zonal mean at target latitude (default 60°N) for a DataArray.
    """
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
    """
    Loads ERA5 data spanning across year boundary from era5_dir,
    extracting only var_name at target_lat before concatenation.
    """
    start_dt = pd.to_datetime(start_time)
    end_dt = pd.to_datetime(end_time)
    years = list(range(start_dt.year, end_dt.year + 1))

    zonal_series_list = []
    for yr in years:
        era_path = era5_dir / f"era5_1deg_{yr}.nc"
        if not era_path.exists():
            raise FileNotFoundError(f"ERA5 file not found: {era_path}")
        print(f"Loading {var_name} from {era_path.name}...")
        with xr.open_dataset(era_path) as ds_yr:
            da_var = ds_yr[var_name]
            # Select target lat first to minimize data transferred
            lat_name = "latitude" if "latitude" in da_var.coords else "lat"
            lon_name = "longitude" if "longitude" in da_var.coords else "lon"
            da_lat = da_var.sel({lat_name: target_lat}, method="nearest")
            da_zonal = da_lat.mean(dim=lon_name)
            # Slice time window relevant for this year
            da_slice = da_zonal.sel(time=slice(start_dt, end_dt))
            zonal_series_list.append(da_slice.load())

    if len(zonal_series_list) == 1:
        da_combined = zonal_series_list[0]
    else:
        da_combined = xr.concat(zonal_series_list, dim="time")

    times = pd.to_datetime(da_combined.time.values)
    values = da_combined.values
    return times, values


def plot_comparison(
    pred_path: pathlib.Path,
    era5_dir: pathlib.Path,
    output_path: pathlib.Path,
    start_time_str: str = "2018-12-13T00:00:00",
    target_lat: float = 60.0,
) -> None:
    print(f"Loading predictions from: {pred_path}")
    ds_pred = xr.open_dataset(pred_path)

    # Determine time coordinates
    if "time" in ds_pred.coords and np.issubdtype(ds_pred.time.dtype, np.datetime64):
        pred_times = pd.to_datetime(ds_pred.time.values)
    else:
        n_steps = ds_pred.sizes.get("time", 100)
        pred_times = pd.date_range(start=start_time_str, periods=n_steps, freq="6h")

    start_time_str = str(pred_times[0])
    end_time_str = str(pred_times[-1])
    print(f"Forecast period: {start_time_str} to {end_time_str} ({len(pred_times)} steps)")

    # Extract level 0 wind zonal mean from predictions
    da_pred_zonal = extract_zonal_mean_60n(ds_pred["eastward_wind_0"], target_lat=target_lat)
    actual_lat = da_pred_zonal.attrs.get("latitude_used", target_lat)

    # Load ERA5 truth efficiently
    era_times, era_values = load_era5_truth(
        start_time=start_time_str,
        end_time=end_time_str,
        era5_dir=era5_dir,
        var_name="eastward_wind_0",
        target_lat=target_lat,
    )
    print(f"Loaded {len(era_times)} ERA5 timestamps ({era_times[0]} to {era_times[-1]}).")

    # Analyze ensemble members
    has_samples = "sample" in da_pred_zonal.dims
    if has_samples:
        n_members = da_pred_zonal.sizes["sample"]
        member_data = da_pred_zonal.values  # shape: (sample, time) or (time, sample)
        if member_data.shape[0] != n_members:
            member_data = member_data.T
        ens_mean = np.mean(member_data, axis=0)
        ens_std = np.std(member_data, axis=0)
        ens_min = np.min(member_data, axis=0)
        ens_max = np.max(member_data, axis=0)
    else:
        n_members = 1
        ens_mean = da_pred_zonal.values
        member_data = np.expand_dims(ens_mean, axis=0)
        ens_std = np.zeros_like(ens_mean)
        ens_min = ens_mean
        ens_max = ens_mean

    # Interpolate / match ERA5 times with forecast times if necessary
    era_series = pd.Series(era_values, index=era_times)
    era_matched = era_series.reindex(pred_times, method="nearest").values

    # Setup 2-panel figure
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 10),
        gridspec_kw={"height_ratios": [3, 1.2]},
        sharex=True,
    )

    # -------------------------------------------------------------
    # Panel 1: Absolute Level 0 Zonal Mean Eastward Wind
    # -------------------------------------------------------------
    # Zero wind reference line (SSW wind reversal threshold)
    ax1.axhline(0, color="firebrick", linestyle="--", linewidth=1.5, alpha=0.8, label="Zero Wind ($U = 0$ m/s, Reversal)")

    # SSW Central Date vertical line (2 Jan 2019)
    ssw_central = pd.Timestamp("2019-01-02 00:00")
    if pred_times[0] <= ssw_central <= pred_times[-1]:
        ax1.axvline(
            ssw_central,
            color="darkmagenta",
            linestyle=":",
            linewidth=2.0,
            alpha=0.9,
            label="SSW Central Date (2 Jan 2019)",
        )
        ax1.text(
            ssw_central + pd.Timedelta(hours=12),
            20,
            "Major SSW Onset",
            color="darkmagenta",
            fontweight="bold",
            fontsize=10,
            rotation=90,
            va="top",
        )

    # Shaded spread
    if has_samples and n_members > 1:
        ax1.fill_between(
            pred_times,
            ens_min,
            ens_max,
            color="#3498db",
            alpha=0.18,
            label="Ensemble Range (Min–Max)",
        )
        ax1.fill_between(
            pred_times,
            ens_mean - ens_std,
            ens_mean + ens_std,
            color="#2980b9",
            alpha=0.28,
            label=r"Ensemble Spread ($\pm 1\sigma$)",
        )

        # Individual members
        for m in range(n_members):
            lbl = "Perturbed Members (1..9)" if m == 1 else ("Unperturbed Control (Member 0)" if m == 0 else None)
            clr = "#2c3e50" if m == 0 else "#2980b9"
            ls = "-" if m == 0 else "-"
            lw = 1.4 if m == 0 else 0.8
            alpha = 0.6 if m == 0 else 0.35
            ax1.plot(pred_times, member_data[m], color=clr, linestyle=ls, linewidth=lw, alpha=alpha, label=lbl)

        # Ensemble Mean
        ax1.plot(
            pred_times,
            ens_mean,
            color="#0052cc",
            linewidth=2.8,
            label=f"ACE2 Ensemble Mean ({n_members} members)",
        )
    else:
        ax1.plot(pred_times, ens_mean, color="#0052cc", linewidth=2.8, label="ACE2 Forecast")

    # ERA5 Truth
    ax1.plot(
        era_times,
        era_values,
        color="black",
        linewidth=2.5,
        linestyle="-",
        label="ERA5 Reanalysis Truth",
    )

    ax1.set_ylabel(r"Zonal Wind $U$ (m s$^{-1}$)", fontsize=13, fontweight="bold")
    ax1.set_title(
        f"SNAPSI Case Study 2 (Jan 2019 SSW): Level 0 Zonal Mean Wind at {actual_lat:.1f}°N\n"
        f"ACE2 {n_members}-Member Ensemble (Blended Nudging, $\\tau = 24$ h) vs. ERA5 Truth",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", framealpha=0.92, fontsize=10, ncol=2)

    # -------------------------------------------------------------
    # Panel 2: Model Bias (Ensemble Mean - ERA5) and Spread
    # -------------------------------------------------------------
    bias = ens_mean - era_matched
    ax2.axhline(0, color="black", linestyle="-", linewidth=1.0, alpha=0.7)
    if pred_times[0] <= ssw_central <= pred_times[-1]:
        ax2.axvline(ssw_central, color="darkmagenta", linestyle=":", linewidth=2.0, alpha=0.9)

    line_bias = ax2.plot(
        pred_times,
        bias,
        color="#d35400",
        linewidth=2.2,
        label="Forecast Bias (Ensemble Mean − ERA5)",
    )
    ax2.set_ylabel(r"Bias (m s$^{-1}$)", fontsize=11, fontweight="bold", color="#d35400")
    ax2.tick_params(axis="y", labelcolor="#d35400")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # Twin axis for spread
    if has_samples and n_members > 1:
        ax2_spread = ax2.twinx()
        line_spread = ax2_spread.plot(
            pred_times,
            ens_std,
            color="#27ae60",
            linewidth=2.0,
            linestyle="-.",
            label=r"Ensemble Spread $\sigma$ (m s$^{-1}$)",
        )
        ax2_spread.set_ylabel(r"Spread $\sigma$ (m s$^{-1}$)", fontsize=11, fontweight="bold", color="#27ae60")
        ax2_spread.tick_params(axis="y", labelcolor="#27ae60")
        # Combined legend
        lines = line_bias + line_spread
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc="upper left", framealpha=0.9, fontsize=9.5)
    else:
        ax2.legend(loc="upper left", framealpha=0.9, fontsize=9.5)

    ax2.set_xlabel("Date (UTC)", fontsize=13, fontweight="bold")
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b\n%Y"))
    fig.autofmt_xdate(rotation=0, ha="center")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Comparison plot successfully saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot Level 0 winds for SNAPSI Case 2 vs ERA5.")
    parser.add_argument(
        "--pred-file",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_output/exp_20181213_ens10_blended_tau24h/autoregressive_predictions.nc"),
        help="Path to prediction NetCDF file.",
    )
    parser.add_argument(
        "--era5-dir",
        type=pathlib.Path,
        default=pathlib.Path("/disco/share/ws359/ERA5_for_ACE"),
        help="Directory containing era5_1deg_YYYY.nc files.",
    )
    parser.add_argument(
        "--output-file",
        type=pathlib.Path,
        default=pathlib.Path("/home/links/ws359/ACE/ACE_output/exp_20181213_ens10_blended_tau24h/case2_level0_u60n_comparison.png"),
        help="Path where output plot will be saved.",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=60.0,
        help="Target latitude (default: 60.0).",
    )
    args = parser.parse_args()

    plot_comparison(
        pred_path=args.pred_file,
        era5_dir=args.era5_dir,
        output_path=args.output_file,
        target_lat=args.lat,
    )


if __name__ == "__main__":
    main()
