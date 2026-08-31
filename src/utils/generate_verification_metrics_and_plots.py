#!/usr/bin/env python3
"""
Comprehensive Model Verification, Line Plotter, and Heatmap Generator.
Computes RMSE, BIAS, ACC, and Min/Max bounds for T, P, U, V, W, Q across lead times
and outputs verification line plots and global spatial error heatmaps with full global maps
(East & West Hemispheres) and lat-lon markings every 30 degrees.
"""

import os
import glob
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.ticker as mticker

# Cartopy import for geographic map projections & coastlines
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate Verification Curves and Spatial Heatmaps")
    parser.add_argument("--fcst_dir", type=str, default="output/20260201/06", help="Directory with forecast NetCDF files")
    parser.add_argument("--truth_dir", type=str, default="/scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral-truth", help="Ground truth NetCDF directory")
    parser.add_argument("--init_time", type=str, default="2026020106", help="Forecast init time YYYYMMDDHH")
    parser.add_argument("--level_idx", type=int, default=10, help="Vertical level index for verification (0-31)")
    parser.add_argument("--lead_hours", nargs="+", type=int, default=list(range(6, 726, 6)), help="Lead times in hours")
    parser.add_argument("--out_dir", type=str, default="verification_results", help="Output directory for plots and summary")
    parser.add_argument("-s", "--show", action="store_true", help="Display plot interactively")
    return parser.parse_args()


def extract_variable(ds, var_name, level_idx):
    """
    Extracts 2D slice (nodes,) for a variable at level_idx, converting log-state values back to physical units.
    """
    var_lower = var_name.lower()
    mapping = {
        "t": ["ln_t_icosahedral", "t_icosahedral", "T", "t"],
        "p": ["ln_p_icosahedral", "p_icosahedral", "P", "p"],
        "u": ["u_icosahedral", "U", "u"],
        "v": ["v_icosahedral", "V", "v"],
        "w": ["w_icosahedral", "W", "w"],
        "q": ["q_icosahedral", "Q", "q"],
    }

    possible_keys = mapping.get(var_lower, [var_name])
    found_key = None
    for k in possible_keys:
        if k in ds:
            found_key = k
            break

    if found_key is None:
        return None

    val = np.squeeze(ds[found_key].values)
    if val.ndim == 2:
        val = val[level_idx]

    # Un-log log-state variables back to physical space
    if found_key.startswith("ln_"):
        val = np.exp(val)

    # Convert pressure to hPa if in Pa
    if var_lower == "p" and np.nanmean(val) > 2000.0:
        val = val / 100.0

    # Convert humidity to g/kg if in kg/kg
    if var_lower == "q" and np.nanmean(val) < 0.1:
        val = val * 1000.0

    return val.astype(np.float32)


def compute_acc(fcst, truth):
    """Computes Anomaly Correlation Coefficient (ACC)."""
    f_anom = fcst - np.nanmean(fcst)
    t_anom = truth - np.nanmean(truth)
    denom = np.sqrt(np.sum(f_anom**2) * np.sum(t_anom**2))
    if denom == 0:
        return 0.0
    return float(np.sum(f_anom * t_anom) / denom)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    init_dt = datetime.strptime(args.init_time, "%Y%m%d%H")

    vars_list = ["T", "P", "U", "V", "W", "Q"]
    var_units = {"T": "K", "P": "hPa", "U": "m/s", "V": "m/s", "W": "Pa/s", "Q": "g/kg"}

    metrics = {
        v: {
            "lead": [],
            "rmse": [],
            "bias": [],
            "acc": [],
            "f_min": [],
            "f_max": [],
            "t_min": [],
            "t_max": [],
            "diff_maps": [],
        }
        for v in vars_list
    }

    print("=" * 155)
    print(f" GENERATING METRICS SUMMARY (Init: {init_dt.strftime('%Y-%m-%d %HZ')} | Level Index: {args.level_idx})")
    print("=" * 155)

    for lead_h in args.lead_hours:
        valid_dt = init_dt + timedelta(hours=lead_h)
        valid_date = valid_dt.strftime("%Y%m%d")
        valid_cycle = f"t{valid_dt.hour:02d}z"

        # Match forecast file
        fcst_file = os.path.join(args.fcst_dir, f"forecast_standard_f{lead_h:04d}h.nc")
        if not os.path.exists(fcst_file):
            fcst_matches = glob.glob(os.path.join(args.fcst_dir, f"*f{lead_h:04d}*.nc"))
            fcst_file = fcst_matches[0] if fcst_matches else None

        # Match ground truth file
        truth_matches = glob.glob(os.path.join(args.truth_dir, f"*{valid_date}.{valid_cycle}*.nc"))
        if not truth_matches:
            truth_matches = glob.glob(os.path.join(args.truth_dir, f"*{valid_date}*.nc"))

        if not fcst_file or not truth_matches or not os.path.exists(fcst_file):
            logging.warning(f"Missing file pair for lead f{lead_h:04d}h. Skipping.")
            continue

        ds_f = xr.open_dataset(fcst_file)
        ds_t = xr.open_dataset(truth_matches[0])

        lats = ds_f["latitude"].values if "latitude" in ds_f else np.linspace(-90, 90, 40962)
        lons_raw = ds_f["longitude"].values if "longitude" in ds_f else np.linspace(0, 360, 40962)

        # Convert 0..360 longitude coordinates to standard -180..+180 space
        lons = ((lons_raw + 180.0) % 360.0) - 180.0

        for v in vars_list:
            f_val = extract_variable(ds_f, v, args.level_idx)
            t_val = extract_variable(ds_t, v, args.level_idx)

            if f_val is not None and t_val is not None and f_val.shape == t_val.shape:
                diff = f_val - t_val
                rmse = float(np.sqrt(np.nanmean(diff**2)))
                bias = float(np.nanmean(diff))
                acc = compute_acc(f_val, t_val)

                metrics[v]["lead"].append(lead_h)
                metrics[v]["rmse"].append(rmse)
                metrics[v]["bias"].append(bias)
                metrics[v]["acc"].append(acc)
                metrics[v]["f_min"].append(float(np.nanmin(f_val)))
                metrics[v]["f_max"].append(float(np.nanmax(f_val)))
                metrics[v]["t_min"].append(float(np.nanmin(t_val)))
                metrics[v]["t_max"].append(float(np.nanmax(t_val)))
                metrics[v]["diff_maps"].append((lead_h, diff, lats, lons))

        ds_f.close()
        ds_t.close()

    # Print Text Metrics Summary Table with Min/Max
    print(f"{'VAR':<5} | {'LEAD':<6} | {'RMSE':<8} | {'BIAS':<8} | {'ACC':<7} | {'FCST MIN/MAX':<22} | {'TRUTH MIN/MAX':<22}")
    print("-" * 105)
    for v in vars_list:
        for i, lead in enumerate(metrics[v]["lead"]):
            f_minmax = f"{metrics[v]['f_min'][i]:.2f} / {metrics[v]['f_max'][i]:.2f}"
            t_minmax = f"{metrics[v]['t_min'][i]:.2f} / {metrics[v]['t_max'][i]:.2f}"
            print(
                f"{v:<5} | f{lead:03d}h  | {metrics[v]['rmse'][i]:<8.4f} | {metrics[v]['bias'][i]:<8.4f} | "
                f"{metrics[v]['acc'][i]:<7.4f} | {f_minmax:<22} | {t_minmax:<22}"
            )
        print("-" * 105)

    # 1. Generate Verification Metric Line Plots (RMSE, BIAS, ACC)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    for v in vars_list:
        if metrics[v]["lead"]:
            axes[0].plot(metrics[v]["lead"], metrics[v]["rmse"], marker="o", label=f"{v} ({var_units[v]})")
            axes[1].plot(metrics[v]["lead"], metrics[v]["bias"], marker="s", label=f"{v} ({var_units[v]})")
            axes[2].plot(metrics[v]["lead"], metrics[v]["acc"], marker="^", label=f"{v}")

    axes[0].set_ylabel("RMSE")
    axes[0].set_title("Root Mean Squared Error (RMSE) vs Lead Time")
    axes[0].grid(True, linestyle="--", alpha=0.6)
    axes[0].legend(loc="upper left", ncol=3)

    axes[1].set_ylabel("BIAS (Fcst - Truth)")
    axes[1].set_title("Mean Bias vs Lead Time")
    axes[1].axhline(0, color="black", linestyle="--", linewidth=1)
    axes[1].grid(True, linestyle="--", alpha=0.6)
    axes[1].legend(loc="upper left", ncol=3)

    axes[2].set_ylabel("ACC Score")
    axes[2].set_xlabel("Forecast Lead Time (Hours)")
    axes[2].set_title("Anomaly Correlation Coefficient (ACC) vs Lead Time")
    axes[2].set_ylim([-0.1, 1.05])
    axes[2].axhline(0.6, color="red", linestyle=":", label="Skill Threshold (0.6)")
    axes[2].grid(True, linestyle="--", alpha=0.6)
    axes[2].legend(loc="lower left", ncol=3)

    plt.tight_layout()
    metrics_plot_path = os.path.join(args.out_dir, "verification_metrics_curves.png")
    plt.savefig(metrics_plot_path, dpi=600)
    if args.show:
        plt.show()
    plt.close()
    logging.info(f"Saved metric curves to: {metrics_plot_path}")

    # 2. Generate Spatial Error Heatmaps with Global Map (-180..+180) & 30-degree Markings
    target_leads = [24, 48, 72, 96, 120]
    fig = plt.figure(figsize=(22, 18))
    gs = GridSpec(len(vars_list), len(target_leads), figure=fig)

    for row_idx, v in enumerate(vars_list):
        diff_dict = {item[0]: (item[1], item[2], item[3]) for item in metrics[v]["diff_maps"]}

        for col_idx, lead_h in enumerate(target_leads):
            if HAS_CARTOPY:
                ax = fig.add_subplot(gs[row_idx, col_idx], projection=ccrs.PlateCarree(central_longitude=0))
                ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="black", alpha=0.8)
                ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="gray", alpha=0.5)
                transform = ccrs.PlateCarree()
            else:
                ax = fig.add_subplot(gs[row_idx, col_idx])
                transform = None

            if lead_h in diff_dict:
                diff_data, lats, lons = diff_dict[lead_h]
                bound = max(np.percentile(np.abs(diff_data), 98), 1e-4)

                if transform is not None:
                    sc = ax.scatter(
                        lons, lats, c=diff_data, cmap="RdBu_r", vmin=-bound, vmax=bound, s=0.8, alpha=0.8, transform=transform
                    )
                else:
                    sc = ax.scatter(
                        lons, lats, c=diff_data, cmap="RdBu_r", vmin=-bound, vmax=bound, s=0.8, alpha=0.8
                    )

                cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
                cbar.ax.tick_params(labelsize=7)

                if row_idx == 0:
                    ax.set_title(f"f{lead_h:04d}h (Valid t+{lead_h}h)", fontsize=11, fontweight="bold")
                if col_idx == 0:
                    ax.set_ylabel(f"{v} Error\n({var_units[v]})", fontsize=11, fontweight="bold")

                # Configure Lat-Lon Gridlines & Tick Markings every 30 degrees (-180..+180)
                if HAS_CARTOPY:
                    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle=":")
                    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 30))
                    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 30))
                    gl.top_labels = False
                    gl.right_labels = False
                    gl.xlabel_style = {"size": 6}
                    gl.ylabel_style = {"size": 6}
                else:
                    ax.set_xticks(np.arange(-180, 181, 30))
                    ax.set_yticks(np.arange(-90, 91, 30))
                    ax.set_xlim([-180, 180])
                    ax.set_ylim([-90, 90])
                    ax.tick_params(labelsize=6)
                    ax.grid(True, linestyle=":", linewidth=0.4, color="gray", alpha=0.5)
            else:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center")
                ax.axis("off")

    plt.suptitle(
        f"Global Error Heatmaps (Forecast - Truth) with Full Globe Coverage\nInit: {init_dt.strftime('%Y-%m-%d %HZ')} | Level Index: {args.level_idx}",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()
    heatmap_plot_path = os.path.join(args.out_dir, "spatial_error_heatmaps_panel.png")
    plt.savefig(heatmap_plot_path, dpi=600)
    if args.show:
        plt.show()
    plt.close()
    logging.info(f"Saved spatial error heatmaps to: {heatmap_plot_path}")


if __name__ == "__main__":
    main()
