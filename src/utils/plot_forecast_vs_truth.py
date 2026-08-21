#!/usr/bin/env python3
"""
Diagnostic Forecast Metrics Summary & Panel Plotter for Terrain-Following Height Coordinates.
Compares reconstructed forecasts against ground truth on matching terrain height levels (target_level).
"""

import os
import glob
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Plot Forecast vs Truth Diagnostics")
    parser.add_argument("--fcst_dir", type=str, default="output_reconstructed")
    parser.add_argument("--truth_dir", type=str, default="../data/nc_truth")
    parser.add_argument("--init_time", type=str, default="2026020100")
    parser.add_argument("--level_idx", type=int, default=10, help="Terrain height level index (0=2m, 10=500m, 31=20km)")
    parser.add_argument("--lead_times", nargs="+", type=int, default=[24, 48, 72, 96, 120])
    parser.add_argument("--vars", nargs="+", type=str, default=["t", "p", "u", "v", "q"])
    parser.add_argument("--out_dir", type=str, default="verification_plots")
    parser.add_argument("--show", action="store_true")
    return parser.parse_args()


def compute_acc(fcst, truth):
    f_anom = fcst - np.nanmean(fcst)
    t_anom = truth - np.nanmean(truth)
    denom = np.sqrt(np.sum(f_anom**2) * np.sum(t_anom**2))
    if denom == 0:
        return 0.0
    return float(np.sum(f_anom * t_anom) / denom)


def extract_field_2d(ds, var_name, level_idx):
    """Safely extracts 2D field at terrain height level_idx, converting log-state variables if present."""
    var_lower = var_name.lower()
    var_upper = var_name.upper()

    possible_keys = [
        f"ln_{var_lower}_icosahedral", f"{var_lower}_icosahedral",
        f"ln_{var_lower}", var_lower, var_upper
    ]

    found_key = None
    for k in possible_keys:
        if k in ds or k in ds.coords:
            found_key = k
            break

    if found_key is None:
        return None

    da = ds[found_key]
    val = np.squeeze(da.values)

    # Slice terrain height level index
    if val.ndim == 3:
        idx = min(level_idx, val.shape[0] - 1)
        val = val[idx]

    # Convert log-state variables back to physical space if present in truth file
    if found_key.startswith("ln_"):
        val = np.exp(val)

    # Convert pressure to hPa if values are in Pa range (~100,000)
    if var_lower == "p" and np.nanmean(val) > 2000.0:
        val = val / 100.0

    # Convert humidity to g/kg if in kg/kg
    if var_lower == "q" and np.nanmean(val) < 0.1:
        val = val * 1000.0

    return val


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        init_dt = datetime.strptime(args.init_time, "%Y%m%d%H")
    except ValueError:
        init_dt = datetime.fromisoformat(args.init_time)

    print("=" * 115)
    print(f" DIAGNOSTIC FORECAST METRICS SUMMARY (Init: {init_dt.strftime('%Y-%m-%d %HZ')} | Terrain Level Index: {args.level_idx})")
    print("=" * 115)
    print(f"{'VAR':<7} | {'LEAD':<8} | {'TRUTH MIN':<11} | {'TRUTH MAX':<11} | {'FCST MIN':<11} | {'FCST MAX':<11} | {'DIFF MIN':<11} | {'DIFF MAX':<11} | {'RMSE':<11} | {'ACC':<7}")
    print("-" * 115)

    for var_name in args.vars:
        panel_fcst, panel_truth, panel_leads = [], [], []

        for lead_h in args.lead_times:
            valid_dt = init_dt + timedelta(hours=lead_h)

            fcst_patterns = [
                os.path.join(args.fcst_dir, f"*f{lead_h:04d}h*.nc"),
                os.path.join(args.fcst_dir, f"*f{lead_h}h*.nc"),
            ]
            fcst_matches = []
            for pat in fcst_patterns:
                fcst_matches = glob.glob(pat)
                if fcst_matches:
                    break

            valid_date_str = valid_dt.strftime("%Y%m%d")
            valid_cycle_str = f"t{valid_dt.hour:02d}z"

            truth_patterns = [
                os.path.join(args.truth_dir, f"gfs.{valid_date_str}.{valid_cycle_str}*.nc"),
                os.path.join(args.truth_dir, f"*{valid_date_str}*{valid_cycle_str}*.nc"),
                os.path.join(args.truth_dir, f"*{valid_date_str}*.nc"),
            ]
            truth_matches = []
            for pat in truth_patterns:
                truth_matches = glob.glob(pat)
                if truth_matches:
                    break

            if not fcst_matches or not truth_matches:
                continue

            ds_f = xr.open_dataset(fcst_matches[0])
            ds_t = xr.open_dataset(truth_matches[0])

            f_2d = extract_field_2d(ds_f, var_name, args.level_idx)
            t_2d = extract_field_2d(ds_t, var_name, args.level_idx)

            ds_f.close()
            ds_t.close()

            if f_2d is not None and t_2d is not None and f_2d.shape == t_2d.shape:
                diff = f_2d - t_2d
                rmse = float(np.sqrt(np.nanmean(diff**2)))
                acc = compute_acc(f_2d, t_2d)

                print(
                    f"{var_name.upper():<7} | f{lead_h:04d}h   | "
                    f"{np.nanmin(t_2d):<11.4f} | {np.nanmax(t_2d):<11.4f} | "
                    f"{np.nanmin(f_2d):<11.4f} | {np.nanmax(f_2d):<11.4f} | "
                    f"{np.nanmin(diff):<11.4f} | {np.nanmax(diff):<11.4f} | "
                    f"{rmse:<11.4f} | {acc:<7.4f}"
                )

                panel_fcst.append(f_2d)
                panel_truth.append(t_2d)
                panel_leads.append(lead_h)

        print("-" * 115)

        # Render Verification Panel Plot
        if panel_fcst:
            fig, axes = plt.subplots(len(panel_leads), 3, figsize=(15, 3 * len(panel_leads)))
            if len(panel_leads) == 1:
                axes = np.expand_dims(axes, axis=0)

            for idx, lead_h in enumerate(panel_leads):
                f_data, t_data = panel_fcst[idx], panel_truth[idx]
                d_data = f_data - t_data
                v_min = min(np.nanmin(t_data), np.nanmin(f_data))
                v_max = max(np.nanmax(t_data), np.nanmax(f_data))

                im0 = axes[idx, 0].imshow(t_data, extent=[0, 360, -90, 90], cmap="viridis", vmin=v_min, vmax=v_max)
                axes[idx, 0].set_title(f"Truth f{lead_h:04d}h ({var_name.upper()})")
                plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)

                im1 = axes[idx, 1].imshow(f_data, extent=[0, 360, -90, 90], cmap="viridis", vmin=v_min, vmax=v_max)
                axes[idx, 1].set_title(f"Forecast f{lead_h:04d}h ({var_name.upper()})")
                plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

                diff_bound = max(abs(np.nanmin(d_data)), abs(np.nanmax(d_data)), 1e-5)
                im2 = axes[idx, 2].imshow(d_data, extent=[0, 360, -90, 90], cmap="RdBu_r", vmin=-diff_bound, vmax=diff_bound)
                axes[idx, 2].set_title("Diff (Fcst - Truth)")
                plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            out_plot_path = os.path.join(args.out_dir, f"verification_panel_{var_name.lower()}_level{args.level_idx}.png")
            plt.savefig(out_plot_path, dpi=200)
            if args.show:
                plt.show()
            plt.close()


if __name__ == "__main__":
    main()
