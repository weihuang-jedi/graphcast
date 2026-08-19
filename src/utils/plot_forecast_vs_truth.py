#!/usr/bin/env python3
"""
Diagnostic Forecast Metrics Summary & Panel Plotter.
Dynamically computes valid target datetimes from initialization time and lead hours
to match ground truth files stored as absolute dates (e.g., gfs.YYYYMMDD.tHHz.1p00.f000.nc).
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
    parser.add_argument("--fcst_dir", type=str, default="output_reconstructed", help="Directory with reconstructed NetCDF forecasts")
    parser.add_argument("--truth_dir", type=str, default="../data/nc_truth", help="Directory with ground truth NetCDF files")
    parser.add_argument("--init_time", type=str, default="2026020100", help="Initialization timestamp (YYYYMMDDHH)")
    parser.add_argument("--level_idx", type=int, default=10, help="Vertical height level index (e.g. 10 = ~500m)")
    parser.add_argument("--lead_times", nargs="+", type=int, default=[480, 960, 1440, 1920, 2400], help="Lead times in hours")
    parser.add_argument("--vars", nargs="+", type=str, default=["t", "p", "u", "v", "q"], help="Variables to evaluate")
    parser.add_argument("--out_dir", type=str, default="verification_plots", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Display plots interactively")
    return parser.parse_args()


def compute_acc(fcst, truth):
    """Computes Anomaly Correlation Coefficient (ACC)."""
    f_anom = fcst - np.mean(fcst)
    t_anom = truth - np.mean(truth)
    denom = np.sqrt(np.sum(f_anom**2) * np.sum(t_anom**2))
    if denom == 0:
        return 0.0
    return float(np.sum(f_anom * t_anom) / denom)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    try:
        init_dt = datetime.strptime(args.init_time, "%Y%m%d%H")
    except ValueError:
        init_dt = datetime.fromisoformat(args.init_time)

    print("=" * 115)
    print(f" DIAGNOSTIC FORECAST METRICS SUMMARY (Init: {init_dt.strftime('%Y-%m-%d %HZ')} | Height Level Index: {args.level_idx})")
    print("=" * 115)
    print(f"{'VAR':<7} | {'LEAD':<8} | {'TRUTH MIN':<11} | {'TRUTH MAX':<11} | {'FCST MIN':<11} | {'FCST MAX':<11} | {'DIFF MIN':<11} | {'DIFF MAX':<11} | {'RMSE':<11} | {'ACC':<7}")
    print("-" * 115)

    for var_name in args.vars:
        var_lower = var_name.lower()
        var_upper = var_name.upper()

        panel_fcst = []
        panel_truth = []
        panel_leads = []

        for lead_h in args.lead_times:
            valid_dt = init_dt + timedelta(hours=lead_h)
            
            # 1. Match reconstructed forecast file
            fcst_patterns = [
                os.path.join(args.fcst_dir, f"*f{lead_h:04d}h*.nc"),
                os.path.join(args.fcst_dir, f"*f{lead_h}h*.nc"),
                os.path.join(args.fcst_dir, f"*f{lead_h:04d}*.nc"),
            ]
            fcst_matches = []
            for pat in fcst_patterns:
                fcst_matches = glob.glob(pat)
                if fcst_matches:
                    break

            # 2. Match ground truth file by computed target valid date (YYYYMMDD) and cycle hour (tHHz)
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

            # Resolve variable key name
            var_f_key = var_lower if var_lower in ds_f else (var_upper if var_upper in ds_f else None)
            var_t_key = var_lower if var_lower in ds_t else (var_upper if var_upper in ds_t else None)

            if var_f_key is None or var_t_key is None:
                ds_f.close()
                ds_t.close()
                continue

            # Extract 2D field at height level
            da_f = ds_f[var_f_key]
            da_t = ds_t[var_t_key]

            h_dim_f = "height" if "height" in da_f.dims else "level"
            h_dim_t = "height" if "height" in da_t.dims else "level"

            f_2d = np.squeeze(da_f.isel({h_dim_f: args.level_idx}).values)
            t_2d = np.squeeze(da_t.isel({h_dim_t: args.level_idx}).values)

            if f_2d.shape == t_2d.shape:
                diff = f_2d - t_2d
                rmse = float(np.sqrt(np.mean(diff**2)))
                acc = compute_acc(f_2d, t_2d)

                print(
                    f"{var_upper:<7} | f{lead_h:04d}h   | "
                    f"{t_2d.min():<11.4f} | {t_2d.max():<11.4f} | "
                    f"{f_2d.min():<11.4f} | {f_2d.max():<11.4f} | "
                    f"{diff.min():<11.4f} | {diff.max():<11.4f} | "
                    f"{rmse:<11.4f} | {acc:<7.4f}"
                )

                panel_fcst.append(f_2d)
                panel_truth.append(t_2d)
                panel_leads.append(lead_h)

            ds_f.close()
            ds_t.close()

        print("-" * 115)

        # Render Verification Panel Plot
        if panel_fcst:
            fig, axes = plt.subplots(len(panel_leads), 3, figsize=(15, 3 * len(panel_leads)))
            if len(panel_leads) == 1:
                axes = np.expand_dims(axes, axis=0)

            for idx, lead_h in enumerate(panel_leads):
                f_data = panel_fcst[idx]
                t_data = panel_truth[idx]
                d_data = f_data - t_data

                v_min = min(t_data.min(), f_data.min())
                v_max = max(t_data.max(), f_data.max())

                # Truth
                im0 = axes[idx, 0].imshow(t_data, extent=[0, 360, -90, 90], cmap="viridis", vmin=v_min, vmax=v_max)
                axes[idx, 0].set_title(f"Truth f{lead_h:04d}h ({var_upper})")
                plt.colorbar(im0, ax=axes[idx, 0], fraction=0.046, pad=0.04)

                # Forecast
                im1 = axes[idx, 1].imshow(f_data, extent=[0, 360, -90, 90], cmap="viridis", vmin=v_min, vmax=v_max)
                axes[idx, 1].set_title(f"Forecast f{lead_h:04d}h ({var_upper})")
                plt.colorbar(im1, ax=axes[idx, 1], fraction=0.046, pad=0.04)

                # Diff
                diff_bound = max(abs(d_data.min()), abs(d_data.max()), 1e-5)
                im2 = axes[idx, 2].imshow(d_data, extent=[0, 360, -90, 90], cmap="RdBu_r", vmin=-diff_bound, vmax=diff_bound)
                axes[idx, 2].set_title(f"Diff (Fcst - Truth)")
                plt.colorbar(im2, ax=axes[idx, 2], fraction=0.046, pad=0.04)

            plt.tight_layout()
            out_plot_path = os.path.join(args.out_dir, f"verification_panel_{var_lower}_level{args.level_idx}.png")
            plt.savefig(out_plot_path, dpi=200)
            if args.show:
                plt.show()
            plt.close()
            logging.info(f"Verification plot saved to: {out_plot_path}")


if __name__ == "__main__":
    main()
