#!/usr/bin/env python3
"""
Evaluates pre-generated NetCDF forecast files (06h to 240h) against ground truth data.
Extracts 2D slices at level_idx (default 10) to match operational verification metrics.
"""

import os
import sys
import glob
import logging
import argparse
import yaml
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# Ensure project root is in sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Pre-generated 240h Forecast NetCDF Files")
    parser.add_argument("-c", "--config", type=str, default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--fcst_dir", type=str, default="output/20260201/06", help="Forecast directory")
    parser.add_argument("--truth_dir", type=str, default="/scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral-truth", help="Ground truth directory")
    parser.add_argument("--level_idx", type=int, default=10, help="Vertical level index for verification (0-31)")
    parser.add_argument("--out_dir", type=str, default="verification_results/rollout_240h", help="Output directory")
    return parser.parse_args()


def extract_variable(ds: xr.Dataset, var_name: str, level_idx: int) -> np.ndarray:
    """
    Extracts 2D slice (nodes,) for a variable at level_idx, un-logging log-state fields if needed.
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

    # Un-log log-state variables if present
    if found_key.startswith("ln_"):
        val = np.exp(val)

    # Convert pressure to hPa if in Pa
    if var_lower == "p" and np.nanmean(val) > 2000.0:
        val = val / 100.0

    # Convert humidity to g/kg if in kg/kg
    if var_lower == "q" and np.nanmean(val) < 0.1:
        val = val * 1000.0

    return val.astype(np.float32)


def compute_acc(fcst: np.ndarray, truth: np.ndarray) -> float:
    """Computes Anomaly Correlation Coefficient (ACC)."""
    f_anom = fcst - np.nanmean(fcst)
    t_anom = truth - np.nanmean(truth)
    denom = np.sqrt(np.sum(f_anom**2) * np.sum(t_anom**2))
    if denom == 0:
        return 0.0
    return float(np.sum(f_anom * t_anom) / denom)


def evaluate_forecast_nc_files():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    pattern = os.path.join(args.fcst_dir, "forecast_standard_f*.nc")
    nc_files = sorted(glob.glob(pattern))

    if not nc_files:
        raise FileNotFoundError(f"No forecast files matching '{pattern}' were found.")

    logging.info(f"[INFO] Found {len(nc_files)} NetCDF forecast files in '{args.fcst_dir}'")

    var_names = ["T", "P", "U", "V", "W", "Q"]
    results = []

    for file_path in nc_files:
        filename = os.path.basename(file_path)
        lead_str = filename.split("_")[-1].replace(".nc", "")
        lead_h = int(lead_str.replace("f", "").replace("h", ""))

        # Find matching ground truth file
        truth_matches = glob.glob(os.path.join(args.truth_dir, f"*f{lead_h:04d}*.nc"))
        if not truth_matches:
            truth_matches = sorted(glob.glob(os.path.join(args.truth_dir, "*.nc")))

        if not truth_matches:
            logging.warning(f"No truth file found for lead {lead_str}. Skipping.")
            continue

        # Match step index in ground truth dataset
        truth_file = truth_matches[0]
        ds_f = xr.open_dataset(file_path)
        ds_t = xr.open_dataset(truth_file)

        for var in var_names:
            f_val = extract_variable(ds_f, var, args.level_idx)
            t_val = extract_variable(ds_t, var, args.level_idx)

            if f_val is not None and t_val is not None and f_val.shape == t_val.shape:
                diff = f_val - t_val
                rmse = float(np.sqrt(np.nanmean(diff**2)))
                bias = float(np.nanmean(diff))
                acc = compute_acc(f_val, t_val)

                results.append({
                    "VAR": var,
                    "LEAD": lead_str,
                    "LEAD_H": lead_h,
                    "RMSE": rmse,
                    "BIAS": bias,
                    "ACC": acc,
                    "FCST_MIN": float(np.nanmin(f_val)),
                    "FCST_MAX": float(np.nanmax(f_val)),
                    "TRUTH_MIN": float(np.nanmin(t_val)),
                    "TRUTH_MAX": float(np.nanmax(t_val)),
                })

        ds_f.close()
        ds_t.close()

    df_res = pd.DataFrame(results)

    # Print Verification Summary Table
    print("\n" + "=" * 105)
    print(f"{'VAR':<5} | {'LEAD':<7} | {'RMSE':<9} | {'BIAS':<9} | {'ACC':<8} | {'FCST MIN/MAX':<24} | {'TRUTH MIN/MAX':<24}")
    print("-" * 105)

    sample_leads = ["f0006h", "f0024h", "f0048h", "f0072h", "f0120h", "f0168h", "f0240h"]
    for _, row in df_res[df_res["LEAD"].isin(sample_leads)].iterrows():
        fcst_range = f"{row['FCST_MIN']:.2f} / {row['FCST_MAX']:.2f}"
        truth_range = f"{row['TRUTH_MIN']:.2f} / {row['TRUTH_MAX']:.2f}"
        print(f"{row['VAR']:<5} | {row['LEAD']:<7} | {row['RMSE']:<9.4f} | {row['BIAS']:<9.4f} | {row['ACC']:<8.4f} | {fcst_range:<24} | {truth_range:<24}")
    print("=" * 105 + "\n")

    # Export CSV & Plot Skill Curves
    csv_out = os.path.join(args.out_dir, "scores_240h_nc_eval.csv")
    df_res.to_csv(csv_out, index=False)
    logging.info(f"[SUCCESS] Verification scores saved to CSV: '{csv_out}'")

    plot_metric_curves(df_res, os.path.join(args.out_dir, "verification_curves_240h.png"))


def plot_metric_curves(df: pd.DataFrame, output_fig: str):
    """Plots RMSE and ACC skill curves across 240 hours."""
    var_names = ["T", "P", "U", "V", "W", "Q"]
    fig, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    axes = axes.flatten()

    for idx, var in enumerate(var_names):
        ax = axes[idx]
        df_var = df[df["VAR"] == var].sort_values("LEAD_H")

        color_rmse = "#1f77b4"
        color_acc = "#2ca02c"

        ax.set_title(f"Variable: {var} (Level Index 10 | 0 to 240 Hours)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast Lead Time (Hours)", fontsize=10)

        # RMSE Curve
        ax.set_ylabel(f"RMSE ({var})", color=color_rmse, fontweight="bold")
        ax.plot(df_var["LEAD_H"], df_var["RMSE"], color=color_rmse, linewidth=2.0, marker="o", label="RMSE")
        ax.tick_params(axis="y", labelcolor=color_rmse)
        ax.grid(True, linestyle="--", alpha=0.5)

        # ACC Curve
        ax2 = ax.twinx()
        ax2.set_ylabel("ACC Skill", color=color_acc, fontweight="bold")
        ax2.plot(df_var["LEAD_H"], df_var["ACC"], color=color_acc, linestyle="--", linewidth=2.0, marker="s", label="ACC")
        ax2.set_ylim(-0.1, 1.05)
        ax2.tick_params(axis="y", labelcolor=color_acc)

        ax.axvline(120, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(240, color="red", linestyle=":", alpha=0.7)

    plt.suptitle("3D GraphCast 240h Forecast Skill Curves (Level Index 10)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=200, bbox_inches="tight")
    plt.show()
    logging.info(f"[SUCCESS] Saved 240h forecast evaluation figure to: '{output_fig}'")


if __name__ == "__main__":
    evaluate_forecast_nc_files()
