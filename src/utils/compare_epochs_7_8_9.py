#!/usr/bin/env python3
"""
Comparative Evaluator for GraphCast Epochs 7, 8, and 9.
Parses forecast NetCDF files from output-epoch7, output-epoch8, and output-epoch9,
computes level-aware verification metrics (RMSE, BIAS, ACC), and generates comparative skill curves.
"""

import os
import sys
import glob
import logging
import argparse
from datetime import datetime, timedelta
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
    parser = argparse.ArgumentParser(description="Compare Verification Metrics for Epochs 7, 8, and 9")
    parser.add_argument(
        "--truth_dir",
        type=str,
        default="/scratch5/purged/Wei.Huang/src/starviewerweathermodel/data/icosahedral-truth",
        help="Ground truth NetCDF directory",
    )
    parser.add_argument("--init_time", type=str, default="2026020106", help="Forecast init time YYYYMMDDHH")
    parser.add_argument("--level_idx", type=int, default=10, help="Vertical level index for verification (0-31)")
    parser.add_argument(
        "--out_dir", type=str, default="verification_results/epoch_comparison", help="Output directory for plots and CSVs"
    )
    return parser.parse_args()


def extract_variable(ds: xr.Dataset, var_name: str, level_idx: int) -> np.ndarray:
    """Extracts 2D slice (nodes,) for a variable at level_idx, un-logging log-state fields if needed."""
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


def evaluate_epoch(epoch_label: str, fcst_dir: str, truth_dir: str, init_dt: datetime, level_idx: int) -> pd.DataFrame:
    """Evaluates all forecast files in a specific epoch directory."""
    pattern = os.path.join(fcst_dir, "forecast_standard_f*.nc")
    nc_files = sorted(glob.glob(pattern))

    if not nc_files:
        logging.warning(f"[WARNING] No files found for {epoch_label} in '{fcst_dir}'")
        return pd.DataFrame()

    logging.info(f"[INFO] Evaluating {epoch_label}: Found {len(nc_files)} NetCDF files in '{fcst_dir}'")

    var_names = ["T", "P", "U", "V", "W", "Q"]
    results = []

    for file_path in nc_files:
        filename = os.path.basename(file_path)
        lead_str = filename.split("_")[-1].replace(".nc", "")
        lead_h = int(lead_str.replace("f", "").replace("h", ""))

        valid_dt = init_dt + timedelta(hours=lead_h)
        valid_date = valid_dt.strftime("%Y%m%d")
        valid_cycle = f"t{valid_dt.hour:02d}z"

        # Match truth file
        truth_matches = glob.glob(os.path.join(truth_dir, f"*{valid_date}.{valid_cycle}*.nc"))
        if not truth_matches:
            truth_matches = glob.glob(os.path.join(truth_dir, f"*{valid_date}*.nc"))

        if not truth_matches:
            logging.warning(f"[{epoch_label}] Missing truth file for lead f{lead_h:04d}h. Skipping.")
            continue

        ds_f = xr.open_dataset(file_path)
        ds_t = xr.open_dataset(truth_matches[0])

        for var in var_names:
            f_val = extract_variable(ds_f, var, level_idx)
            t_val = extract_variable(ds_t, var, level_idx)

            if f_val is not None and t_val is not None and f_val.shape == t_val.shape:
                diff = f_val - t_val
                rmse = float(np.sqrt(np.nanmean(diff**2)))
                bias = float(np.nanmean(diff))
                acc = compute_acc(f_val, t_val)

                results.append({
                    "EPOCH": epoch_label,
                    "VAR": var,
                    "LEAD": lead_str,
                    "LEAD_H": lead_h,
                    "RMSE": rmse,
                    "BIAS": bias,
                    "ACC": acc,
                })

        ds_f.close()
        ds_t.close()

    return pd.DataFrame(results)


def plot_comparative_curves(df_all: pd.DataFrame, output_fig: str, level_idx: int):
    """Plots comparative RMSE and ACC skill curves across Epochs 7, 8, and 9."""
    var_names = ["T", "P", "U", "V", "W", "Q"]
    epochs = df_all["EPOCH"].unique()
    colors = {"Epoch 7": "#1f77b4", "Epoch 8": "#ff7f0e", "Epoch 9": "#2ca02c"}

    fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
    axes = axes.flatten()

    for idx, var in enumerate(var_names):
        ax = axes[idx]

        for ep in epochs:
            df_ep = df_all[(df_all["EPOCH"] == ep) & (df_all["VAR"] == var)].sort_values("LEAD_H")
            if not df_ep.empty:
                color = colors.get(ep, "#333333")
                ax.plot(
                    df_ep["LEAD_H"],
                    df_ep["RMSE"],
                    color=color,
                    linewidth=2.0,
                    marker="o",
                    markersize=3,
                    label=f"{ep} RMSE",
                )

        ax.set_title(f"Variable: {var} (Level Index {level_idx})", fontsize=11, fontweight="bold")
        ax.set_xlabel("Forecast Lead Time (Hours)", fontsize=10)
        ax.set_ylabel(f"RMSE ({var})", fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5)

        # Plot ACC on right Y-axis
        ax2 = ax.twinx()
        for ep in epochs:
            df_ep = df_all[(df_all["EPOCH"] == ep) & (df_all["VAR"] == var)].sort_values("LEAD_H")
            if not df_ep.empty:
                color = colors.get(ep, "#333333")
                ax2.plot(
                    df_ep["LEAD_H"],
                    df_ep["ACC"],
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    alpha=0.85,
                    label=f"{ep} ACC",
                )

        ax2.set_ylabel("ACC Skill", color="#2ca02c", fontweight="bold")
        ax2.set_ylim(-0.1, 1.05)

        # Build combined legend
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=7, ncol=2)

        ax.axvline(120, color="gray", linestyle=":", alpha=0.7)
        ax.axvline(240, color="red", linestyle=":", alpha=0.7)

    plt.suptitle(f"GraphCast Forecast Skill Progression across Epochs 7, 8, and 9 (0-240h)", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=200, bbox_inches="tight")
    # plt.show()
    logging.info(f"[SUCCESS] Saved comparative figure to: '{output_fig}'")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    init_dt = datetime.strptime(args.init_time, "%Y%m%d%H")

    epoch_dirs = {
        "Epoch 7": "output-epoch7/20260201/06",
        "Epoch 8": "output-epoch8/20260201/06",
        "Epoch 9": "output-epoch9/20260201/06",
    }

    df_list = []
    for epoch_label, fcst_dir in epoch_dirs.items():
        if os.path.exists(fcst_dir):
            df_ep = evaluate_epoch(epoch_label, fcst_dir, args.truth_dir, init_dt, args.level_idx)
            if not df_ep.empty:
                df_list.append(df_ep)

    if not df_list:
        raise FileNotFoundError("No valid forecast output directories were found.")

    df_all = pd.concat(df_list, ignore_index=True)

    # Save summary CSV
    csvname=f"epoch_comparison_scores_L{args.level_idx}.csv"
    csv_out = os.path.join(args.out_dir, csvname)
    df_all.to_csv(csv_out, index=False)
    logging.info(f"[SUCCESS] Comparative metrics exported to CSV: '{csv_out}'")

    # Generate comparative plot panel
    imgname=f'epoch_comparison_curves_L{args.level_idx}.png'
    plot_comparative_curves(df_all, os.path.join(args.out_dir, imgname), args.level_idx)


if __name__ == "__main__":
    main()
