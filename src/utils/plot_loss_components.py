#!/usr/bin/env python3
"""
Plots itemized 3D GraphCast physical loss components from PyTorch Lightning metrics.csv.
Saves a clean multi-panel figure showing loss progression over steps/epochs.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt


def find_latest_metrics_csv(log_dir="lightning_logs/standard_direct_m6"):
    """Finds the most recent metrics.csv file in lightning_logs."""
    csv_files = glob.glob(os.path.join(log_dir, "version_*", "metrics.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No metrics.csv files found under '{log_dir}'")

    latest_csv = max(csv_files, key=os.path.getmtime)
    print(f"[INFO] Parsing latest metrics file: {latest_csv}")
    return latest_csv


def resolve_column(df, base_name):
    """Finds matching column name whether PyTorch Lightning logs it raw or with _step/_epoch suffix."""
    candidates = [
        base_name,
        f"{base_name}_step",
        f"{base_name}_epoch",
        base_name.replace("train/", "train_"),
    ]
    for col in candidates:
        if col in df.columns:
            return col
    return None


def plot_loss_metrics(csv_path, output_fig="verification_results/loss_components_panel.png"):
    df = pd.read_csv(csv_path)

    if "step" not in df.columns:
        raise KeyError("Expected 'step' column in metrics.csv")

    # Clean step data by converting to numeric and sorting
    df["step"] = pd.to_numeric(df["step"], errors="coerce")
    df_clean = df.dropna(subset=["step"]).sort_values("step")

    # Canonical metric definitions (Base Name -> Display Title)
    loss_targets = {
        "train/loss": "Total Combined Loss",
        "train/loss_mse": "Base State MSE",
        "train/loss_momentum_h": "3D Horizontal Momentum Residual",
        "train/loss_momentum_v": "Vertical Momentum Residual",
        "train/loss_continuity": "3D Mass Continuity Residual",
        "train/loss_variance": "Symmetric Range & Variance Loss",
        "train/loss_wind_ke": "Wind Kinetic Energy MSE",
        "train/loss_wind_dir": "Wind Direction Cosine Penalty",
        "train/loss_mass_drift": "Global Surface Pressure Mass Drift",
        "train/loss_moisture_penalty": "Global Moisture Conservation Penalty",
    }

    # Map available CSV columns to display titles
    active_metrics = {}
    for base_key, title in loss_targets.items():
        matched_col = resolve_column(df_clean, base_key)
        if matched_col:
            active_metrics[matched_col] = title

    print(f"[INFO] Resolved {len(active_metrics)} active loss columns for plotting.")

    if not active_metrics:
        print("[WARNING] No canonical loss columns matched. Falling back to all numerical columns.")
        active_metrics = {c: c for c in df_clean.columns if "loss" in c}

    n_metrics = len(active_metrics)
    cols = 2
    rows = (n_metrics + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(14, 3.2 * rows), sharex=True)
    axes = axes.flatten()

    for idx, (col_name, title) in enumerate(active_metrics.items()):
        ax = axes[idx]
        
        # Filter valid numerical entries for this specific column
        sub_df = df_clean.dropna(subset=[col_name]).copy()
        sub_df[col_name] = pd.to_numeric(sub_df[col_name], errors="coerce")
        sub_df = sub_df.dropna(subset=[col_name])

        if not sub_df.empty:
            ax.plot(sub_df["step"], sub_df[col_name], color="#1f77b4", linewidth=2.0, marker="o", markersize=4, label="Train")
            ax.set_title(title, fontsize=11, fontweight="bold")
            ax.set_ylabel("Value")
            ax.grid(True, linestyle="--", alpha=0.5)

            # Plot validation counterpart if present
            val_base = col_name.replace("train/", "val/").replace("train_", "val_")
            matched_val_col = resolve_column(df_clean, val_base)
            if matched_val_col:
                val_df = df_clean.dropna(subset=[matched_val_col]).copy()
                val_df[matched_val_col] = pd.to_numeric(val_df[matched_val_col], errors="coerce")
                val_df = val_df.dropna(subset=[matched_val_col])
                if not val_df.empty:
                    ax.plot(val_df["step"], val_df[matched_val_col], color="#ff7f0e", linestyle="--", linewidth=2.0, marker="s", markersize=4, label="Val")
            
            ax.legend(loc="upper right", fontsize=9)
        else:
            ax.text(0.5, 0.5, "No Numeric Data", ha="center", va="center", transform=ax.transAxes)

    # Remove extra subplot axes
    for j in range(idx + 1, len(axes)):
        fig.delaxes(axes[j])

    for ax in axes[-cols:]:
        ax.set_xlabel("Global Step", fontsize=10, fontweight="bold")

    plt.suptitle("3D GraphCast Itemized Physical Loss Components Progression", fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.savefig(output_fig, dpi=200, bbox_inches="tight")
    plt.show()
    print(f"[SUCCESS] Saved loss curves panel figure to: '{output_fig}'")


if __name__ == "__main__":
    csv_file = find_latest_metrics_csv()
    plot_loss_metrics(csv_file)
