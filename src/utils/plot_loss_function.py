#!/usr/bin/env python3
"""
3D GraphCast Model - Stage M3 Loss Function Analyzer
Plots Total Loss, Per-Variable MSE (P, Q, T, U, V), and Zero-Mean Anomaly Penalty.
"""

import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt


class WeatherLossPlotter:
    def __init__(self, metrics_path=None):
        self.metrics_path = metrics_path if metrics_path else self._find_latest_csv()
        self.df = None

    def _find_latest_csv(self):
        log_dirs = sorted(glob.glob("lightning_logs/standard_direct/version_*"))

        if not log_dirs:
            raise FileNotFoundError("Error: No training log directories (lightning_logs/stage_m3/version_*) found.")

        latest_log_dir = log_dirs[-1]
        csv_path = os.path.join(latest_log_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Error: Found directory {latest_log_dir}, but metrics.csv is missing.")

        return csv_path

    def load_data(self):
        print(f"[PLOTTER] Loading metrics from: {self.metrics_path}")
        self.df = pd.read_csv(self.metrics_path)

    def generate_chart(self, output_img="analysis_loss.png", show_plot=True):
        if self.df is None:
            self.load_data()

        # Clean step-level metric data
        step_df = self.df.dropna(subset=['train/loss']).sort_values('step').copy()
        if step_df.empty:
            raise ValueError("No valid step data found in CSV.")

        ema_alpha = 0.08
        cols_to_smooth = [
            'train/loss', 'val/loss',
            'train/loss_P', 'train/loss_Q', 'train/loss_T',
            'train/loss_U', 'train/loss_V', 'train/loss_zero_mean'
        ]

        for col in cols_to_smooth:
            if col in step_df.columns:
                step_df[f'{col}_ema'] = step_df[col].ewm(alpha=ema_alpha, adjust=False).mean()

        fig, axes = plt.subplots(2, 2, figsize=(13, 8.5), sharex=True)
        fig.suptitle('3D GraphCast Stage M3 Loss & Penalty Convergence', fontsize=15, fontweight='bold')

        # Panel 1: Total Training & Validation Loss
        ax1 = axes[0, 0]
        ax1.plot(step_df['step'], step_df['train/loss_ema'], label='Train Loss (EMA)', color='#1f77b4', linewidth=2.0)
        if 'val/loss_ema' in step_df.columns:
            ax1.plot(step_df['step'], step_df['val/loss_ema'], label='Val Loss (EMA)', color='#ff7f0e', linestyle='--', linewidth=2.0)
        ax1.set_ylabel('Loss', fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend(loc='best')
        ax1.set_title('(A) Overall Training & Validation Loss', fontweight='bold')

        # Panel 2: Atmospheric State Losses (P, T, Q)
        ax2 = axes[0, 1]
        if 'train/loss_T_ema' in step_df.columns:
            ax2.plot(step_df['step'], step_df['train/loss_T_ema'], label='Temperature (T)', color='#2ca02c', linewidth=1.8)
        if 'train/loss_P_ema' in step_df.columns:
            ax2.plot(step_df['step'], step_df['train/loss_P_ema'], label='Pressure (P)', color='#d62728', linewidth=1.8)
        if 'train/loss_Q_ema' in step_df.columns:
            ax2.plot(step_df['step'], step_df['train/loss_Q_ema'], label='Humidity (Q)', color='#9467bd', linewidth=1.8)
        ax2.set_ylabel('Weighted MSE', fontsize=11)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend(loc='best')
        ax2.set_title('(B) Atmospheric Scalar Loss Components (P, T, Q)', fontweight='bold')

        # Panel 3: Momentum Vector Losses (U, V)
        ax3 = axes[1, 0]
        if 'train/loss_U_ema' in step_df.columns:
            ax3.plot(step_df['step'], step_df['train/loss_U_ema'], label='Zonal Wind (U)', color='#17becf', linewidth=1.8)
        if 'train/loss_V_ema' in step_df.columns:
            ax3.plot(step_df['step'], step_df['train/loss_V_ema'], label='Meridional Wind (V)', color='#8c564b', linewidth=1.8)
        ax3.set_xlabel('Training Steps', fontsize=11)
        ax3.set_ylabel('Weighted MSE', fontsize=11)
        ax3.grid(True, linestyle='--', alpha=0.5)
        ax3.legend(loc='best')
        ax3.set_title('(C) Wind Vector Loss Components (U, V)', fontweight='bold')

        # Panel 4: Orthogonal Zero-Mean Drift Penalty
        ax4 = axes[1, 1]
        if 'train/loss_zero_mean_ema' in step_df.columns:
            ax4.plot(step_df['step'], step_df['train/loss_zero_mean_ema'], label='Zero-Mean Anomaly Penalty', color='#e377c2', linewidth=2.0)
        ax4.set_xlabel('Training Steps', fontsize=11)
        ax4.set_ylabel('Penalty', fontsize=11)
        ax4.grid(True, linestyle='--', alpha=0.5)
        ax4.legend(loc='best')
        ax4.set_title('(D) Global Mean Anomaly Drift Penalty', fontweight='bold')

        plt.tight_layout()
        plt.savefig(output_img, bbox_inches='tight', dpi=150)
        print(f"[SUCCESS] Loss breakdown chart saved to: {output_img}")

        if show_plot:
            plt.show()
        plt.close()


def main():
    parser = argparse.ArgumentParser(description="Analyze loss function breakdown from PyTorch Lightning CSV metrics.")
    parser.add_argument("-i", "--input", default=None, help="Path to metrics.csv")
    parser.add_argument("-o", "--output", default="loss_analysis.png", help="Output chart filename.")
    parser.add_argument("--no-show", action="store_true", help="Do not display interactive pop-up plot.")
    args = parser.parse_args()

    try:
        plotter = WeatherLossPlotter(metrics_path=args.input)
        plotter.generate_chart(output_img=args.output, show_plot=not args.no_show)
    except Exception as e:
        print(f"[ERROR] {e}")


if __name__ == "__main__":
    main()
