#!/usr/bin/env python3
"""
utils/plot_forecast_panel_comparison.py
----------------------------------------
Generates a 4x7 panel plot for all 7 dynamic state variables (T, u, v, w, q, rho, p)
at a selected vertical level using grid interpolation for smooth polar rendering.
Computes and displays comprehensive diagnostic statistics: Min, Max, Mean, STD, Bias, RMSE, and ACC.
"""

import os
import logging
import argparse
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

R_D = 287.058


def compute_acc(forecast: np.ndarray, truth: np.ndarray, climatology: np.ndarray = None) -> float:
    """
    Computes the Anomaly Correlation Coefficient (ACC).
    If climatology is not provided, computes anomaly relative to spatial mean.
    """
    if climatology is None:
        f_anom = forecast - np.nanmean(forecast)
        t_anom = truth - np.nanmean(truth)
    else:
        f_anom = forecast - climatology
        t_anom = truth - climatology

    numerator = np.sum(f_anom * t_anom)
    denominator = np.sqrt(np.sum(f_anom**2) * np.sum(t_anom**2))

    if denominator == 0:
        return 0.0
    return float(numerator / denominator)


def extract_physical_fields(nc_file: str, target_level: int = 0) -> tuple[dict, np.ndarray, np.ndarray]:
    """Reads dynamic variables from NetCDF and extracts coordinates for target level."""
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"[ERROR] Input file not found: '{nc_file}'")

    ds = xr.open_dataset(nc_file)

    def get_var(candidates):
        for c in candidates:
            if c in ds or c in ds.coords:
                val = np.squeeze(ds[c].values)
                if val.ndim == 2:  # (levels, nodes)
                    idx = min(target_level, val.shape[0] - 1)
                    return val[idx]
                elif val.ndim == 1:  # (nodes,)
                    return val
        return None

    lons = get_var(['longitude', 'lon', 'lons'])
    lats = get_var(['latitude', 'lat', 'lats'])

    if lons is None or lats is None:
        raise KeyError(f"[ERROR] Missing lat/lon coordinates in '{nc_file}'")

    # 1. Temperature T (K)
    ln_t = get_var(['ln_t_icosahedral', 'ln_t', 't_icosahedral', 't', 'T'])
    if ln_t is not None:
        mean_t = np.nanmean(ln_t)
        t_k = np.exp(ln_t) if mean_t < 10.0 else ln_t
    else:
        t_k = np.full_like(lats, 273.15)

    # 2. Pressure p (hPa)
    ln_p = get_var(['ln_p_icosahedral', 'ln_p', 'p_icosahedral', 'p', 'P'])
    if ln_p is not None:
        mean_p = np.nanmean(ln_p)
        p_hpa = np.exp(ln_p) / 100.0 if mean_p < 20.0 else ln_p
        if np.nanmean(p_hpa) > 2000.0:
            p_hpa = p_hpa / 100.0
    else:
        p_hpa = np.full_like(lats, 1013.25)

    # 3. Specific Humidity q (g/kg)
    q_val = get_var(['q_icosahedral', 'q', 'Q'])
    if q_val is not None:
        q_gkg = q_val * 1000.0 if np.nanmean(q_val) < 0.1 else q_val
    else:
        q_gkg = np.zeros_like(lats)

    # 4. Zonal Wind u (m/s)
    u_ms = get_var(['u_icosahedral', 'u', 'U'])
    if u_ms is None:
        u_ms = np.zeros_like(lats)

    # 5. Meridional Wind v (m/s)
    v_ms = get_var(['v_icosahedral', 'v', 'V'])
    if v_ms is None:
        v_ms = np.zeros_like(lats)

    # 6. Vertical Velocity w (Pa/s)
    w_pas = get_var(['w_icosahedral', 'w', 'W'])
    if w_pas is None:
        w_pas = np.zeros_like(lats)

    # 7. Density rho (kg/m3)
    ln_rho = get_var(['ln_rho_icosahedral', 'ln_rho', 'rho_icosahedral', 'rho', 'RHO'])
    if ln_rho is not None:
        rho_kgm3 = np.exp(ln_rho) if np.nanmean(ln_rho) < 2.0 else ln_rho
    else:
        rho_kgm3 = (p_hpa * 100.0) / (R_D * np.maximum(t_k, 1e-3))

    fields = {
        'T': t_k,
        'U': u_ms,
        'V': v_ms,
        'W': w_pas,
        'Q': q_gkg,
        'RHO': rho_kgm3,
        'P': p_hpa
    }

    ds.close()
    return fields, lons, lats


def interpolate_to_regular_grid(lons: np.ndarray, lats: np.ndarray, data: np.ndarray, grid_lon: np.ndarray, grid_lat: np.ndarray):
    """Interpolates unstructured icosahedral node data onto a 2D regular grid."""
    lons_clean = np.where(lons > 180.0, lons - 360.0, lons)
    points = np.column_stack([lons_clean, lats])
    grid_z = griddata(points, data, (grid_lon, grid_lat), method='linear')

    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z_near = griddata(points, data, (grid_lon, grid_lat), method='nearest')
        grid_z[nan_mask] = grid_z_near[nan_mask]

    return grid_z


def print_diagnostic_statistics(var_order: list, var_units: dict, f_fields: dict, t_fields: dict, level_idx: int):
    """Prints a formatted diagnostic statistics table to the terminal."""
    header = f"{'VAR':<5} | {'TRUTH MIN':<10} | {'TRUTH MAX':<10} | {'FCST MIN':<10} | {'FCST MAX':<10} | {'BIAS (MEAN)':<10} | {'RMSE':<10} | {'ACC':<7}"
    line = "=" * 92

    print("\n" + line)
    print(f" FORECAST DIAGNOSTIC METRICS SUMMARY (Level Index: {level_idx + 1})")
    print(line)
    print(header)
    print("-" * 92)

    stats_summary = {}

    for var in var_order:
        f_nodes = f_fields[var]
        t_nodes = t_fields[var]

        t_min, t_max = np.nanmin(t_nodes), np.nanmax(t_nodes)
        f_min, f_max = np.nanmin(f_nodes), np.nanmax(f_nodes)

        bias = np.nanmean(f_nodes) - np.nanmean(t_nodes)
        rmse = np.sqrt(np.nanmean((f_nodes - t_nodes) ** 2))
        acc = compute_acc(f_nodes, t_nodes)

        stats_summary[var] = {
            'rmse': rmse,
            'acc': acc,
            'bias': bias
        }

        unit_str = f"({var_units[var]})"
        print(f"{var:<5} | {t_min:<10.4f} | {t_max:<10.4f} | {f_min:<10.4f} | {f_max:<10.4f} | {bias:<10.4f} | {rmse:<10.4f} | {acc:<7.4f}")

    print(line + "\n")
    return stats_summary


def make_comparison_panel_plot(
    forecast_nc: str,
    truth_nc: str,
    x0_nc: str = None,
    level_idx: int = 0,
    output_png: str = "forecast_diagnostic_panel.png",
    show: bool = False
):
    print(f"\n[PANEL PLOTTER] Loading Forecast : '{forecast_nc}'")
    print(f"[PANEL PLOTTER] Loading Truth    : '{truth_nc}'")

    f_fields, lons, lats = extract_physical_fields(forecast_nc, target_level=level_idx)
    t_fields, _, _ = extract_physical_fields(truth_nc, target_level=level_idx)

    if x0_nc and os.path.exists(x0_nc):
        print(f"[PANEL PLOTTER] Loading Initial X0: '{x0_nc}'")
        x0_fields, _, _ = extract_physical_fields(x0_nc, target_level=level_idx)
    else:
        x0_fields = None

    var_order = ['T', 'U', 'V', 'W', 'Q', 'RHO', 'P']
    var_units = {'T': 'K', 'U': 'm/s', 'V': 'm/s', 'W': 'Pa/s', 'Q': 'g/kg', 'RHO': 'kg/m³', 'P': 'hPa'}
    var_titles = {
        'T': 'Temp T (K)',
        'U': 'Zonal Wind U (m/s)',
        'V': 'Merid Wind V (m/s)',
        'W': 'Vert Vel W (Pa/s)',
        'Q': 'Humidity Q (g/kg)',
        'RHO': 'Density ρ (kg/m³)',
        'P': 'Pressure P (hPa)'
    }

    # Print summary statistics table to stdout
    stats_summary = print_diagnostic_statistics(var_order, var_units, f_fields, t_fields, level_idx)

    reg_lon = np.linspace(-180, 180, 360)
    reg_lat = np.linspace(-90, 90, 180)
    grid_lon, grid_lat = np.meshgrid(reg_lon, reg_lat)

    fig = plt.figure(figsize=(28, 14))
    proj = ccrs.PlateCarree()

    rows = 4
    cols = 7

    for col_idx, var_name in enumerate(var_order):
        f_nodes = f_fields[var_name]
        t_nodes = t_fields[var_name]

        if x0_fields is not None:
            x0_nodes = x0_fields[var_name]
            delt_total_nodes = f_nodes - x0_nodes
        else:
            delt_total_nodes = f_nodes - t_nodes

        delt_error_nodes = f_nodes - t_nodes

        f_grid = interpolate_to_regular_grid(lons, lats, f_nodes, grid_lon, grid_lat)
        t_grid = interpolate_to_regular_grid(lons, lats, t_nodes, grid_lon, grid_lat)
        delt_total_grid = interpolate_to_regular_grid(lons, lats, delt_total_nodes, grid_lon, grid_lat)
        delt_error_grid = interpolate_to_regular_grid(lons, lats, delt_error_nodes, grid_lon, grid_lat)

        rmse_val = stats_summary[var_name]['rmse']
        acc_val = stats_summary[var_name]['acc']

        row_data = [
            (t_grid, f"Truth: {var_titles[var_name]}", 'viridis', False),
            (f_grid, f"Fcst: {var_titles[var_name]}\n(ACC: {acc_val:.3f})", 'viridis', False),
            (delt_total_grid, f"ΔX (Fcst - X0): {var_name}", 'coolwarm', True),
            (delt_error_grid, f"Error (Fcst - Truth): {var_name}\n(RMSE: {rmse_val:.3f} {var_units[var_name]})", 'coolwarm', True)
        ]

        if var_name == 'T':
            vmin_state, vmax_state, vlim_total, vlim_error = 190.0, 320.0, 20.0, 20.0
        elif var_name == 'P':
            vmin_state, vmax_state, vlim_total, vlim_error = 850.0, 1000.0, 50.0, 50.0
        elif var_name == 'U':
            vmin_state, vmax_state, vlim_total, vlim_error = -50.0, 50.0, 20.0, 20.0
        elif var_name == 'V':
            vmin_state, vmax_state, vlim_total, vlim_error = -50.0, 50.0, 20.0, 20.0
        elif var_name == 'W':
            vmin_state, vmax_state, vlim_total, vlim_error = -5.0, 5.0, 2.0, 2.0
        elif var_name == 'Q':
            vmin_state, vmax_state, vlim_total, vlim_error = 0.0, 15.0, 5.0, 5.0
        elif var_name == 'RHO':
            vmin_state, vmax_state, vlim_total, vlim_error = 0.75, 1.5, 0.25, 0.25

        for row_idx, (data_grid, title, cmap, is_diff) in enumerate(row_data):
            ax = fig.add_subplot(rows, cols, row_idx * cols + col_idx + 1, projection=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black', alpha=0.7, facecolor='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5)

            if is_diff:
                vlim = vlim_total if row_idx == 2 else vlim_error
                sc = ax.pcolormesh(grid_lon, grid_lat, data_grid, cmap=cmap, vmin=-vlim, vmax=vlim, shading='auto', transform=proj)
            else:
                sc = ax.pcolormesh(grid_lon, grid_lat, data_grid, cmap=cmap, vmin=vmin_state, vmax=vmax_state, shading='auto', transform=proj)

            ax.set_title(title, fontsize=8.5, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85)
            cbar.ax.tick_params(labelsize=7)

    plt.suptitle(
        f"AIDA GNN Multi-Variable Forecast Diagnostic Panel | Level Index: {level_idx + 1} (1=Surface, 32=Top)",
        fontsize=16, fontweight='bold', y=0.99
    )

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_png, dpi=250, bbox_inches='tight')
    print(f"[SUCCESS] Multi-panel diagnostic comparison saved to: '{output_png}'\n")
    if show:
        plt.show()
    plt.close()
    logging.info(f"Plot saved to: {output_png}")


def main():
    parser = argparse.ArgumentParser(description="Multi-Variable 4x7 Diagnostic Panel Plotter for AIDA GNN Forecasts")
    parser.add_argument("-f", "--forecast", required=True, help="Path to forecast NetCDF file")
    parser.add_argument("-t", "--truth", required=True, help="Path to ground truth NetCDF file")
    parser.add_argument("-z", "--x0", help="Path to initial state X0 NetCDF file (for computing Total Increment ΔX)")
    parser.add_argument("-l", "--level", type=int, default=0, help="Vertical level index to plot (0=Surface, 31=Top)")
    parser.add_argument("-o", "--output", default="aida_forecast_comparison_panel.png", help="Destination PNG plot path")
    parser.add_argument("-s", "--show", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    make_comparison_panel_plot(
        forecast_nc=args.forecast,
        truth_nc=args.truth,
        x0_nc=args.x0,
        level_idx=args.level,
        output_png=args.output,
        show=args.show
    )


if __name__ == "__main__":
    main()
