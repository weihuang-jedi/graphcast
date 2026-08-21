#!/usr/bin/env python3
"""
utils/plot_forecast_panel_comparison.py
----------------------------------------
Generates a 4x7 panel plot for all 7 dynamic state variables (T, u, v, w, q, rho, p)
at a selected vertical level using grid interpolation for smooth polar rendering.

Panels:
  Row 1: Truth (X_truth)
  Row 2: Forecast (X_forecast)
  Row 3: Total Increment ΔX_total (X_forecast - X_0)
  Row 4: Forecast Error (X_forecast - X_truth)
"""

import argparse
import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from scipy.interpolate import griddata

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="cartopy")

R_D = 287.058


def extract_physical_fields(nc_file: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Reads dynamic variables from NetCDF and extracts coordinates."""
    if not os.path.exists(nc_file):
        raise FileNotFoundError(f"[ERROR] Input file not found: '{nc_file}'")

    ds = xr.open_dataset(nc_file)

    def get_var(candidates):
        for c in candidates:
            if c in ds:
                val = ds[c].values
                if val.ndim == 3:
                    val = val[0]
                return val
        return None

    lons = get_var(['longitude', 'lon'])
    lats = get_var(['latitude', 'lat'])

    if lons is None or lats is None:
        raise KeyError(f"[ERROR] Missing lat/lon coordinates in '{nc_file}'")

    # 1. Temperature T (K)
    ln_t = get_var(['ln_t_icosahedral', 'ln_t', 't_icosahedral', 't'])
    t_k = np.exp(ln_t) if np.nanmean(ln_t) < 10.0 else ln_t

    # 2. Pressure p (hPa)
    ln_p = get_var(['ln_p_icosahedral', 'ln_p', 'p_icosahedral', 'p'])
    p_hpa = np.exp(ln_p) / 100.0 if np.nanmean(ln_p) < 20.0 else ln_p
    if np.nanmean(p_hpa) > 2000.0:
        p_hpa = p_hpa / 100.0

    # 3. Specific Humidity q (g/kg)
    q_val = get_var(['q_icosahedral', 'q'])
    q_gkg = q_val * 1000.0 if np.nanmean(q_val) < 0.1 else q_val

    # 4. Zonal Wind u (m/s)
    u_ms = get_var(['u_icosahedral', 'u'])

    # 5. Meridional Wind v (m/s)
    v_ms = get_var(['v_icosahedral', 'v'])

    # 6. Vertical Velocity w (Pa/s)
    w_pas = get_var(['w_icosahedral', 'w'])

    # 7. Density rho (kg/m3)
    ln_rho = get_var(['ln_rho_icosahedral', 'ln_rho', 'rho_icosahedral', 'rho'])
    if ln_rho is not None:
        rho_kgm3 = np.exp(ln_rho) if np.nanmean(ln_rho) < 2.0 else ln_rho
    else:
        rho_kgm3 = (p_hpa * 100.0) / (R_D * t_k)

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
    
    # Fill remaining polar boundary NaNs with nearest neighbor
    nan_mask = np.isnan(grid_z)
    if np.any(nan_mask):
        grid_z_near = griddata(points, data, (grid_lon, grid_lat), method='nearest')
        grid_z[nan_mask] = grid_z_near[nan_mask]
        
    return grid_z


def make_comparison_panel_plot(
    forecast_nc: str,
    truth_nc: str,
    x0_nc: str = None,
    level_idx: int = 0,
    output_png: str = "forecast_diagnostic_panel.png"
):
    print(f"\n[PANEL PLOTTER] Loading Forecast : '{forecast_nc}'")
    print(f"[PANEL PLOTTER] Loading Truth    : '{truth_nc}'")

    f_fields, lons, lats = extract_physical_fields(forecast_nc)
    t_fields, _, _ = extract_physical_fields(truth_nc)

    if x0_nc and os.path.exists(x0_nc):
        print(f"[PANEL PLOTTER] Loading Initial X0: '{x0_nc}'")
        x0_fields, _, _ = extract_physical_fields(x0_nc)
    else:
        x0_fields = None

    # Target regular interpolation grid (1.0 degree resolution)
    reg_lon = np.linspace(-180, 180, 360)
    reg_lat = np.linspace(-90, 90, 180)
    grid_lon, grid_lat = np.meshgrid(reg_lon, reg_lat)

    var_order = ['T', 'U', 'V', 'W', 'Q', 'RHO', 'P']
    var_titles = {
        'T': 'Temp T (K)',
        'U': 'Zonal Wind U (m/s)',
        'V': 'Merid Wind V (m/s)',
        'W': 'Vert Vel W (Pa/s)',
        'Q': 'Humidity Q (g/kg)',
        'RHO': 'Density ρ (kg/m³)',
        'P': 'Pressure P (hPa)'
    }

    fig = plt.figure(figsize=(28, 14))
    proj = ccrs.PlateCarree()

    rows = 4
    cols = 7

    for col_idx, var_name in enumerate(var_order):
        # Extract 2D node slices at selected level
        f_nodes = f_fields[var_name][level_idx]
        t_nodes = t_fields[var_name][level_idx]

        # Compute Deltas on nodes first
        if x0_fields is not None:
            x0_nodes = x0_fields[var_name][level_idx]
            delt_total_nodes = f_nodes - x0_nodes
        else:
            delt_total_nodes = f_nodes - t_nodes

        delt_error_nodes = f_nodes - t_nodes

        # Interpolate 2D fields for continuous pcolormesh rendering
        f_grid = interpolate_to_regular_grid(lons, lats, f_nodes, grid_lon, grid_lat)
        t_grid = interpolate_to_regular_grid(lons, lats, t_nodes, grid_lon, grid_lat)
        delt_total_grid = interpolate_to_regular_grid(lons, lats, delt_total_nodes, grid_lon, grid_lat)
        delt_error_grid = interpolate_to_regular_grid(lons, lats, delt_error_nodes, grid_lon, grid_lat)

        row_data = [
            (t_grid, f"Truth: {var_titles[var_name]}", 'viridis', False),
            (f_grid, f"Forecast: {var_titles[var_name]}", 'viridis', False),
            (delt_total_grid, f"ΔX (Fcst - X0): {var_name}", 'coolwarm', True),
            (delt_error_grid, f"Error (Fcst - Truth): {var_name}", 'coolwarm', True)
        ]

        vmin_state = min(np.nanmin(t_grid), np.nanmin(f_grid))
        vmax_state = max(np.nanmax(t_grid), np.nanmax(f_grid))

        vlim_total = max(abs(np.nanmin(delt_total_grid)), abs(np.nanmax(delt_total_grid))) or 1e-3
        vlim_error = max(abs(np.nanmin(delt_error_grid)), abs(np.nanmax(delt_error_grid))) or 1e-3

        print(f'var name: {var_name}, min: {vmin_state}, max: {vmax_state}')
        print(f'var name: {var_name}, vlim_total: {vlim_total}, vlim_error: {vlim_error}')

        if var_name == 'T':
            vmin_state = 190.0
            vmax_state = 320.0
            vlim_total = 20.0
            vlim_error = 20.0
        elif var_name == 'P':
            vmin_state = 850.0
            vmax_state = 1000.0
            vlim_total = 50.0
            vlim_error = 50.0
        elif var_name == 'U':
            vmin_state = -50.0
            vmax_state =  50.0
            vlim_total = 20.0
            vlim_error = 20.0
        elif var_name == 'V':
            vmin_state = -50.0
            vmax_state =  50.0
            vlim_total = 20.0
            vlim_error = 20.0
        elif var_name == 'W':
            vmin_state = -5.0
            vmax_state =  5.0
            vlim_total = 2.0
            vlim_error = 2.0
        elif var_name == 'Q':
            vmin_state =  0.0
            vmax_state =  5.0
            vlim_total = 5.0
            vlim_error = 5.0
        elif var_name == 'RHO':
            vmin_state =  0.75
            vmax_state =  1.5
            vlim_total = 0.25
            vlim_error = 0.25

        # print(f'\t\tvar name: {var_name}, new min: {vmin_state}, new max: {vmax_state}')
        # print(f'\t\tvar name: {var_name}, new vlim_total: {vlim_total}, new vlim_error: {vlim_error}')

        for row_idx, (data_grid, title, cmap, is_diff) in enumerate(row_data):
            ax = fig.add_subplot(rows, cols, row_idx * cols + col_idx + 1, projection=proj)
            ax.add_feature(cfeature.COASTLINE, linewidth=0.6, color='black', alpha=0.7, facecolor='gray')
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, color='gray', alpha=0.5)

            if is_diff:
                vlim = vlim_total if row_idx == 2 else vlim_error
                sc = ax.pcolormesh(grid_lon, grid_lat, data_grid, cmap=cmap, vmin=-vlim, vmax=vlim, shading='auto', transform=proj)
            else:
                sc = ax.pcolormesh(grid_lon, grid_lat, data_grid, cmap=cmap, vmin=vmin_state, vmax=vmax_state, shading='auto', transform=proj)

            ax.set_title(title, fontsize=9, fontweight='bold')

            cbar = plt.colorbar(sc, ax=ax, orientation='horizontal', pad=0.04, shrink=0.85)
            cbar.ax.tick_params(labelsize=7)

    plt.suptitle(
        f"AIDA GNN Multi-Variable Forecast Diagnostic Panel | Level Index: {level_idx + 1} (1=Surface, 32=Top)",
        fontsize=16, fontweight='bold', y=0.99
    )

    os.makedirs(os.path.dirname(output_png) or ".", exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(output_png, dpi=250, bbox_inches='tight')
    plt.show()
    plt.close()
    print(f"[SUCCESS] Multi-panel diagnostic comparison saved to: '{output_png}'\n")


def main():
    parser = argparse.ArgumentParser(description="Multi-Variable 4x7 Diagnostic Panel Plotter for AIDA GNN Forecasts")
    parser.add_argument("-f", "--forecast", required=True, help="Path to forecast NetCDF file")
    parser.add_argument("-t", "--truth", required=True, help="Path to ground truth NetCDF file")
    parser.add_argument("-z", "--x0", help="Path to initial state X0 NetCDF file (for computing Total Increment ΔX)")
    parser.add_argument("-l", "--level", type=int, default=0, help="Vertical level index to plot (0=Surface, 31=Top)")
    parser.add_argument("-o", "--output", default="aida_forecast_comparison_panel.png", help="Destination PNG plot path")

    args = parser.parse_args()

    make_comparison_panel_plot(
        forecast_nc=args.forecast,
        truth_nc=args.truth,
        x0_nc=args.x0,
        level_idx=args.level,
        output_png=args.output
    )


if __name__ == "__main__":
    main()
