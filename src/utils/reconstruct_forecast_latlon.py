#!/usr/bin/env python3
"""
Cartesian 3D Spherical Regridder with Terrain-Following Vertical Interpolation.
1. Regrids M6 icosahedral mesh forecasts (40,962 nodes) to regular Lat-Lon grids using Gaussian 4-NN.
2. Vertically interpolates dynamic terrain-following pressure layers to standard isobaric levels.
"""

import os
import glob
import argparse
import numpy as np
import xarray as xr
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d


def extract_node_coords(ds):
    """Extracts 1D node latitude and longitude arrays from an icosahedral dataset."""
    for lat_key in ["latitude", "lat", "lat_icosahedral", "lats"]:
        if lat_key in ds.coords or lat_key in ds.data_vars:
            lats = np.ravel(np.asarray(ds[lat_key].values, dtype=np.float32))
            break
    else:
        raise KeyError("Could not find latitude variable in icosahedral dataset.")

    for lon_key in ["longitude", "lon", "lon_icosahedral", "lons"]:
        if lon_key in ds.coords or lon_key in ds.data_vars:
            lons = np.ravel(np.asarray(ds[lon_key].values, dtype=np.float32))
            break
    else:
        raise KeyError("Could not find longitude variable in icosahedral dataset.")

    # Convert radians to degrees if necessary
    if np.abs(lats).max() <= 1.58:
        lats = np.degrees(lats)
    if np.abs(lons).max() <= 3.15 and np.min(lons) < 0:
        lons = np.degrees(lons)

    lons = np.mod(lons, 360.0)
    return lats, lons


def extract_target_grid(ds):
    """Extracts 1D latitude and longitude coordinate axes for the output regular grid."""
    for lat_key in ["latitude", "lat"]:
        if lat_key in ds.coords or lat_key in ds.data_vars:
            lats = np.unique(np.ravel(ds[lat_key].values))
            break
    else:
        raise KeyError("Could not find latitude variable in target grid dataset.")

    for lon_key in ["longitude", "lon"]:
        if lon_key in ds.coords or lon_key in ds.data_vars:
            lons = np.unique(np.ravel(ds[lon_key].values))
            break
    else:
        raise KeyError("Could not find longitude variable in target grid dataset.")

    lons = np.mod(lons, 360.0)
    return np.sort(lats), np.sort(lons)


def terrain_to_isobaric(field_3d, p_terrain_3d, target_pressures):
    """
    Vertically interpolates 3D fields from terrain-following pressure layers to standard isobaric levels.
    Deduplicates identical pressure coordinates per column to eliminate divide-by-zero warnings and NaNs.
    """
    num_levels, num_nodes = field_3d.shape
    out_isobaric = np.zeros((len(target_pressures), num_nodes), dtype=np.float32)

    for i in range(num_nodes):
        p_col = np.nan_to_num(p_terrain_3d[:, i], nan=1000.0)
        v_col = np.nan_to_num(field_3d[:, i], nan=0.0)

        # Sort column in ascending order by pressure
        sort_idx = np.argsort(p_col)
        p_sorted = p_col[sort_idx]
        v_sorted = v_col[sort_idx]

        # Deduplicate identical pressure levels to prevent x_hi - x_lo = 0
        p_unique, unique_idx = np.unique(p_sorted, return_index=True)
        v_unique = v_sorted[unique_idx]

        if len(p_unique) >= 2:
            f_interp = interp1d(
                p_unique,
                v_unique,
                bounds_error=False,
                fill_value="extrapolate",
                assume_sorted=True,
            )
            out_isobaric[:, i] = f_interp(target_pressures)
        else:
            # Fallback for degenerate single-level columns
            out_isobaric[:, i] = v_col[0]

    return out_isobaric


def regrid_forecasts(fcst_dir, truth_ref, grid_ref, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Icosahedral Mesh Node Coordinates
    print(f"[REGRID] Loading icosahedral mesh reference: {grid_ref}")
    ds_grid = xr.open_dataset(grid_ref)
    node_lats, node_lons = extract_node_coords(ds_grid)
    num_nodes = len(node_lats)
    ds_grid.close()

    print(f"[REGRID] Found {num_nodes} icosahedral mesh nodes.")

    # Convert icosahedral node lats/lons to 3D Cartesian coordinates
    rad_lat = np.radians(node_lats)
    rad_lon = np.radians(node_lons)
    x_nodes = np.cos(rad_lat) * np.cos(rad_lon)
    y_nodes = np.cos(rad_lat) * np.sin(rad_lon)
    z_nodes = np.sin(rad_lat)
    node_pts = np.column_stack((x_nodes, y_nodes, z_nodes))

    tree = cKDTree(node_pts)

    # 2. Extract Regular Output Grid Coordinates
    print(f"[REGRID] Loading target grid reference: {truth_ref}")
    ds_truth = xr.open_dataset(truth_ref)
    target_lats, target_lons = extract_target_grid(ds_truth)
    ds_truth.close()

    print(f"[REGRID] Target grid resolution: {len(target_lats)} x {len(target_lons)}")

    lon_grid, lat_grid = np.meshgrid(target_lons, target_lats)
    t_rad_lat = np.radians(lat_grid.ravel())
    t_rad_lon = np.radians(lon_grid.ravel())

    target_pts = np.column_stack(
        (
            np.cos(t_rad_lat) * np.cos(t_rad_lon),
            np.cos(t_rad_lat) * np.sin(t_rad_lon),
            np.sin(t_rad_lat),
        )
    )

    # 3. Query 4 Nearest Neighbors with Gaussian Distance Weighting (M6 sigma = 0.0125)
    dists, indices = tree.query(target_pts, k=4)
    sigma = 0.0125
    weights = np.exp(-(dists**2) / (2 * sigma**2))
    weights /= weights.sum(axis=-1, keepdims=True)

    # Standard GFS isobaric pressure levels (32 levels in hPa)
    target_pressures = np.array(
        [
            1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700, 650, 600,
            550, 500, 450, 400, 350, 300, 250, 225, 200, 175, 150, 125, 100, 70,
            50, 30, 20, 10,
        ],
        dtype=np.float32,
    )

    # 4. Process Forecast Files
    fcst_files = sorted(glob.glob(os.path.join(fcst_dir, "*.nc")))
    print(f"[REGRID] Regridding {len(fcst_files)} forecast files to lat-lon grid...")

    for fpath in fcst_files:
        fname = os.path.basename(fpath)
        ds_in = xr.open_dataset(fpath)

        # Extract 3D pressure field P(level, node) for vertical interpolation
        if "P" in ds_in:
            p_terrain_3d = np.squeeze(ds_in["P"].values)
        elif "p" in ds_in:
            p_terrain_3d = np.squeeze(ds_in["p"].values)
        else:
            p_terrain_3d = None

        out_vars = {}
        for var in ["P", "Q", "T", "U", "V", "W"]:
            if var in ds_in:
                val = np.squeeze(ds_in[var].values)  # (32, num_nodes)

                # Vertical terrain-following to isobaric interpolation if pressure is available
                if p_terrain_3d is not None and var != "P":
                    val_isobaric = terrain_to_isobaric(
                        val, p_terrain_3d, target_pressures
                    )
                else:
                    val_isobaric = val

                # Spatial Gaussian 4-NN regridding: (32, 40962) -> (32, n_lat, n_lon)
                regrid_val = np.sum(
                    val_isobaric[:, indices] * weights[None, :, :], axis=-1
                )
                out_vars[var.lower()] = (
                    ["level", "latitude", "longitude"],
                    regrid_val.reshape(
                        len(target_pressures), len(target_lats), len(target_lons)
                    ),
                )

        ds_out = xr.Dataset(
            data_vars=out_vars,
            coords={
                "level": target_pressures,
                "latitude": target_lats,
                "longitude": target_lons,
            },
            attrs=ds_in.attrs,
        )

        ds_out.to_netcdf(os.path.join(out_dir, f"reconstructed_{fname}"))
        ds_in.close()

    print(f"[SUCCESS] Regridding complete. Files saved to '{out_dir}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Regrid M6 icosahedral mesh forecasts to standard Lat-Lon grids."
    )
    parser.add_argument(
        "--fcst_dir", required=True, help="Directory containing raw forecast .nc files"
    )
    parser.add_argument(
        "--truth_ref",
        required=True,
        help="Regular Lat-Lon target reference NetCDF file",
    )
    parser.add_argument(
        "--grid_ref",
        required=True,
        help="Icosahedral M6 mesh topology reference NetCDF file",
    )
    parser.add_argument(
        "--out_dir", required=True, help="Output directory for reconstructed NetCDFs"
    )
    args = parser.parse_args()

    regrid_forecasts(args.fcst_dir, args.truth_ref, args.grid_ref, args.out_dir)


if __name__ == "__main__":
    main()
