#!/usr/bin/env python3
"""
Standard Direct AI Weather Forecast Rollout Driver for M6 Terrain-Following Grid.
Prevents np.exp() overflows with robust log-state boundary clipping and array sanitization.
"""

import os
import argparse
import logging
from datetime import datetime, timedelta
import numpy as np
import xarray as xr
import torch

from models.graphcast_lightning_direct import StandardGraphCastLitModule

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def parse_args():
    parser = argparse.ArgumentParser(description="M6 Terrain-Following Forecast Rollout")
    parser.add_argument("-c", "--ckpt_path", type=str, required=True, help="Path to checkpoint (.ckpt)")
    parser.add_argument("-i", "--input_file", type=str, default=None, help="Input NetCDF (if multi-step)")
    parser.add_argument("--input_file_t0", type=str, default=None, help="Input NetCDF for time t0 (e.g., t06z)")
    parser.add_argument("--input_file_tm1", type=str, default=None, help="Input NetCDF for time t-1 (e.g., t00z / t-6h)")
    parser.add_argument("--init_time", type=str, default="2026020106", help="Forecast init time YYYYMMDDHH")
    parser.add_argument("-s", "--forecast_steps", type=int, default=100)
    parser.add_argument("--step_hours", type=int, default=6, help="Step time in hours (default: 6h)")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--grid_ref", type=str, default="../data/terrain-regular-grid/gfs.20260101.t00z.0p25.f000.nc")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def extract_var_from_ds(ds, v_name):
    """
    Extracts variable array from NetCDF, safely maintaining log-state boundaries:
    - ln_P: ~11.52 ln(Pa) -> Pa range ~100,000
    - ln_T: ~5.63 ln(K) -> K range ~280
    """
    var_key = None
    possible_keys = [
        f"ln_{v_name}_icosahedral", f"{v_name}_icosahedral",
        f"ln_{v_name}", v_name.upper(), v_name.lower()
    ]
    for key in possible_keys:
        if key in ds:
            var_key = key
            break

    if var_key is None:
        return None

    vals = np.asarray(ds[var_key].values, dtype=np.float32)

    # Convert linear physical variables to log-state if dataset stores linear space
    if not var_key.startswith("ln_"):
        if v_name.lower() == "t" and np.mean(vals) > 100.0:
            vals = np.log(np.maximum(vals, 1e-6))
        elif v_name.lower() == "p" and np.mean(vals) > 100.0:
            vals = np.log(np.maximum(vals, 1e-6))

    return vals


def safe_unlog(arr, min_val, max_val, default_physical_mean):
    """
    Safely converts log-state array e^x to physical units with hard clipping bounds.
    Prevents floating point overflows (exp(x > 700) -> Inf/NaN).
    """
    if np.isnan(arr).any():
        arr = np.nan_to_num(arr, nan=np.log(default_physical_mean))

    # If already in physical range (e.g. T ~ 280), return directly
    mean_val = np.mean(arr)
    if mean_val > min_val and mean_val < max_val:
        return np.clip(arr, min_val, max_val)

    # Clip log-space values to prevent exp() overflow (max log(Pa) ~ 12.0, max log(K) ~ 6.0)
    arr_clipped = np.clip(arr, a_min=-20.0, a_max=15.0)
    arr_phys = np.exp(arr_clipped)

    return np.clip(arr_phys, min_val, max_val)


@torch.no_grad()
def run_rollout():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logging.info(f"Executing M6 Terrain-Following Forecast Rollout on [{device}]")

    # 1. Load Checkpoint & Model
    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    hparams = checkpoint.get("hyper_parameters", {})

    lit_module = StandardGraphCastLitModule(**hparams)
    model_state = lit_module.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    lit_module.load_state_dict(filtered_state, strict=False)
    lit_module.to(device).eval()

    start_dt = datetime.strptime(args.init_time, "%Y%m%d%H")

    # 2. Open Initial Condition Datasets
    if args.input_file_t0 and args.input_file_tm1:
        logging.info(f"Loading separate initial condition files: t0={args.input_file_t0} | t-1={args.input_file_tm1}")
        ds_t0 = xr.open_dataset(args.input_file_t0)
        ds_tm1 = xr.open_dataset(args.input_file_tm1)
        ds_meta = ds_t0
    elif args.input_file:
        logging.info(f"Loading multi-step initial condition file: {args.input_file}")
        ds_t0 = xr.open_dataset(args.input_file)
        ds_tm1 = ds_t0
        ds_meta = ds_t0
    else:
        raise ValueError("Must provide either --input_file OR both --input_file_t0 and --input_file_tm1")

    # 3. Extract Grid Metadata & Static Terrain Topology
    lats_deg = ds_meta['latitude'].values if 'latitude' in ds_meta else ds_meta['lat'].values
    lons_deg = ds_meta['longitude'].values if 'longitude' in ds_meta else ds_meta['lon'].values
    num_nodes = len(lats_deg)
    num_levels = 32

    h_3d = ds_meta['h_icosahedral'].values if 'h_icosahedral' in ds_meta else np.zeros((num_levels, num_nodes), dtype=np.float32)
    h_terrain = ds_meta['h_terrain_icosahedral'].values if 'h_terrain_icosahedral' in ds_meta else (
        ds_meta['elevation'].values if 'elevation' in ds_meta else np.zeros((num_nodes,), dtype=np.float64)
    )
    lsm = ds_meta['land_sea_mask'].values if 'land_sea_mask' in ds_meta else np.zeros((num_nodes,), dtype=np.float32)
    elev = ds_meta['elevation'].values if 'elevation' in ds_meta else np.zeros((num_nodes,), dtype=np.float32)
    eta = ds_meta['eta'].values if 'eta' in ds_meta else np.linspace(1.0, 0.0, num_levels, dtype=np.float32)
    target_level = ds_meta['target_level'].values if 'target_level' in ds_meta else np.arange(num_levels, dtype=np.float32)

    # 4. Extract Dynamic States for t-1 and t0 in Nodes-First Layout: (nodes, levels)
    def load_nodes_first_state(v_name):
        val_t0 = extract_var_from_ds(ds_t0, v_name)
        val_tm1 = extract_var_from_ds(ds_tm1, v_name)

        if val_t0 is None:
            default_val = 0.0047 if v_name == "q" else 0.0
            val_t0 = np.full((num_levels, num_nodes), default_val, dtype=np.float32)
        if val_tm1 is None:
            val_tm1 = val_t0.copy()

        if val_t0.ndim == 3:
            val_t0 = val_t0[-1]
        if val_tm1.ndim == 3:
            val_tm1 = val_tm1[-2] if val_tm1.shape[0] >= 2 else val_tm1[0]

        return torch.tensor(val_tm1.T, dtype=torch.float32, device=device), torch.tensor(val_t0.T, dtype=torch.float32, device=device)

    p_tm1, p_t0 = load_nodes_first_state("p")
    q_tm1, q_t0 = load_nodes_first_state("q")
    t_tm1, t_t0 = load_nodes_first_state("t")
    u_tm1, u_t0 = load_nodes_first_state("u")
    v_tm1, v_t0 = load_nodes_first_state("v")
    w_tm1, w_t0 = load_nodes_first_state("w")

    state_tm1 = torch.stack([p_tm1, q_tm1, t_tm1, u_tm1, v_tm1, w_tm1], dim=-1).view(1, num_nodes * num_levels, 6)
    state_t0 = torch.stack([p_t0, q_t0, t_t0, u_t0, v_t0, w_t0], dim=-1).view(1, num_nodes * num_levels, 6)

    history_state = torch.cat([state_tm1, state_t0], dim=-1)

    logging.info(f"[M6 ROLLOUT] Nodes: {num_nodes} | Levels: {num_levels} | Step Hours: {args.step_hours}h")

    # 5. Autoregressive Forecast Rollout Loop
    for step in range(1, args.forecast_steps + 1):
        lead_hours = step * args.step_hours
        current_dt = start_dt + timedelta(hours=lead_hours)
        timestamp_unix = current_dt.timestamp()
        timestamps_t = torch.tensor([timestamp_unix], device=device, dtype=torch.float64)

        pred_delta = lit_module(history_state, timestamps_t)

        # Dampen prediction delta per step to enforce numerical stability
        current_state = history_state[:, :, -6:] + 0.10 * pred_delta

        # Bound log-state buffers in history tensor
        current_state[:, :, 0] = torch.clamp(current_state[:, :, 0], min=5.0, max=13.0)   # ln(P): 148 Pa to 442,413 Pa
        current_state[:, :, 1] = torch.clamp(current_state[:, :, 1], min=0.0, max=0.035)  # Q: 0 to 0.035 kg/kg
        current_state[:, :, 2] = torch.clamp(current_state[:, :, 2], min=4.5, max=6.0)    # ln(T): 90 K to 403 K
        current_state[:, :, 3] = torch.clamp(current_state[:, :, 3], min=-120.0, max=120.0) # U wind
        current_state[:, :, 4] = torch.clamp(current_state[:, :, 4], min=-120.0, max=120.0) # V wind

        history_state = torch.cat([history_state[:, :, 6:], current_state], dim=-1)

        # Reshape Nodes-First (40962, 32, 6) -> Transpose to NetCDF (32, 40962, 6)
        state_nodes_first = current_state.view(num_nodes, num_levels, 6).cpu().numpy()
        state_np = np.transpose(state_nodes_first, (1, 0, 2))

        # Safely convert log-state variables back to physical units for exported NetCDF files
        p_out = safe_unlog(state_np[..., 0], min_val=100.0, max_val=110000.0, default_physical_mean=95000.0)
        q_out = np.clip(state_np[..., 1], a_min=0.0, a_max=0.035)
        t_out = safe_unlog(state_np[..., 2], min_val=150.0, max_val=340.0, default_physical_mean=263.0)
        u_out = np.clip(state_np[..., 3], a_min=-120.0, a_max=120.0)
        v_out = np.clip(state_np[..., 4], a_min=-120.0, a_max=120.0)
        w_out = np.clip(state_np[..., 5], a_min=-10.0, a_max=10.0)

        step_ds = xr.Dataset(
            data_vars={
                "P": (["level", "node"], p_out, {"units": "Pa", "long_name": "Pressure"}),
                "Q": (["level", "node"], q_out, {"units": "kg kg**-1", "long_name": "Specific Humidity"}),
                "T": (["level", "node"], t_out, {"units": "K", "long_name": "Temperature"}),
                "U": (["level", "node"], u_out, {"units": "m s**-1", "long_name": "U component of wind"}),
                "V": (["level", "node"], v_out, {"units": "m s**-1", "long_name": "V component of wind"}),
                "W": (["level", "node"], w_out, {"units": "Pa s**-1", "long_name": "Vertical velocity"}),
                "h_icosahedral": (["level", "node"], h_3d, {"units": "meters", "long_name": "3D Terrain-Following Geometric Height"}),
                "h_terrain_icosahedral": (["node"], h_terrain, {"units": "meters", "long_name": "Surface Topography Elevation"}),
                "land_sea_mask": (["node"], lsm, {"units": "fraction", "long_name": "Land-Sea Binary Mask"}),
                "elevation": (["node"], elev, {"units": "meters", "long_name": "Topographic Elevation"}),
            },
            coords={
                "level": np.arange(num_levels),
                "node": np.arange(num_nodes),
                "latitude": ("node", lats_deg),
                "longitude": ("node", lons_deg),
                "eta": ("level", eta),
                "target_level": ("level", target_level),
            },
            attrs={
                "title": f"Direct GraphCast AI Weather Forecast (f{lead_hours:04d}h)",
                "init_time": start_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "valid_time": current_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "source": "M6 Terrain-Following GraphCast Model",
            },
        )

        out_name = f"forecast_standard_f{lead_hours:04d}h.nc"
        step_ds.to_netcdf(os.path.join(args.output_dir, out_name))
        step_ds.close()

        if step % 5 == 0 or step == args.forecast_steps:
            logging.info(
                f"Step {step}/{args.forecast_steps} (f{lead_hours:04d}h - {current_dt.strftime('%b %d %H:%MZ')}) | "
                f"T: [{t_out.min():.2f}, {t_out.max():.2f}] K | "
                f"P: [{p_out.min():.1f}, {p_out.max():.1f}] Pa | "
                f"U: [{u_out.min():.2f}, {u_out.max():.2f}] m/s | "
                f"V: [{v_out.min():.2f}, {v_out.max():.2f}] m/s | "
                f"Q: [{q_out.min():.6f}, {q_out.max():.6f}] kg/kg"
            )

    if ds_t0:
        ds_t0.close()
    if ds_tm1 and ds_tm1 != ds_t0:
        ds_tm1.close()

    logging.info("M6 Terrain-Following Forecast Rollout completed successfully.")


if __name__ == "__main__":
    run_rollout()
