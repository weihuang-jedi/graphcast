#!/usr/bin/env python3
"""
Standard Direct AI Weather Forecast Rollout Driver.
Maintains strict Nodes-First memory layout during rollout and exports standard (level, node) NetCDFs.
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
    parser = argparse.ArgumentParser(description="Standard Direct Forecast Rollout")
    parser.add_argument("-c", "--ckpt_path", type=str, required=True)
    parser.add_argument("-i", "--input_file", type=str, required=True)
    parser.add_argument("--init_time", type=str, default="2026020100")
    parser.add_argument("-s", "--forecast_steps", type=int, default=100)
    parser.add_argument("--step_hours", type=int, default=24)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--grid_ref", type=str, default="../data/starviewergraphcast-grid/global_icosahedral_m4.20220101.t00z.1p00.f000.nc")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


@torch.no_grad()
def run_rollout():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    logging.info(f"Executing Nodes-First Forecast Rollout on [{device}]")

    checkpoint = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    hparams = checkpoint.get("hyper_parameters", {})

    lit_module = StandardGraphCastLitModule(**hparams)
    model_state = lit_module.state_dict()
    filtered_state = {k: v for k, v in state_dict.items() if k in model_state and model_state[k].shape == v.shape}
    lit_module.load_state_dict(filtered_state, strict=False)
    lit_module.to(device).eval()

    start_dt = datetime.strptime(args.init_time, "%Y%m%d%H")
    ds_init = xr.open_dataset(args.input_file)

    lats_deg = ds_init['latitude'].values if 'latitude' in ds_init else ds_init['lat'].values
    lons_deg = ds_init['longitude'].values if 'longitude' in ds_init else ds_init['lon'].values
    # Extract node count dynamically from input file
    num_nodes = len(lats_deg)  # Automatically evaluates to 40962 for M6
    num_levels = 32

    # Verify size match
    expected_size = num_nodes * num_levels  # 40,962 * 32 = 1,310,784
    logging.info(f"[M6 ROLLOUT] Nodes: {num_nodes} | Levels: {num_levels} | Total 3D points: {expected_size}")

    # Extract initial condition fields and format in Nodes-First layout: (nodes=2562, levels=32)
    def extract_var_nodes_first(v_name):
        var_key = [k for k in [f"{v_name}_icosahedral", v_name.upper(), v_name.lower()] if k in ds_init]
        if not var_key:
            val0, val1 = np.zeros((num_nodes, num_levels)), np.zeros((num_nodes, num_levels))
        else:
            vals = np.asarray(ds_init[var_key[0]].values, dtype=np.float32)
            if vals.ndim == 4:
                vals = vals[0]
            if vals.ndim == 3 and vals.shape[0] >= 2:
                val0, val1 = vals[-2], vals[-1]  # (32, 2562)
            elif vals.ndim == 3 and vals.shape[0] == 1:
                val0, val1 = vals[0], vals[0]
            else:
                val0, val1 = vals, vals

            # Transpose (levels=32, nodes=2562) -> (nodes=2562, levels=32)
            val0 = np.transpose(val0, (1, 0))
            val1 = np.transpose(val1, (1, 0))

        return torch.tensor(val0, dtype=torch.float32, device=device), torch.tensor(val1, dtype=torch.float32, device=device)

    p_t0, p_t1 = extract_var_nodes_first("p")
    q_t0, q_t1 = extract_var_nodes_first("q")
    t_t0, t_t1 = extract_var_nodes_first("t")
    u_t0, u_t1 = extract_var_nodes_first("u")
    v_t0, v_t1 = extract_var_nodes_first("v")
    w_t0, w_t1 = extract_var_nodes_first("w")

    state_t0 = torch.stack([p_t0, q_t0, t_t0, u_t0, v_t0, w_t0], dim=-1).view(1, num_nodes * num_levels, 6)
    state_t1 = torch.stack([p_t1, q_t1, t_t1, u_t1, v_t1, w_t1], dim=-1).view(1, num_nodes * num_levels, 6)

    history_state = torch.cat([state_t0, state_t1], dim=-1)

    for step in range(1, args.forecast_steps + 1):
        lead_hours = step * args.step_hours
        current_dt = start_dt + timedelta(hours=lead_hours)
        timestamp_unix = current_dt.timestamp()
        timestamps_t = torch.tensor([timestamp_unix], device=device, dtype=torch.float64)

        pred_delta = lit_module(history_state, timestamps_t)
        current_state = history_state[:, :, -6:] + pred_delta

        # Physical clamping for Q (0 <= Q <= 0.035 kg/kg)
        current_state[:, :, 1] = torch.clamp(current_state[:, :, 1], min=0.0, max=0.035)

        history_state = torch.cat([history_state[:, :, 6:], current_state], dim=-1)

        # Reshape Nodes-First (2562, 32, 6) -> Transpose back to standard NetCDF (32, 2562, 6)
        state_nodes_first = current_state.view(num_nodes, num_levels, 6).cpu().numpy()
        state_np = np.transpose(state_nodes_first, (1, 0, 2))  # (levels=32, nodes=2562, 6)

        step_ds = xr.Dataset(
            data_vars={
                "P": (["level", "node"], state_np[..., 0]),
                "Q": (["level", "node"], state_np[..., 1]),
                "T": (["level", "node"], state_np[..., 2]),
                "U": (["level", "node"], state_np[..., 3]),
                "V": (["level", "node"], state_np[..., 4]),
                "W": (["level", "node"], state_np[..., 5]),
            },
            coords={"level": np.arange(num_levels), "node": np.arange(num_nodes), "latitude": ("node", lats_deg), "longitude": ("node", lons_deg)},
            attrs={
                "title": f"Direct GraphCast AI Weather Forecast (f{lead_hours:04d}h)",
                "init_time": start_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "valid_time": current_dt.strftime("%Y-%m-%d %H:%M:%S UTC"),
            },
        )

        out_name = f"forecast_standard_f{lead_hours:04d}h.nc"
        step_ds.to_netcdf(os.path.join(args.output_dir, out_name))
        step_ds.close()

        if step % 10 == 0 or step == args.forecast_steps:
            logging.info(
                f"Step {step}/{args.forecast_steps} (f{lead_hours:04d}h - {current_dt.strftime('%b %d')}) | "
                f"T: [{state_np[..., 2].min():.2f}, {state_np[..., 2].max():.2f}] K | "
                f"U: [{state_np[..., 3].min():.2f}, {state_np[..., 3].max():.2f}] m/s | "
                f"V: [{state_np[..., 4].min():.2f}, {state_np[..., 4].max():.2f}] m/s | "
                f"Q: [{state_np[..., 1].min():.6f}, {state_np[..., 1].max():.6f}] kg/kg"
            )

    ds_init.close()
    logging.info("Nodes-First Forecast Rollout completed successfully.")


if __name__ == "__main__":
    run_rollout()
