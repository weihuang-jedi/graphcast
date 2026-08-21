#!/usr/bin/env python3
"""
3D GraphCast Lightning Training Executable for Standard Direct AI Weather Forecasting.
Formats batches into strict Nodes-First (40962, 32) layout for M6 terrain-following grids.
"""

import os
import argparse
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
import xarray as xr
import numpy as np

from models.graphcast_lightning_direct import StandardGraphCastLitModule


class IcosahedralZarrDataset(Dataset):
    def __init__(self, zarr_path: str, history_steps: int = 2, forecast_steps: int = 2):
        super().__init__()
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr dataset store not found at '{zarr_path}'")

        self.ds = xr.open_zarr(zarr_path)
        self.history_steps = history_steps
        self.forecast_steps = forecast_steps

        # Support both log-state and standard variable naming
        var_p = "ln_p_icosahedral" if "ln_p_icosahedral" in self.ds else ("p_icosahedral" if "p_icosahedral" in self.ds else "p")
        var_t = "ln_t_icosahedral" if "ln_t_icosahedral" in self.ds else ("t_icosahedral" if "t_icosahedral" in self.ds else "t")
        var_u = "u_icosahedral" if "u_icosahedral" in self.ds else ("U" if "U" in self.ds else "u")

        self.time_len = self.ds.sizes.get("time", self.ds.sizes.get("step", 0))
        self.valid_indices = self.time_len - (history_steps + forecast_steps)

        self.var_keys = [
            var_p,
            "q_icosahedral" if "q_icosahedral" in self.ds else "q",
            var_t,
            var_u,
            "v_icosahedral" if "v_icosahedral" in self.ds else "v",
            "w_icosahedral" if "w_icosahedral" in self.ds else "w",
        ]

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        t_start = idx
        t_target = idx + self.history_steps

        # Load history sequence across all 6 variables
        input_vars = []
        for key in self.var_keys:
            if key in self.ds:
                da = self.ds[key].isel(time=slice(t_start, t_start + self.history_steps))
                val = np.asarray(da.values, dtype=np.float32)  # (history_steps, levels, nodes)
                input_vars.append(val)

        # Shape: (history_steps, levels, nodes, 6 vars)
        input_arr = np.stack(input_vars, axis=-1)
        num_levels, n_nodes = input_arr.shape[1], input_arr.shape[2]

        # Transpose to Nodes-First Layout: (nodes, levels, history_steps, vars)
        input_nodes_first = np.transpose(input_arr, (2, 1, 0, 3))
        x_flat = torch.tensor(input_nodes_first, dtype=torch.float32).reshape(n_nodes * num_levels, self.history_steps * 6)

        # Load target state
        target_vars = []
        for key in self.var_keys:
            if key in self.ds:
                da = self.ds[key].isel(time=t_target)
                val = np.asarray(da.values, dtype=np.float32)  # (levels, nodes)
                target_vars.append(val)

        target_arr = np.stack(target_vars, axis=-1)  # (levels, nodes, 6)
        target_nodes_first = np.transpose(target_arr, (1, 0, 2))  # (nodes, levels, 6)
        y_flat = torch.tensor(target_nodes_first, dtype=torch.float32).reshape(n_nodes * num_levels, 6)

        time_val = self.ds["time"].isel(time=t_target).values
        timestamp_unix = float(np.datetime64(time_val, "s").astype(int))

        return x_flat, y_flat, torch.tensor(timestamp_unix, dtype=torch.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Direct AI Weather Model")
    parser.add_argument("-c", "--config", type=str, default="config_standard.yaml")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    print("=" * 80)
    print(" INITIALIZING DIRECT GRAPHCAST TRAINING RUN (M6 GRID)")
    print("=" * 80)

    lit_module = StandardGraphCastLitModule(cfg=cfg)

    logger = CSVLogger("lightning_logs", name="standard_direct_m6")
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="graphcast_m6_{epoch:02d}_{val_loss:.4f}",
        save_top_k=3,
        monitor="val_loss",
        mode="min",
        save_last=True,
    )
    lr_monitor = LearningRateMonitor(logging_interval="step")

    train_params = cfg.get("training_params", {})
    trainer = pl.Trainer(
        max_epochs=train_params.get("max_epochs", 25),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 32,
        callbacks=[checkpoint_callback, lr_monitor],
        logger=logger,
        log_every_n_steps=10,
    )

    zarr_path = cfg.get("paths", {}).get("zarr_store", "data/gfs_icosahedral_m6.zarr")
    print(f"Loading Zarr training store: {zarr_path}")

    dataset = IcosahedralZarrDataset(zarr_path=zarr_path, history_steps=cfg.get("model_params", {}).get("history_steps", 2))

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    num_workers = train_params.get("num_workers", 4)
    batch_size = train_params.get("batch_size", 1)  # Set to 1 for M6 memory safety

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    print(f"Starting M6 training across {len(train_ds)} train samples and {len(val_ds)} val samples...")
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    main()
