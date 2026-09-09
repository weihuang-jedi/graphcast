#!/usr/bin/env python3
"""
3D GraphCast Lightning Training Executable for Standard Direct AI Weather Forecasting.
Supports multi-step target loading for autoregressive training loss backpropagation.
Includes rich epoch-level progress callbacks, detailed itemized loss component logging,
GPU memory telemetry, seamless checkpoint state resumption, and per-epoch checkpoint persistence.
"""

import os
import time
import argparse
import logging
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, TQDMProgressBar, Callback
from pytorch_lightning.loggers import CSVLogger
import xarray as xr
import numpy as np

from models.graphcast_lightning_direct import StandardGraphCastLitModule

# Optimize CUDA matrix multiplication performance on NVIDIA Tensor Core GPUs (e.g., H100)
torch.set_float32_matmul_precision("high")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class EpochProgressLogger(Callback):
    """Custom Lightning Callback to log detailed progress, execution times, itemized loss components, and VRAM usage."""

    def __init__(self):
        super().__init__()
        self.epoch_start_time = None

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.epoch_start_time = time.time()
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0
        logging.info(f"--- [EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs} START] --- | Peak VRAM Allocated: {gpu_mem:.2f} GB")

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        elapsed = time.time() - self.epoch_start_time if self.epoch_start_time else 0.0
        
        # Safely extract main and itemized loss metrics logged during training
        metrics = trainer.callback_metrics
        train_loss = metrics.get("train_loss", metrics.get("train/loss", None))
        val_loss = metrics.get("val_loss", metrics.get("val/loss", None))
        
        loss_mse = metrics.get("train/loss_mse", None)
        loss_mom_h = metrics.get("train/loss_momentum_h", None)
        loss_mom_v = metrics.get("train/loss_momentum_v", None)
        loss_conti = metrics.get("train/loss_continuity", None)
        loss_mass = metrics.get("train/loss_mass_drift", None)

        lr = "N/A"
        if trainer.optimizers:
            lr = f"{trainer.optimizers[0].param_groups[0]['lr']:.2e}"

        train_str = f"{train_loss.item():.4f}" if train_loss is not None else "N/A"
        val_str = f"{val_loss.item():.4f}" if val_loss is not None else "N/A"
        mse_str = f"{loss_mse.item():.4f}" if loss_mse is not None else "N/A"
        mom_h_str = f"{loss_mom_h.item():.4f}" if loss_mom_h is not None else "N/A"
        mom_v_str = f"{loss_mom_v.item():.4f}" if loss_mom_v is not None else "N/A"
        conti_str = f"{loss_conti.item():.4f}" if loss_conti is not None else "N/A"
        mass_str = f"{loss_mass.item():.4f}" if loss_mass is not None else "N/A"
        gpu_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) if torch.cuda.is_available() else 0.0

        logging.info(
            f"--- [EPOCH {trainer.current_epoch + 1}/{trainer.max_epochs} COMPLETE] --- | "
            f"Time: {elapsed:.1f}s | Train Loss: {train_str} | Val Loss: {val_str} | "
            f"MSE: {mse_str} | Mom_H: {mom_h_str} | Mom_V: {mom_v_str} | Continuity: {conti_str} | "
            f"Mass_Drift: {mass_str} | LR: {lr} | VRAM: {gpu_mem:.2f} GB"
        )


class IcosahedralZarrDataset(Dataset):
    def __init__(self, zarr_path: str, history_steps: int = 2, rollout_steps: int = 6):
        super().__init__()
        if not os.path.exists(zarr_path):
            raise FileNotFoundError(f"Zarr dataset store not found at '{zarr_path}'")

        self.ds = xr.open_zarr(zarr_path)
        self.history_steps = history_steps
        self.rollout_steps = rollout_steps

        var_p = "ln_p_icosahedral" if "ln_p_icosahedral" in self.ds else ("p_icosahedral" if "p_icosahedral" in self.ds else "p")
        var_t = "ln_t_icosahedral" if "ln_t_icosahedral" in self.ds else ("t_icosahedral" if "t_icosahedral" in self.ds else "t")
        var_u = "u_icosahedral" if "u_icosahedral" in self.ds else ("U" if "U" in self.ds else "u")

        self.time_len = self.ds.sizes.get("time", self.ds.sizes.get("step", 0))
        self.valid_indices = self.time_len - (history_steps + rollout_steps)

        self.var_keys = [
            var_p,
            "q_icosahedral" if "q_icosahedral" in self.ds else "q",
            var_t,
            var_u,
            "v_icosahedral" if "v_icosahedral" in self.ds else "v",
            "w_icosahedral" if "w_icosahedral" in self.ds else "w",
        ]

        logging.info(
            f"[DATASET] Zarr Time Length: {self.time_len} | Valid Samples: {self.valid_indices} | "
            f"History Steps: {history_steps} | Rollout Steps: {rollout_steps}"
        )
        logging.info(f"[DATASET] Resolved Variable Keys: {self.var_keys}")

    def __len__(self):
        return max(0, self.valid_indices)

    def __getitem__(self, idx):
        t_start = idx
        t_history_end = idx + self.history_steps

        input_vars = []
        for key in self.var_keys:
            if key in self.ds:
                da = self.ds[key].isel(time=slice(t_start, t_history_end))
                val = np.asarray(da.values, dtype=np.float32)
                input_vars.append(val)

        input_arr = np.stack(input_vars, axis=-1)
        num_levels, n_nodes = input_arr.shape[1], input_arr.shape[2]

        input_nodes_first = np.transpose(input_arr, (2, 1, 0, 3))
        x_flat = torch.tensor(input_nodes_first, dtype=torch.float32).reshape(n_nodes * num_levels, self.history_steps * 6)

        target_seq = []
        for step in range(self.rollout_steps):
            t_target = t_history_end + step
            step_vars = []
            for key in self.var_keys:
                if key in self.ds:
                    da = self.ds[key].isel(time=t_target)
                    val = np.asarray(da.values, dtype=np.float32)
                    step_vars.append(val)

            step_arr = np.stack(step_vars, axis=-1)
            step_nodes_first = np.transpose(step_arr, (1, 0, 2))
            step_flat = torch.tensor(step_nodes_first, dtype=torch.float32).reshape(n_nodes * num_levels, 6)
            target_seq.append(step_flat)

        y_seq_flat = torch.stack(target_seq, dim=0)

        time_val = self.ds["time"].isel(time=t_history_end).values
        timestamp_unix = float(np.datetime64(time_val, "s").astype(int))

        return x_flat, y_seq_flat, torch.tensor(timestamp_unix, dtype=torch.float64)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Direct AI Weather Model")
    parser.add_argument("-c", "--config", type=str, default="config.yaml")
    parser.add_argument("--ckpt", type=str, default=None, help="Path to checkpoint to resume training from")
    return parser.parse_args()


def main():
    args = parse_args()

    if os.path.exists(args.config):
        with open(args.config, "r") as f:
            cfg = yaml.safe_load(f)
        logging.info(f"Successfully loaded configuration from '{args.config}'")
    else:
        cfg = {}
        logging.warning(f"Config file '{args.config}' not found. Falling back to default settings.")

    print("=" * 80)
    print(" INITIALIZING MULTI-STEP DIRECT GRAPHCAST TRAINING RUN (M6 GRID)")
    print("=" * 80)

    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logging.info(f"[HARDWARE] Device: {device_name} | GPU Count: {torch.cuda.device_count()}")

    lit_module = StandardGraphCastLitModule(cfg=cfg)

    total_params = sum(p.numel() for p in lit_module.parameters() if p.requires_grad)
    logging.info(f"[MODEL DIAGNOSTICS] Total Trainable Parameters: {total_params / 1e6:.2f} Million")

    train_params = cfg.get("training_params", {})
    accum_grad = train_params.get("accumulate_grad_batches", 4)
    batch_size = train_params.get("batch_size", 1)
    max_epochs = train_params.get("max_epochs", 25)

    logging.info(
        f"[TRAINING PARAMS] Batch Size: {batch_size} | Accumulate Grads: {accum_grad} | "
        f"Effective Batch Size: {batch_size * accum_grad} | Max Epochs: {max_epochs}"
    )
    logging.info(f"[LOSS WEIGHTS] {cfg.get('loss_weights', {})}")

    logger = CSVLogger("lightning_logs", name="standard_direct_m6")

    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints",
        filename="graphcast_m6_epoch_{epoch:02d}",
        save_top_k=-1,          # Retains all epoch checkpoints
        every_n_epochs=1,       # Forces save on every single epoch
        save_last=True,         # Keeps last.ckpt updated for fast resumption
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")
    epoch_logger = EpochProgressLogger()
    progress_bar = TQDMProgressBar(refresh_rate=50)

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=torch.cuda.device_count() if torch.cuda.is_available() else 1,
        precision="bf16-mixed" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 32,
        accumulate_grad_batches=accum_grad,
        callbacks=[checkpoint_callback, lr_monitor, epoch_logger, progress_bar],
        logger=logger,
        log_every_n_steps=10,
    )

    zarr_path = cfg.get("paths", {}).get("zarr_store", "data/gfs_icosahedral_m6.zarr")
    logging.info(f"[DATA] Loading Zarr training store from: '{zarr_path}'")

    dataset = IcosahedralZarrDataset(
        zarr_path=zarr_path,
        history_steps=cfg.get("model_params", {}).get("history_steps", 2),
        rollout_steps=cfg.get("model_params", {}).get("rollout_steps", 6),
    )

    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_ds, val_ds = torch.utils.data.random_split(dataset, [train_size, val_size])

    num_workers = train_params.get("num_workers", 4)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    ckpt_path = args.ckpt
    if ckpt_path is None and os.path.exists("checkpoints/last.ckpt"):
        ckpt_path = "checkpoints/last.ckpt"

    if ckpt_path and os.path.exists(ckpt_path):
        ckpt_meta = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        resumed_epoch = ckpt_meta.get("epoch", 0)
        global_step = ckpt_meta.get("global_step", 0)

        # Pre-load state dict into lightning module non-strictly to absorb missing buffers smoothly
        if "state_dict" in ckpt_meta:
            missing, unexpected = lit_module.load_state_dict(ckpt_meta["state_dict"], strict=False)
            if missing:
                logging.info(f"[RESUME] Non-strict pre-load absorbed missing keys: {missing}")

        logging.info(
            f"[RESUME] Found valid checkpoint: '{ckpt_path}' | "
            f"Resuming at Epoch {resumed_epoch + 1} (Global Step {global_step})"
        )
    else:
        ckpt_path = None
        logging.info("[START] Starting new training run from scratch...")

    logging.info(f"[FIT] Executing trainer.fit across {len(train_ds)} train samples and {len(val_ds)} val samples...")
    trainer.fit(lit_module, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=ckpt_path)
    logging.info("[SUCCESS] Training completed successfully.")


if __name__ == "__main__":
    main()
