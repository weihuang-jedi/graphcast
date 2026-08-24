#!/usr/bin/env python3
"""
Standard Direct AI Weather Forecasting PyTorch Lightning Module.
Trains end-to-end on Log-State physical targets X_full (ln_P, Q, ln_T, U, V, W).
Includes multi-step autoregressive rollout unrolling during training loss calculation.
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple, Optional

from models.graphcast import DeepGraphCastModel


class StandardGraphCastLitModule(pl.LightningModule):
    def __init__(self, cfg: Dict[str, Any], **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.num_levels = cfg.get("num_levels", 32)
        self.num_nodes = cfg.get("num_nodes", 40962)  # M6 grid default

        # Log-State Standardization [ln_P, Q, ln_T, U, V, W]
        # mu_vals = cfg.get("mu_vals", [11.52, 0.0047, 5.57, 0.0, 0.0, 0.0])
        # sigma_vals = cfg.get("sigma_vals", [0.15, 0.0045, 0.10, 5.0, 5.0, 0.1])

        # Order: [ln_P, Q, ln_T, U, V, W]
        # Setting sigma_U = 1.0 and sigma_V = 1.0 amplifies backpropagated gradients by 5x
        # mu_vals = cfg.get("mu_vals", [11.52, 0.0047, 5.57, 0.0, 0.0, 0.0])
        # sigma_vals = cfg.get("sigma_vals", [0.15, 0.0045, 0.10, 1.0, 1.0, 0.1])

        # Order: [ln_P, Q, ln_T, U, V, W]
        # Asymmetric sigma: tightens U and V scaling to force un-dampened wind gradients
        mu_vals = cfg.get("mu_vals", [11.52, 0.0047, 5.57, 0.0, 0.0, 0.0])
        sigma_vals = cfg.get("sigma_vals", [0.25, 0.0045, 0.15, 2.0, 2.0, 0.1])

        self.register_buffer("mu", torch.tensor(mu_vals, dtype=torch.float32).view(1, 1, 6))
        self.register_buffer("sigma", torch.tensor(sigma_vals, dtype=torch.float32).view(1, 1, 6))

        # Model Architecture
        self.in_channels = cfg.get("in_channels", 15)
        self.out_channels = cfg.get("out_channels", 6)
        self.latent_dim = cfg.get("latent_dim", 256)
        self.processor_layers = cfg.get("processor_layers", 16)
        self.hierarchy_levels = cfg.get("hierarchy_levels", [6, 5, 4, 3, 2, 1, 0])
        self.history_steps = cfg.get("history_steps", 2)
        self.rollout_steps = cfg.get("rollout_steps", 2)

        self.model = DeepGraphCastModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            latent_dim=self.latent_dim,
            processor_layers=self.processor_layers,
            hierarchy_levels=self.hierarchy_levels,
            history_steps=self.history_steps,
        )

        self.learning_rate = cfg.get("learning_rate", 1.0e-04)
        
        # Resolve loss weights safely across root or nested dict
        loss_cfg = cfg.get("loss_weights", {})
        w_p = loss_cfg.get("weight_P", 2.0)
        w_q = loss_cfg.get("weight_Q", 1.0)
        w_t = loss_cfg.get("weight_T", 1.0)
        w_u = loss_cfg.get("weight_U", 5.0)  # 5x wind momentum penalty
        w_v = loss_cfg.get("weight_V", 5.0)  # 5x wind momentum penalty
        w_w = loss_cfg.get("weight_W", 1.0)

        self.register_buffer(
            "loss_weights",
            torch.tensor([w_p, w_q, w_t, w_u, w_v, w_w], dtype=torch.float32).view(1, 1, 1, 6),
        )

        self.lambda_moisture = loss_cfg.get("lambda_moisture", cfg.get("lambda_moisture", 10.0))
        self.register_buffer("static_features", None)

    def _get_static_surface_features(self, device: torch.device) -> torch.Tensor:
        if self.static_features is None:
            grid_path = self.cfg.get("grid_ref", "")

            if os.path.exists(grid_path):
                ds_grid = xr.open_dataset(grid_path)
                lsm = ds_grid["land_sea_mask"].values if "land_sea_mask" in ds_grid else np.zeros((self.num_nodes,))
                elev = ds_grid["elevation"].values if "elevation" in ds_grid else np.zeros((self.num_nodes,))
                ds_grid.close()
            else:
                lsm = np.zeros((self.num_nodes,), dtype=np.float32)
                elev = np.zeros((self.num_nodes,), dtype=np.float32)

            lsm_nodes_first = torch.tensor(lsm, dtype=torch.float32).unsqueeze(-1).repeat(1, self.num_levels).unsqueeze(-1)
            elev_nodes_first = torch.tensor(elev / 1000.0, dtype=torch.float32).unsqueeze(-1).repeat(1, self.num_levels).unsqueeze(-1)

            feat_list = [lsm_nodes_first, elev_nodes_first]

            if self.in_channels >= 15:
                lev_idx = torch.arange(self.num_levels, dtype=torch.float32).unsqueeze(0)
                terrain_decay = torch.exp(-lev_idx / 10.0).unsqueeze(-1)
                z_flat = (lev_idx * 0.8).unsqueeze(-1).repeat(self.num_nodes, 1, 1)
                z_terrain_3d = z_flat + (elev_nodes_first * terrain_decay)
                feat_list.append(z_terrain_3d)

            static_cat = torch.cat(feat_list, dim=-1)
            self.static_features = static_cat.view(self.num_nodes * self.num_levels, -1).to(device)

        return self.static_features

    def forward(self, x_raw: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        batch_size, num_flat, num_vars = x_raw.shape

        x_t0 = (x_raw[:, :, :6] - self.mu) / self.sigma
        x_t1 = (x_raw[:, :, 6:12] - self.mu) / self.sigma
        x_norm = torch.cat([x_t0, x_t1], dim=-1)

        static_feat = self._get_static_surface_features(x_raw.device)
        static_flat = static_feat.unsqueeze(0).repeat(batch_size, 1, 1)
        x_input = torch.cat([x_norm, static_flat], dim=-1)

        pred_delta_norm = self.model(x_input, None, timestamps)
        pred_delta = pred_delta_norm * self.sigma
        return pred_delta

    def _compute_loss(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], stage_name: str) -> torch.Tensor:
        x_raw, y_target_seq, timestamps = batch
        batch_size = x_raw.shape[0]

        # y_target_seq shape: (B, rollout_steps, num_nodes * num_levels, 6)
        if y_target_seq.ndim == 3:
            y_target_seq = y_target_seq.unsqueeze(1)

        current_history = x_raw.clone()
        total_unrolled_loss = 0.0

        # Multi-Step Autoregressive Unrolling during backprop
        for k in range(self.rollout_steps):
            step_ts = timestamps + (k * 21600.0)  # Advance 6 hours in seconds per rollout step
            pred_delta = self.forward(current_history, step_ts)

            # State prediction at step k
            x_latest = current_history[:, :, -6:]
            pred_full = x_latest + pred_delta

            target_k = y_target_seq[:, k, :, :] if y_target_seq.ndim == 4 else y_target_seq[:, 0, :, :]

            # Compute standardized step MSE loss
            pred_norm = (pred_full.view(batch_size, self.num_nodes, self.num_levels, 6) - self.mu.view(1, 1, 1, 6)) / self.sigma.view(1, 1, 1, 6)
            target_norm = (target_k.view(batch_size, self.num_nodes, self.num_levels, 6) - self.mu.view(1, 1, 1, 6)) / self.sigma.view(1, 1, 1, 6)

            weighted_sq_error = ((pred_norm - target_norm) ** 2) * self.loss_weights
            step_mse = torch.mean(weighted_sq_error)

            # Global moisture conservation penalty
            delta_q = pred_delta.view(batch_size, self.num_nodes, self.num_levels, 6)[:, :, :, 1]
            global_moisture_drift = torch.mean(torch.sum(delta_q, dim=2))
            moisture_penalty = global_moisture_drift ** 2

            total_unrolled_loss += step_mse + (self.lambda_moisture * moisture_penalty)

            # Autoregressively update history tensor for next unrolled step
            current_history = torch.cat([current_history[:, :, 6:], pred_full], dim=-1)

        total_loss = total_unrolled_loss / self.rollout_steps

        key_alias = "val_loss" if stage_name == "val" else "train_loss"
        self.log(key_alias, total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage_name}/loss", total_loss, prog_bar=True, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.get("max_epochs", 25))
        return [optimizer], [scheduler]
