#!/usr/bin/env python3
"""
Standard Direct AI Weather Forecasting PyTorch Lightning Module.
Trains end-to-end on full physical state targets X_full (P, Q, T, U, V, W).

Features:
- Standardized per-variable normalization (mu, sigma) to prevent moisture explosion and loss swamping.
- Strict Nodes-First memory layout: (B, num_nodes * num_levels, C).
- Support for 15 input channels:
    * 12 dynamic history channels (2 steps x 6 vars)
    * 2 static surface channels (land_sea_mask, elevation)
    * 1 3D vertical terrain/height coordinate channel
- Optional layer mass-weighted MSE loss for terrain-following (sigma/eta) vertical coordinates.
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
    """
    Standard Direct AI Weather Forecasting Lightning Module.
    Predicts state increments delta_X directly in physical space.
    """

    def __init__(self, cfg: Dict[str, Any], **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.num_levels = cfg.get("num_levels", 32)
        self.num_nodes = cfg.get("num_nodes", 40962)  # Default for M6 grid

        # Empirical Dataset Means (mu) and Standard Deviations (sigma)
        # Order: [P (hPa), Q (kg/kg), T (K), U (m/s), V (m/s), W (m/s)]
        # mu_vals = cfg.get("mu_vals", [950.0, 0.0047, 263.0, 0.0, 0.0, 0.0])
        # sigma_vals = cfg.get("sigma_vals", [50.0, 0.0045, 25.0, 12.0, 12.0, 0.1])
        # For log-state inputs (ln_p, q, ln_t, u, v, w):
        # ln_P: ~11.5 ln(Pa), Q: ~0.0047, ln_T: ~5.57 ln(K), U: 0.0, V: 0.0, W: 0.0
        mu_vals = cfg.get("mu_vals", [11.5, 0.0047, 5.57, 0.0, 0.0, 0.0])
        sigma_vals = cfg.get("sigma_vals", [0.15, 0.0045, 0.12, 12.0, 12.0, 0.1])

        self.register_buffer("mu", torch.tensor(mu_vals, dtype=torch.float32).view(1, 1, 6))
        self.register_buffer("sigma", torch.tensor(sigma_vals, dtype=torch.float32).view(1, 1, 6))

        # Model Architecture
        self.in_channels = cfg.get("in_channels", 15)  # 12 dynamic + 2 static surface + 1 3D terrain
        self.out_channels = cfg.get("out_channels", 6)
        self.latent_dim = cfg.get("latent_dim", 256)
        self.processor_layers = cfg.get("processor_layers", 16)
        self.hierarchy_levels = cfg.get("hierarchy_levels", [6, 5, 4, 3, 2, 1, 0])
        self.history_steps = cfg.get("history_steps", 2)

        self.model = DeepGraphCastModel(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            latent_dim=self.latent_dim,
            processor_layers=self.processor_layers,
            hierarchy_levels=self.hierarchy_levels,
            history_steps=self.history_steps,
        )

        self.learning_rate = cfg.get("learning_rate", 1.0e-04)
        self.loss_weights_dict = cfg.get("loss_weights", {})

        # Loss weight multipliers per variable [P, Q, T, U, V, W]
        w_p = self.loss_weights_dict.get("weight_P", 1.0)
        w_q = self.loss_weights_dict.get("weight_Q", 1.0)
        w_t = self.loss_weights_dict.get("weight_T", 1.0)
        w_u = self.loss_weights_dict.get("weight_U", 2.5)  # Boosted for kinetic wind energy
        w_v = self.loss_weights_dict.get("weight_V", 2.5)  # Boosted for kinetic wind energy
        w_w = self.loss_weights_dict.get("weight_W", 1.0)

        self.register_buffer(
            "loss_weights",
            torch.tensor([w_p, w_q, w_t, w_u, w_v, w_w], dtype=torch.float32).view(1, 1, 1, 6),
        )

        # Static features cache
        self.register_buffer("static_features", None)

    def forward(self, x_raw: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        """
        x_raw: (B, N_flat, 12) raw un-normalized physical history (t-1, t0)
        timestamps: (B,) Unix epoch timestamps
        Returns predicted state increment delta_X in physical units: (B, N_flat, 6)
        """
        batch_size, n_flat, num_vars = x_raw.shape

        # 1. Normalize t-1 and t0 history steps separately
        x_t0 = (x_raw[:, :, :6] - self.mu) / self.sigma
        x_t1 = (x_raw[:, :, 6:12] - self.mu) / self.sigma
        x_norm = torch.cat([x_t0, x_t1], dim=-1)  # (B, N_flat, 12)

        # 2. Extract or construct static features
        num_static = self.in_channels - 12 if self.in_channels > 12 else 3
        if hasattr(self, "static_features") and self.static_features is not None:
            static_flat = self.static_features.to(x_norm.device)
            if static_flat.dim() == 2:
                static_flat = static_flat.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            static_flat = torch.zeros((batch_size, n_flat, num_static), device=x_norm.device, dtype=x_norm.dtype)

        # 3. Align static feature dimensions if needed
        if static_flat.shape[1] != n_flat:
            if static_flat.shape[1] * self.num_levels == n_flat:
                num_cols = static_flat.shape[-1]
                static_flat = static_flat.unsqueeze(2).expand(-1, -1, self.num_levels, -1).reshape(batch_size, n_flat, num_cols)
            else:
                static_flat = torch.zeros((batch_size, n_flat, num_static), device=x_norm.device, dtype=x_norm.dtype)

        # 4. Concatenate history and static features for GNN input
        x_input = torch.cat([x_norm, static_flat], dim=-1)  # (B, N_flat, in_channels)

        # 5. Forward pass through underlying DeepGraphCastModel
        pred_delta_norm = self.model(x_input, None, timestamps)

        # 6. Un-normalize delta back to physical units
        pred_delta = pred_delta_norm * self.sigma

        return pred_delta

    def _compute_loss(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        stage_name: str,
        dp_mass_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Computes standardized MSE loss across full state predictions.
        Supports mass-weighting by layer pressure thickness (dp) for terrain-following coordinates.
        """
        x_raw, y_target_full, timestamps = batch
        batch_size = y_target_full.shape[0]

        # Reshape target dynamically based on input batch tensor shape
        if y_target_full.dim() == 2:
            y_target_full = y_target_full.view(batch_size, -1, 6)

        # Ensure y_target_full is 4D: (B, N_nodes, N_levels, 6)
        if y_target_full.dim() == 3:
            y_target_full = y_target_full.view(batch_size, -1, self.num_levels, 6)

        # Predict increment in physical units
        pred_delta_flat = self.forward(x_raw, timestamps)  # (B, N_flat, 6)
        pred_delta = pred_delta_flat.view(batch_size, -1, self.num_levels, 6)

        # Unflatten t0 latest physical state
        x_latest = x_raw[:, :, 6:12].view(batch_size, -1, self.num_levels, 6)

        # Full predicted physical state
        pred_full = x_latest + pred_delta

        # Standardized loss calculation in normalized space
        pred_norm = (pred_full - self.mu.view(1, 1, 1, 6)) / self.sigma.view(1, 1, 1, 6)
        target_norm = (y_target_full - self.mu.view(1, 1, 1, 6)) / self.sigma.view(1, 1, 1, 6)

        # Per-variable squared error: (B, nodes, levels, 6)
        sq_error = (pred_norm - target_norm) ** 2

        # Optional layer mass-weighting for terrain-following coordinates
        if dp_mass_weight is not None:
            norm_dp = dp_mass_weight / torch.mean(dp_mass_weight)
            sq_error = sq_error * norm_dp

        # Apply loss weight scaling factors per variable [P, Q, T, U, V, W]
        weighted_sq_error = sq_error * self.loss_weights  # (B, nodes, levels, 6)

        # Per-variable mean losses
        loss_p = torch.mean(weighted_sq_error[:, :, :, 0])
        loss_q = torch.mean(weighted_sq_error[:, :, :, 1])
        loss_t = torch.mean(weighted_sq_error[:, :, :, 2])
        loss_u = torch.mean(weighted_sq_error[:, :, :, 3])
        loss_v = torch.mean(weighted_sq_error[:, :, :, 4])
        loss_w = torch.mean(weighted_sq_error[:, :, :, 5])

        total_loss = loss_p + loss_q + loss_t + loss_u + loss_v + loss_w

        # Logging
        key_alias = "val_loss" if stage_name == "val" else "train_loss"
        self.log(key_alias, total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage_name}/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage_name}/loss_P", loss_p, sync_dist=True)
        self.log(f"{stage_name}/loss_Q", loss_q, sync_dist=True)
        self.log(f"{stage_name}/loss_T", loss_t, sync_dist=True)
        self.log(f"{stage_name}/loss_U", loss_u, sync_dist=True)
        self.log(f"{stage_name}/loss_V", loss_v, sync_dist=True)
        self.log(f"{stage_name}/loss_W", loss_w, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-5,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.cfg.get("max_epochs", 25),
        )
        return [optimizer], [scheduler]
