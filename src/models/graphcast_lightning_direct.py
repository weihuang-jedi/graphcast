# models/graphcast_lightning_direct.py
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
        self.num_nodes = cfg.get("num_nodes", 40962)  # M6 grid

        # Log-state standardization constants [ln_P, Q, ln_T, U, V, W]
        # mu_vals = cfg.get("mu_vals", [11.52, 0.0047, 5.57, 0.0, 0.0, 0.0])
        # sigma_vals = cfg.get("sigma_vals", [0.25, 0.0045, 0.15, 12.0, 12.0, 0.1])

        # Order: [ln_P, Q, ln_T, U, V, W]
        # Reduced sigma_U and sigma_V from 12.0 to 6.0 to amplify backpropagated wind gradients
        mu_vals = cfg.get("mu_vals", [11.52, 0.0047, 5.57, 0.0, 0.0, 0.0])
        sigma_vals = cfg.get("sigma_vals", [0.15, 0.0045, 0.10, 5.0, 5.0, 0.1])

        self.register_buffer("mu", torch.tensor(mu_vals, dtype=torch.float32).view(1, 1, 6))
        self.register_buffer("sigma", torch.tensor(sigma_vals, dtype=torch.float32).view(1, 1, 6))

        self.in_channels = cfg.get("in_channels", 15)  # 12 dynamic + 2 surface + 1 3D terrain
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

        w_p = self.loss_weights_dict.get("weight_P", 1.0)
        w_q = self.loss_weights_dict.get("weight_Q", 1.0)
        w_t = self.loss_weights_dict.get("weight_T", 1.0)
        w_u = self.loss_weights_dict.get("weight_U", 2.5)
        w_v = self.loss_weights_dict.get("weight_V", 2.5)
        w_w = self.loss_weights_dict.get("weight_W", 1.0)

        self.register_buffer(
            "loss_weights",
            torch.tensor([w_p, w_q, w_t, w_u, w_v, w_w], dtype=torch.float32).view(1, 1, 1, 6),
        )

        self.register_buffer("static_features", None)

    def _get_static_surface_features(self, device: torch.device) -> torch.Tensor:
        if self.static_features is None:
            grid_path = self.cfg.get(
                "grid_ref",
                "../data/starviewergraphcast-grid/global_icosahedral_m6.20220101.t00z.1p00.f000.nc",
            )

            if os.path.exists(grid_path):
                ds_grid = xr.open_dataset(grid_path)
                lsm = ds_grid["land_sea_mask"].values if "land_sea_mask" in ds_grid else np.zeros((self.num_nodes,))
                elev = ds_grid["elevation"].values if "elevation" in ds_grid else np.zeros((self.num_nodes,))
                ds_grid.close()
            else:
                lsm = np.zeros((self.num_nodes,), dtype=np.float32)
                elev = np.zeros((self.num_nodes,), dtype=np.float32)

            lsm_nodes_first = (
                torch.tensor(lsm, dtype=torch.float32)
                .unsqueeze(-1)
                .repeat(1, self.num_levels)
                .unsqueeze(-1)
            )

            elev_nodes_first = (
                torch.tensor(elev / 1000.0, dtype=torch.float32)
                .unsqueeze(-1)
                .repeat(1, self.num_levels)
                .unsqueeze(-1)
            )

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
        x_raw, y_target_full, timestamps = batch
        batch_size = y_target_full.shape[0]

        if y_target_full.ndim == 3:
            y_target_full = y_target_full.view(batch_size, self.num_nodes, self.num_levels, 6)

        pred_delta = self.forward(x_raw, timestamps).view(batch_size, self.num_nodes, self.num_levels, 6)
        x_latest = x_raw[:, :, 6:12].view(batch_size, self.num_nodes, self.num_levels, 6)

        pred_full = x_latest + pred_delta

        pred_norm = (pred_full - self.mu.view(1, 1, 1, 6)) / self.sigma.view(1, 1, 1, 6)
        target_norm = (y_target_full - self.mu.view(1, 1, 1, 6)) / self.sigma.view(1, 1, 1, 6)

        weighted_sq_error = ((pred_norm - target_norm) ** 2) * self.loss_weights

        total_loss = torch.mean(weighted_sq_error)

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
