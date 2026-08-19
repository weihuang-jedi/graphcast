# models/graphcast_lightning_direct.py
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Dict, Any, Tuple
import xarray as xr
import numpy as np

from models.graphcast import DeepGraphCastModel


class StandardGraphCastLitModule(pl.LightningModule):
    """
    Standard Direct AI Weather Forecasting Module with Per-Variable Normalization.
    Strictly aligns tensor memory layouts in Nodes-First order: (nodes=2562, levels=32).
    """

    def __init__(self, cfg: Dict[str, Any], **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        self.num_levels = cfg.get("num_levels", 32)
        self.num_nodes = cfg.get("num_nodes", 2562)

        # Empirical Dataset Means and Standard Deviations across (P, Q, T, U, V, W)
        mu_vals = [950.0, 0.0047, 263.0, 0.0, 0.0, 0.0]
        sigma_vals = [50.0, 0.0045, 25.0, 12.0, 12.0, 0.1]

        self.register_buffer("mu", torch.tensor(mu_vals, dtype=torch.float32).view(1, 1, 6))
        self.register_buffer("sigma", torch.tensor(sigma_vals, dtype=torch.float32).view(1, 1, 6))

        self.model = DeepGraphCastModel(
            in_channels=cfg.get("in_channels", 14),
            out_channels=cfg.get("out_channels", 6),
            latent_dim=cfg.get("latent_dim", 256),
            processor_layers=cfg.get("processor_layers", 16),
            hierarchy_levels=cfg.get("hierarchy_levels", [4, 3, 2, 1, 0]),
            history_steps=cfg.get("history_steps", 2),
        )

        self.learning_rate = cfg.get("learning_rate", 1.0e-04)
        self.register_buffer("static_features", None)

    # Insert inside StandardGraphCastLitModule in models/graphcast_lightning_direct.py

    def _rotate_mesh_to_spherical_winds(self, pred_raw: torch.Tensor, lats_rad: torch.Tensor, lons_rad: torch.Tensor) -> torch.Tensor:
        """
        Transforms icosahedral local node velocity predictions into true Earth zonal (U) 
        and meridional (V) spherical wind components, eliminating icosahedral seam shadows.
        """
        u_raw = pred_raw[:, :, :, 3]
        v_raw = pred_raw[:, :, :, 4]
    
        sin_lon = torch.sin(lons_rad).unsqueeze(0).unsqueeze(1) # (1, 1, 2562)
        cos_lon = torch.cos(lons_rad).unsqueeze(0).unsqueeze(1)
        sin_lat = torch.sin(lats_rad).unsqueeze(0).unsqueeze(1)
        cos_lat = torch.cos(lats_rad).unsqueeze(0).unsqueeze(1)

        # Spherical coordinate projection
        u_sph = u_raw * cos_lon - v_raw * sin_lon
        v_sph = (u_raw * sin_lon + v_raw * cos_lon) * sin_lat + v_raw * cos_lat

        pred_corrected = pred_raw.clone()
        pred_corrected[:, :, :, 3] = u_sph
        pred_corrected[:, :, :, 4] = v_sph
        return pred_corrected

    def _get_static_surface_features(self, device: torch.device) -> torch.Tensor:
        """Constructs static surface features in Nodes-First layout: (nodes=2562, levels=32, 2)."""
        if self.static_features is None:
            grid_path = self.cfg.get(
                "grid_ref",
                "/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/starviewergraphcast-grid/global_icosahedral_m4.20220101.t00z.1p00.f000.nc",
            )
            ds_grid = xr.open_dataset(grid_path)
            lsm = ds_grid["land_sea_mask"].values if "land_sea_mask" in ds_grid else np.zeros((self.num_nodes,))
            elev = ds_grid["elevation"].values if "elevation" in ds_grid else np.zeros((self.num_nodes,))
            ds_grid.close()

            # Layout: (nodes=2562, levels=32, 1)
            lsm_nodes_first = torch.tensor(lsm, dtype=torch.float32).unsqueeze(-1).repeat(1, self.num_levels).unsqueeze(-1)
            elev_nodes_first = torch.tensor(elev / 1000.0, dtype=torch.float32).unsqueeze(-1).repeat(1, self.num_levels).unsqueeze(-1)

            static_cat = torch.cat([lsm_nodes_first, elev_nodes_first], dim=-1)  # (2562, 32, 2)
            self.static_features = static_cat.view(self.num_nodes * self.num_levels, 2).to(device)

        return self.static_features

    def forward(self, x_raw: torch.Tensor, timestamps: torch.Tensor) -> torch.Tensor:
        batch_size, num_flat, num_vars = x_raw.shape

        x_t0 = (x_raw[:, :, :6] - self.mu) / self.sigma
        x_t1 = (x_raw[:, :, 6:12] - self.mu) / self.sigma
        x_norm = torch.cat([x_t0, x_t1], dim=-1)

        static_feat = self._get_static_surface_features(x_raw.device)
        static_flat = static_feat.unsqueeze(0).repeat(batch_size, 1, 1)
        x_input = torch.cat([x_norm, static_flat], dim=-1)  # (B, 81984, 14)

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

        pred_norm = (pred_full - self.mu) / self.sigma
        target_norm = (y_target_full - self.mu) / self.sigma

        loss_p = torch.mean((pred_norm[:, :, :, 0] - target_norm[:, :, :, 0]) ** 2)
        loss_q = torch.mean((pred_norm[:, :, :, 1] - target_norm[:, :, :, 1]) ** 2)
        loss_t = torch.mean((pred_norm[:, :, :, 2] - target_norm[:, :, :, 2]) ** 2)
        loss_u = torch.mean((pred_norm[:, :, :, 3] - target_norm[:, :, :, 3]) ** 2)
        loss_v = torch.mean((pred_norm[:, :, :, 4] - target_norm[:, :, :, 4]) ** 2)
        loss_w = torch.mean((pred_norm[:, :, :, 5] - target_norm[:, :, :, 5]) ** 2)

        total_loss = loss_p + loss_q + loss_t + loss_u + loss_v + loss_w

        key_alias = "val_loss" if stage_name == "val" else "train_loss"
        self.log(key_alias, total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage_name}/loss", total_loss, prog_bar=True, sync_dist=True)
        self.log(f"{stage_name}/loss_P", loss_p, sync_dist=True)
        self.log(f"{stage_name}/loss_Q", loss_q, sync_dist=True)
        self.log(f"{stage_name}/loss_T", loss_t, sync_dist=True)
        self.log(f"{stage_name}/loss_U", loss_u, sync_dist=True)
        self.log(f"{stage_name}/loss_V", loss_v, sync_dist=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._compute_loss(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._compute_loss(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.get("max_epochs", 25))
        return [optimizer], [scheduler]
