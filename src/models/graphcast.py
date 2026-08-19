#!/usr/bin/env python3
"""
GraphCast 3D Atmospheric Neural Network Module.
"""

import torch
import torch.nn as nn
import numpy as np
from models.stage_manager import MultiStageAtmosphericManager


class DeepGraphCastModel(nn.Module):
    def __init__(
        self,
        in_channels: int = 12,
        out_channels: int = 6,
        num_levels: int = 32,
        num_nodes: int = 2562,
        hidden_dim: int = 128,
        stage: str = "M3",
        climatology_file: str = "/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/climatology_m0_m1_m2_diurnal.nc",
        **kwargs,
    ):
        super().__init__()
        self.num_levels = num_levels
        self.num_nodes = num_nodes
        self.num_vars = out_channels
        self.stage = stage

        # Encoder: Projects input features (in_channels=12) to hidden space
        self.encoder = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Processor: Multi-layer perceptron processing
        self.processor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Decoder: Maps back to physical variable updates (6 variables)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        # Stage Manager for physical invariant constraints
        self.stage_mgr = MultiStageAtmosphericManager(
            stage=stage, climatology_file=climatology_file
        )

        # Register Mesh Coordinate Buffers
        lat_linspace = torch.linspace(-np.pi / 2.0, np.pi / 2.0, num_nodes)
        lon_linspace = torch.linspace(0.0, 2.0 * np.pi, num_nodes)
        self.register_buffer("mesh_lat", lat_linspace)
        self.register_buffer("mesh_lon", lon_linspace)

    def forward(self, x, edge_index=None, timestamps=None):
        b_size = x.shape[0]

        # 1. Forward Pass through Neural Network
        h = self.encoder(x)
        h = self.processor(h) + h
        pred_delta = self.decoder(h)

        # 2. Reshape predictions to (batch, levels, nodes, vars)
        m_reshaped = pred_delta.view(b_size, self.num_levels, self.num_nodes, self.num_vars)

        x_dict = {
            "P": m_reshaped[:, :, :, 0],
            "Q": m_reshaped[:, :, :, 1],
            "T": m_reshaped[:, :, :, 2],
            "U": m_reshaped[:, :, :, 3],
            "V": m_reshaped[:, :, :, 4],
            "W": m_reshaped[:, :, :, 5],
        }

        # Extract timestamp for diurnal-seasonal climatology lookup
        timestamp_unix = (
            timestamps[0].item()
            if timestamps is not None and timestamps.numel() > 0
            else None
        )

        # 3. Enforce Physical Constraints via Stage Manager
        x_dict = self.stage_mgr.enforce_stage_constraints(
            x_dict,
            lat_rad=self.mesh_lat,
            lon_rad=self.mesh_lon,
            timestamp_unix=timestamp_unix,
        )

        # Reconstruct output tensor
        out_constrained = torch.stack(
            [x_dict["P"], x_dict["Q"], x_dict["T"], x_dict["U"], x_dict["V"], x_dict["W"]],
            dim=-1,
        )
        return out_constrained.view(b_size, self.num_levels * self.num_nodes, self.num_vars)
