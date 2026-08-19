#!/usr/bin/env python3
"""
Multi-Stage Atmospheric Decomposition & Invariant Constraint Manager.
Decomposes 3D global non-hydrostatic fields into spectral/spatial stages:
- Stage M0: X0(z, t) -> Global vertical mean profile per level z.
- Stage M1: X1(lat, z, t) -> Zonally symmetric latitudinal perturbation with C2 balanced wind U1.
- Stage M2: X2(lon, lat, z, t) -> Stationary planetary waves (m=1..3) + 2D Moisture Q2(lat, z, t) + Meridional wind V2.
- Stage M3/M4: Higher-order transient baroclinic / non-hydrostatic residual stages.
"""

import os
import logging
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.nn.functional as F

from datetime import datetime, timezone, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MultiStageAtmosphericManager(nn.Module):
    """
    Multi-Stage Atmospheric Decomposition Manager using 1,464-step (366 days x 4 cycles)
    Diurnal-Seasonal Multi-Year Climatology.
    """

    def __init__(
        self,
        stage: str = "M2",
        g: float = 9.80665,
        Omega: float = 7.292115e-5,
        a: float = 6371000.0,
        rho0: float = 1.225,
        climatology_file: str = "/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/climatology_m0_m1_m2_diurnal.nc",
    ):
        super().__init__()
        self.stage = stage.upper()
        self.g = g
        self.Omega = Omega
        self.a = a
        self.rho0 = rho0

        # Savitzky-Golay 1D smoothing kernel (kernel_size=9) for C2 differentiability
        kernel = torch.tensor(
            [-21, 14, 39, 54, 59, 54, 39, 14, -21], dtype=torch.float32
        ) / 231.0
        self.register_buffer("savgol_kernel", kernel.view(1, 1, 9))

        # Load 1,464-step diurnal-seasonal climatology table if available
        self.clim_ds = None
        clim_path = climatology_file
        if not os.path.exists(clim_path):
            clim_path = os.path.join(os.path.dirname(__file__), "..", climatology_file)

        if os.path.exists(clim_path):
            logging.info(f"Loading Diurnal-Seasonal Climatology Table: {clim_path}")
            self.clim_ds = xr.open_dataset(clim_path)
        else:
            logging.warning(
                f"Climatology table not found at '{climatology_file}'. "
                f"Fallback spatial-mean mode will be used."
            )

    def smooth_1d_savgol(self, x_bin: torch.Tensor) -> torch.Tensor:
        """Applies C2 Savitzky-Golay 1D smoothing across latitudinal bins (dim=-1)."""
        num_levels, n_bins = x_bin.shape
        x_padded = F.pad(x_bin.unsqueeze(1), (4, 4), mode="replicate")
        x_smoothed = F.conv1d(
            x_padded, self.savgol_kernel.to(device=x_bin.device, dtype=x_bin.dtype)
        )
        return x_smoothed.squeeze(1)

    # Add to MultiStageAtmosphericManager in models/stage_manager.py

    def get_static_surface_features(self, device: torch.device, num_levels: int = 32, num_nodes: int = 2562):
        """
        Returns normalized land_sea_mask and elevation expanded across 32 levels.
        Shapes: (num_levels * num_nodes, 1)
        """
        if not hasattr(self, "_cached_static_features"):
            import xarray as xr
            zarr_path = "/scratch4/NAGAPE/epic/Wei.Huang/src/starviewergraphcast/data/gfs_icosahedral_m4.zarr"
            ds = xr.open_zarr(zarr_path)
    
            lsm = ds['land_sea_mask'].values if 'land_sea_mask' in ds else np.zeros((num_nodes,))
            elev = ds['elevation'].values if 'elevation' in ds else np.zeros((num_nodes,))
            ds.close()

            # Normalize elevation (meters -> kilometers)
            elev_norm = elev / 1000.0

            # Expand across 32 height levels
            lsm_3d = torch.tensor(lsm, dtype=torch.float32).unsqueeze(0).repeat(num_levels, 1)      # (32, 2562)
            elev_3d = torch.tensor(elev_norm, dtype=torch.float32).unsqueeze(0).repeat(num_levels, 1) # (32, 2562)

            self._cached_lsm = lsm_3d.view(-1, 1).to(device)
            self._cached_elev = elev_3d.view(-1, 1).to(device)
            self._cached_static_features = True

        return self._cached_lsm.to(device), self._cached_elev.to(device)

    # Update get_clim_slice method inside MultiStageAtmosphericManager
    def get_clim_slice(self, timestamp_val):
        if hasattr(timestamp_val, "item"):
            timestamp_val = timestamp_val.item()
        if isinstance(timestamp_val, (int, float)) and timestamp_val > 1e11:
            timestamp_val = timestamp_val / 1e9
    
        dt = datetime.fromtimestamp(float(timestamp_val), tz=timezone.utc)
        target_doy = dt.timetuple().tm_yday
        target_cycle = (dt.hour // 6) * 6

        return self.clim_ds.sel(dayofyear=target_doy, cycle_hour=target_cycle)

    def get_clim_slice(self, timestamp_val):
        if hasattr(timestamp_val, "item"):
            timestamp_val = timestamp_val.item()
        # Handle nanosecond timestamps (e.g. > 1e12)
        if timestamp_val > 1e11:
            timestamp_val = timestamp_val / 1e9

        dt = datetime.fromtimestamp(timestamp_val, tz=timezone.utc)
        target_doy = dt.timetuple().tm_yday
        target_cycle = (dt.hour // 6) * 6
        return self.clim_ds.sel(dayofyear=target_doy, cycle_hour=target_cycle)

    def compute_m0_baseline(
        self,
        init_ds,
        device: torch.device,
        lat_rad: torch.Tensor,
        timestamp_unix: float = None,
    ) -> torch.Tensor:
        """
        Computes X0(z, t): 1D Global Vertical Mean Profile per level z for current timestamp.
        Returns tensor of shape (1, num_levels * n_nodes, 6).
        """
        if timestamp_unix is not None and self.clim_ds is not None:
            slice_ds = self.get_clim_slice(timestamp_unix)
            p0_z = torch.tensor(
                slice_ds["P0_clim"].values, device=device, dtype=torch.float32
            ).unsqueeze(-1)
            t0_z = torch.tensor(
                slice_ds["T0_clim"].values, device=device, dtype=torch.float32
            ).unsqueeze(-1)
        else:
            var_p = "p_icosahedral" if "p_icosahedral" in init_ds else ("P" if "P" in init_ds else "p")
            var_t = "t_icosahedral" if "t_icosahedral" in init_ds else ("T" if "T" in init_ds else "t")

            p_data = init_ds[var_p]
            t_data = init_ds[var_t]

            if "time" in p_data.dims:
                p_data = p_data.isel(time=-1)
                t_data = t_data.isel(time=-1)
            elif "step" in p_data.dims:
                p_data = p_data.isel(step=-1)
                t_data = t_data.isel(step=-1)

            p_np = np.asarray(p_data.values, dtype=np.float32)
            t_np = np.asarray(t_data.values, dtype=np.float32)

            while p_np.ndim > 2:
                p_np = p_np[0]
                t_np = t_np[0]

            p0_z = torch.tensor(p_np.mean(axis=1, keepdims=True), device=device, dtype=torch.float32)
            t0_z = torch.tensor(t_np.mean(axis=1, keepdims=True), device=device, dtype=torch.float32)

        num_levels = p0_z.shape[0]
        n_nodes = lat_rad.shape[0]

        p0_nodes = p0_z.expand(num_levels, n_nodes)
        t0_nodes = t0_z.expand(num_levels, n_nodes)
        zeros_nodes = torch.zeros_like(p0_nodes)

        x0_nodes = torch.stack(
            [p0_nodes, zeros_nodes, t0_nodes, zeros_nodes, zeros_nodes, zeros_nodes],
            dim=-1,
        )
        return x0_nodes.unsqueeze(0).reshape(1, num_levels * n_nodes, 6).contiguous()

    def compute_m1_balanced_wind(
        self, p1_lat: torch.Tensor, lat_rad_1d: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes zonal wind jet U1(lat, z, t) from P1(lat, z, t) using C2 Savitzky-Golay
        filtering and Hermite equatorial blending to eliminate saw-tooth noise.
        """
        num_levels, n_lats = p1_lat.shape
        p1_pa = p1_lat * 100.0 if p1_lat.abs().max() < 2000.0 else p1_lat

        p1_smoothed = self.smooth_1d_savgol(p1_pa)
        p1_smoothed = self.smooth_1d_savgol(p1_smoothed)

        dphi = lat_rad_1d[1:] - lat_rad_1d[:-1]
        mean_dphi = torch.mean(dphi).clamp(min=1e-5)

        dp1_dphi = torch.zeros_like(p1_smoothed)
        dp1_dphi[:, 1:-1] = (p1_smoothed[:, 2:] - p1_smoothed[:, :-2]) / (
            lat_rad_1d[2:] - lat_rad_1d[:-2]
        ).unsqueeze(0)
        dp1_dphi[:, 0] = (p1_smoothed[:, 1] - p1_smoothed[:, 0]) / dphi[0]
        dp1_dphi[:, -1] = (p1_smoothed[:, -1] - p1_smoothed[:, -2]) / dphi[-1]

        d2p1_dphi2 = torch.zeros_like(p1_smoothed)
        d2p1_dphi2[:, 1:-1] = (
            p1_smoothed[:, 2:] - 2.0 * p1_smoothed[:, 1:-1] + p1_smoothed[:, :-2]
        ) / (mean_dphi**2)

        f = 2.0 * self.Omega * torch.sin(lat_rad_1d)
        beta = 2.0 * self.Omega * torch.cos(lat_rad_1d) / self.a

        f_sign = torch.sign(f)
        f_sign[f_sign == 0] = 1.0
        f_clamped = f_sign * torch.clamp(
            torch.abs(f), min=2.0 * self.Omega * np.sin(np.radians(15.0))
        )
        u_geo = -(1.0 / (self.rho0 * f_clamped * self.a)).unsqueeze(0) * dp1_dphi

        beta_clamped = torch.clamp(beta, min=1e-12)
        u_beta = -(1.0 / (self.rho0 * beta_clamped * (self.a**2))).unsqueeze(0) * d2p1_dphi2

        eq_idx = torch.abs(torch.rad2deg(lat_rad_1d)) <= 5.0
        u_beta[:, eq_idx] = u_beta[:, eq_idx].mean(dim=-1, keepdim=True)

        lat_deg_abs = torch.abs(torch.rad2deg(lat_rad_1d))
        x = torch.clamp((lat_deg_abs - 5.0) / 15.0, min=0.0, max=1.0).unsqueeze(0)
        w_hermite = 3.0 * (x**2) - 2.0 * (x**3)

        u1_balanced = w_hermite * u_geo + (1.0 - w_hermite) * u_beta
        u1_final = self.smooth_1d_savgol(u1_balanced)
        return torch.clamp(u1_final, min=-65.0, max=75.0)

    def enforce_stage_constraints(
        self, x_dict: dict, lat_rad: torch.Tensor, lon_rad: torch.Tensor = None, timestamp_unix: float = None
    ) -> dict:
        """
        Enforces Stage M0, M1, or M2 physical invariants grounded in 1,464-step climatology.
        """
        target_dtype = x_dict["P"].dtype
        num_levels = x_dict["P"].shape[1]
        n_nodes = lat_rad.shape[0]

        lats_deg = torch.rad2deg(lat_rad)
        bin_edges = torch.linspace(-90.0, 90.0, 181, device=lat_rad.device)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
        bin_indices = torch.bucketize(lats_deg, bin_edges) - 1
        bin_indices = torch.clamp(bin_indices, 0, 179)

        if self.stage == "M0":
            x_dict["U"].zero_()
            x_dict["V"].zero_()
            x_dict["W"].zero_()
            x_dict["Q"].zero_()

        elif self.stage in ["M1", "M2"]:
            # 1. Read M0/M1 directly from 1,464-step diurnal-seasonal climatology
            if timestamp_unix is not None and self.clim_ds is not None:
                slice_ds = self.get_clim_slice(timestamp_unix)

                p1_clim_bin = torch.tensor(
                    slice_ds["P1_clim"].values, device=lat_rad.device, dtype=target_dtype
                )
                t1_clim_bin = torch.tensor(
                    slice_ds["T1_clim"].values, device=lat_rad.device, dtype=target_dtype
                )

                p1_clim_bin = self.smooth_1d_savgol(p1_clim_bin)
                t1_clim_bin = self.smooth_1d_savgol(t1_clim_bin)

                for b in range(180):
                    mask = bin_indices == b
                    if mask.any():
                        x_dict["P"][..., mask] = p1_clim_bin[:, b].unsqueeze(-1)
                        x_dict["T"][..., mask] = t1_clim_bin[:, b].unsqueeze(-1)

            # 2. Ring-average P1 and T1 across latitude bins (d/d_lambda = 0)
            for key in ["P", "T"]:
                for b in range(180):
                    mask = bin_indices == b
                    if mask.any():
                        ring_mean = torch.mean(x_dict[key][..., mask], dim=-1, keepdim=True).to(target_dtype)
                        x_dict[key][..., mask] = ring_mean

            # 3. Derive U1 geostrophically from P1 with Equatorial Beta-Blend
            p1_field = x_dict["P"].to(torch.float32)
            p1_bin_list = []
            for b in range(180):
                mask = bin_indices == b
                if mask.any():
                    p1_bin_list.append(p1_field[..., mask].mean(dim=-1))
                else:
                    p1_bin_list.append(torch.zeros(p1_field.shape[:-1], device=lat_rad.device))

            p1_bin = torch.stack(p1_bin_list, dim=-1)
            bin_centers_rad = torch.deg2rad(bin_centers).to(device=lat_rad.device, dtype=torch.float32)

            u1_target_list = []
            for b in range(p1_bin.shape[0]):
                u1_bal = self.compute_m1_balanced_wind(p1_bin[b], bin_centers_rad)
                u1_target_list.append(u1_bal)

            u1_bin = torch.stack(u1_target_list, dim=0).to(dtype=target_dtype)

            # Polar Tapering (|lat| > 82 deg) to eliminate polar spikes
            lat_abs = torch.abs(bin_centers)
            polar_mask = lat_abs > 82.0
            if polar_mask.any():
                taper = torch.cos(
                    torch.deg2rad((lat_abs[polar_mask] - 82.0) / 8.0 * 90.0)
                ).to(device=lat_rad.device, dtype=target_dtype)
                u1_bin[..., polar_mask] *= taper

            x_dict["U"] = torch.zeros_like(x_dict["P"])
            for b in range(180):
                mask = bin_indices == b
                if mask.any():
                    x_dict["U"][..., mask] = u1_bin[..., b].unsqueeze(-1)

            if self.stage == "M1":
                # Strict zeros for unmodeled fields in M1
                zeros_ref = torch.zeros_like(x_dict["P"])
                x_dict["V"] = zeros_ref
                x_dict["W"] = zeros_ref
                x_dict["Q"] = zeros_ref

            elif self.stage == "M2":
                # 4. Populate 2D Specific Humidity Q2(lat, z, t)
                if timestamp_unix is not None and self.clim_ds is not None and "Q2_clim" in self.clim_ds:
                    slice_ds = self.get_clim_slice(timestamp_unix)
                    q2_clim_bin = torch.tensor(
                        slice_ds["Q2_clim"].values, device=lat_rad.device, dtype=target_dtype
                    )
                    q2_clim_bin = torch.clamp(self.smooth_1d_savgol(q2_clim_bin), min=0.0)

                    for b in range(180):
                        mask = bin_indices == b
                        if mask.any():
                            x_dict["Q"][..., mask] = q2_clim_bin[:, b].unsqueeze(-1)
                else:
                    x_dict["Q"].zero_()

                # 5. Add Stationary Planetary Waves m=1..3 for P2 and T2
                dp2_dlon = torch.zeros_like(x_dict["P"])

                if lon_rad is not None and timestamp_unix is not None and self.clim_ds is not None and "P2_A_clim" in self.clim_ds:
                    slice_ds = self.get_clim_slice(timestamp_unix)
                    p2_a = torch.tensor(slice_ds["P2_A_clim"].values, device=lat_rad.device, dtype=target_dtype)
                    p2_b = torch.tensor(slice_ds["P2_B_clim"].values, device=lat_rad.device, dtype=target_dtype)
                    t2_a = torch.tensor(slice_ds["T2_A_clim"].values, device=lat_rad.device, dtype=target_dtype)
                    t2_b = torch.tensor(slice_ds["T2_B_clim"].values, device=lat_rad.device, dtype=target_dtype)

                    lon_3d = lon_rad.unsqueeze(0).unsqueeze(0)  # (1, 1, n_nodes)

                    p2_field = torch.zeros_like(x_dict["P"])
                    t2_field = torch.zeros_like(x_dict["T"])

                    for m in range(1, 4):
                        cos_m = torch.cos(m * lon_3d)
                        sin_m = torch.sin(m * lon_3d)

                        for b in range(180):
                            mask = bin_indices == b
                            if mask.any():
                                p2_wave = p2_a[:, b, m-1].unsqueeze(-1) * cos_m[..., mask] + \
                                          p2_b[:, b, m-1].unsqueeze(-1) * sin_m[..., mask]
                                t2_wave = t2_a[:, b, m-1].unsqueeze(-1) * cos_m[..., mask] + \
                                          t2_b[:, b, m-1].unsqueeze(-1) * sin_m[..., mask]

                                p2_field[..., mask] += p2_wave
                                t2_field[..., mask] += t2_wave

                                # dP2/d_lambda for meridional wind V2
                                dp2_dlon[..., mask] += m * (-p2_a[:, b, m-1].unsqueeze(-1) * sin_m[..., mask] + \
                                                             p2_b[:, b, m-1].unsqueeze(-1) * cos_m[..., mask])

                    x_dict["P"] += p2_field
                    x_dict["T"] += t2_field

                # 6. Derive Meridional Geostrophic Wind V2 from dP2/d_lambda
                f = 2.0 * self.Omega * torch.sin(lat_rad).unsqueeze(0).unsqueeze(0)
                f_sign = torch.sign(f)
                f_sign[f_sign == 0] = 1.0
                f_clamped = f_sign * torch.clamp(torch.abs(f), min=2.0 * self.Omega * np.sin(np.radians(15.0)))

                cos_lat = torch.cos(lat_rad).clamp(min=1e-3).unsqueeze(0).unsqueeze(0)
                dp2_pa = dp2_dlon * 100.0 if dp2_dlon.abs().max() < 2000.0 else dp2_dlon

                v2_geo = (1.0 / (self.rho0 * f_clamped * self.a * cos_lat)) * dp2_pa
                x_dict["V"] = torch.clamp(v2_geo, min=-45.0, max=45.0)
                x_dict["W"].zero_()

            elif self.stage == "M3":
                # 1. Enforce Stage M0 + M1 + M2 background climatology (P, T, U, V, Q)
                x_dict = self.enforce_stage_constraints(x_dict, lat_rad, lon_rad, timestamp_unix=timestamp_unix)

                # 2. Stage M3 unlocks dynamic, ageostrophic meridional wind (V) and full moisture (Q)
                # We preserve the neural network's dynamic predicted residual for synoptic waves
                if "V_pred" in x_dict:
                    x_dict["V"] = x_dict["V"] + x_dict["V_pred"]

                # 3. Apply high-latitude polar dampening on vertical velocity W to preserve CFL stability
                if "W" in x_dict:
                    lat_abs = torch.abs(torch.rad2deg(lat_rad))
                    polar_mask = lat_abs > 80.0
                    if polar_mask.any():
                        taper = torch.cos(torch.deg2rad((lat_abs[polar_mask] - 80.0) / 10.0 * 90.0)).to(
                            device=lat_rad.device, dtype=x_dict["W"].dtype
                        )
                        x_dict["W"][..., polar_mask] *= taper

        return x_dict
