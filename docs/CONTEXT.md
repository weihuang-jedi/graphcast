# CONTEXT.md: AIDA GNN Weather Model (M6 Mesh & Terrain-Following Upgrade)

## 1. Project Overview & Objective
This project implements a direct, end-to-end 3D GraphCast AI weather forecasting system on an icosahedral global grid. The model predicts physical atmospheric state increments ($\Delta X$) across 6 primary dynamic state variables ($P, Q, T, U, V, W$) over 32 vertical levels.

The pipeline has been upgraded from a 400 km flat $z$-coordinate grid to a high-resolution **M6 icosahedral mesh (~100 km resolution)** operating on a **hybrid terrain-following ($\eta$) coordinate system**.

---

## 2. Core Architectural & Mathematical Choices

### A. End-to-End Direct Prediction in Log-State Space
- **Elimination of Climatology Backgrounds:** Rather than decomposing fields into baseline climatology tables ($X_0 + X_1 + X_2 + X_3$), the model directly predicts full physical state transitions ($X_{t+\Delta t} = X_t + \Delta X_{\text{model}}$).
- **Logarithmic State Space:** To maintain numerical stability across multi-order-of-magnitude variables:
  - Pressure: $\ln P \sim 11.52 \text{ ln(Pa)}$
  - Temperature: $\ln T \sim 5.57 \text{ ln(K)}$
  - Humidity: $Q \text{ (kg/kg)}$
  - Winds: $U, V \text{ (m/s)}$, $W \text{ (Pa/s)}$
- **Standardization Buffers:** Input state channels are standardized using log-space empirical parameters ($\mu_{\ln P} = 11.52, \mu_{\ln T} = 5.57, \sigma_{\ln P} = 0.25, \sigma_{\ln T} = 0.15$).

### B. M6 Mesh Topology & Memory Alignment
- **Node Scale:** M6 grid containing **40,962 horizontal nodes** $\times$ **32 vertical levels** = **1,310,784 total 3D points**.
- **Hierarchy Levels:** `hierarchy_levels: [6, 5, 4, 3, 2, 1, 0]`.
- **Nodes-First Tensor Memory Layout:** Input arrays are formatted in strict `(nodes=40962, levels=32)` ordering. This prevents horizontal message-passing stride mismatches across vertical level boundaries (which previously produced $10^\circ\text{--}160^\circ$ longitude S-shape shadows).

### C. Terrain-Following ($\eta$) Vertical Coordinates
- **3D Topographic Coordinate Feature:** The model accepts 15 input channels (12 dynamic history channels + 2 static 2D surface channels `land_sea_mask` & `elevation` + 1 3D terrain height channel $Z_{3\text{D}}$).
- **Vertical Regridding:** In post-processing, 3D terrain-following pressure layers $P(k, i)$ are interpolated back to standard GFS isobaric pressure levels ($1000\text{ hPa} \dots 10\text{ hPa}$) using deduplicated 1D vertical spline interpolation (`terrain_to_isobaric`).

---

## 3. Key Pipeline Components

| Script | Path | Function |
| :--- | :--- | :--- |
| **Model Module** | `models/graphcast_lightning_direct.py` | PyTorch Lightning module with log-state standardization, 15 input channels, and weighted loss scaling. |
| **Training Executable** | `training/training.py` | PyTorch dataloader for Zarr datasets (`gfs_icosahedral_m6.zarr`) with Nodes-First memory layout formatting. |
| **Slurm Script** | `training.slurm` | Slurm job runner with `expandable_segments:True` and `batch_size: 1` + `accumulate_grad: 4` for GPU memory safety. |
| **Rollout Driver** | `rollout/forecast.py` | Autoregressive forecast driver supporting single or separate $t_0 / t_{-6\text{h}}$ NetCDF files and rollout wind step inflation (`--alpha_wind`). |
| **Lat-Lon Regridder** | `utils/reconstruct_forecast_latlon.py` | 3D KD-Tree Gaussian regridder ($\sigma = 0.0125$) with 1D vertical terrain-to-isobaric spline interpolation. |
| **Diagnostic Panel** | `utils/plot_forecast_panel_comparison.py` | Generates 4x7 diagnostic panel maps and outputs terminal summary tables (Min, Max, Bias, RMSE, ACC). |

---

## 4. Current Performance Metrics

Diagnostic metrics evaluated at 24-hour lead time ($f0024h$, level index 10):

```text
============================================================================================
 FORECAST DIAGNOSTIC METRICS SUMMARY (Level Index: 11)
============================================================================================
VAR   | TRUTH MIN  | TRUTH MAX  | FCST MIN   | FCST MAX   | BIAS (MEAN) | RMSE       | ACC
--------------------------------------------------------------------------------------------
T     | 222.3583   | 309.9827   | 234.2701   | 307.9674   | 0.3431      | 4.1330     | 0.9539
U     | -46.9057   | 35.7544    | -19.9893   | 18.9196    | -0.3804     | 5.8013     | 0.7470
V     | -42.3732   | 39.1350    | -14.9264   | 13.8108    | 0.1207      | 6.1149     | 0.4826
W     | -4.7900    | 4.2347     | -0.3320    | 1.0422     | -0.0075     | 0.3469     | 0.1705
Q     | 0.0001     | 29.1588    | 0.0000     | 18.8489    | 0.1475      | 2.7688     | 0.8861
RHO   | 1.0556     | 1.4833     | 1.0609     | 1.4366     | -0.0022     | 0.0202     | 0.9350
P     | 898.3631   | 983.1854   | 891.2394   | 978.8201   | -0.3648     | 4.8897     | 0.9012
============================================================================================

Thermodynamics & Pressure: Excellent spatial pattern correlation (ACC>0.88–0.95).

Kinetic Wind Recovery: Step inflation factor (--alpha_wind 2.1) applied in rollout/forecast.py counteracts MSE regression damping to preserve kinetic wind energy.

5. Standard Execution Workflow
A. Training
Bash
sbatch training.slurm
B. Rollout Forecast (6-hour steps)
Bash
PYTHONPATH=. python rollout/forecast.py \
    -c checkpoints/last.ckpt \
    --input_file_t0 ${truthdir}/icosahedral_logstate_m6.20260201.t06z.0p25.f000.nc \
    --input_file_tm1 ${truthdir}/icosahedral_logstate_m6.20260201.t00z.0p25.f000.nc \
    --init_time 2026020106 \
    -s 20 \
    --step_hours 6 \
    --alpha_wind 2.1 \
    --output_dir output \
    --device cpu
C. Lat-Lon Reconstruction & Diagnostic Panel
Bash
# 1. Regrid to 0.25-degree lat-lon grid
PYTHONPATH=. python utils/reconstruct_forecast_latlon.py \
    --fcst_dir output \
    --grid_ref ${truthdir}/icosahedral_logstate_m6.20260201.t06z.0p25.f000.nc \
    --truth_ref ${griddir}/gfs.20260101.t00z.0p25.f000.nc \
    --out_dir output_reconstructed

# 2. Generate 4x7 Diagnostic Panel Plot
genpanel.sh
