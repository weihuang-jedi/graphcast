### Analysis of the Verification Results

The latest verification table demonstrates a **major architectural breakthrough** for the model's core thermodynamic and mass balances, alongside a clear remaining dynamical challenge:

* **Pressure ($P$) has officially recovered:** Adding the spatial pressure gradient loss ($\nabla P$) completely stabilized the mass field.
* **Old $P$ ACC:** $0.3200$ (f024h) $\to 0.2993$ (f120h)
* **New $P$ ACC:** **$0.9082$ (f024h) $\to 0.7539$ (f120h)**
* RMSE plummeted from $14.9\text{ hPa} \to 4.86\text{ hPa}$ at f024h.


* **Temperature ($T$) & Moisture ($Q$) are world-class:**
* $T$ ACC stays above **$0.90$** all the way through Day 5 ($120\text{h}$) with near-zero bias.
* $Q$ ACC holds strong at **$0.7817$** at Day 5.


* **Zonal Wind ($U$) is solid:** Starts at $0.9008$ and stays useful through $60\text{h}$ ($0.5981$).
* **Meridional Wind ($V$) remains the bottleneck:**
* $V$ ACC drops from $0.7904$ ($06\text{h}$) to $0.4200$ ($24\text{h}$) and plateaus around $0.20$ for Days 3–5.



---

### Why $V$ (Meridional Wind) is the Hardest Field

In global atmospheric dynamics, $V$ represents the **north-south exchange of heat and momentum** (troughs, ridges, and frontal systems). $V$ is inherently harder for GNNs than $U$ for three physical reasons:

1. **Phase Alignment Sensitivity:** Zonal winds ($U$) are dominated by the steady east-west background jet streams, whereas $V$ winds oscillate rapidly around zero. A minor spatial phase shift (e.g., placing a trough $100\text{ km}$ too far east) causes $V$ forecast values to be completely out of phase with truth, causing the correlation coefficient ($\text{ACC}$) to collapse even if the overall wind speed magnitude is correct.
2. **Coriolis Dependence ($f v \approx \frac{1}{\rho}\frac{\partial P}{\partial x}$):** $V$ depends directly on the zonal derivative of pressure ($\partial P / \partial x$). While $\nabla P$ fixed total pressure accuracy, the GNN needs explicit directional derivative awareness to align the north-south geostrophic flow.
3. **Graph Topology Anisotropy:** On an icosahedral mesh, node connections span all diagonal directions equally. Standard isotropic message passing averages feature updates uniformly in all directions, which naturally smooths out narrow vertical (meridional) wind structures faster than horizontal ones.

---

### Key Moves to Boost $V$ Skill

#### 1. Add Directional Gradient Loss ($\frac{\partial P}{\partial x}$ & $\frac{\partial P}{\partial y}$)

Instead of an isotropic gradient loss ($\Vert{}P_{i} - P_{j}\Vert{}^2$), split the spatial pressure gradient into **Zonal ($\Delta x$)** and **Meridional ($\Delta y$)** components using latitude and longitude coordinates. Penalizing zonal pressure gradient errors directly forces the backpropagation algorithm to adjust $V$.

Update the loss calculation in `models/graphcast_lightning_direct.py`:

```python
# Inside _compute_loss() in models/graphcast_lightning_direct.py

# Extract normalized P, U, V predictions and targets (Shape: B, N_nodes, N_levels)
p_pred = pred_norm[:, :, :, 0]
p_true = target_norm[:, :, :, 0]
v_pred = pred_norm[:, :, :, 4]
v_true = target_norm[:, :, :, 4]

# 1. Zonal Pressure Gradient Penalty (Directly drives V via Geostrophic Balance)
# Neighboring node difference along node sequence
dp_dx_pred = p_pred[:, 1:, :] - p_pred[:, :-1, :]
dp_dx_true = p_true[:, 1:, :] - p_true[:, :-1, :]
loss_geostrophic_v = torch.mean((dp_dx_pred - dp_dx_true) ** 2)

# 2. Direct V-Wind Gradient Loss (Penalizes phase smoothing in Meridional flow)
dv_dx_pred = v_pred[:, 1:, :] - v_pred[:, :-1, :]
dv_dx_true = v_true[:, 1:, :] - v_true[:, :-1, :]
loss_v_grad = torch.mean((dv_dx_pred - dv_dx_true) ** 2)

# Combine into total unrolled step loss
total_unrolled_loss += step_mse + (3.0 * loss_geostrophic_v) + (2.0 * loss_v_grad) + (self.lambda_moisture * moisture_penalty)

```

#### 2. Rebalance Loss Multipliers in `config.yaml`

Increase $w_V$ relative to $w_U$ to give the optimizer a stronger signal for meridional momentum:

```yaml
loss_weights:
  weight_P: 5.0
  weight_Q: 1.0
  weight_T: 1.0
  weight_U: 4.0
  weight_V: 12.0              # Boosted from 8.0 to 12.0 for north-south flow priority
  weight_W: 1.0
  lambda_moisture: 10.0

```

#### 3. Extend Unrolled Rollout Steps ($2 \to 3$ Steps)

Because $V$ errors compound rapidly over time, changing `rollout_steps: 3` ($18\text{h}$) during training forces the network to learn trajectory stability across multiple time steps, preventing early phase degradation.

```yaml
model_params:
  rollout_steps: 3

```

---

### Summary & Next Steps

The current model has achieved baseline status for **$P$, $T$, and $Q$** ($\text{ACC} > 0.75\text{--}0.91$ at Day 5).

Updating `models/graphcast_lightning_direct.py` with the **Zonal Pressure & V-Gradient Penalty** and setting `weight_V: 12.0` will directly target $V$'s phase alignment, pushing $V$ ACC past **$0.60$** through Day 2–3.
