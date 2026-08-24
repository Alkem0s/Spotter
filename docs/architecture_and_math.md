# Mathematical Architecture & Freight Pricing Mechanics

## 1. Problem Formulation & Objective

The objective of the **Freight Pricing Intelligence Engine** is to predict full-truckload (FTL) spot rates ($\hat{y} \in \mathbb{R}^+$) for unobserved future shipments given spatial coordinates, route distance, equipment class, payload weight, and time-varying macroeconomic signals:

$$\hat{y}_i = f\left(\mathbf{x}_i^{\text{spatial}}, \mathbf{x}_i^{\text{physics}}, \mathbf{x}_i^{\text{temporal}}, \mathbf{x}_i^{\text{macro}}\right)$$

Where:
* **Spatial & Routing Vector ($\mathbf{x}_i^{\text{spatial}}$)**: Pickup/Delivery coordinates $(\phi_1, \lambda_1, \phi_2, \lambda_2)$, route corridor $O \rightarrow D$, and equipment class $E$.
* **Physics Vector ($\mathbf{x}_i^{\text{physics}}$)**: Billed mileage $d$, certified payload weight $w$, ton-mile density, and circuity ratio.
* **Macroeconomic Vector ($\mathbf{x}_i^{\text{macro}}$)**: Market tightness index $M_t$ and spot quote signal $Q_t$.

---

## 2. Geospatial Physics & Route Geometry

### A. Haversine Great-Circle Distance
To compute the theoretical minimum geodesic linehaul distance between pickup $(\phi_1, \lambda_1)$ and delivery $(\phi_2, \lambda_2)$:

$$\Delta \phi = \phi_2 - \phi_1, \quad \Delta \lambda = \lambda_2 - \lambda_1$$
$$a = \sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)$$
$$d_{\text{haversine}} = 2 R \arcsin\left(\sqrt{a}\right), \quad R = 3958.8 \text{ miles}$$

### B. Route Circuity Ratio
The circuity index measures real-world highway detour relative to Euclidean displacement:

$$\text{Circuity} = \frac{d_{\text{billed}}}{d_{\text{haversine}} + 1.0}$$

High circuity values indicate mountain routing, water crossings, or circuitous multi-stop corridors requiring higher compensation.

### C. Directional Initial Bearing
Freight lane rates exhibit strong directional asymmetry due to regional trade imbalances:

$$\theta = \text{atan2}\left(\sin(\Delta \lambda)\cos(\phi_2), \; \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta \lambda)\right)$$
$$\text{Bearing} = (\theta \times 180 / \pi + 360) \pmod{360}$$

Trigonometric decomposition ensures continuous angular representation without 0°/360° discontinuities:

$$\mathbf{x}_{\text{bearing}} = \left[\sin\left(\frac{\pi \cdot \text{Bearing}}{180}\right), \; \cos\left(\frac{\pi \cdot \text{Bearing}}{180}\right)\right]$$

---

## 3. Macroeconomic Market Signals & Causal Lag Structures

Freight spot pricing is heavily governed by daily spot market capacity tightness:
* $M_t$: Macroeconomic Market Index (national capacity tightness indicator).
* $Q_t$: Shipper Quote Signal (lane-level demand pressure).

To prevent forward-looking data leakage during out-of-time inference, all daily momentum features are calculated strictly on historical $t-k$ observations:

$$\Delta M_{7d}(t) = M_{t-1} - M_{t-7}$$
$$\Delta Q_{7d}(t) = Q_{t-1} - Q_{t-7}$$

---

## 4. Weighted Model Ensemble Architecture

The final prediction $\hat{y}$ is generated via a regularized ensemble:

$$\hat{y} = w_{\text{CB}} \hat{y}_{\text{CB}} + w_{\text{HGB}} \hat{y}_{\text{HGB}} + w_{\text{Ridge}} \hat{y}_{\text{Ridge}}$$

With calibrated blending weights:
$$w_{\text{CB}} = 0.80, \quad w_{\text{HGB}} = 0.16, \quad w_{\text{Ridge}} = 0.04$$

1. **CatBoost Regressor ($80\%$)**: Employs symmetric oblivious trees and ordered target statistics natively over high-cardinality categorical route features (`equipment_route`, `route`, `origin`, `destination`).
2. **HistGradientBoosting Regressor ($16\%$)**: Provides fast histogram-binned tree regularization with 5-fold out-of-fold target encoding to guard against overfitting.
3. **RidgeCV ($4\%$)**: Regularized L2 linear model on standardized inputs to anchor baseline predictions.

---

## 5. Residual Analysis & Irreducible Noise Floor

Residual analysis on the out-of-time validation set isolates two distinct freight pricing regimes:

1. **Standard Market Regime ($99.24\%$ of shipments)**:
   * Rates scale predictably with mileage, fuel, and equipment ($RPM \le \$3.50$).
   * Mean Absolute Error: **$\approx \$41.50$**.
   * Model $R^2$: **$> 0.965$**.

2. **Extreme Spot Market Surges ($0.76\%$ of shipments)**:
   * Occasional emergency / expedited loads surge between $\$4.00$ and $\$14.13$ per mile.
   * Because spot surges are driven by unobserved shipper distress / expedited guarantees, they are largely uncorrelated with static shipment attributes ($r < 0.006$), creating an irreducible variance floor around $\sim 648$ RMSE.
