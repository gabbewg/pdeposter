"""
vasicek_mc_surface.py

Vasicek zero-coupon bond price P(r, tau) via Monte Carlo simulation,
where tau = T - t is time to expiry.

For each grid point (r_i, tau_j > 0) we estimate
    P = E[exp(-∫_0^{tau_j} r_s ds) | r_0 = r_i]
by simulating N OU paths with Euler-Maruyama and the trapezoidal rule.
At tau=0, P=1 exactly for all r.

OU parameters (kappa, mu, sigma) are imported from estimate_ou.py.
"""

from estimate_ou import load_csv, fit_vasicek, infer_dt
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Load OU parameters from Sweden data (same as se_analytisk_vasicek)
# ---------------------------------------------------------------------------
_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Sverige 10-årig historiska data för avkastning i obligationer.csv",
)
_dates, _r_data = load_csv(_csv)
_dt_data = infer_dt(_dates)
kappa, mu, sigma, _ = fit_vasicek(_r_data, _dt_data)
r0 = float(_r_data[-1])

print(
    f"OU parameters (Sweden): kappa={kappa:.6f}  mu={mu:.6f}  sigma={sigma:.6f}")
print(f"Current short rate:      r0={r0:.6f}  ({r0*100:.3f}%)")
print()

# ---------------------------------------------------------------------------
# Closed-form Vasicek (benchmark for error check)
# ---------------------------------------------------------------------------


def _B(tau):
    return (1.0 - np.exp(-kappa * tau)) / kappa


def _A(tau):
    B = _B(tau)
    R_inf = mu - sigma**2 / (2.0 * kappa**2)
    return (B - tau) * R_inf - sigma**2 * B**2 / (4.0 * kappa)


def vasicek_price(r, tau):
    return np.exp(_A(tau) - _B(tau) * r)


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
T = 10.0
N = 5000        # MC paths
dt_sim = 0.01   # Euler-Maruyama step size
Nr, Nt = 25, 25

r_vals = np.linspace(0.0, 0.12, Nr)
# tau in [0, T] — tau=0 gives P=1 exactly; simulate only tau>0
tau_vals = np.linspace(0.0, T, Nt)   # 25 points: 0, T/(Nt-1), …, T

# ---------------------------------------------------------------------------
# Monte Carlo surface  (vectorised over paths; loop over grid points)
# ---------------------------------------------------------------------------
np.random.seed(42)

P_MC = np.zeros((Nr, Nt))
P_anal = np.zeros((Nr, Nt))

print("Running Monte Carlo surface simulation …")
for j, tau in enumerate(tau_vals):
    if tau == 0.0:
        # Terminal condition: bond matures now, price = 1 everywhere
        P_MC[:, j] = 1.0
        P_anal[:, j] = 1.0
        continue

    n_steps = max(2, int(round(tau / dt_sim)))
    dt_j = tau / n_steps   # exact so paths span exactly tau
    sqrt_dt_j = np.sqrt(dt_j)

    # Shape (Nr, N): all r starting values × all paths simultaneously
    r_cur = np.tile(r_vals[:, np.newaxis], (1, N))   # (Nr, N)
    acc = 0.5 * r_cur.copy()                        # trapezoidal first half-weight

    for k in range(n_steps):
        Z = np.random.standard_normal((Nr, N))
        r_cur = r_cur + kappa * (mu - r_cur) * dt_j + sigma * sqrt_dt_j * Z
        # full weight for interior points, half weight for last
        acc += r_cur if k < n_steps - 1 else 0.5 * r_cur

    integral = dt_j * acc
    P_MC[:, j] = np.exp(-integral).mean(axis=1)
    P_anal[:, j] = vasicek_price(r_vals, tau)

print("Done.")

# ---------------------------------------------------------------------------
# Convergence sanity check
# ---------------------------------------------------------------------------
# exclude tau=0 from error (P=1 exactly there by construction)
max_err = np.abs(P_MC[:, 1:] - P_anal[:, 1:]).max()
print(f"\nMax |P_MC - P_analytical| across grid (tau>0): {max_err:.4e}\n")

# ---------------------------------------------------------------------------
# 3-D surface plot  (style matches fig5_fdm_surface.png)
# ---------------------------------------------------------------------------
R_grid, TAU_grid = np.meshgrid(r_vals, tau_vals, indexing="ij")   # (Nr, Nt)

fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    R_grid, TAU_grid, P_MC,
    cmap=cm.viridis, alpha=0.85,
    rstride=1, cstride=1,
    linewidth=0, antialiased=True,
    edgecolor="none",
)
ax.plot_wireframe(
    R_grid, TAU_grid, P_MC,
    rstride=1, cstride=1,
    color="white", linewidth=0.4, alpha=0.5,
)

# Terminal condition P(r, tau=0) = 1  — front edge, all rates
ax.plot(r_vals, np.zeros_like(r_vals), np.ones_like(r_vals),
        color="red", lw=2.0, alpha=0.8, zorder=10,
        label=r"$P(r,\,0)=1$")

# Current short rate r0 reference curve across all tau
if r_vals[0] <= r0 <= r_vals[-1]:
    p_r0 = vasicek_price(r0, tau_vals)
    ax.plot(np.full_like(tau_vals, r0), tau_vals, p_r0,
            color="black", lw=1.6, ls="--", alpha=0.75, zorder=10,
            label=rf"$r=r_0\approx{r0:.4f}$")

ax.set_xlabel("short rate $r$", labelpad=10)
ax.set_ylabel(r"time to maturity $\tau$ (years)", labelpad=10)
ax.set_zlabel(r"$P(r,\,\tau)$", labelpad=8)
ax.set_title(
    "Vasicek bond price surface — Monte Carlo\n"
    rf"($\kappa={kappa:.4f},\ \mu={mu:.4f},\ \sigma={sigma:.4f},\ T={T}$)",
    pad=15,
)
ax.view_init(elev=10, azim=30, roll=0)
ax.set_zlim(min(0.6, P_MC.min() - 0.02), 1.05)
ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=20, pad=0.08)
cbar.set_label(r"$P(r,\,\tau)$")

fig.tight_layout()
plt.savefig("se_vasicek_mc_surface.png", dpi=300, bbox_inches="tight")
print("Saved se_vasicek_mc_surface.png")
fig.savefig("mcfinalgraf.png", dpi=100)
print("Saved mcfinalgraf.png")
plt.show()
