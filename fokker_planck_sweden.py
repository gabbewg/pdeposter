"""
fokker_planck_sweden.py

3-D surface of the analytical OU transition density p(x, t) under the
Fokker-Planck equation
    d_t p = -d_x( theta (mu - x) p ) + (sigma^2 / 2) d_xx p

calibrated on the Swedish 10-year bond yield series (same parameters as
se_analytisk_vasicek / vasicek_mc_surface).

For an OU process started at X_0 = x0, X_t ~ N(m(t), v(t)) with
    m(t) = x0 e^{-theta t} + mu (1 - e^{-theta t})
    v(t) = sigma^2 / (2 theta) * (1 - e^{-2 theta t})

so the closed-form density is Gaussian:
    p(x, t) = exp(-(x - m(t))^2 / (2 v(t))) / sqrt(2 pi v(t))

Visual style matches the other result figures (Vasicek surfaces).
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_ou import load_csv, fit_vasicek, infer_dt

# ---------------------------------------------------------------------------
# Load OU parameters from Sweden data
# ---------------------------------------------------------------------------
_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Sverige 10-årig historiska data för avkastning i obligationer.csv",
)
_dates, _r_data = load_csv(_csv)
_dt_data = infer_dt(_dates)
theta, mu, sigma, _ = fit_vasicek(_r_data, _dt_data)
x0 = float(_r_data[-1])  # latest observation as starting point

print(f"OU parameters (Sweden): theta={theta:.6f}  mu={mu:.6f}  sigma={sigma:.6f}")
print(f"Initial value:          x0={x0:.6f}  ({x0*100:.3f}%)")
print()

# ---------------------------------------------------------------------------
# Closed-form OU density
# ---------------------------------------------------------------------------
def ou_mean(t):
    return x0 * np.exp(-theta * t) + mu * (1.0 - np.exp(-theta * t))

def ou_var(t):
    return sigma**2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * t))

def ou_pdf(x, t):
    m = ou_mean(t)
    v = ou_var(t)
    return np.exp(-0.5 * (x - m) ** 2 / v) / np.sqrt(2.0 * np.pi * v)

# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------
# Half-life of mean reversion roughly ln(2)/theta; choose t-window covering
# several half-lives so the stationary regime is reached.
t_min = 0.05
t_max = 4.0 * np.log(2.0) / theta

# Stationary std and a wide window around mu
v_inf = sigma**2 / (2.0 * theta)
sd_inf = np.sqrt(v_inf)
x_min = min(x0, mu) - 4.0 * sd_inf
x_max = max(x0, mu) + 4.0 * sd_inf

Nt, Nx = 60, 80
t_vals = np.linspace(t_min, t_max, Nt)
x_vals = np.linspace(x_min, x_max, Nx)

X, T = np.meshgrid(x_vals, t_vals, indexing="xy")  # (Nt, Nx)
P = ou_pdf(X, T)

# ---------------------------------------------------------------------------
# 3-D surface plot (style matches vasicek_mc_surface.py)
# ---------------------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, T, P,
    cmap=cm.viridis, alpha=0.85,
    rstride=1, cstride=1,
    linewidth=0, antialiased=True,
    edgecolor="none",
)
ax.plot_wireframe(
    X, T, P,
    rstride=2, cstride=2,
    color="white", linewidth=0.4, alpha=0.5,
)

# Stationary Gaussian as reference at the back edge (boundary, solid red)
p_inf = np.exp(-0.5 * (x_vals - mu) ** 2 / v_inf) / np.sqrt(2 * np.pi * v_inf)
ax.plot(x_vals, np.full_like(x_vals, t_max), p_inf,
        color="red", lw=2.0, alpha=0.85, zorder=10,
        label=r"stationary $\mathcal{N}(\mu,\,\sigma^2/2\theta)$")

# Mean curve m(tau) — slice through the surface (dashed black)
m_curve = ou_mean(t_vals)
p_on_mean = ou_pdf(m_curve, t_vals)
ax.plot(m_curve, t_vals, p_on_mean,
        color="black", lw=1.6, ls="--", alpha=0.85, zorder=10,
        label=r"mean $\mathbb{E}[r_\tau]$")

ax.set_xlabel(r"short rate $r$", labelpad=10)
ax.set_ylabel(r"time $\tau$ (years)", labelpad=10)
ax.set_zlabel(r"$p(r,\,\tau)$", labelpad=8)
ax.set_title(
    r"OU-täthet $p(r,\tau)$ — Fokker–Planck (Sverige)" "\n"
    rf"($\theta={theta:.4f},\ \mu={mu:.4f},\ \sigma={sigma:.4f},\ r_0={x0:.4f}$)",
    pad=15,
)
ax.view_init(elev=25, azim=-60)
ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=20, pad=0.08)
cbar.set_label(r"$p(r,\,\tau)$")

fig.tight_layout()
out_path = "fokker_planck_sweden.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"Saved {out_path}")
plt.show()
