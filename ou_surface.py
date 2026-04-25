"""
3D-ytplot av den analytiska tätheten p(x, t) för Ornstein-Uhlenbeck-processen.

  p(x, t) = 1 / sqrt(2*pi*v(t)) * exp( -(x - m(t))^2 / (2*v(t)) )

med
  m(t) = x_0 * exp(-theta*t) + mu * (1 - exp(-theta*t))
  v(t) = sigma^2 / (2*theta) * (1 - exp(-2*theta*t))

Projektarbete SF1693, KTH VT 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Parametrar ---------------------------------------------------------
theta = 1.0
mu = 1.0
sigma = 0.5
x0 = -1.0

# --- Analytiska uttryck -------------------------------------------------
def ou_mean(t):
    return x0 * np.exp(-theta * t) + mu * (1.0 - np.exp(-theta * t))

def ou_var(t):
    return sigma**2 / (2.0 * theta) * (1.0 - np.exp(-2.0 * theta * t))

def ou_pdf(x, t):
    m = ou_mean(t)
    v = ou_var(t)
    return np.exp(-0.5 * (x - m)**2 / v) / np.sqrt(2.0 * np.pi * v)

# --- Rutnät --------------------------------------------------------------
# Starta vid t = t_min > 0 eftersom Var(0) = 0 ger en delta-fördelning.
t_min, t_max = 0.05, 6.0
x_min, x_max = -2.0, 3.0

Nt, Nx = 200, 240
t_vals = np.linspace(t_min, t_max, Nt)
x_vals = np.linspace(x_min, x_max, Nx)
X, Tg = np.meshgrid(x_vals, t_vals)
P = ou_pdf(X, Tg)

# --- Figur ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    X, Tg, P,
    cmap=cm.viridis,
    rstride=2, cstride=2,
    linewidth=0, antialiased=True,
    alpha=0.92,
)

# Medelvärdeskurva m(t) inritad i ytan på höjden p(m(t), t)
m_curve = ou_mean(t_vals)
p_on_mean = ou_pdf(m_curve, t_vals)
ax.plot(m_curve, t_vals, p_on_mean, color="red", lw=2.0,
        label=r"$\mathbb{E}[X_t]$")

# Stationär gauss som referens i bakkanten
v_inf = sigma**2 / (2 * theta)
p_inf = np.exp(-0.5 * (x_vals - mu)**2 / v_inf) / np.sqrt(2 * np.pi * v_inf)
ax.plot(x_vals, np.full_like(x_vals, t_max), p_inf,
        color="black", lw=1.6, ls="--",
        label=r"Stationär $\mathcal{N}(\mu,\,\sigma^2/2\theta)$")

ax.set_xlabel(r"$x$", labelpad=8)
ax.set_ylabel(r"$t$", labelpad=8)
ax.set_zlabel(r"$p(x,\,t)$", labelpad=6)
ax.set_title(
    rf"OU-täthet $p(x,t)$:  $\theta={theta},\ \mu={mu},\ "
    rf"\sigma={sigma},\ x_0={x0}$"
)
ax.view_init(elev=28, azim=-58)
ax.legend(loc="upper left", fontsize=9)

fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08, label=r"$p(x,t)$")
fig.tight_layout()

out_path = "ou_surface.png"
fig.savefig(out_path, dpi=150)
print(f"Sparade {out_path}")

plt.show()
