import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Parametrar ---------------------------------------------------------
theta = 0.488935
mu = 0.068000
sigma = 0.004988

R_inf = mu - sigma**2 / (2.0 * theta**2)

# --- Vasicek-koefficienter ----------------------------------------------


def B_coef(tau):
    return (1.0 - np.exp(-theta * tau)) / theta


def A_coef(tau):
    B = B_coef(tau)
    return (B - tau) * R_inf - sigma**2 * B**2 / (4.0 * theta)


def vasicek_price(r, tau):
    return np.exp(A_coef(tau) - B_coef(tau) * r)


# --- Rutnät --------------------------------------------------------------
tau_min, tau_max = 0.0, 10.0   # löptid i år
r_min, r_max = 0.00, 0.08     # kortränta

Ntau, Nr = 200, 240
tau_vals = np.linspace(tau_min, tau_max, Ntau)
r_vals = np.linspace(r_min, r_max, Nr)
R, Tau = np.meshgrid(r_vals, tau_vals)
P = vasicek_price(R, Tau)

# --- Figur ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection="3d")

surf = ax.plot_surface(
    R, Tau, P,
    cmap=cm.viridis,
    rstride=2, cstride=2,
    linewidth=0, antialiased=True,
    alpha=0.92,
)

# P(r, 0) = 1 i framkanten (terminalvillkoret)
ax.plot(r_vals, np.zeros_like(r_vals), np.ones_like(r_vals),
        color="red", lw=2.0,
        label=r"$P(r,\,0)=1$")

# Pris vid r = R_inf (långsiktig ränta) som funktion av tau
p_at_Rinf = vasicek_price(R_inf, tau_vals)
ax.plot(np.full_like(tau_vals, R_inf), tau_vals, p_at_Rinf,
        color="black", lw=1.6, ls="--",
        label=rf"$r=R_\infty\approx{R_inf:.4f}$")

ax.set_xlabel(r"short rate $r$", labelpad=10)
ax.set_ylabel(r"time to maturity $\tau$ (years)", labelpad=10)
ax.set_zlabel(r"$P(r,\,\tau)$", labelpad=6)
ax.set_title(
    rf"Vasicek-obligationspris $P(r,\tau)=e^{{A(\tau)-B(\tau)r}}$:"
    rf"  $\theta={theta},\ \mu={mu},\ \sigma={sigma}$"
)
ax.view_init(elev=28, azim=-58)
ax.legend(loc="upper right", fontsize=9)

fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08,
             label=r"$P(r,\tau)$")
fig.tight_layout()

plt.show()
