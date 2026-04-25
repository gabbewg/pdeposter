r"""
Härledning av Vasicek-modellens obligationspris via affine ansats,
samt 3D-ytplot av priset P(r, tau) som funktion av kortränta r och löptid
tau = T - t.

================================================================
1. Modell
================================================================
Under det riskneutrala måttet Q antas korträntan följa en
Ornstein-Uhlenbeck-process:

    dr_t = theta*(mu - r_t) dt + sigma dW_t,    theta, sigma > 0.

Priset vid tid t på en nollkupongsobligation med förfall T är

    P(t, T) = E^Q[ exp(-int_t^T r_s ds) | r_t = r ].

================================================================
2. PDE via Feynman-Kac
================================================================
Feynman-Kac ger att P(t, T, r) löser

    dP/dt + theta*(mu - r) * dP/dr + (1/2) sigma^2 * d^2P/dr^2 - r*P = 0,

med terminalvillkor P(T, T, r) = 1.

================================================================
3. Affine ansats
================================================================
Eftersom drift och diffusionens kvadrat är affina i r ansätter vi

    P(t, T, r) = exp( A(t,T) - B(t,T) * r ),     A(T,T) = B(T,T) = 0.

Då gäller
    dP/dt   = (A_t - B_t * r) * P,
    dP/dr   = -B * P,
    d^2P/dr^2 = B^2 * P.

Insatt i PDE:n och dividerat med P:

    A_t - B_t*r - theta*(mu - r)*B + (1/2) sigma^2 B^2 - r = 0.

Samla termer i r:

    [ -B_t + theta*B - 1 ] * r + [ A_t - theta*mu*B + (1/2) sigma^2 B^2 ] = 0.

För att detta skall hålla för alla r krävs att båda hakparenteserna
är noll, vilket ger ett system av ODE:n med terminalvillkor:

    B_t = theta*B - 1,                                  B(T,T) = 0,
    A_t = theta*mu*B - (1/2) sigma^2 B^2,               A(T,T) = 0.

================================================================
4. Lös B
================================================================
Inför löptid tau = T - t (notera att d/dtau = -d/dt). ODE:n blir

    dB/dtau = 1 - theta*B,    B(0) = 0,

en linjär första ordningens ekvation med lösningen

    B(tau) = (1 - exp(-theta*tau)) / theta.

================================================================
5. Lös A
================================================================
Eftersom A(T,T) = 0:

    A(t,T) = - int_t^T A_s ds
           = int_0^tau [ (1/2) sigma^2 B(u)^2 - theta*mu*B(u) ] du.

Beräkning av integralerna ger

    int_0^tau B(u) du   = (tau - B(tau)) / theta,
    int_0^tau B(u)^2 du = (tau - B(tau) - (theta/2) B(tau)^2) / theta^2.

Med "långsiktig ränta" R_inf := mu - sigma^2 / (2 theta^2) blir

    A(t,T) = (B(tau) - tau) * R_inf  -  sigma^2 * B(tau)^2 / (4 theta).

================================================================
6. Vasiceks formel
================================================================
Alltså

    P(t, T, r) = exp( A(t,T) - B(t,T) * r )

med
    B(tau)  = (1 - exp(-theta*tau)) / theta,
    A(t,T)  = (B(tau) - tau) * R_inf - sigma^2 B(tau)^2 / (4*theta),
    R_inf   = mu - sigma^2 / (2 theta^2).

Detta är ett affint termstrukturresultat: log P är linjär i r.

Projektarbete SF1693, KTH VT 2026.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# --- Parametrar ---------------------------------------------------------
theta = 0.35
mu = 0.0283
sigma = 0.008

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

ax.set_xlabel(r"$r$", labelpad=8)
ax.set_ylabel(r"$\tau=T-t$", labelpad=8)
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
