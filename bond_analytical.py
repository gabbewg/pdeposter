import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_ou import load_csv, fit_vasicek, infer_dt


def plot_country(label, csv_name, out_name, final_name=None,
                 r_min=0.0, r_max=0.08, tau_max=10.0,
                 Ntau=200, Nr=240,
                 elev=10, azim=30, roll=0):
    csv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), csv_name)
    dates, r_data = load_csv(csv_path)
    dt = infer_dt(dates)
    theta, mu, sigma, _ = fit_vasicek(r_data, dt)
    r0 = float(r_data[-1])

    print(f"{label}: theta={theta:.6f}  mu={mu:.6f}  sigma={sigma:.6f}  r0={r0:.6f}")

    R_inf = mu - sigma**2 / (2.0 * theta**2)

    def B_coef(tau):
        return (1.0 - np.exp(-theta * tau)) / theta

    def A_coef(tau):
        B = B_coef(tau)
        return (B - tau) * R_inf - sigma**2 * B**2 / (4.0 * theta)

    def vasicek_price(r, tau):
        return np.exp(A_coef(tau) - B_coef(tau) * r)

    # Widen r-window if r0 is outside default
    if r0 > r_max:
        r_max = max(r_max, r0 + 0.02)
    if r0 < r_min:
        r_min = min(r_min, r0 - 0.005)

    tau_vals = np.linspace(0.0, tau_max, Ntau)
    r_vals = np.linspace(r_min, r_max, Nr)
    R, Tau = np.meshgrid(r_vals, tau_vals)
    P = vasicek_price(R, Tau)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        R, Tau, P,
        cmap=cm.viridis,
        rstride=2, cstride=2,
        linewidth=0, antialiased=True,
        alpha=0.92,
    )

    # Terminal condition P(r, 0) = 1
    ax.plot(r_vals, np.zeros_like(r_vals), np.ones_like(r_vals),
            color="red", lw=2.0,
            label=r"$P(r,\,0)=1$")

    # Pris vid r = r0 (dagens kortränta) som funktion av tau
    p_at_r0 = vasicek_price(r0, tau_vals)
    ax.plot(np.full_like(tau_vals, r0), tau_vals, p_at_r0,
            color="black", lw=1.6, ls="--",
            label=rf"$r=r_0\approx{r0:.4f}$")

    ax.set_xlabel(r"short rate $r$", labelpad=10)
    ax.set_ylabel(r"time to maturity $\tau$ (years)", labelpad=10)
    ax.set_zlabel(r"$P(r,\,\tau)$", labelpad=6)
    ax.set_title(
        rf"Vasicek-obligationspris ({label}):"
        rf"  $\theta={theta:.4f},\ \mu={mu:.4f},\ \sigma={sigma:.4f}$"
    )
    ax.view_init(elev=elev, azim=azim, roll=roll)
    ax.legend(loc="upper right", fontsize=9)

    fig.colorbar(surf, ax=ax, shrink=0.55, aspect=14, pad=0.08,
                 label=r"$P(r,\tau)$")
    fig.tight_layout()
    fig.savefig(out_name, dpi=300, bbox_inches="tight")
    print(f"Saved {out_name}")
    if final_name:
        fig.savefig(final_name, dpi=100)
        print(f"Saved {final_name}")
    plt.close(fig)


if __name__ == "__main__":
    plot_country(
        label="Sverige",
        csv_name="Sverige 10-årig historiska data för avkastning i obligationer.csv",
        out_name="se_analytisk_vasicek.png",
        final_name="swedenanalfinal.png",
        r_min=0.0, r_max=0.08,
    )
    plot_country(
        label="Indien",
        csv_name="India 10-Year Bond Yield Historical Data.csv",
        out_name="in_analytisk_vasicek.png",
        final_name="indiaanalfinal.png",
        r_min=0.04, r_max=0.10,
    )
