"""
Vasicek bond pricing by finite differences (Crank-Nicolson).

Solves the term-structure PDE:

    dP/dt + a(b - r) dP/dr + (1/2) sigma^2 d^2P/dr^2 - r P = 0

with terminal condition P(r, T) = 1, on a grid in (r, t) space.

Outputs:
  - The full price surface P(r, t) on the grid
  - A scipy interpolator so we can query P at arbitrary (r, t)
  - A 3D mesh plot of the surface
  - A comparison with the closed-form solution
"""

import numpy as np
from scipy.sparse import diags, eye as speye
from scipy.sparse.linalg import spsolve, splu
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt
from matplotlib import cm


def solve_fdm(a, b, sigma, T, r_min, r_max, I, N):
    """Run Crank-Nicolson on the Vasicek PDE. Returns r_grid, t_grid, P, abs_err."""
    def closed_form(r, t):
        tau = T - t
        B = (1.0 - np.exp(-a * tau)) / a
        log_A = (B - tau) * (b - sigma**2 / (2 * a**2)) - (sigma**2 / (4 * a)) * B**2
        return np.exp(log_A - B * r)

    dr = (r_max - r_min) / I
    dt = T / N
    r_grid = np.linspace(r_min, r_max, I + 1)
    t_grid = np.linspace(0.0, T, N + 1)

    r_int = r_grid[1:-1]
    drift = a * (b - r_int)
    alpha = -drift / (2 * dr) + sigma**2 / (2 * dr**2)
    beta  = -sigma**2 / dr**2 - r_int
    gamma =  drift / (2 * dr) + sigma**2 / (2 * dr**2)

    L = diags([alpha[1:], beta, gamma[:-1]], offsets=[-1, 0, 1], format="csc")
    Imat  = speye(I - 1, format="csc")
    A_lhs = (Imat - 0.5 * dt * L).tocsc()
    B_rhs = (Imat + 0.5 * dt * L).tocsc()
    solve_lhs = splu(A_lhs).solve   # factor once, reuse every timestep

    P = np.empty((N + 1, I + 1))
    P[N, :] = 1.0

    for n in range(N, 0, -1):
        t_curr, t_next = t_grid[n], t_grid[n - 1]
        P_L_curr, P_R_curr = closed_form(r_min, t_curr), closed_form(r_max, t_curr)
        P_L_next, P_R_next = closed_form(r_min, t_next), closed_form(r_max, t_next)
        P[n,     0]  = P_L_curr
        P[n,    -1]  = P_R_curr
        P[n - 1, 0]  = P_L_next
        P[n - 1, -1] = P_R_next

        rhs = B_rhs @ P[n, 1:-1]
        rhs[0]  += 0.5 * dt * alpha[0]  * (P_L_curr + P_L_next)
        rhs[-1] += 0.5 * dt * gamma[-1] * (P_R_curr + P_R_next)
        P[n - 1, 1:-1] = solve_lhs(rhs)

    R, Tg = np.meshgrid(r_grid, t_grid, indexing="xy")
    abs_err = np.abs(P - closed_form(R, Tg))
    return r_grid, t_grid, P, abs_err


def solve_and_plot(label, a, b, sigma, r0, T=10.0,
                   r_min=0.0, r_max=0.08, I=200, N=500,
                   tag=None):
    """Solve the Vasicek PDE and produce the 3D plots."""
    tag = tag or label.lower()
    r_grid, t_grid, P, abs_err = solve_fdm(a, b, sigma, T, r_min, r_max, I, N)
    R, Tg = np.meshgrid(r_grid, t_grid, indexing="xy")

    print("=" * 60)
    print(f"{label}:  a={a:.5f}  b={b:.5f}  sigma={sigma:.5f}  r0={r0:.5f}")
    print(f"Grid: {I+1} points in r, {N+1} points in t")
    print(f"  dr = {(r_max-r_min)/I:.5f},  dt = {T/N:.5f}")
    print(f"max |P_FDM - P_exact|        = {abs_err.max():.3e}")
    print(f"max |P_FDM - P_exact| at t=0 = {abs_err[0].max():.3e}")
    print()

    # ------------------------------------------------------------------
    # Interpolator P(r, t)
    # ------------------------------------------------------------------
    interp = RegularGridInterpolator(
        points=(t_grid, r_grid),
        values=P,
        method="cubic",
        bounds_error=False,
        fill_value=None,
    )

    # ------------------------------------------------------------------
    # 3D surface plot
    # ------------------------------------------------------------------
    stride_r, stride_t = 10, 25
    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")

    # Plot against time-to-maturity tau = T - t (forward in tau: 0 at maturity, T today)
    Tau = T - Tg
    tau_grid = T - t_grid

    surf = ax.plot_surface(
        R, Tau, P,
        rstride=stride_t, cstride=stride_r,
        cmap=cm.viridis, alpha=0.85,
        linewidth=0, antialiased=True,
        edgecolor="none",
    )
    ax.plot_wireframe(
        R, Tau, P,
        rstride=stride_t, cstride=stride_r,
        color="white", linewidth=0.4, alpha=0.6,
    )

    # Terminal condition edge: P(r, tau=0) = 1
    ax.plot(r_grid, np.zeros_like(r_grid), np.ones_like(r_grid),
            color="red", lw=2.0, alpha=0.7, zorder=10,
            label=r"$P(r,\,\tau=0)=1$")

    # Dashed line tracing P(r0, tau) at today's short rate
    P_r0 = interp(np.column_stack([t_grid, np.full_like(t_grid, r0)]))
    ax.plot(np.full_like(tau_grid, r0), tau_grid, P_r0,
            color="black", lw=1.6, ls="--", alpha=0.65, zorder=10,
            label=rf"$r=r_0\approx{r0:.4f}$")

    ax.set_xlabel("short rate $r$", labelpad=10)
    ax.set_ylabel(r"time to maturity $\tau$ (years)", labelpad=10)
    ax.set_zlabel(r"$P(r, \tau)$", labelpad=8)
    ax.set_title(f"Vasicek bond price surface ({label}) from finite differences\n"
                 f"(Crank-Nicolson, max error vs. closed form: {abs_err.max():.1e})",
                 pad=15)
    ax.view_init(elev=10, azim=-72)
    ax.set_zlim(min(0.6, P.min() - 0.02), 1.05)
    ax.legend(loc="upper left", framealpha=0.95)

    cbar = fig.colorbar(surf, shrink=0.55, aspect=20, pad=0.08)
    cbar.set_label(r"$P(r, \tau)$")

    fig.tight_layout()
    surface_name = f"fig5_fdm_surface_{tag}.png"
    fig.savefig(surface_name, dpi=200, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Error surface (log scale)
    # ------------------------------------------------------------------
    fig2 = plt.figure(figsize=(10, 6))
    ax2  = fig2.add_subplot(111, projection="3d")
    ax2.plot_surface(
        R, Tau, np.log10(abs_err + 1e-16),
        rstride=stride_t, cstride=stride_r,
        cmap=cm.magma, alpha=0.9, edgecolor="none",
    )
    ax2.set_xlabel("short rate $r$")
    ax2.set_ylabel(r"time to maturity $\tau$ (years)")
    ax2.set_zlabel(r"$\log_{10}\,|P_{\mathrm{FDM}} - P_{\mathrm{exact}}|$")
    ax2.set_title(f"Pointwise FDM error ({label}, log scale)", pad=15)
    ax2.view_init(elev=10, azim=-72)
    fig2.tight_layout()
    error_name = f"fig6_fdm_error_{tag}.png"
    fig2.savefig(error_name, dpi=200, bbox_inches="tight")

    print(f"Saved:\n  {surface_name}\n  {error_name}\n")


def study_convergence(label, a, b, sigma, T=10.0, r_min=0.0, r_max=0.08,
                      grids=((25, 60), (50, 125), (100, 250), (200, 500),
                             (400, 1000), (800, 2000), (1600, 4000),
                             (3200, 8000), (6400, 16000)),
                      tag=None):
    """Run FDM at successively halved (dr, dt) and plot max error vs h on log-log."""
    tag = tag or label.lower()
    drs, dts, errs = [], [], []
    print(f"Convergence study ({label}):")
    print(f"{'I':>5} {'N':>6} {'dr':>10} {'dt':>10} {'max err':>12}")
    for I, N in grids:
        _, _, _, abs_err = solve_fdm(a, b, sigma, T, r_min, r_max, I, N)
        dr, dt = (r_max - r_min) / I, T / N
        e = abs_err.max()
        drs.append(dr); dts.append(dt); errs.append(e)
        print(f"{I:>5} {N:>6} {dr:>10.5f} {dt:>10.5f} {e:>12.3e}")
    drs, dts, errs = map(np.array, (drs, dts, errs))

    # Empirical order from successive refinement
    orders = np.log2(errs[:-1] / errs[1:])
    print(f"  empirical order (log2 ratio): {np.array2string(orders, precision=2)}")
    print(f"  final tolerance (finest grid): {errs[-1]:.3e}")
    print(f"  best   tolerance (any grid)  : {errs.min():.3e} "
          f"at (I, N) = {grids[int(np.argmin(errs))]}")
    print()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.loglog(drs, errs, "o-", color="#1954A6", lw=1.8, ms=7,
              label=r"max $|P_{\mathrm{FDM}} - P_{\mathrm{exact}}|$")
    # Reference slope-2 line, anchored at the coarsest point
    ref = errs[0] * (drs / drs[0]) ** 2
    ax.loglog(drs, ref, "k--", lw=1.2, alpha=0.7,
              label=r"slope $2$ reference ($\mathcal{O}(h^2)$)")
    # Mark the best (lowest-error) point — the practical tolerance floor
    i_best = int(np.argmin(errs))
    ax.loglog(drs[i_best], errs[i_best], "o", ms=12, mfc="none",
              mec="#C8102E", mew=2.0,
              label=rf"floor $\approx {errs[i_best]:.1e}$")

    ax.set_xlabel(r"grid spacing $\Delta r$ (log scale; $\Delta t \propto \Delta r$)")
    ax.set_ylabel(r"max error vs.\ closed form (log scale)")
    ax.set_title(f"Crank-Nicolson FDM convergence ({label}) -- log-log", pad=10)
    ax.grid(True, which="major", ls="-",  lw=0.6, alpha=0.45)
    ax.grid(True, which="minor", ls=":",  lw=0.5, alpha=0.25)
    ax.legend(loc="upper right", framealpha=0.95)
    ax.invert_xaxis()    # finer grids to the right
    fig.tight_layout()
    out_name = f"fig7_fdm_convergence_{tag}.png"
    fig.savefig(out_name, dpi=200, bbox_inches="tight")
    print(f"Saved:\n  {out_name}\n")


# ----------------------------------------------------------------------
# Run for both countries (parameters from estimate_ou.py, fit 2020-2026)
# ----------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

solve_and_plot(
    label="Sweden 10Y",
    a=0.357848, b=0.028277, sigma=0.007999, r0=0.027570,
    tag="sweden",
)

solve_and_plot(
    label="India 10Y",
    a=0.488935, b=0.068000, sigma=0.004988, r0=0.069330,
    tag="india",
)

study_convergence(
    label="Sweden 10Y",
    a=0.357848, b=0.028277, sigma=0.007999,
    tag="sweden",
)

plt.show()
