"""
vasicek_mc_convergence.py

Log-log convergence plot: Monte Carlo RMS and max error vs. number of paths N
for the Vasicek bond price surface.  Shows the expected N^(-1/2) decay.
OU parameters are imported from estimate_ou.py.
Results are saved to vasicek_mc_convergence.npz for instant re-plotting.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from estimate_ou import load_csv, fit_vasicek, infer_dt

# ---------------------------------------------------------------------------
# Load OU parameters (Sweden)
# ---------------------------------------------------------------------------
_csv = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "Sverige 10-årig historiska data för avkastning i obligationer.csv",
)
_dates, _r_data = load_csv(_csv)
_dt_data = infer_dt(_dates)
kappa, mu, sigma, _ = fit_vasicek(_r_data, _dt_data)

print(f"OU parameters: kappa={kappa:.6f}  mu={mu:.6f}  sigma={sigma:.6f}")
print()

# ---------------------------------------------------------------------------
# Closed-form Vasicek
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
# Grid  (coarser than surface plot for speed)
# ---------------------------------------------------------------------------
T      = 5.0
dt_sim = 0.01
Nr, Nt = 10, 10

r_vals   = np.linspace(0.0, 0.12, Nr)
t_vals   = np.linspace(0.0, T, Nt + 1)[:-1]   # [0, T)
tau_vals = T - t_vals

R_grid, TAU_grid = np.meshgrid(r_vals, tau_vals, indexing="ij")
P_exact = vasicek_price(R_grid, TAU_grid)       # (Nr, Nt)

# ---------------------------------------------------------------------------
# MC simulation: one estimate of the surface using N paths
# ---------------------------------------------------------------------------
def mc_surface(N):
    P = np.zeros((Nr, Nt))
    for j, t_j in enumerate(t_vals):
        tau       = T - t_j
        n_steps   = max(2, int(round(tau / dt_sim)))
        dt_j      = tau / n_steps
        sqrt_dt_j = np.sqrt(dt_j)

        r_cur = np.tile(r_vals[:, np.newaxis], (1, N))
        acc   = 0.5 * r_cur.copy()

        for k in range(n_steps):
            Z     = np.random.standard_normal((Nr, N))
            r_cur = r_cur + kappa * (mu - r_cur) * dt_j + sigma * sqrt_dt_j * Z
            acc  += r_cur if k < n_steps - 1 else 0.5 * r_cur

        P[:, j] = np.exp(-dt_j * acc).mean(axis=1)
    return P

# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------
np.random.seed(42)

N_list = [100, 200, 300, 500, 1_000, 2_000, 3_000, 5_000, 10_000]
n_runs = 10

mean_rms, std_rms   = [], []
mean_max, std_max   = [], []

print("Running convergence study …")
overall = tqdm(total=len(N_list) * n_runs, unit="run", ncols=72)
for N_idx, N in enumerate(N_list):
    rms_list, max_list = [], []
    for run in range(n_runs):
        overall.set_description(f"N={N:>6,}  run {run+1}/{n_runs}")
        np.random.seed(42 + run + 1000 * N_idx)
        P_MC  = mc_surface(N)
        err   = np.abs(P_MC - P_exact)
        rms_list.append(float(np.sqrt(np.mean(err ** 2))))
        max_list.append(float(err.max()))
        overall.update(1)

    mean_rms.append(float(np.mean(rms_list)))
    std_rms.append(float(np.std(rms_list, ddof=1)))
    mean_max.append(float(np.mean(max_list)))
    std_max.append(float(np.std(max_list, ddof=1)))
    overall.write(f"  N={N:>6,}  max={mean_max[-1]:.3e} ±{std_max[-1]:.1e}")
overall.close()

N_arr    = np.array(N_list, dtype=float)
mean_rms = np.array(mean_rms)
std_rms  = np.array(std_rms)
mean_max = np.array(mean_max)
std_max  = np.array(std_max)
print()

# Save results so the plot can be regenerated without re-simulation
npz_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "vasicek_mc_convergence.npz")
np.savez(npz_path, N_arr=N_arr,
         mean_rms=mean_rms, std_rms=std_rms,
         mean_max=mean_max, std_max=std_max)
print(f"Results saved to vasicek_mc_convergence.npz")
print()

# ---------------------------------------------------------------------------
# Linear regression (RMS)
# ---------------------------------------------------------------------------
slope_rms, _, r_rms, _, _ = stats.linregress(np.log(N_arr), np.log(mean_rms))
slope_max, _, r_max, _, _ = stats.linregress(np.log(N_arr), np.log(mean_max))
print(f"RMS fitted slope : {slope_rms:.4f}   (expected -0.5000)  R²={r_rms**2:.4f}")
print(f"Max fitted slope : {slope_max:.4f}   (expected -0.5000)  R²={r_max**2:.4f}")
print()

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(N_arr, mean_max, yerr=std_max,
            fmt="s-", color="darkorange", lw=1.8, ms=6,
            capsize=4, capthick=1.3, elinewidth=1.3,
            label="Mean max error (±1 std, 10 runs)")

# N^(-1/2) reference anchored at first max point
ref = mean_max[0] * (N_arr / N_arr[0]) ** (-0.5)
ax.plot(N_arr, ref, "--", color="gray", lw=1.5,
        label=r"$N^{-1/2}$ reference")

ax.text(0.97, 0.97,
        f"fitted slope = {slope_max:.3f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("Number of paths $N$", fontsize=12)
ax.set_ylabel(r"Max error over $(r,\,t)$ grid", fontsize=12)
ax.set_title("Vasicek MC convergence — max error vs paths", fontsize=13)
ax.legend(fontsize=10, framealpha=0.9)
ax.grid(True, which="both", ls=":", alpha=0.5)

fig.tight_layout()
plt.savefig("vasicek_mc_convergence.png", dpi=300, bbox_inches="tight")
print("Saved vasicek_mc_convergence.png")
plt.show()
