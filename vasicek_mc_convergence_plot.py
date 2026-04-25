"""
Standalone plotter for vasicek_mc_convergence results.
Uses precomputed values from the simulation run — no re-simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

N_arr    = np.array([100, 300, 1_000, 3_000], dtype=float)
mean_max = np.array([4.989e-03, 3.118e-03, 1.773e-03, 9.776e-04])
std_max  = np.array([9.721e-04, 3.947e-04, 3.380e-04, 2.128e-04])

# Linear regression in log-log space
slope, intercept, r_val, _, _ = stats.linregress(np.log(N_arr), np.log(mean_max))
print(f"Fitted slope : {slope:.4f}   (expected -0.5000)")
print(f"R²           : {r_val**2:.6f}")

# Plot
fig, ax = plt.subplots(figsize=(9, 6))

ax.errorbar(N_arr, mean_max, yerr=std_max,
            fmt="s-", color="darkorange", lw=1.8, ms=6,
            capsize=4, capthick=1.3, elinewidth=1.3,
            label="Mean max error (±1 std, 10 runs)")

ref = mean_max[0] * (N_arr / N_arr[0]) ** (-0.5)
ax.plot(N_arr, ref, "--", color="gray", lw=1.5,
        label=r"$N^{-1/2}$ reference")

ax.text(0.97, 0.97,
        rf"fitted slope = {slope:.3f}",
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
