"""
Compare the calibrated Vasicek yield curve to observed Swedish market yields.

Inputs:
    - Sverige 10-årig historiska data för avkastning i obligationer.csv  (calibration)
    - riksbanken_data.csv  (Referensränta + SE statsobligation 2/5/7/10 år)

Outputs:
    - vasicek_vs_market_sweden.png   (3-panel figure for the poster)
"""

import csv
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

from estimate_ou import fit_vasicek, infer_dt, load_csv

# ---------------------------------------------------------------------------
# 1. Calibrate Sweden parameters from the historical 10Y yield CSV.
# ---------------------------------------------------------------------------
SE_HIST_PATH = "Sverige 10-årig historiska data för avkastning i obligationer.csv"

hist_dates, r_hist = load_csv(SE_HIST_PATH)
dt = infer_dt(hist_dates)
theta, mu, sigma, _ = fit_vasicek(r_hist, dt)
R_inf = mu - sigma ** 2 / (2.0 * theta ** 2)

print(f"Sweden Vasicek calibration: theta={theta:.4f}  mu={mu:.4f}  "
      f"sigma={sigma:.4f}  R_inf={R_inf:.4f}")


# ---------------------------------------------------------------------------
# 2. Vasicek price  P(r, tau) = exp(A(tau) - B(tau) r).
# ---------------------------------------------------------------------------
def B_coef(tau):
    return (1.0 - np.exp(-theta * tau)) / theta


def A_coef(tau):
    B = B_coef(tau)
    return (B - tau) * R_inf - sigma ** 2 * B ** 2 / (4.0 * theta)


def vasicek_price(r, tau):
    tau = np.asarray(tau, dtype=float)
    return np.exp(A_coef(tau) - B_coef(tau) * r)


# ---------------------------------------------------------------------------
# 3. Load Riksbanken's market data (long format → pivot by date).
# ---------------------------------------------------------------------------
RB_PATH = "riksbanken_data.csv"
SERIES_SHORT = "Referensränta"
SERIES_BONDS = [
    ("SE statsobligation 2 år", 2.0),
    ("SE statsobligation 5 år", 5.0),
    ("SE statsobligation 7 år", 7.0),
    ("SE statsobligation 10 år", 10.0),
]

by_date = defaultdict(dict)
with open(RB_PATH, encoding="utf-8-sig", newline="") as f:
    reader = csv.reader(f, delimiter=";")
    next(reader)
    for row in reader:
        d, _group, series, val = row
        v = val.replace(",", ".").replace("−", "-").strip()
        by_date[d][series] = float(v) / 100.0


def snapshot(target_date_str):
    """Most recent date <= target where all 5 series are present."""
    target = datetime.strptime(target_date_str, "%Y-%m-%d")
    cand = sorted(
        (d for d in by_date if datetime.strptime(d, "%Y-%m-%d") <= target),
        reverse=True,
    )
    for d in cand:
        rec = by_date[d]
        if SERIES_SHORT in rec and all(s in rec for s, _ in SERIES_BONDS):
            r0 = rec[SERIES_SHORT]
            pts = np.array(
                [(tau, rec[s]) for s, tau in SERIES_BONDS], dtype=float
            )
            return d, r0, pts[:, 0], pts[:, 1]
    raise ValueError(f"No complete snapshot on or before {target_date_str}")


# ---------------------------------------------------------------------------
# 4. Plot — three snapshots across the calibration window.
# ---------------------------------------------------------------------------
SNAPSHOT_DATES = ["2022-06-01", "2024-01-02", "2026-04-17"]
KTH_BLUE = "#1954A6"
KTH_DARK = "#0b2d5c"
ACCENT = "#e0b84a"
ACCENT_DK = "#9c7c1f"

tau_grid = np.linspace(0.0, 10.0, 300)
P_at_Rinf = vasicek_price(R_inf, tau_grid)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.6), sharey=True)

for ax, target in zip(axes, SNAPSHOT_DATES):
    d_used, r0, taus, ys = snapshot(target)
    P_model = vasicek_price(r0, tau_grid)
    P_market = np.exp(-ys * taus)

    ax.plot(tau_grid, P_model, color=KTH_BLUE, lw=2.0,
            label="Vasicek model")
    ax.scatter(taus, P_market, color=ACCENT, edgecolor=ACCENT_DK,
               s=70, zorder=5, label="Market (Riksbanken)")
    ax.plot(tau_grid, P_at_Rinf, color="gray", ls=":", lw=1.0,
            label=rf"$P(R_\infty,\tau),\ R_\infty={R_inf*100:.2f}\%$")

    ax.set_xlim(-0.3, 10.3)
    ax.set_xlabel(r"Time to maturity $\tau$ (years)")
    ax.set_title(rf"{d_used}   $r_0={r0*100:.2f}\%$", fontsize=11)
    ax.grid(alpha=0.3)

axes[0].set_ylabel(r"$P(r,\tau)$")
axes[0].legend(loc="best", fontsize=8, framealpha=0.92)

fig.suptitle(
    "Sweden: Vasicek-implied bond prices vs. market data  "
    rf"($\theta={theta:.3f},\ \mu={mu*100:.2f}\%,\ "
    rf"\sigma={sigma*100:.2f}\%$)",
    fontsize=12,
)
fig.tight_layout()

OUT = "vasicek_vs_market_sweden.png"
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved {OUT}")
