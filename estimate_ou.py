"""
Estimate Vasicek / Ornstein-Uhlenbeck parameters from bond yield CSVs.

Model:  dr = theta (mu - r) dt + sigma dW

Discretized AR(1):  r_{t+dt} = alpha + beta * r_t + eps,  eps ~ N(0, s^2)
where:
    beta  = exp(-theta dt)
    alpha = mu (1 - beta)
    s^2   = sigma^2 (1 - exp(-2 theta dt)) / (2 theta)

Recovered:
    theta = -ln(beta) / dt
    mu    = alpha / (1 - beta)
    sigma = sqrt( s^2 * 2 theta / (1 - exp(-2 theta dt)) )
"""

import csv
from datetime import datetime
import numpy as np


def load_csv(path):
    """Return (dates, yields_decimal) sorted ascending in time."""
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.reader(f))

    header = rows[0]
    date_col = 0
    price_col = 1  # both files: "Price" / "Senaste" in column 1

    dates, yields = [], []
    for row in rows[1:]:
        d_raw = row[date_col].strip()
        p_raw = row[price_col].strip()

        # Date: try ISO first, then MM/DD/YYYY
        try:
            d = datetime.strptime(d_raw, "%Y-%m-%d")
        except ValueError:
            d = datetime.strptime(d_raw, "%m/%d/%Y")

        # Price: Sweden uses comma decimal, India uses dot
        if "," in p_raw and "." not in p_raw:
            p = float(p_raw.replace(",", "."))
        else:
            p = float(p_raw)

        dates.append(d)
        yields.append(p / 100.0)  # percent -> decimal

    order = np.argsort(dates)
    dates = [dates[i] for i in order]
    yields = np.array([yields[i] for i in order])
    return dates, yields


def infer_dt(dates):
    """Average step in years from datetimes."""
    deltas = np.diff([d.timestamp() for d in dates])
    mean_seconds = deltas.mean()
    return mean_seconds / (365.25 * 24 * 3600)


def fit_vasicek(r, dt):
    """OLS on r_{k+1} = alpha + beta r_k + eps. Returns (theta, mu, sigma, residual_std)."""
    x = r[:-1]
    y = r[1:]
    n = len(x)

    x_bar = x.mean()
    y_bar = y.mean()
    Sxx = ((x - x_bar) ** 2).sum()
    Sxy = ((x - x_bar) * (y - y_bar)).sum()

    beta = Sxy / Sxx
    alpha = y_bar - beta * x_bar
    resid = y - (alpha + beta * x)
    s2 = (resid ** 2).sum() / (n - 2)  # unbiased

    theta = -np.log(beta) / dt
    mu = alpha / (1.0 - beta)
    sigma = np.sqrt(s2 * 2.0 * theta / (1.0 - np.exp(-2.0 * theta * dt)))

    return theta, mu, sigma, np.sqrt(s2)


def report(label, path):
    dates, r = load_csv(path)
    dt = infer_dt(dates)
    theta, mu, sigma, s = fit_vasicek(r, dt)

    print("=" * 64)
    print(label)
    print("=" * 64)
    print(f"  file        : {path}")
    print(f"  observations: {len(r)}  ({dates[0].date()}  ->  {dates[-1].date()})")
    print(f"  dt (years)  : {dt:.6f}   (~{dt*365.25:.2f} days)")
    print()
    print(f"  r0 (latest) : {r[-1]:.6f}   ({r[-1]*100:.3f}%)")
    print(f"  sample mean : {r.mean():.6f}   ({r.mean()*100:.3f}%)")
    print(f"  sample std  : {r.std(ddof=1):.6f}")
    print()
    print(f"  Vasicek MLE / OLS estimates:")
    print(f"    theta (mean reversion) = {theta:.6f}   (half-life {np.log(2)/theta:.3f} yr)")
    print(f"    mu    (long-run mean)  = {mu:.6f}   ({mu*100:.3f}%)")
    print(f"    sigma (vol)            = {sigma:.6f}")
    print(f"    residual std (s)       = {s:.6f}")
    print()


if __name__ == "__main__":
    report(
        "INDIA  10-Year Bond Yield",
        "India 10-Year Bond Yield Historical Data.csv",
    )
    report(
        "SVERIGE 10-Year Bond Yield",
        "Sverige 10-årig historiska data för avkastning i obligationer.csv",
    )
