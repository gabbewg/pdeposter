"""
Vasicek bond pricing: Monte Carlo vs. closed form.

Model:  dr = a(b - r) dt + sigma dW
Price:  P(t,T) = E^Q[ exp(-∫_t^T r_u du) | r_t ]
"""

import numpy as np


# ----------------------------------------------------------------------
# Closed form (the benchmark)
# ----------------------------------------------------------------------
def vasicek_closed_form(r0, a, b, sigma, t, T):
    """Analytical bond price P(t,T) = A(t,T) * exp(-B(t,T) * r_t)."""
    tau = T - t
    B = (1.0 - np.exp(-a * tau)) / a
    log_A = (B - tau) * (b - sigma**2 / (2 * a**2)) - (sigma**2 / (4 * a)) * B**2
    return np.exp(log_A - B * r0)


# ----------------------------------------------------------------------
# Monte Carlo with EXACT transitions
# ----------------------------------------------------------------------
def vasicek_mc_exact(r0, a, b, sigma, t, T, n_paths, n_steps, seed=None):
    """
    Simulate r using the exact Gaussian transition, then trapezoidal-integrate
    each path and average exp(-integral).

    Returns: estimate, standard error, 95% CI half-width.
    """
    rng = np.random.default_rng(seed)
    dt = (T - t) / n_steps

    # Exact transition: r_{k+1} | r_k ~ N(r_k * m + b*(1-m), v)
    m = np.exp(-a * dt)
    v = (sigma**2 / (2 * a)) * (1.0 - np.exp(-2 * a * dt))
    sd = np.sqrt(v)

    # Simulate all paths in parallel, shape (n_paths, n_steps + 1)
    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        r[:, k + 1] = r[:, k] * m + b * (1 - m) + sd * Z[:, k]

    # Trapezoidal rule along each path
    integral = dt * (0.5 * r[:, 0] + r[:, 1:-1].sum(axis=1) + 0.5 * r[:, -1])
    discount = np.exp(-integral)

    P_hat = discount.mean()
    se = discount.std(ddof=1) / np.sqrt(n_paths)
    return P_hat, se, 1.96 * se


# ----------------------------------------------------------------------
# Monte Carlo with EULER discretization (for comparison)
# ----------------------------------------------------------------------
def vasicek_mc_euler(r0, a, b, sigma, t, T, n_paths, n_steps, seed=None):
    """Same as above but with Euler-Maruyama instead of exact transitions."""
    rng = np.random.default_rng(seed)
    dt = (T - t) / n_steps
    sqrt_dt = np.sqrt(dt)

    r = np.empty((n_paths, n_steps + 1))
    r[:, 0] = r0
    Z = rng.standard_normal((n_paths, n_steps))
    for k in range(n_steps):
        r[:, k + 1] = r[:, k] + a * (b - r[:, k]) * dt + sigma * sqrt_dt * Z[:, k]

    integral = dt * (0.5 * r[:, 0] + r[:, 1:-1].sum(axis=1) + 0.5 * r[:, -1])
    discount = np.exp(-integral)

    P_hat = discount.mean()
    se = discount.std(ddof=1) / np.sqrt(n_paths)
    return P_hat, se, 1.96 * se


# ----------------------------------------------------------------------
# Demo
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Parameters
    r0    = 0.03    # current short rate, 3%
    a     = 0.10    # mean-reversion speed
    b     = 0.05    # long-run mean rate, 5%
    sigma = 0.01    # volatility
    t, T  = 0.0, 5.0   # 5-year bond

    print("=" * 64)
    print("VASICEK BOND PRICING")
    print("=" * 64)
    print(f"Parameters: r0={r0}, a={a}, b={b}, sigma={sigma}")
    print(f"Maturity: T - t = {T - t} years")
    print()

    # Benchmark
    P_exact = vasicek_closed_form(r0, a, b, sigma, t, T)
    print(f"Closed form:           P(t,T) = {P_exact:.8f}")
    print()

    # MC with exact transitions
    np.random.seed(42)
    P_mc, se, ci = vasicek_mc_exact(
        r0, a, b, sigma, t, T,
        n_paths=200_000, n_steps=250, seed=42,
    )
    err = P_mc - P_exact
    print(f"MC (exact transitions, N=200k, M=250):")
    print(f"  estimate  = {P_mc:.8f}")
    print(f"  std error = {se:.2e}")
    print(f"  95% CI    = [{P_mc - ci:.8f}, {P_mc + ci:.8f}]")
    print(f"  error vs closed form = {err:+.2e}  ({err / se:+.2f} std errors)")
    print()

    # MC with Euler (coarser comparison)
    P_eu, se_eu, ci_eu = vasicek_mc_euler(
        r0, a, b, sigma, t, T,
        n_paths=200_000, n_steps=250, seed=42,
    )
    err_eu = P_eu - P_exact
    print(f"MC (Euler-Maruyama,  N=200k, M=250):")
    print(f"  estimate  = {P_eu:.8f}")
    print(f"  std error = {se_eu:.2e}")
    print(f"  error vs closed form = {err_eu:+.2e}  ({err_eu / se_eu:+.2f} std errors)")
    print()

    # Convergence: vary N, fix M
    print("-" * 64)
    print("Convergence in N (exact transitions, M=250):")
    print("-" * 64)
    print(f"{'N':>10}  {'estimate':>12}  {'std err':>10}  {'error':>11}")
    for N in [1_000, 10_000, 100_000, 500_000]:
        P_mc, se, _ = vasicek_mc_exact(r0, a, b, sigma, t, T, N, 250, seed=42)
        print(f"{N:>10,}  {P_mc:>12.8f}  {se:>10.2e}  {P_mc - P_exact:>+11.2e}")