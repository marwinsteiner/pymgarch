"""Portfolio risk metrics on model forecasts, and VaR backtests.

value_at_risk / expected_shortfall report POSITIVE loss numbers at tail
probability alpha (5% VaR = the loss exceeded with 5% probability).

method="simulation" (default) draws future return paths through the fitted
model's simulate() -- exact for every model and horizon, including Student-t
and copula tails. method="analytic" uses the one-step-per-horizon Gaussian
(or Student-t, when the model's stage-2 distribution is t) quantile on the
analytic covariance forecast -- fast, but normal/t-elliptical only.
"""

from __future__ import annotations

import numpy as np
from scipy import stats as sps


def _portfolio_draws(result, weights, horizon, n_paths, seed) -> np.ndarray:
    sim = result.simulate(horizon=horizon, n_paths=n_paths, seed=seed)
    rets = sim["returns"] if isinstance(sim, dict) else sim
    return np.einsum("hmn,n->hm", np.asarray(rets), np.asarray(weights, float))


def value_at_risk(
    result,
    weights,
    alpha: float = 0.05,
    horizon: int = 1,
    method: str = "simulation",
    n_paths: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Per-horizon portfolio value-at-risk (positive loss units)."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    w = np.asarray(weights, dtype=float)
    if method == "simulation":
        p = _portfolio_draws(result, w, horizon, n_paths, seed)
        return -np.quantile(p, alpha, axis=1)
    if method == "analytic":
        fc = result.forecast(horizon=horizon)
        sig = np.sqrt(np.einsum("i,hij,j->h", w, fc.covariances, w))
        nu = getattr(result, "_dcc_coefs", lambda: (0, 0, 0, None))()[3]
        if nu is not None:
            q = sps.t.ppf(alpha, df=nu) * np.sqrt((nu - 2.0) / nu)
        else:
            q = sps.norm.ppf(alpha)
        return -q * sig
    raise ValueError("method must be 'simulation' or 'analytic'")


def expected_shortfall(
    result,
    weights,
    alpha: float = 0.05,
    horizon: int = 1,
    method: str = "simulation",
    n_paths: int = 10000,
    seed: int | None = None,
) -> np.ndarray:
    """Per-horizon portfolio expected shortfall (positive loss units)."""
    if not 0.0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    w = np.asarray(weights, dtype=float)
    if method == "simulation":
        p = _portfolio_draws(result, w, horizon, n_paths, seed)
        q = np.quantile(p, alpha, axis=1, keepdims=True)
        tail = np.where(p <= q, p, np.nan)
        return -np.nanmean(tail, axis=1)
    if method == "analytic":
        fc = result.forecast(horizon=horizon)
        sig = np.sqrt(np.einsum("i,hij,j->h", w, fc.covariances, w))
        nu = getattr(result, "_dcc_coefs", lambda: (0, 0, 0, None))()[3]
        if nu is not None:
            # ES of the covariance-standardized t
            s = np.sqrt((nu - 2.0) / nu)
            q = sps.t.ppf(alpha, df=nu)
            es_std = s * sps.t.pdf(q, df=nu) * (nu + q**2) / ((nu - 1.0) * alpha)
        else:
            es_std = sps.norm.pdf(sps.norm.ppf(alpha)) / alpha
        return es_std * sig
    raise ValueError("method must be 'simulation' or 'analytic'")


def var_coverage(losses: np.ndarray, var: np.ndarray, alpha: float = 0.05) -> dict:
    """Kupiec (1995) POF and Christoffersen (1998) VaR backtests.

    losses : realized portfolio losses (positive numbers = losses).
    var : the corresponding VaR forecasts.
    Returns the violation count/rate and LR statistics with p-values for
    unconditional coverage (Kupiec), independence, and conditional coverage.
    """
    losses = np.asarray(losses, dtype=float)
    var = np.asarray(var, dtype=float)
    if losses.shape != var.shape:
        raise ValueError("losses and var must have equal length")
    hits = (losses > var).astype(int)
    n = hits.shape[0]
    x = int(hits.sum())
    pi = x / n if n else 0.0

    def _safe_log(v):
        return np.log(np.clip(v, 1e-300, None))

    # Kupiec unconditional coverage
    lr_uc = -2.0 * (
        (n - x) * _safe_log(1 - alpha)
        + x * _safe_log(alpha)
        - (n - x) * _safe_log(1 - pi)
        - x * _safe_log(pi if x else 1.0)
    )
    p_uc = float(sps.chi2.sf(lr_uc, 1))

    # Christoffersen independence: first-order Markov transition counts
    h0, h1 = hits[:-1], hits[1:]
    n00 = int(np.sum((h0 == 0) & (h1 == 0)))
    n01 = int(np.sum((h0 == 0) & (h1 == 1)))
    n10 = int(np.sum((h0 == 1) & (h1 == 0)))
    n11 = int(np.sum((h0 == 1) & (h1 == 1)))
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi1 = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    ll_ind = (
        n00 * _safe_log(1 - pi01)
        + n01 * _safe_log(pi01 if n01 else 1.0)
        + n10 * _safe_log(1 - pi11)
        + n11 * _safe_log(pi11 if n11 else 1.0)
    )
    ll_null = (n00 + n10) * _safe_log(1 - pi1) + (n01 + n11) * _safe_log(
        pi1 if (n01 + n11) else 1.0
    )
    lr_ind = -2.0 * (ll_null - ll_ind)
    p_ind = float(sps.chi2.sf(lr_ind, 1))

    lr_cc = lr_uc + lr_ind
    p_cc = float(sps.chi2.sf(lr_cc, 2))
    return {
        "n": n,
        "violations": x,
        "rate": pi,
        "expected_rate": alpha,
        "kupiec_stat": float(lr_uc),
        "kupiec_pvalue": p_uc,
        "independence_stat": float(lr_ind),
        "independence_pvalue": p_ind,
        "conditional_stat": float(lr_cc),
        "conditional_pvalue": p_cc,
    }
