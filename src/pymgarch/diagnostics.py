"""Model diagnostics: the Engle-Sheppard test of constant correlation.

Implements the test as specified in Engle & Sheppard (2001, sec. 3.3):
residuals are standardized univariately and then *jointly* whitened by the
symmetric inverse square root of the constant-correlation estimate,
z_t = Rbar^{-1/2} D_t^{-1} (r_t - mu_t). Under the null of constant
correlation the off-diagonal outer products z_it z_jt - Rbar*_ij are serially
uncorrelated with zero mean, so a pooled OLS regression of the stacked pair
products on a constant and their lags has all coefficients zero:

    stat = delta' (X'X) delta / s2  ~  chi2(n_lags + 1).

Note: rmgarch's DCCtest deviates from the paper (it skips the Rbar^{-1/2}
whitening and uses a reversed-regression algebra); pymgarch follows the
paper, so statistics differ from rmgarch's while the accept/reject decision
agrees on clear-cut data. The replication fixture asserts the decision.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats as sps

from .correlation import correlation_targets
from .models import _build_marginals


@dataclass
class DCCTestResult:
    statistic: float
    pvalue: float
    n_lags: int
    df: int
    null: str = "constant conditional correlation"

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<DCCTestResult stat={self.statistic:.4f} "
            f"p={self.pvalue:.4g} df={self.df}>"
        )


def dcc_test(
    returns=None,
    n_lags: int = 1,
    marginals=None,
    eps: np.ndarray | None = None,
) -> DCCTestResult:
    """Engle-Sheppard (2001) test of constant conditional correlation.

    Pass returns (marginals fitted as in DCC.fit; customize via `marginals`)
    or precomputed standardized residuals via `eps` (T, N). A small p-value
    rejects constant correlation, motivating DCC-family dynamics.
    """
    if eps is None:
        if returns is None:
            raise ValueError("provide returns or eps")
        mset, _, _ = _build_marginals(returns, marginals)
        eps = mset.std_resid
    else:
        eps = np.asarray(eps, dtype=float)
        if eps.ndim != 2:
            raise ValueError("eps must be (T, N)")
    T, N = eps.shape
    if N < 2:
        raise ValueError("the test needs at least two assets")

    Rbar, _ = correlation_targets(eps)
    vals, vecs = np.linalg.eigh(Rbar)
    inv_sqrt = (vecs / np.sqrt(vals)) @ vecs.T
    z = eps @ inv_sqrt.T  # jointly standardized: Cov(z) ~ I under H0

    # stacked off-diagonal outer products, demeaned under H0 (E[z_i z_j] = 0)
    pairs = [(i, j) for i in range(N) for j in range(i + 1, N)]
    ys = []
    xs = []
    for i, j in pairs:
        op = z[:, i] * z[:, j]
        y = op[n_lags:]
        X = np.column_stack(
            [np.ones(T - n_lags)]
            + [op[n_lags - k : T - k] for k in range(1, n_lags + 1)]
        )
        ys.append(y)
        xs.append(X)
    y = np.concatenate(ys)
    X = np.vstack(xs)

    delta, *_ = np.linalg.lstsq(X, y, rcond=None)
    e = y - X @ delta
    s2 = float(e @ e) / y.shape[0]
    stat = float(delta @ (X.T @ X) @ delta) / s2
    df = n_lags + 1
    pvalue = float(sps.chi2.sf(stat, df))
    return DCCTestResult(statistic=stat, pvalue=pvalue, n_lags=n_lags, df=df)


__all__ = ["DCCTestResult", "dcc_test"]
