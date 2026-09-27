"""Rolling re-estimation with out-of-sample one-step forecasts.

Parity target: rmgarch's dccroll/gogarchroll. The implementation exploits
the fact that a filtered path IS the sequence of one-step-ahead conditional
covariances -- sigma^2_t and R_t depend only on information through t-1 --
so each refit block needs one fit plus one filter pass, not a per-step
forecast loop:

    for each refit point r: fit on the estimation window ending at r, filter
    from the window start through r + refit_every, and read the filtered
    H_t for t in (r, r + refit_every] as the OOS forecasts.

Works for every model exposing fit()/filter() and a conditional covariance
path: DCC, ADCC, CCC, GO-GARCH, BEKK, and CopulaGARCH (for the copula the
covariance D_t R_t D_t uses the copula correlation, exact under the
Gaussian copula and an approximation otherwise -- documented caveat).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .results import _coerce_returns


@dataclass
class RollResult:
    """Out-of-sample one-step forecasts from a rolling scheme."""

    index: pd.Index  # OOS target dates
    realized: np.ndarray  # (T_oos, N) realized returns
    covariances: np.ndarray  # (T_oos, N, N) one-step-ahead H forecasts
    refit_points: list[int]
    params: list[dict]  # fitted stage-2 params per refit
    window: int
    refit_every: int
    expanding: bool
    names: list[str] = field(default_factory=list)

    @property
    def nobs(self) -> int:
        return int(self.realized.shape[0])

    @property
    def correlations(self) -> np.ndarray:
        s = np.sqrt(np.einsum("tii->ti", self.covariances))
        return self.covariances / np.einsum("ti,tj->tij", s, s)

    def portfolio_sigma(self, weights: np.ndarray) -> np.ndarray:
        w = np.asarray(weights, dtype=float)
        return np.sqrt(np.einsum("i,tij,j->t", w, self.covariances, w))

    def var_forecasts(self, weights: np.ndarray, alpha: float = 0.05) -> np.ndarray:
        """One-step Gaussian portfolio VaR path (positive loss numbers)."""
        from scipy import stats as sps

        return -sps.norm.ppf(alpha) * self.portfolio_sigma(weights)

    def coverage_test(self, weights: np.ndarray, alpha: float = 0.05) -> dict:
        """Kupiec POF and Christoffersen tests of the Gaussian VaR path."""
        from .risk import var_coverage

        losses = -(self.realized @ np.asarray(weights, dtype=float))
        return var_coverage(losses, self.var_forecasts(weights, alpha), alpha)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<RollResult T_oos={self.nobs} refits={len(self.refit_points)} "
            f"window={self.window}{' expanding' if self.expanding else ''}>"
        )


def _one_block(model, returns, fit_start, r, block_end, fit_kwargs):
    frame = returns.iloc[fit_start:r]
    fitted = model.fit(frame, **fit_kwargs)
    flt = fitted.filter(returns.iloc[fit_start:block_end])
    H = np.asarray(flt.conditional_covariances)
    # positions r..block_end-1 within [fit_start, block_end) are the OOS
    # one-step-ahead covariances for those dates
    oos = H[r - fit_start : block_end - fit_start]
    params = dict(getattr(fitted, "params", {}) or {})
    return oos, params


def roll(
    model,
    returns,
    window: int = 1000,
    refit_every: int = 25,
    expanding: bool = False,
    n_jobs: int = 1,
    **fit_kwargs,
) -> RollResult:
    """Rolling refit + out-of-sample one-step covariance forecasts.

    model : an unfitted pymgarch model instance (DCC(), ADCC(), CCC(),
        GOGARCH(), BEKK(), CopulaGARCH()).
    window : estimation window length (ignored start when expanding=True).
    refit_every : re-estimate after this many OOS steps; between refits the
        fitted parameters are filtered forward.
    fit_kwargs : forwarded to model.fit() (e.g. compute_se=False is set by
        default for speed).
    """
    y, names, index = _coerce_returns(returns)
    if names is None:
        names = [f"y{i}" for i in range(y.shape[1])]
    frame = (
        returns
        if isinstance(returns, pd.DataFrame)
        else pd.DataFrame(y, columns=names, index=index)
    )
    T = y.shape[0]
    if window >= T:
        raise ValueError(f"window {window} must be < sample size {T}")
    fit_kwargs.setdefault("compute_se", False)
    sig = getattr(type(model), "fit", None)
    if sig is not None and "compute_se" not in sig.__code__.co_varnames:
        fit_kwargs.pop("compute_se", None)

    refit_points = list(range(window, T, refit_every))
    blocks = [
        (0 if expanding else r - window, r, min(r + refit_every, T))
        for r in refit_points
    ]
    if n_jobs != 1:
        from joblib import Parallel, delayed

        results = Parallel(n_jobs=n_jobs)(
            delayed(_one_block)(model, frame, fs, r, be, fit_kwargs)
            for fs, r, be in blocks
        )
    else:
        results = [
            _one_block(model, frame, fs, r, be, fit_kwargs) for fs, r, be in blocks
        ]

    covs = np.concatenate([oos for oos, _ in results], axis=0)
    params = [p for _, p in results]
    oos_index = frame.index[window:T]
    return RollResult(
        index=oos_index,
        realized=y[window:T],
        covariances=covs,
        refit_points=refit_points,
        params=params,
        window=window,
        refit_every=refit_every,
        expanding=expanding,
        names=names,
    )
