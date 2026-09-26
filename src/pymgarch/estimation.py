"""Stage-2 maximum likelihood for the (A)DCC correlation parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from .correlation import adcc_delta, correlation_targets, dcc_path
from .distributions import mvnorm_llt, mvt_llt

_PENALTY = 1e10
_EPS_BOUND = 1e-6


@dataclass(frozen=True)
class ParamLayout:
    """Packing order of the stage-2 vector: (a, b[, g][, nu])."""

    asymmetric: bool
    studentt: bool

    @property
    def size(self) -> int:
        return 2 + int(self.asymmetric) + int(self.studentt)

    @property
    def names(self) -> list[str]:
        names = ["alpha", "beta"]
        if self.asymmetric:
            names.append("gamma")
        if self.studentt:
            names.append("nu")
        return names

    def unpack(self, x: np.ndarray) -> tuple[float, float, float, float | None]:
        a, b = float(x[0]), float(x[1])
        g = float(x[2]) if self.asymmetric else 0.0
        nu = float(x[-1]) if self.studentt else None
        return a, b, g, nu

    def pack(self, a: float, b: float, g: float = 0.0, nu: float | None = None):
        x = [a, b]
        if self.asymmetric:
            x.append(g)
        if self.studentt:
            if nu is None:
                raise ValueError("nu required for Student-t layout")
            x.append(nu)
        return np.asarray(x, dtype=float)


def stage2_llt(
    x: np.ndarray,
    eps: np.ndarray,
    Sbar: np.ndarray,
    Nbar: np.ndarray,
    layout: ParamLayout,
) -> np.ndarray | None:
    """Per-observation stage-2 log-likelihood of eps, or None on failure.

    This is the full multivariate density of the standardized residuals
    (not the correlation-only component), so summing it with the marginal
    -log sigma Jacobian gives the joint likelihood rmgarch reports.
    """
    a, b, g, nu = layout.unpack(x)
    delta = adcc_delta(Sbar, Nbar) if layout.asymmetric else 0.0
    if a < 0.0 or b < 0.0 or g < 0.0 or a + b + delta * g >= 1.0:
        return None
    if layout.studentt and (nu is None or nu <= 2.0):
        return None
    path = dcc_path(eps, a, b, g, Sbar, Nbar)
    if not path.ok:
        return None
    ndim = eps.shape[1]
    if layout.studentt:
        return mvt_llt(path.logdet, path.quad, nu, ndim)
    return mvnorm_llt(path.logdet, path.quad, ndim)


def composite_pairs(n_assets: int, scheme: str) -> list[tuple[int, int]]:
    """Asset pairs for the composite likelihood.

    "contiguous": (0,1), (1,2), ... -- O(N) pairs, the Engle-Shephard-
    Sheppard (2008) recommendation for large cross-sections.
    "all": every pair -- O(N^2), exhausts the bivariate information.
    """
    if scheme == "contiguous":
        return [(i, i + 1) for i in range(n_assets - 1)]
    if scheme == "all":
        return [
            (i, j) for i in range(n_assets) for j in range(i + 1, n_assets)
        ]
    raise ValueError("pairs must be 'contiguous' or 'all'")


def stage2_llt_composite(
    x: np.ndarray,
    eps: np.ndarray,
    Sbar: np.ndarray,
    Nbar: np.ndarray,
    layout: ParamLayout,
    pairs: list[tuple[int, int]],
) -> np.ndarray | None:
    """Composite per-observation objective: mean over pairs of the bivariate
    stage-2 log-likelihood. Each pair runs its own 2x2 recursion with the
    corresponding submatrices of the full targets, so the cost is O(T P)
    instead of O(T N^2)-with-N^3-Cholesky per evaluation."""
    a, b, g, nu = layout.unpack(x)
    if a < 0.0 or b < 0.0 or g < 0.0:
        return None
    if layout.studentt and (nu is None or nu <= 2.0):
        return None
    total = np.zeros(eps.shape[0])
    for i, j in pairs:
        sub = np.ascontiguousarray(eps[:, (i, j)])
        Sp = Sbar[np.ix_((i, j), (i, j))]
        Np = Nbar[np.ix_((i, j), (i, j))]
        if layout.asymmetric:
            if a + b + adcc_delta(Sp, Np) * g >= 1.0:
                return None
        elif a + b >= 1.0:
            return None
        path = dcc_path(sub, a, b, g, Sp, Np)
        if not path.ok:
            return None
        if layout.studentt:
            total += mvt_llt(path.logdet, path.quad, nu, 2)
        else:
            total += mvnorm_llt(path.logdet, path.quad, 2)
    return total / len(pairs)


@dataclass
class Stage2Fit:
    params: np.ndarray
    layout: ParamLayout
    llt: np.ndarray
    Sbar: np.ndarray
    Nbar: np.ndarray
    delta: float
    converged: bool
    message: str
    method: str = "full"
    pairs: list[tuple[int, int]] | None = None


def fit_stage2(
    eps: np.ndarray,
    layout: ParamLayout,
    method: str = "full",
    pairs_scheme: str = "contiguous",
) -> Stage2Fit:
    """Estimate (a, b[, g][, nu]) by SLSQP with correlation targeting.

    method="composite" replaces the full N-dimensional likelihood with the
    mean of bivariate pair likelihoods (Engle, Shephard and Sheppard 2008),
    making estimation feasible for large N. Point estimates are consistent;
    the reported joint likelihood is still evaluated on the full model.
    """
    if method not in ("full", "composite"):
        raise ValueError("method must be 'full' or 'composite'")
    Sbar, Nbar = correlation_targets(eps)
    delta = adcc_delta(Sbar, Nbar) if layout.asymmetric else 0.0
    pairs = None
    if method == "composite":
        pairs = composite_pairs(eps.shape[1], pairs_scheme)
        deltas = [
            adcc_delta(Sbar[np.ix_(p, p)], Nbar[np.ix_(p, p)]) for p in pairs
        ]
        delta_con = max(deltas) if layout.asymmetric else 0.0
    else:
        delta_con = delta

    def objective_llt(x: np.ndarray):
        if method == "composite":
            return stage2_llt_composite(x, eps, Sbar, Nbar, layout, pairs)
        return stage2_llt(x, eps, Sbar, Nbar, layout)

    def negll(x: np.ndarray) -> float:
        llt = objective_llt(x)
        if llt is None or not np.all(np.isfinite(llt)):
            return _PENALTY
        return -float(np.sum(llt))

    # a + b + delta * g <= 1 - eps as a linear inequality (>= 0 form); for
    # the composite objective every pair must satisfy its own condition, so
    # the binding constant is the max over pair deltas
    def stationarity(x: np.ndarray) -> float:
        a, b, g, _ = layout.unpack(x)
        return 1.0 - _EPS_BOUND - (a + b + delta_con * g)

    bounds = [(0.0, 0.999), (0.0, 0.999)]
    if layout.asymmetric:
        bounds.append((0.0, 0.999))
    if layout.studentt:
        bounds.append((2.05, 300.0))

    starts = []
    for a0, b0 in [(0.02, 0.95), (0.05, 0.90), (0.10, 0.80)]:
        starts.append(layout.pack(a0, b0, g=0.02, nu=8.0 if layout.studentt else None))

    best = None
    for x0 in starts:
        res = minimize(
            negll,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": stationarity}],
            options={"maxiter": 500, "ftol": 1e-10},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None

    # per-obs stage-2 llt reported on the FULL model even under composite
    # estimation, so joint likelihoods stay comparable across methods
    llt = stage2_llt(best.x, eps, Sbar, Nbar, layout)
    if llt is None:
        if method == "composite":
            # pair-wise constraints do not bound the full-model delta, so the
            # full intercept can be non-PSD at a valid composite optimum
            raise RuntimeError(
                "composite estimate is infeasible for the full N-dimensional "
                f"model (params {best.x}); refit with method='full' or "
                "reduce the asymmetry"
            )
        raise RuntimeError(
            "stage-2 estimation failed: optimizer terminated at an invalid "
            f"parameter vector {best.x} ({best.message})"
        )
    return Stage2Fit(
        params=np.asarray(best.x, dtype=float),
        layout=layout,
        llt=llt,
        Sbar=Sbar,
        Nbar=Nbar,
        delta=delta,
        converged=bool(best.success),
        message=str(best.message),
        method=method,
        pairs=pairs,
    )
