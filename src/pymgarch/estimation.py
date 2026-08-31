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


def fit_stage2(eps: np.ndarray, layout: ParamLayout) -> Stage2Fit:
    """Estimate (a, b[, g][, nu]) by SLSQP with correlation targeting."""
    Sbar, Nbar = correlation_targets(eps)
    delta = adcc_delta(Sbar, Nbar) if layout.asymmetric else 0.0

    def negll(x: np.ndarray) -> float:
        llt = stage2_llt(x, eps, Sbar, Nbar, layout)
        if llt is None or not np.all(np.isfinite(llt)):
            return _PENALTY
        return -float(np.sum(llt))

    # a + b + delta * g <= 1 - eps as a linear inequality (>= 0 form)
    def stationarity(x: np.ndarray) -> float:
        a, b, g, _ = layout.unpack(x)
        return 1.0 - _EPS_BOUND - (a + b + delta * g)

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

    llt = stage2_llt(best.x, eps, Sbar, Nbar, layout)
    if llt is None:
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
    )
