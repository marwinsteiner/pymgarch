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
    delta: float | None = None,
) -> np.ndarray | None:
    """Per-observation stage-2 log-likelihood of eps, or None on failure.

    This is the full multivariate density of the standardized residuals
    (not the correlation-only component), so summing it with the marginal
    -log sigma Jacobian gives the joint likelihood rmgarch reports.
    delta may be passed precomputed (it is a constant of the targets) to
    avoid an eigendecomposition per call in optimizer/FD loops.
    """
    a, b, g, nu = layout.unpack(x)
    if delta is None:
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


def _pair_targets(
    Sbar: np.ndarray,
    Nbar: np.ndarray,
    pairs: list[tuple[int, int]],
    asymmetric: bool,
) -> list[tuple[tuple[int, int], np.ndarray, np.ndarray, float]]:
    """Precompute per-pair (indices, Sp, Np, delta_p) — constants of the
    optimization; delta_p is only derived (an eigendecomposition) when the
    asymmetric term needs it."""
    out = []
    for i, j in pairs:
        Sp = Sbar[np.ix_((i, j), (i, j))]
        Np = Nbar[np.ix_((i, j), (i, j))]
        dp = adcc_delta(Sp, Np) if asymmetric else 0.0
        out.append(((i, j), Sp, Np, dp))
    return out


def stage2_llt_composite(
    x: np.ndarray,
    eps: np.ndarray,
    Sbar: np.ndarray,
    Nbar: np.ndarray,
    layout: ParamLayout,
    pairs: list[tuple[int, int]],
    pair_targets: list | None = None,
) -> np.ndarray | None:
    """Composite per-observation objective: mean over pairs of the bivariate
    stage-2 log-likelihood. Each pair runs its own 2x2 recursion with the
    corresponding submatrices of the full targets, so the cost is O(T P)
    instead of O(T N^2)-with-N^3-Cholesky per evaluation.

    pair_targets: precomputed output of _pair_targets; passed by fit_stage2
    and the SE machinery so the per-pair submatrices and deltas are not
    re-derived on every objective evaluation.
    """
    if not pairs:
        raise ValueError("composite estimation needs at least two assets")
    a, b, g, nu = layout.unpack(x)
    if a < 0.0 or b < 0.0 or g < 0.0:
        return None
    if layout.studentt and (nu is None or nu <= 2.0):
        return None
    if pair_targets is None:
        pair_targets = _pair_targets(Sbar, Nbar, pairs, layout.asymmetric)
    # feasibility once, against the binding (max) pair delta
    if layout.asymmetric:
        max_dp = max(dp for _, _, _, dp in pair_targets)
        if a + b + max_dp * g >= 1.0:
            return None
    elif a + b >= 1.0:
        return None
    total = np.zeros(eps.shape[0])
    for (i, j), Sp, Np, _dp in pair_targets:
        sub = np.ascontiguousarray(eps[:, (i, j)])
        path = dcc_path(sub, a, b, g, Sp, Np)
        if not path.ok:
            return None
        if layout.studentt:
            total += mvt_llt(path.logdet, path.quad, nu, 2)
        else:
            total += mvnorm_llt(path.logdet, path.quad, 2)
    return total / len(pair_targets)


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
    pair_targets: list | None = None  # precomputed (idx, Sp, Np, delta_p)

    def objective_llt(self, x: np.ndarray, eps: np.ndarray) -> np.ndarray | None:
        """The stage-2 objective this fit optimized, bound to its targets —
        the single source of truth for SE machinery (Godambe sandwiches must
        differentiate the objective that was actually maximized)."""
        if self.method == "composite":
            return stage2_llt_composite(
                x, eps, self.Sbar, self.Nbar, self.layout, self.pairs,
                pair_targets=self.pair_targets,
            )
        return stage2_llt(x, eps, self.Sbar, self.Nbar, self.layout, delta=self.delta)


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
    pair_targets = None
    if method == "composite":
        if eps.shape[1] < 2:
            raise ValueError("composite estimation needs at least two assets")
        pairs = composite_pairs(eps.shape[1], pairs_scheme)
        pair_targets = _pair_targets(Sbar, Nbar, pairs, layout.asymmetric)

    def objective_llt(x: np.ndarray):
        if method == "composite":
            return stage2_llt_composite(
                x, eps, Sbar, Nbar, layout, pairs, pair_targets=pair_targets
            )
        return stage2_llt(x, eps, Sbar, Nbar, layout, delta=delta)

    def negll(x: np.ndarray) -> float:
        llt = objective_llt(x)
        if llt is None or not np.all(np.isfinite(llt)):
            return _PENALTY
        return -float(np.sum(llt))

    # a + b + delta * g <= 1 - eps as a linear inequality (>= 0 form). The
    # FULL-model delta binds for the composite method too: it dominates every
    # pair delta (eigenvalue interlacing), so pair feasibility follows, the
    # joint likelihood at the optimum is always reportable, and no valid
    # composite estimate is discarded post hoc.
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

    # per-obs stage-2 llt reported on the FULL model even under composite
    # estimation, so joint likelihoods stay comparable across methods; the
    # full-delta constraint above guarantees this is evaluable at any
    # constraint-feasible optimum, so None here means plain optimizer failure
    llt = stage2_llt(best.x, eps, Sbar, Nbar, layout, delta=delta)
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
        method=method,
        pairs=pairs,
        pair_targets=pair_targets,
    )
