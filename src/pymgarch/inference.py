"""Two-stage covariance estimation (Engle & Sheppard 2001).

The estimator stacks the per-observation scores of every marginal likelihood
and of the stage-2 correlation likelihood into an M-estimator moment vector
g_t(theta), theta = (phi_1, ..., phi_N, psi), and computes

    Avar(theta_hat) = A^{-1} B A^{-T} / T,
    A = (1/T) sum_t d g_t / d theta',   B = (1/T) sum_t g_t g_t'.

A is block lower-triangular because marginal scores do not depend on psi.
All derivatives are central finite differences of per-observation likelihood
contributions; marginal likelihoods are re-evaluated through arch's own
variance recursions (see marginals.MarginalSet._recompute).

Known approximation, shared with rmgarch: the correlation targets Sbar/Nbar
are held fixed at their point estimates, so targeting uncertainty is ignored.
"""

from __future__ import annotations

import warnings

import numpy as np

from .estimation import Stage2Fit, stage2_llt
from .marginals import MarginalSet

_H_SCORE = 6e-6  # relative step for per-obs score differences
_H_JAC = 1e-4  # relative step for jacobian-of-score differences


def _step(x: float, rel: float) -> float:
    return rel * max(1.0, abs(float(x)))


def _fd_column(func, x: np.ndarray, j: int, h: float) -> np.ndarray:
    """Central difference of a vector-valued func wrt x[j], with one-sided
    fallback when a perturbed point is infeasible (returns None)."""
    xp = x.copy()
    xm = x.copy()
    xp[j] += h
    xm[j] -= h
    fp = func(xp)
    fm = func(xm)
    if fp is not None and fm is not None:
        return (fp - fm) / (2.0 * h)
    f0 = func(x)
    if f0 is None:
        raise RuntimeError("likelihood not evaluable at the fitted parameters")
    if fp is not None:
        return (fp - f0) / h
    if fm is not None:
        return (f0 - fm) / h
    raise RuntimeError(f"likelihood not evaluable near fitted parameter {j}")


def two_stage_vcov(mset: MarginalSet, fit: Stage2Fit, llt_fn=None) -> dict:
    """Sandwich covariance of the stage-2 parameters.

    Returns {"vcov": (K_psi, K_psi), "se": (K_psi,), "method": str}.
    Falls back to a stage-2-only OPG sandwich with a warning if the stacked
    bread matrix is numerically singular.

    llt_fn(psi, eps) -> per-obs stage-2 log-likelihood (or None if psi is
    infeasible) can be supplied to reuse the sandwich for stage-2 objectives
    other than the plain (A)DCC one (e.g. copula likelihoods); it defaults
    to the (A)DCC stage-2 likelihood at fit's targets.
    """
    T = mset.nobs
    layout = fit.layout
    k_psi = layout.size
    k_phi = mset.total_params
    K = k_phi + k_psi
    psi_hat = fit.params
    eps_hat = mset.std_resid

    if llt_fn is None:

        def s2_llt(psi: np.ndarray, eps: np.ndarray):
            return stage2_llt(psi, eps, fit.Sbar, fit.Nbar, layout)

    else:
        s2_llt = llt_fn

    # ---- per-observation scores G (T, K) --------------------------------
    G = np.zeros((T, K))
    offsets = np.cumsum([0] + [mset.n_params(i) for i in range(mset.nassets)])
    for i in range(mset.nassets):
        phi = mset.full_params(i)

        def l1(p, i=i):
            llt = mset.per_obs_loglik(i, p)
            return llt if np.all(np.isfinite(llt)) else None

        for j in range(phi.shape[0]):
            h = _step(phi[j], _H_SCORE)
            G[:, offsets[i] + j] = _fd_column(l1, phi, j, h)

    def l2_at_psi(psi):
        return s2_llt(psi, eps_hat)

    for j in range(k_psi):
        h = _step(psi_hat[j], _H_SCORE)
        G[:, k_phi + j] = _fd_column(l2_at_psi, psi_hat, j, h)

    B = G.T @ G / T

    # ---- bread matrix A --------------------------------------------------
    A = np.zeros((K, K))

    # A11: per-asset Hessian blocks of the mean marginal log-likelihood
    for i in range(mset.nassets):
        phi = mset.full_params(i)
        npar = phi.shape[0]

        def mean_score_i(p, i=i, npar=npar):
            def l1(pp, i=i):
                llt = mset.per_obs_loglik(i, pp)
                return llt if np.all(np.isfinite(llt)) else None

            out = np.empty(npar)
            for j in range(npar):
                h = _step(p[j], _H_SCORE)
                out[j] = float(np.mean(_fd_column(l1, p, j, h)))
            return out

        block = slice(offsets[i], offsets[i] + npar)
        for j in range(npar):
            h = _step(phi[j], _H_JAC)
            xp = phi.copy()
            xm = phi.copy()
            xp[j] += h
            xm[j] -= h
            A[block, offsets[i] + j] = (mean_score_i(xp) - mean_score_i(xm)) / (2.0 * h)

    # mean stage-2 psi-score as a function of the eps matrix
    def mean_psi_score(eps: np.ndarray) -> np.ndarray:
        def l2(psi):
            return s2_llt(psi, eps)

        out = np.empty(k_psi)
        for j in range(k_psi):
            h = _step(psi_hat[j], _H_SCORE)
            out[j] = float(np.mean(_fd_column(l2, psi_hat, j, h)))
        return out

    # A21: derivative of the mean psi-score wrt each marginal parameter;
    # perturbing phi_i only changes column i of eps.
    for i in range(mset.nassets):
        phi = mset.full_params(i)
        for j in range(phi.shape[0]):
            h = _step(phi[j], _H_JAC)
            xp = phi.copy()
            xm = phi.copy()
            xp[j] += h
            xm[j] -= h
            eps_p = eps_hat.copy()
            eps_m = eps_hat.copy()
            eps_p[:, i] = mset.std_resid_at(i, xp)
            eps_m[:, i] = mset.std_resid_at(i, xm)
            A[k_phi:, offsets[i] + j] = (
                mean_psi_score(eps_p) - mean_psi_score(eps_m)
            ) / (2.0 * h)

    # A22: Hessian of the mean stage-2 log-likelihood wrt psi
    def mean_psi_score_at(psi: np.ndarray) -> np.ndarray:
        def l2(p):
            return s2_llt(p, eps_hat)

        out = np.empty(k_psi)
        for j in range(k_psi):
            h = _step(psi[j], _H_SCORE)
            out[j] = float(np.mean(_fd_column(l2, psi, j, h)))
        return out

    for j in range(k_psi):
        h = _step(psi_hat[j], _H_JAC)
        xp = psi_hat.copy()
        xm = psi_hat.copy()
        xp[j] += h
        xm[j] -= h
        A[k_phi:, k_phi + j] = (mean_psi_score_at(xp) - mean_psi_score_at(xm)) / (
            2.0 * h
        )

    # ---- assemble --------------------------------------------------------
    try:
        Ainv = np.linalg.inv(A)
        avar = Ainv @ B @ Ainv.T / T
        vcov = avar[k_phi:, k_phi:]
        se = np.sqrt(np.diag(vcov))
        if np.all(np.isfinite(se)):
            return {"vcov": vcov, "se": se, "method": "two-stage-robust"}
        raise np.linalg.LinAlgError("non-finite standard errors")
    except np.linalg.LinAlgError:
        warnings.warn(
            "stacked two-stage bread matrix is singular; falling back to a "
            "stage-2-only sandwich that ignores marginal estimation error",
            RuntimeWarning,
            stacklevel=2,
        )
        A22 = A[k_phi:, k_phi:]
        B22 = B[k_phi:, k_phi:]
        Ainv = np.linalg.pinv(A22)
        vcov = Ainv @ B22 @ Ainv.T / T
        return {
            "vcov": vcov,
            "se": np.sqrt(np.diag(vcov)),
            "method": "stage2-robust",
        }
