"""Scalar and diagonal BEKK (Engle and Kroner 1995) with variance targeting.

The one model in pymgarch that does not decompose into arch marginals plus a
correlation stage: the conditional covariance is modeled directly on the
demeaned returns u_t = r_t - mu,

    H_t = C + (a a') o (u_{t-1} u_{t-1}') + (b b') o H_{t-1},

the Hadamard form of diagonal BEKK (A = diag(a), B = diag(b)); scalar BEKK
is a = a * ones. Variance targeting sets C = Sigma o (1 - a a' - b b') with
Sigma the sample covariance, so C is determined by (a, b) and must remain
PSD -- guaranteed in the scalar case by a^2 + b^2 < 1, checked explicitly in
the diagonal case. Estimation is Gaussian QML.

Validation note: unlike DCC/GO-GARCH/copula-GARCH there is no maintained R
reference implementation to replicate against (mgarchBEKK is dead), so the
test suite relies on simulation-recovery and closed-form forecast checks.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from ._kernels import bekk_recursion
from .inference import qml_vcov
from .results import MGARCHForecast, _coerce_returns

_PENALTY = 1e10
_EPS_BOUND = 1e-6


def _targeting_intercept(
    avec: np.ndarray, bvec: np.ndarray, Sigma: np.ndarray
) -> np.ndarray:
    """C = Sigma o (1 - aa' - bb'): the single definition of the variance-
    targeting intercept, shared by estimation, forecasting, filtering, and
    simulation."""
    return Sigma * (1.0 - np.outer(avec, avec) - np.outer(bvec, bvec))


def _bekk_eval(
    u: np.ndarray,
    avec: np.ndarray,
    bvec: np.ndarray,
    Sigma: np.ndarray,
    want_path: bool = False,
    check_psd: bool = True,
):
    """One recursion pass: per-obs Gaussian log-likelihood (and, when
    want_path, the H path), or None if the parameters are infeasible.

    check_psd=False skips the intercept eigendecomposition: valid for the
    scalar variant, where C = (1 - a^2 - b^2) * Sigma is PSD iff the scalar
    stationarity bound (already enforced) holds.
    """
    avec = np.ascontiguousarray(avec, dtype=np.float64)
    bvec = np.ascontiguousarray(bvec, dtype=np.float64)
    if np.any(avec**2 + bvec**2 >= 1.0):
        return None
    C = _targeting_intercept(avec, bvec, Sigma)
    if check_psd and np.linalg.eigvalsh(C)[0] < -1e-12:
        return None
    flag, llt, Hpath, _ = bekk_recursion(
        np.ascontiguousarray(u),
        avec,
        bvec,
        np.ascontiguousarray(C, dtype=np.float64),
        np.ascontiguousarray(Sigma, dtype=np.float64),
        1 if want_path else 0,
    )
    if flag != 0 or not np.all(np.isfinite(llt)):
        return None
    return (llt, Hpath) if want_path else llt


def _bekk_llt(u, avec, bvec, Sigma):
    """Back-compat objective wrapper: llt only, full PSD check."""
    return _bekk_eval(u, avec, bvec, Sigma, want_path=False, check_psd=True)


@dataclass
class BEKKResult:
    variant: str  # "scalar" | "diagonal"
    names: list[str]
    index: pd.Index
    mu: np.ndarray
    Sigma: np.ndarray  # variance target (sample covariance of u)
    avec: np.ndarray
    bvec: np.ndarray
    H: np.ndarray  # (T, N, N) conditional covariances
    loglikelihood: float
    llt: np.ndarray
    vcov: np.ndarray | None = None
    se: np.ndarray | None = None
    se_method: str | None = None
    converged: bool = True
    message: str = ""
    filtered: bool = False

    model = "BEKK"

    @property
    def nobs(self) -> int:
        return int(self.H.shape[0])

    @property
    def nassets(self) -> int:
        return int(self.H.shape[1])

    @property
    def psi(self) -> np.ndarray:
        if self.variant == "scalar":
            return np.array([self.avec[0], self.bvec[0]])
        return np.concatenate([self.avec, self.bvec])

    @property
    def psi_names(self) -> list[str]:
        if self.variant == "scalar":
            return ["a", "b"]
        return [f"a.{n}" for n in self.names] + [f"b.{n}" for n in self.names]

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.psi_names, [float(v) for v in self.psi]))

    @property
    def std_errors(self) -> dict[str, float] | None:
        if self.se is None:
            return None
        return dict(zip(self.psi_names, [float(v) for v in self.se]))

    @property
    def conditional_covariances(self) -> np.ndarray:
        return self.H

    @property
    def conditional_correlations(self) -> np.ndarray:
        s = np.sqrt(np.einsum("tii->ti", self.H))
        return self.H / np.einsum("ti,tj->tij", s, s)

    @property
    def num_params(self) -> int:
        # psi plus the N estimated means; the targeting Sigma is excluded by
        # the same convention DCC applies to its correlation target
        return len(self.psi) + self.nassets

    @property
    def aic(self) -> float:
        return -2.0 * self.loglikelihood + 2.0 * self.num_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglikelihood + self.num_params * float(np.log(self.nobs))

    def summary(self) -> str:
        title = f"{self.variant.capitalize()} BEKK (variance targeting)"
        lines = [title, "=" * len(title)]
        lines.append(f"Assets: {self.nassets}   Obs: {self.nobs}")
        lines.append(
            f"Log-likelihood: {self.loglikelihood:.4f}   "
            f"AIC: {self.aic:.2f}   BIC: {self.bic:.2f}"
        )
        lines.append("")
        lines.append(f"{'param':<10}{'coef':>12}{'std err':>12}{'z':>10}")
        for j, name in enumerate(self.psi_names):
            c = self.psi[j]
            if self.se is not None and np.isfinite(self.se[j]) and self.se[j] > 0:
                z = c / self.se[j]
                lines.append(f"{name:<10}{c:>12.6f}{self.se[j]:>12.6f}{z:>10.3f}")
            else:
                lines.append(f"{name:<10}{c:>12.6f}{'--':>12}{'--':>10}")
        if self.se_method:
            lines.append(f"Covariance: {self.se_method}")
            lines.append(
                "  (single-stage QML; mean and targeting-Sigma estimation "
                "error held fixed, see docs on inference)"
            )
        if not self.converged:
            lines.append(f"WARNING: optimizer did not converge: {self.message}")
        if self.filtered:
            lines.append("(filtered result: parameters fixed, no estimation)")
        return "\n".join(lines)

    # -- forecasting -------------------------------------------------------

    def forecast(self, horizon: int = 1) -> MGARCHForecast:
        """Closed-form covariance forecasts.

        E[H_{T+1}] is deterministic; for h >= 2,
        E[H_{T+h}] = C + M o E[H_{T+h-1}] with M = a a' + b b'.
        Returns an MGARCHForecast (variances/correlations/covariances),
        matching the family-wide forecast contract. Filtered results
        forecast from their own (updated) terminal state.
        """
        if horizon < 1:
            raise ValueError("horizon must be >= 1")
        u_last = getattr(self, "_u_last", None)
        if u_last is None:
            raise RuntimeError("terminal state missing; refit the model")
        C = _targeting_intercept(self.avec, self.bvec, self.Sigma)
        aa = np.outer(self.avec, self.avec)
        bb = np.outer(self.bvec, self.bvec)
        covs = np.empty((horizon, self.nassets, self.nassets))
        covs[0] = C + aa * np.outer(u_last, u_last) + bb * self.H[-1]
        for h in range(1, horizon):
            covs[h] = C + (aa + bb) * covs[h - 1]
        s = np.sqrt(np.einsum("hii->hi", covs))
        corrs = covs / np.einsum("hi,hj->hij", s, s)
        return MGARCHForecast(
            horizon=horizon,
            method="analytic",
            variances=np.einsum("hii->hi", covs).copy(),
            correlations=corrs,
            covariances=covs,
        )

    # -- filtering ---------------------------------------------------------

    def filter(self, returns) -> BEKKResult:
        y, names, index = _coerce_returns(returns)
        if y.shape[0] < 1:
            raise ValueError("filter needs at least one observation")
        if y.shape[1] != self.nassets:
            raise ValueError(f"expected {self.nassets} columns, got {y.shape[1]}")
        u = y - self.mu
        out = _bekk_eval(u, self.avec, self.bvec, self.Sigma, want_path=True)
        if out is None:
            raise RuntimeError("BEKK recursion failed on new data")
        llt, Hpath = out
        new = BEKKResult(
            variant=self.variant,
            names=names or self.names,
            index=index,
            mu=self.mu.copy(),
            Sigma=self.Sigma.copy(),
            avec=self.avec.copy(),
            bvec=self.bvec.copy(),
            H=Hpath,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            filtered=True,
        )
        new._u_last = u[-1]
        return new

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<BEKKResult {self.variant} N={self.nassets} T={self.nobs} "
            f"ll={self.loglikelihood:.2f}>"
        )


class BEKK:
    """Scalar or diagonal BEKK with variance targeting, Gaussian QML."""

    def __init__(self, variant: str = "scalar"):
        if variant not in ("scalar", "diagonal"):
            raise ValueError("variant must be 'scalar' or 'diagonal'")
        self.variant = variant

    def fit(self, returns, compute_se: bool = True) -> BEKKResult:
        y, names, index = _coerce_returns(returns)
        if names is None:
            names = [f"y{i}" for i in range(y.shape[1])]
        T, N = y.shape
        if T <= N:
            raise ValueError(
                f"BEKK needs more observations than assets (T={T}, N={N}): "
                "the targeting covariance would be singular"
            )
        mu = y.mean(axis=0)
        u = y - mu
        Sigma = u.T @ u / T

        scalar = self.variant == "scalar"

        def unpack(x: np.ndarray):
            if scalar:
                return np.full(N, x[0]), np.full(N, x[1])
            return x[:N].copy(), x[N:].copy()

        def negll(x: np.ndarray) -> float:
            avec, bvec = unpack(x)
            # scalar: C PSD follows from the stationarity constraint, skip
            # the per-eval eigendecomposition
            llt = _bekk_eval(u, avec, bvec, Sigma, check_psd=not scalar)
            if llt is None:
                return _PENALTY
            return -float(np.sum(llt))

        if scalar:
            k = 2
            starts = [np.array([0.25, 0.95]), np.array([0.35, 0.90])]
            constraints = [
                {"type": "ineq", "fun": lambda x: 1.0 - _EPS_BOUND - x[0] ** 2 - x[1] ** 2}
            ]
            bounds = [(0.0, 0.9995)] * k
        else:
            k = 2 * N
            # warm-start the diagonal fit from the scalar optimum
            scalar_fit = BEKK("scalar").fit(
                pd.DataFrame(y, columns=names), compute_se=False
            )
            a0, b0 = scalar_fit.avec[0], scalar_fit.bvec[0]
            starts = [np.concatenate([np.full(N, a0), np.full(N, b0)])]
            constraints = [
                {
                    "type": "ineq",
                    "fun": lambda x, i=i: 1.0 - _EPS_BOUND - x[i] ** 2 - x[N + i] ** 2,
                }
                for i in range(N)
            ]
            # smooth PSD constraint on the targeting intercept: per-asset
            # bounds do NOT imply Hadamard PSD for heterogeneous loadings,
            # and a bare penalty cliff stalls SLSQP's line search
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda x: float(
                        np.linalg.eigvalsh(
                            _targeting_intercept(*unpack(x), Sigma)
                        )[0]
                    ),
                }
            )
            # ARCH loadings may be mixed-sign (negative cross news impact is
            # a distinct, valid diagonal BEKK); persistence stays nonnegative
            bounds = [(-0.9995, 0.9995)] * N + [(0.0, 0.9995)] * N

        best = None
        for x0 in starts:
            res = minimize(
                negll,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=constraints,
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if best is None or res.fun < best.fun:
                best = res
        avec, bvec = unpack(np.asarray(best.x, dtype=float))
        # sign normalization: avec enters only through aa', so a global flip
        # is likelihood-invariant -- pin the largest-|a| element positive
        if not scalar and avec[int(np.argmax(np.abs(avec)))] < 0:
            avec = -avec
        out = _bekk_eval(u, avec, bvec, Sigma, want_path=True)
        if out is None:
            raise RuntimeError(
                "BEKK estimation failed: the optimizer terminated at "
                f"infeasible parameters {best.x} ({best.message}); check for "
                "degenerate or near-nonstationary returns"
            )
        llt, Hpath = out
        result = BEKKResult(
            variant=self.variant,
            names=names,
            index=index,
            mu=mu,
            Sigma=Sigma,
            avec=avec,
            bvec=bvec,
            H=Hpath,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            converged=bool(best.success),
            message=str(best.message),
        )
        result._u_last = u[-1]
        if compute_se:
            # sign-normalized parameters, not best.x, so SEs sit at the
            # reported estimate
            psi_hat = (
                np.array([avec[0], bvec[0]])
                if scalar
                else np.concatenate([avec, bvec])
            )

            def llt_fn(x):
                av, bv = unpack(x)
                return _bekk_eval(u, av, bv, Sigma, check_psd=not scalar)

            vc = qml_vcov(llt_fn, psi_hat)
            result.vcov = vc["vcov"]
            result.se = vc["se"]
            result.se_method = vc["method"]
        return result


def simulate_bekk(
    a: float | np.ndarray,
    b: float | np.ndarray,
    Sigma: np.ndarray,
    nobs: int = 1000,
    burn: int = 500,
    mu: np.ndarray | None = None,
    seed: int | None = None,
) -> dict:
    """Simulate from a targeting-parameterized (scalar/diagonal) BEKK DGP."""
    Sigma = np.asarray(Sigma, dtype=float)
    N = Sigma.shape[0]
    avec = np.atleast_1d(np.asarray(a, dtype=float))
    bvec = np.atleast_1d(np.asarray(b, dtype=float))
    if avec.shape == (1,):
        avec = np.full(N, avec[0])
    if bvec.shape == (1,):
        bvec = np.full(N, bvec[0])
    if avec.shape != (N,) or bvec.shape != (N,):
        raise ValueError(
            f"a and b must be scalars or length-{N} vectors matching Sigma"
        )
    if burn < 0 or nobs < 1:
        raise ValueError("nobs must be >= 1 and burn >= 0")
    C = _targeting_intercept(avec, bvec, Sigma)
    if np.linalg.eigvalsh(C)[0] < -1e-12:
        raise ValueError("targeting intercept is not PSD for these (a, b)")
    mu = np.zeros(N) if mu is None else np.asarray(mu, dtype=float)
    rng = np.random.default_rng(seed)
    total = nobs + burn
    aa = np.outer(avec, avec)
    bb = np.outer(bvec, bvec)
    H = Sigma.copy()
    u = np.empty((total, N))
    Hs = np.empty((total, N, N))
    for t in range(total):
        if t > 0:
            up = u[t - 1]
            H = C + aa * np.outer(up, up) + bb * H
        chol = np.linalg.cholesky(H)
        u[t] = chol @ rng.standard_normal(N)
        Hs[t] = H
    return {"returns": mu + u[burn:], "H": Hs[burn:]}
