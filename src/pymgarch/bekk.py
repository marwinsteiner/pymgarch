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
from .results import _coerce_returns

_PENALTY = 1e10
_EPS_BOUND = 1e-6


def _bekk_llt(u: np.ndarray, avec: np.ndarray, bvec: np.ndarray, Sigma: np.ndarray):
    """Per-obs Gaussian log-likelihood under targeting, or None if infeasible."""
    coef = 1.0 - np.outer(avec, avec) - np.outer(bvec, bvec)
    C = Sigma * coef
    if np.any(avec**2 + bvec**2 >= 1.0):
        return None
    if np.linalg.eigvalsh(C)[0] < -1e-12:
        return None
    flag, llt, _, _ = bekk_recursion(
        np.ascontiguousarray(u),
        np.ascontiguousarray(avec, dtype=np.float64),
        np.ascontiguousarray(bvec, dtype=np.float64),
        np.ascontiguousarray(C, dtype=np.float64),
        np.ascontiguousarray(Sigma, dtype=np.float64),
    )
    if flag != 0 or not np.all(np.isfinite(llt)):
        return None
    return llt


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
        return len(self.psi)

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
        lines.append(f"{'param':<10}{'coef':>12}{'std err':>12}")
        for j, name in enumerate(self.psi_names):
            se = (
                f"{self.se[j]:>12.6f}"
                if self.se is not None and np.isfinite(self.se[j])
                else f"{'--':>12}"
            )
            lines.append(f"{name:<10}{self.psi[j]:>12.6f}{se}")
        if self.se_method:
            lines.append(f"Covariance: {self.se_method}")
        if not self.converged:
            lines.append(f"WARNING: optimizer did not converge: {self.message}")
        if self.filtered:
            lines.append("(filtered result: parameters fixed, no estimation)")
        return "\n".join(lines)

    # -- forecasting -------------------------------------------------------

    def forecast(self, horizon: int = 1):
        """Closed-form covariance forecasts.

        E[H_{T+1}] is deterministic; for h >= 2,
        E[H_{T+h}] = C + M o E[H_{T+h-1}] with M = a a' + b b'.
        Returns {"covariances": (h, N, N), "correlations": (h, N, N)}.
        """
        if self.filtered:
            raise NotImplementedError("forecast from the fitted result instead")
        u_last = getattr(self, "_u_last", None)
        if u_last is None:
            raise RuntimeError("terminal state missing; refit the model")
        coef = 1.0 - np.outer(self.avec, self.avec) - np.outer(self.bvec, self.bvec)
        C = self.Sigma * coef
        M = np.outer(self.avec, self.avec) + np.outer(self.bvec, self.bvec)
        covs = np.empty((horizon, self.nassets, self.nassets))
        covs[0] = (
            C
            + np.outer(self.avec, self.avec) * np.outer(u_last, u_last)
            + np.outer(self.bvec, self.bvec) * self.H[-1]
        )
        for h in range(1, horizon):
            covs[h] = C + M * covs[h - 1]
        s = np.sqrt(np.einsum("hii->hi", covs))
        corrs = covs / np.einsum("hi,hj->hij", s, s)
        return {"covariances": covs, "correlations": corrs}

    # -- filtering ---------------------------------------------------------

    def filter(self, returns) -> BEKKResult:
        y, names, index = _coerce_returns(returns)
        if y.shape[1] != self.nassets:
            raise ValueError(f"expected {self.nassets} columns, got {y.shape[1]}")
        u = y - self.mu
        llt = _bekk_llt(u, self.avec, self.bvec, self.Sigma)
        if llt is None:
            raise RuntimeError("BEKK recursion failed on new data")
        coef = 1.0 - np.outer(self.avec, self.avec) - np.outer(self.bvec, self.bvec)
        _, _, Hpath, _ = bekk_recursion(
            np.ascontiguousarray(u),
            self.avec,
            self.bvec,
            np.ascontiguousarray(self.Sigma * coef),
            np.ascontiguousarray(self.Sigma),
        )
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
        mu = y.mean(axis=0)
        u = y - mu
        Sigma = u.T @ u / T

        def unpack(x: np.ndarray):
            if self.variant == "scalar":
                return np.full(N, x[0]), np.full(N, x[1])
            return x[:N].copy(), x[N:].copy()

        def negll(x: np.ndarray) -> float:
            avec, bvec = unpack(x)
            llt = _bekk_llt(u, avec, bvec, Sigma)
            if llt is None:
                return _PENALTY
            return -float(np.sum(llt))

        if self.variant == "scalar":
            k = 2
            starts = [np.array([0.25, 0.95]), np.array([0.35, 0.90])]
            constraints = [
                {"type": "ineq", "fun": lambda x: 1.0 - _EPS_BOUND - x[0] ** 2 - x[1] ** 2}
            ]
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
        bounds = [(0.0, 0.9995)] * k

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
        llt = _bekk_llt(u, avec, bvec, Sigma)
        if llt is None:
            raise RuntimeError(
                f"BEKK estimation terminated at infeasible parameters {best.x}"
            )
        coef = 1.0 - np.outer(avec, avec) - np.outer(bvec, bvec)
        _, _, Hpath, _ = bekk_recursion(
            np.ascontiguousarray(u),
            np.ascontiguousarray(avec),
            np.ascontiguousarray(bvec),
            np.ascontiguousarray(Sigma * coef),
            np.ascontiguousarray(Sigma),
        )
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

            def llt_fn(x):
                av, bv = unpack(x)
                return _bekk_llt(u, av, bv, Sigma)

            vc = qml_vcov(llt_fn, np.asarray(best.x, dtype=float))
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
    avec = np.full(N, float(a)) if np.isscalar(a) else np.asarray(a, dtype=float)
    bvec = np.full(N, float(b)) if np.isscalar(b) else np.asarray(b, dtype=float)
    coef = 1.0 - np.outer(avec, avec) - np.outer(bvec, bvec)
    C = Sigma * coef
    if np.linalg.eigvalsh(C)[0] < -1e-12:
        raise ValueError("targeting intercept is not PSD for these (a, b)")
    mu = np.zeros(N) if mu is None else np.asarray(mu, dtype=float)
    rng = np.random.default_rng(seed)
    total = nobs + burn
    H = Sigma.copy()
    u = np.empty((total, N))
    Hs = np.empty((total, N, N))
    for t in range(total):
        if t > 0:
            up = u[t - 1]
            H = C + np.outer(avec, avec) * np.outer(up, up) + np.outer(
                bvec, bvec
            ) * H
        chol = np.linalg.cholesky(H)
        u[t] = chol @ rng.standard_normal(N)
        Hs[t] = H
    return {"returns": mu + u[burn:], "H": Hs[burn:]}
