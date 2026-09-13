"""GO-GARCH (van der Weide 2002) via ICA, in the rmgarch/tsmarch mold.

Model: r_t = mu + A f_t with statistically independent factors f_t, each
following a univariate GARCH process with unit unconditional variance (the
ICA whitening step enforces the scale). The conditional covariance is
H_t = A D_t A' with D_t = diag of factor conditional variances.

Because the factors are independent, the joint likelihood decomposes into
the sum of univariate factor likelihoods plus the constant Jacobian
-T log|det A| -- estimation is exactly N univariate arch fits after ICA.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._kernels import garch_forward
from .ica import fastica
from .marginals import MarginalSet, UnivariateSpec
from .results import _coerce_returns


@dataclass
class GOGARCHForecast:
    horizon: int
    method: str
    factor_variances: np.ndarray  # (h, N)
    covariances: np.ndarray  # (h, N, N)
    correlations: np.ndarray  # (h, N, N)
    n_paths: int | None = None


@dataclass
class GOGARCHResult:
    names: list[str]
    index: pd.Index
    mu: np.ndarray  # (N,) sample means removed before ICA
    A: np.ndarray  # (N, N) mixing matrix
    U: np.ndarray  # (N, N) unmixing matrix, U = A^{-1}
    mset: MarginalSet  # factor marginals (Zero-mean GARCH on f_t)
    loglikelihood: float
    llt: np.ndarray
    ica_converged: bool
    filtered: bool = False

    model = "GOGARCH"

    @property
    def nobs(self) -> int:
        return self.mset.nobs

    @property
    def nassets(self) -> int:
        return len(self.names)

    @property
    def factor_sigma(self) -> np.ndarray:
        """(T, N) conditional volatilities of the factors."""
        override = self.__dict__.get("_filtered_sigma")
        return override if override is not None else self.mset.sigma

    @property
    def conditional_covariances(self) -> np.ndarray:
        d2 = self.factor_sigma**2
        # H_t = A diag(d2_t) A'
        return np.einsum("ik,tk,jk->tij", self.A, d2, self.A)

    @property
    def conditional_correlations(self) -> np.ndarray:
        H = self.conditional_covariances
        s = np.sqrt(np.einsum("tii->ti", H))
        return H / np.einsum("ti,tj->tij", s, s)

    @property
    def num_params(self) -> int:
        # factor GARCH parameters; the ICA rotation is not counted, matching
        # rmgarch's convention (A is a method-of-moments style estimate)
        return self.mset.total_params

    @property
    def aic(self) -> float:
        return -2.0 * self.loglikelihood + 2.0 * self.num_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglikelihood + self.num_params * float(np.log(self.nobs))

    def summary(self) -> str:
        lines = ["GO-GARCH (ICA)", "=============="]
        lines.append(f"Assets: {self.nassets}   Obs: {self.nobs}")
        lines.append(
            f"Joint log-likelihood: {self.loglikelihood:.4f}   "
            f"AIC: {self.aic:.2f}   BIC: {self.bic:.2f}"
        )
        lines.append(f"|det A|: {abs(np.linalg.det(self.A)):.6f}")
        lines.append("")
        lines.append(f"{'factor':<8}{'omega':>10}{'alpha':>10}{'beta':>10}{'persist':>10}")
        for i in range(self.mset.nassets):
            fam = self.mset.garch_family(i)
            if fam is None:
                lines.append(f"f{i:<7}{'(non-GARCH factor spec)':>40}")
                continue
            vol_p, p, o, _q = fam
            al = float(np.sum(vol_p[1 : 1 + p]))
            gm = float(np.sum(vol_p[1 + p : 1 + p + o]))
            be = float(np.sum(vol_p[1 + p + o :]))
            lines.append(
                f"f{i:<7}{vol_p[0]:>10.4f}{al:>10.4f}{be:>10.4f}{al + be + 0.5 * gm:>10.4f}"
            )
        if not self.ica_converged:
            lines.append("WARNING: fastICA did not converge; results may be unstable")
        if self.filtered:
            lines.append("(filtered result: parameters fixed, no estimation)")
        return "\n".join(lines)

    # -- forecasting -------------------------------------------------------

    def forecast(
        self,
        horizon: int = 1,
        method: str = "analytic",
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> GOGARCHForecast:
        if self.filtered:
            raise NotImplementedError(
                "forecasting from a filtered result is not supported; "
                "forecast from the fitted result instead"
            )
        if method == "analytic":
            fvar = np.empty((horizon, self.mset.nassets))
            for i, res in enumerate(self.mset.results):
                fc = res.forecast(horizon=horizon, reindex=False)
                fvar[:, i] = np.asarray(fc.variance, dtype=float)[0]
        elif method == "simulation":
            rng = np.random.default_rng(seed)
            fvar = np.empty((horizon, self.mset.nassets))
            for i in range(self.mset.nassets):
                fam = self.mset.garch_family(i)
                if fam is None:
                    raise NotImplementedError(
                        "simulation forecasts need GARCH/GJR factor specs"
                    )
                vol_p, p, o, q = fam
                u_lags, s2_lags = self.mset.sim_state(i)
                z = rng.standard_normal((horizon, n_paths))
                _, sig2 = garch_forward(
                    vol_p,
                    p,
                    o,
                    q,
                    z,
                    np.tile(u_lags[:, None], (1, n_paths)),
                    np.tile(s2_lags[:, None], (1, n_paths)),
                )
                fvar[:, i] = sig2.mean(axis=1)
        else:
            raise ValueError("method must be 'analytic' or 'simulation'")
        cov = np.einsum("ik,hk,jk->hij", self.A, fvar, self.A)
        s = np.sqrt(np.einsum("hii->hi", cov))
        corr = cov / np.einsum("hi,hj->hij", s, s)
        return GOGARCHForecast(
            horizon,
            method,
            fvar,
            cov,
            corr,
            n_paths if method == "simulation" else None,
        )

    # -- filtering ---------------------------------------------------------

    def filter(self, returns) -> GOGARCHResult:
        """Apply the fitted rotation and factor parameters to new returns."""
        y, names, index = _coerce_returns(returns)
        if y.shape[1] != self.nassets:
            raise ValueError(f"expected {self.nassets} columns, got {y.shape[1]}")
        factors = (y - self.mu) @ self.U.T
        sigma = np.empty_like(factors)
        for i in range(self.mset.nassets):
            _, s = self.mset.filter_new_data(i, factors[:, i])
            sigma[:, i] = s
        llt = _gogarch_llt(factors, sigma, self.A)
        new = GOGARCHResult(
            names=names or self.names,
            index=index,
            mu=self.mu.copy(),
            A=self.A.copy(),
            U=self.U.copy(),
            mset=self.mset,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            ica_converged=self.ica_converged,
            filtered=True,
        )
        # filtered paths differ from the stored mset paths; freeze them
        new.__dict__["_filtered_sigma"] = sigma
        return new

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<GOGARCHResult N={self.nassets} T={self.nobs} "
            f"ll={self.loglikelihood:.2f}>"
        )


def _gogarch_llt(factors: np.ndarray, sigma: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Gaussian per-obs joint log-likelihood of returns given factor paths."""
    log2pi = float(np.log(2.0 * np.pi))
    _, logdet_A = np.linalg.slogdet(A)
    z2 = (factors / sigma) ** 2
    ll_factors = -0.5 * (log2pi + 2.0 * np.log(sigma) + z2).sum(axis=1)
    return ll_factors - logdet_A


class GOGARCH:
    """GO-GARCH with ICA rotation and arch GARCH factors.

    dist: factor innovation distribution, passed to the factor spec
    ("normal" default; "t" gives Student-t factors).
    """

    def __init__(self, dist: str = "normal"):
        self.dist = dist

    def fit(
        self,
        returns,
        spec: UnivariateSpec | None = None,
        seed: int = 0,
    ) -> GOGARCHResult:
        y, names, index = _coerce_returns(returns)
        if names is None:
            names = [f"y{i}" for i in range(y.shape[1])]
        mu = y.mean(axis=0)
        ica = fastica(y - mu, seed=seed)
        factors = (y - mu) @ ica.U.T
        fdf = pd.DataFrame(
            factors, columns=[f"f{i}" for i in range(factors.shape[1])], index=index
        )
        if spec is None:
            spec = UnivariateSpec(mean="Zero", dist=self.dist)
        elif spec.mean != "Zero":
            raise ValueError(
                "GO-GARCH factors are zero-mean by construction; "
                "use a spec with mean='Zero'"
            )
        mset = MarginalSet.from_returns(fdf, spec)
        # joint ll: sum of factor lls (arch) + Jacobian of r = A f
        _, logdet_A = np.linalg.slogdet(ica.A)
        llt = np.zeros(mset.nobs)
        for i in range(mset.nassets):
            llt += mset.per_obs_loglik(i)
        llt -= logdet_A
        return GOGARCHResult(
            names=names,
            index=index,
            mu=mu,
            A=ica.A,
            U=ica.U,
            mset=mset,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            ica_converged=ica.converged,
        )


def gogarch_cov_path(A: np.ndarray, factor_sigma: np.ndarray) -> np.ndarray:
    """H_t = A diag(sigma_t^2) A' for external use (fixtures, diagnostics)."""
    return np.einsum("ik,tk,jk->tij", A, factor_sigma**2, A)


def gogarch_corr_from_cov(H: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.einsum("tii->ti", H))
    return H / np.einsum("ti,tj->tij", s, s)
