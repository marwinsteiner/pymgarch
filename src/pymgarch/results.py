"""Result containers: fitted paths, forecasting, filtering, summaries."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .correlation import cov2cor, dcc_path, one_step_q
from .distributions import mvnorm_llt, mvt_llt, sample_standardized
from .estimation import ParamLayout
from .marginals import MarginalSet


@dataclass
class MGARCHForecast:
    horizon: int
    method: str
    variances: np.ndarray  # (h, N)
    correlations: np.ndarray  # (h, N, N)
    covariances: np.ndarray  # (h, N, N)
    n_paths: int | None = None


@dataclass
class MGARCHResult:
    model: str  # "DCC" | "ADCC" | "CCC"
    dist: str  # "norm" | "t"
    names: list[str]
    index: pd.Index
    sigma: np.ndarray  # (T, N) conditional volatilities
    eps: np.ndarray  # (T, N) standardized residuals
    R: np.ndarray  # (T, N, N) conditional correlations
    logdet: np.ndarray
    quad: np.ndarray
    Sbar: np.ndarray
    Nbar: np.ndarray
    psi: np.ndarray  # stage-2 parameters, possibly empty (CCC)
    psi_names: list[str]
    layout: ParamLayout | None
    loglikelihood: float
    llt: np.ndarray  # per-obs joint log-likelihood
    q_last: np.ndarray | None = None
    mset: MarginalSet | None = None
    vcov: np.ndarray | None = None
    se: np.ndarray | None = None
    se_method: str | None = None
    converged: bool = True
    message: str = ""
    filtered: bool = False
    extras: dict = field(default_factory=dict)

    # -- basic accessors ---------------------------------------------------

    @property
    def nobs(self) -> int:
        return int(self.eps.shape[0])

    @property
    def nassets(self) -> int:
        return int(self.eps.shape[1])

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.psi_names, [float(v) for v in self.psi]))

    @property
    def std_errors(self) -> dict[str, float] | None:
        if self.se is None:
            return None
        return dict(zip(self.psi_names, [float(v) for v in self.se]))

    @property
    def conditional_correlations(self) -> np.ndarray:
        return self.R

    @property
    def conditional_covariances(self) -> np.ndarray:
        H = np.empty_like(self.R)
        for t in range(self.nobs):
            s = self.sigma[t]
            H[t] = self.R[t] * np.outer(s, s)
        return H

    @property
    def num_params(self) -> int:
        k = len(self.psi)
        if self.mset is not None:
            k += self.mset.total_params
        return k

    @property
    def aic(self) -> float:
        return -2.0 * self.loglikelihood + 2.0 * self.num_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglikelihood + self.num_params * float(np.log(self.nobs))

    def _dcc_coefs(self) -> tuple[float, float, float, float | None]:
        if self.layout is None:
            return 0.0, 0.0, 0.0, None
        return self.layout.unpack(self.psi)

    # -- forecasting -------------------------------------------------------

    def forecast(
        self,
        horizon: int = 1,
        method: str = "analytic",
        n_paths: int = 1000,
        seed: int | None = None,
    ) -> MGARCHForecast:
        if self.filtered:
            raise NotImplementedError(
                "forecasting from a filtered result is not supported in v0.1; "
                "forecast from the fitted result instead"
            )
        if self.mset is None:
            raise ValueError("forecasting requires the fitted marginals")
        if method == "analytic":
            return self._forecast_analytic(horizon)
        if method == "simulation":
            return self._forecast_simulation(horizon, n_paths, seed)
        raise ValueError("method must be 'analytic' or 'simulation'")

    def _marginal_variance_forecasts(self, horizon: int) -> np.ndarray:
        out = np.empty((horizon, self.nassets))
        for i, res in enumerate(self.mset.results):
            fc = res.forecast(horizon=horizon, reindex=False)
            out[:, i] = np.asarray(fc.variance, dtype=float)[0]
        return out

    def _forecast_analytic(self, horizon: int) -> MGARCHForecast:
        a, b, g, _ = self._dcc_coefs()
        if self.model == "ADCC":
            raise NotImplementedError(
                "no standard analytic multi-step correlation forecast exists "
                "for ADCC; use method='simulation'"
            )
        variances = self._marginal_variance_forecasts(horizon)
        correlations = np.empty((horizon, self.nassets, self.nassets))
        if self.model == "CCC":
            correlations[:] = self.Sbar
        else:
            q1 = one_step_q(self.q_last, self.eps[-1], a, b, g, self.Sbar, self.Nbar)
            r1 = cov2cor(q1)
            correlations[0] = r1
            # Engle-Sheppard approximation: R_{T+h} decays to Sbar at rate a+b
            for h in range(1, horizon):
                w = (a + b) ** h
                correlations[h] = (1.0 - w) * self.Sbar + w * r1
        covariances = np.empty_like(correlations)
        for h in range(horizon):
            d = np.sqrt(variances[h])
            covariances[h] = correlations[h] * np.outer(d, d)
        return MGARCHForecast(horizon, "analytic", variances, correlations, covariances)

    def _forecast_simulation(
        self, horizon: int, n_paths: int, seed: int | None
    ) -> MGARCHForecast:
        a, b, g, nu = self._dcc_coefs()
        N = self.nassets
        rng = np.random.default_rng(seed)
        omega = (1.0 - a - b) * self.Sbar - g * self.Nbar
        fams = [self.mset.garch_family(i) for i in range(N)]
        if any(f is None for f in fams):
            bad = [self.mset.names[i] for i, f in enumerate(fams) if f is None]
            raise NotImplementedError(
                f"simulation forecasts need GARCH/GJR marginals; not: {bad}"
            )
        states = [self.mset.sim_state(i) for i in range(N)]
        # replicate lag state across paths
        u_lags = [np.tile(st[0][:, None], (1, n_paths)) for st in states]
        s2_lags = [np.tile(st[1][:, None], (1, n_paths)) for st in states]
        Q = np.tile(self.q_last[None, :, :], (n_paths, 1, 1))
        eps_prev = np.tile(self.eps[-1][None, :], (n_paths, 1))

        variances = np.zeros((horizon, N))
        correlations = np.zeros((horizon, N, N))
        covariances = np.zeros((horizon, N, N))

        for h in range(horizon):
            # marginal variances from lag state (vectorized across paths)
            sig2 = np.empty((n_paths, N))
            for i, (vol_p, p, o, q) in enumerate(fams):
                s2 = np.full(n_paths, vol_p[0])
                for lag in range(p):
                    s2 += vol_p[1 + lag] * u_lags[i][lag] ** 2
                for lag in range(o):
                    neg = u_lags[i][lag] < 0.0
                    s2 += vol_p[1 + p + lag] * (u_lags[i][lag] ** 2) * neg
                for lag in range(q):
                    s2 += vol_p[1 + p + o + lag] * s2_lags[i][lag]
                sig2[:, i] = s2
            # correlation state update from previous shocks
            outer = np.einsum("mi,mj->mij", eps_prev, eps_prev)
            Q = omega[None, :, :] + a * outer + b * Q
            if g > 0.0:
                neg = np.minimum(eps_prev, 0.0)
                Q = Q + g * np.einsum("mi,mj->mij", neg, neg)
            d = np.sqrt(np.einsum("mii->mi", Q))
            Rm = Q / np.einsum("mi,mj->mij", d, d)
            # draw correlated standardized shocks path by path
            z = np.empty((n_paths, N))
            for m in range(n_paths):
                chol = np.linalg.cholesky(Rm[m])
                z[m] = sample_standardized(rng, chol, 1, self.dist, nu)[0]
            u = np.sqrt(sig2) * z
            # accumulate forecast moments
            variances[h] = sig2.mean(axis=0)
            correlations[h] = Rm.mean(axis=0)
            sig = np.sqrt(sig2)
            covariances[h] = np.einsum(
                "mij,mi,mj->ij", Rm, sig, sig
            ) / n_paths
            # roll lag states
            for i in range(N):
                if u_lags[i].shape[0] > 0:
                    u_lags[i] = np.vstack([u[None, :, i], u_lags[i][:-1]])
                if s2_lags[i].shape[0] > 0:
                    s2_lags[i] = np.vstack([sig2[None, :, i], s2_lags[i][:-1]])
            eps_prev = z
        return MGARCHForecast(
            horizon, "simulation", variances, correlations, covariances, n_paths
        )

    # -- filtering ---------------------------------------------------------

    def filter(self, returns) -> "MGARCHResult":
        """Apply fitted parameters to a new return sample (no re-estimation).

        Uses the fitted correlation targets Sbar/Nbar and Q_1 = Sbar.
        """
        if self.mset is None:
            raise ValueError("filtering requires the fitted marginals")
        y, names, index = _coerce_returns(returns)
        if y.shape[1] != self.nassets:
            raise ValueError(
                f"expected {self.nassets} columns, got {y.shape[1]}"
            )
        sigma = np.empty_like(y)
        resid = np.empty_like(y)
        for i in range(self.nassets):
            r, s = self.mset.filter_new_data(i, y[:, i])
            resid[:, i] = r
            sigma[:, i] = s
        eps = resid / sigma
        a, b, g, nu = self._dcc_coefs()
        if self.model == "CCC":
            llt2, logdet, quad, R, q_last = _constant_corr_eval(
                eps, self.Sbar, self.dist, nu
            )
        else:
            path = dcc_path(eps, a, b, g, self.Sbar, self.Nbar)
            if not path.ok:
                raise RuntimeError("correlation recursion failed on new data")
            ndim = eps.shape[1]
            if self.dist == "t":
                llt2 = mvt_llt(path.logdet, path.quad, nu, ndim)
            else:
                llt2 = mvnorm_llt(path.logdet, path.quad, ndim)
            logdet, quad, R, q_last = path.logdet, path.quad, path.R, path.q_last
        llt = llt2 - np.sum(np.log(sigma), axis=1)
        return MGARCHResult(
            model=self.model,
            dist=self.dist,
            names=names or self.names,
            index=index,
            sigma=sigma,
            eps=eps,
            R=R,
            logdet=logdet,
            quad=quad,
            Sbar=self.Sbar,
            Nbar=self.Nbar,
            psi=self.psi.copy(),
            psi_names=list(self.psi_names),
            layout=self.layout,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            q_last=q_last,
            mset=None,
            filtered=True,
        )

    # -- reporting ---------------------------------------------------------

    def summary(self) -> str:
        lines = []
        title = f"{self.model} ({'Student-t' if self.dist == 't' else 'Gaussian'})"
        lines.append(title)
        lines.append("=" * len(title))
        lines.append(f"Assets: {self.nassets}   Obs: {self.nobs}")
        lines.append(
            f"Joint log-likelihood: {self.loglikelihood:.4f}   "
            f"AIC: {self.aic:.2f}   BIC: {self.bic:.2f}"
        )
        if len(self.psi):
            lines.append("")
            lines.append(f"{'param':<8}{'coef':>12}{'std err':>12}{'z':>10}")
            for j, name in enumerate(self.psi_names):
                c = self.psi[j]
                if self.se is not None and np.isfinite(self.se[j]) and self.se[j] > 0:
                    z = c / self.se[j]
                    lines.append(f"{name:<8}{c:>12.6f}{self.se[j]:>12.6f}{z:>10.3f}")
                else:
                    lines.append(f"{name:<8}{c:>12.6f}{'--':>12}{'--':>10}")
            if self.se_method:
                lines.append(f"Covariance: {self.se_method}")
            if self.se_method == "stage2-robust":
                lines.append(
                    "  (marginal estimation error ignored; see docs on inference)"
                )
        if not self.converged:
            lines.append(f"WARNING: optimizer did not converge: {self.message}")
        if self.filtered:
            lines.append("(filtered result: parameters fixed, no estimation)")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<MGARCHResult {self.model}-{self.dist} N={self.nassets} "
            f"T={self.nobs} ll={self.loglikelihood:.2f}>"
        )


def _constant_corr_eval(eps, R0, dist, nu):
    """Per-obs stage-2 quantities under a constant correlation matrix."""
    T, N = eps.shape
    chol = np.linalg.cholesky(R0)
    logdet_val = 2.0 * float(np.sum(np.log(np.diag(chol))))
    x = np.linalg.solve(chol, eps.T)
    quad = np.sum(x * x, axis=0)
    logdet = np.full(T, logdet_val)
    if dist == "t":
        llt = mvt_llt(logdet, quad, nu, N)
    else:
        llt = mvnorm_llt(logdet, quad, N)
    R = np.tile(R0[None, :, :], (T, 1, 1))
    return llt, logdet, quad, R, R0.copy()


def _coerce_returns(returns):
    """Normalize input to (ndarray (T,N), names, index)."""
    if isinstance(returns, pd.DataFrame):
        y = returns.to_numpy(dtype=float)
        return y, [str(c) for c in returns.columns], returns.index
    y = np.asarray(returns, dtype=float)
    if y.ndim != 2:
        raise ValueError("returns must be 2-dimensional (T, N)")
    return y, None, pd.RangeIndex(y.shape[0])
