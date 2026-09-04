"""Copula-GARCH: Gaussian and Student-t copulas over arch marginals.

Two-stage construction (Patton 2006; rmgarch's cgarch): arch marginals give
standardized residuals eps_it, the probability integral transform maps them
to uniforms u_it, and a copula with constant or DCC-driven correlation ties
them together. The joint log-likelihood decomposes as

    sum_i loglik_marginal_i  +  sum_t log c(u_t; R_t, ...).

Implementation note: the Student-t copula is parameterized through the
covariance-standardized t family (unit-variance margins). A copula is
invariant to strictly monotone marginal rescaling, so this is the same t
copula as the textbook one, but it keeps the DCC recursion inputs
unit-variance and reuses the existing mvt kernel unchanged. A corollary used
by the tests: a Gaussian copula with parametric normal margins reproduces the
plain DCC model exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

from .correlation import cov2cor, dcc_path, one_step_q
from .distributions import mvnorm_llt, mvt_llt
from .estimation import ParamLayout
from .inference import two_stage_vcov
from .marginals import MarginalSet
from .models import _build_marginals
from .results import _coerce_returns

_CLIP = 1e-12
_PENALTY = 1e10


# -- probability integral transforms --------------------------------------


def _std_t_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    """CDF of the unit-variance (covariance-standardized) Student-t."""
    return stats.t.cdf(x * np.sqrt(nu / (nu - 2.0)), df=nu)


def _std_t_ppf(u: np.ndarray, nu: float) -> np.ndarray:
    return stats.t.ppf(u, df=nu) * np.sqrt((nu - 2.0) / nu)


def _std_t_logpdf(x: np.ndarray, nu: float) -> np.ndarray:
    s = np.sqrt(nu / (nu - 2.0))
    return stats.t.logpdf(x * s, df=nu) + np.log(s)


def pit_parametric(mset: MarginalSet, eps: np.ndarray) -> np.ndarray:
    """PIT through each marginal's fitted arch distribution.

    Supports Normal and (standardized) Student-t marginals; other arch
    distributions need margins="empirical". Marginal shape parameters are
    held at their fitted values.
    """
    u = np.empty_like(eps)
    for i in range(mset.nassets):
        res = mset.results[i]
        dist_name = type(res.model.distribution).__name__
        if dist_name == "Normal":
            u[:, i] = stats.norm.cdf(eps[:, i])
        elif dist_name == "StudentsT":
            nu_i = float(np.asarray(res.params)[-1])
            u[:, i] = _std_t_cdf(eps[:, i], nu_i)
        else:
            raise NotImplementedError(
                f"parametric PIT not implemented for arch distribution "
                f"{dist_name!r}; use margins='empirical'"
            )
    return np.clip(u, _CLIP, 1.0 - _CLIP)


def pit_empirical(eps: np.ndarray) -> np.ndarray:
    """Rank-based PIT, rescaled by T/(T+1) to stay inside (0, 1)."""
    T = eps.shape[0]
    u = np.empty_like(eps)
    for i in range(eps.shape[1]):
        u[:, i] = stats.rankdata(eps[:, i]) / (T + 1.0)
    return u


def kendall_correlation(eps: np.ndarray) -> np.ndarray:
    """R = sin(pi/2 * tau) pairwise, eigenvalue-clipped to be PD.

    Kendall's tau is invariant under the monotone PIT, so it can be computed
    on the standardized residuals directly.
    """
    N = eps.shape[1]
    R = np.eye(N)
    for i in range(N):
        for j in range(i + 1, N):
            tau = stats.kendalltau(eps[:, i], eps[:, j]).statistic
            R[i, j] = R[j, i] = np.sin(0.5 * np.pi * tau)
    vals, vecs = np.linalg.eigh(R)
    if vals[0] <= 1e-10:
        vals = np.clip(vals, 1e-10, None)
        R = cov2cor((vecs * vals) @ vecs.T)
    return R


# -- copula log-densities --------------------------------------------------


def _copula_llt_dynamic(
    eta: np.ndarray, a: float, b: float, nu: float | None
) -> np.ndarray | None:
    """log c(u_t) along a DCC path on the copula shocks eta; None on failure."""
    T, N = eta.shape
    Sbar = cov2cor(eta.T @ eta / T)
    path = dcc_path(eta, a, b, 0.0, Sbar, None)
    if not path.ok:
        return None
    if nu is None:
        joint = mvnorm_llt(path.logdet, path.quad, N)
        margins = stats.norm.logpdf(eta).sum(axis=1)
    else:
        joint = mvt_llt(path.logdet, path.quad, nu, N)
        margins = _std_t_logpdf(eta, nu).sum(axis=1)
    return joint - margins


def _copula_llt_static(
    eta: np.ndarray, R: np.ndarray, nu: float | None
) -> np.ndarray:
    T, N = eta.shape
    chol = np.linalg.cholesky(R)
    logdet = np.full(T, 2.0 * float(np.sum(np.log(np.diag(chol)))))
    x = np.linalg.solve(chol, eta.T)
    quad = np.sum(x * x, axis=0)
    if nu is None:
        joint = mvnorm_llt(logdet, quad, N)
        margins = stats.norm.logpdf(eta).sum(axis=1)
    else:
        joint = mvt_llt(logdet, quad, nu, N)
        margins = _std_t_logpdf(eta, nu).sum(axis=1)
    return joint - margins


def _eta_from_u(u: np.ndarray, copula: str, nu: float | None) -> np.ndarray:
    if copula == "gaussian":
        return stats.norm.ppf(u)
    return _std_t_ppf(u, nu)


# -- result ----------------------------------------------------------------


@dataclass
class CopulaGARCHResult:
    copula: str  # "gaussian" | "t"
    dynamics: str  # "dcc" | "static"
    margins: str  # "parametric" | "empirical"
    names: list[str]
    index: pd.Index
    mset: MarginalSet
    u: np.ndarray  # (T, N) PIT values
    eta: np.ndarray  # (T, N) copula shocks at fitted parameters
    R: np.ndarray  # (T, N, N) copula correlation path
    psi: np.ndarray
    psi_names: list[str]
    loglikelihood: float
    llt: np.ndarray  # per-obs joint (marginals + copula)
    llt_copula: np.ndarray
    Sbar: np.ndarray
    q_last: np.ndarray | None = None
    vcov: np.ndarray | None = None
    se: np.ndarray | None = None
    se_method: str | None = None
    converged: bool = True
    message: str = ""
    filtered: bool = False

    model = "CGARCH"

    @property
    def nobs(self) -> int:
        return int(self.u.shape[0])

    @property
    def nassets(self) -> int:
        return int(self.u.shape[1])

    @property
    def params(self) -> dict[str, float]:
        return dict(zip(self.psi_names, [float(v) for v in self.psi]))

    @property
    def std_errors(self) -> dict[str, float] | None:
        if self.se is None:
            return None
        return dict(zip(self.psi_names, [float(v) for v in self.se]))

    @property
    def copula_correlations(self) -> np.ndarray:
        return self.R

    @property
    def num_params(self) -> int:
        return self.mset.total_params + len(self.psi)

    @property
    def aic(self) -> float:
        return -2.0 * self.loglikelihood + 2.0 * self.num_params

    @property
    def bic(self) -> float:
        return -2.0 * self.loglikelihood + self.num_params * float(np.log(self.nobs))

    def summary(self) -> str:
        title = (
            f"Copula-GARCH ({self.copula} copula, {self.dynamics} dynamics, "
            f"{self.margins} margins)"
        )
        lines = [title, "=" * len(title)]
        lines.append(f"Assets: {self.nassets}   Obs: {self.nobs}")
        lines.append(
            f"Joint log-likelihood: {self.loglikelihood:.4f}   "
            f"AIC: {self.aic:.2f}   BIC: {self.bic:.2f}"
        )
        if len(self.psi):
            lines.append("")
            lines.append(f"{'param':<8}{'coef':>12}{'std err':>12}")
            for j, name in enumerate(self.psi_names):
                se = (
                    f"{self.se[j]:>12.6f}"
                    if self.se is not None and np.isfinite(self.se[j])
                    else f"{'--':>12}"
                )
                lines.append(f"{name:<8}{self.psi[j]:>12.6f}{se}")
            if self.se_method:
                lines.append(f"Covariance: {self.se_method}")
        if not self.converged:
            lines.append(f"WARNING: optimizer did not converge: {self.message}")
        if self.filtered:
            lines.append("(filtered result: parameters fixed, no estimation)")
        return "\n".join(lines)

    # -- simulation-based forecasting --------------------------------------

    def simulate(self, horizon: int, n_paths: int = 1000, seed: int | None = None):
        """Simulate future returns through the fitted copula.

        Returns {"returns": (h, n_paths, N), "covariances": (h, N, N)} where
        covariances are sample covariances of the simulated returns per step.
        Requires parametric margins and GARCH/GJR marginal volatilities.
        """
        if self.margins != "parametric":
            raise NotImplementedError(
                "simulation requires parametric margins (empirical quantile "
                "inversion is not implemented)"
            )
        if self.filtered:
            raise NotImplementedError("simulate from the fitted result instead")
        rng = np.random.default_rng(seed)
        N = self.nassets
        a, b, nu = self._coefs()
        fams = [self.mset.garch_family(i) for i in range(N)]
        if any(f is None for f in fams):
            raise NotImplementedError("simulation needs GARCH/GJR marginals")
        states = [self.mset.sim_state(i) for i in range(N)]
        u_lags = [np.tile(st[0][:, None], (1, n_paths)) for st in states]
        s2_lags = [np.tile(st[1][:, None], (1, n_paths)) for st in states]
        if self.dynamics == "dcc":
            Q = np.tile(self.q_last[None, :, :], (n_paths, 1, 1))
            eta_prev = np.tile(self.eta[-1][None, :], (n_paths, 1))
            omega = (1.0 - a - b) * self.Sbar
        rets = np.empty((horizon, n_paths, N))
        covs = np.empty((horizon, N, N))
        marg_dists = [
            (type(r.model.distribution).__name__, np.asarray(r.params)[-1])
            for r in self.mset.results
        ]
        for h in range(horizon):
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
            # copula correlation state and shock draw
            if self.dynamics == "dcc":
                outer = np.einsum("mi,mj->mij", eta_prev, eta_prev)
                Q = omega[None, :, :] + a * outer + b * Q
                d = np.sqrt(np.einsum("mii->mi", Q))
                Rm = Q / np.einsum("mi,mj->mij", d, d)
            eta_new = np.empty((n_paths, N))
            for m in range(n_paths):
                Rmat = Rm[m] if self.dynamics == "dcc" else self.R[0]
                chol = np.linalg.cholesky(Rmat)
                z = chol @ rng.standard_normal(N)
                if nu is None:
                    eta_new[m] = z
                else:
                    w = rng.chisquare(nu)
                    eta_new[m] = z * np.sqrt((nu - 2.0) / w)
            # copula shocks -> uniforms -> marginal standardized residuals
            if nu is None:
                u_sim = stats.norm.cdf(eta_new)
            else:
                u_sim = _std_t_cdf(eta_new, nu)
            u_sim = np.clip(u_sim, _CLIP, 1.0 - _CLIP)
            eps = np.empty_like(u_sim)
            for i, (dname, dparam) in enumerate(marg_dists):
                if dname == "Normal":
                    eps[:, i] = stats.norm.ppf(u_sim[:, i])
                else:
                    eps[:, i] = _std_t_ppf(u_sim[:, i], float(dparam))
            mu = np.array([self.mset.mean_offset(i) for i in range(N)])
            u_ret = np.sqrt(sig2) * eps
            rets[h] = mu + u_ret
            covs[h] = np.cov(u_ret.T)
            for i in range(N):
                if u_lags[i].shape[0] > 0:
                    u_lags[i] = np.vstack([u_ret[None, :, i], u_lags[i][:-1]])
                if s2_lags[i].shape[0] > 0:
                    s2_lags[i] = np.vstack([sig2[None, :, i], s2_lags[i][:-1]])
            if self.dynamics == "dcc":
                eta_prev = eta_new
        return {"returns": rets, "covariances": covs}

    def _coefs(self) -> tuple[float, float, float | None]:
        p = self.params
        return (
            p.get("alpha", 0.0),
            p.get("beta", 0.0),
            p.get("nu", None) if self.copula == "t" else None,
        )

    # -- filtering ---------------------------------------------------------

    def filter(self, returns) -> CopulaGARCHResult:
        """Apply fitted marginal and copula parameters to new returns."""
        y, names, index = _coerce_returns(returns)
        if y.shape[1] != self.nassets:
            raise ValueError(f"expected {self.nassets} columns, got {y.shape[1]}")
        sigma = np.empty_like(y)
        resid = np.empty_like(y)
        for i in range(self.nassets):
            r, s = self.mset.filter_new_data(i, y[:, i])
            resid[:, i] = r
            sigma[:, i] = s
        eps = resid / sigma
        if self.margins == "parametric":
            u = pit_parametric(self.mset, eps)
        else:
            u = pit_empirical(eps)
        a, b, nu = self._coefs()
        eta = _eta_from_u(u, self.copula, nu)
        if self.dynamics == "dcc":
            path = dcc_path(eta, a, b, 0.0, self.Sbar, None)
            if not path.ok:
                raise RuntimeError("copula correlation recursion failed on new data")
            N = eta.shape[1]
            if nu is None:
                joint = mvnorm_llt(path.logdet, path.quad, N)
                margins = stats.norm.logpdf(eta).sum(axis=1)
            else:
                joint = mvt_llt(path.logdet, path.quad, nu, N)
                margins = _std_t_logpdf(eta, nu).sum(axis=1)
            llt_c = joint - margins
            R, q_last = path.R, path.q_last
        else:
            llt_c = _copula_llt_static(eta, self.R[0], nu)
            R = np.tile(self.R[0][None, :, :], (eta.shape[0], 1, 1))
            q_last = None
        # marginal per-obs lls on the new data (fitted params, new paths)
        ll1 = np.zeros(eta.shape[0])
        for i in range(self.nassets):
            dist_name = type(self.mset.results[i].model.distribution).__name__
            if dist_name == "Normal":
                ll1 += stats.norm.logpdf(eps[:, i]) - np.log(sigma[:, i])
            elif dist_name == "StudentsT":
                nu_i = float(np.asarray(self.mset.results[i].params)[-1])
                ll1 += _std_t_logpdf(eps[:, i], nu_i) - np.log(sigma[:, i])
            else:  # pragma: no cover
                raise NotImplementedError
        llt = ll1 + llt_c
        return CopulaGARCHResult(
            copula=self.copula,
            dynamics=self.dynamics,
            margins=self.margins,
            names=names or self.names,
            index=index,
            mset=self.mset,
            u=u,
            eta=eta,
            R=R,
            psi=self.psi.copy(),
            psi_names=list(self.psi_names),
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            llt_copula=llt_c,
            Sbar=self.Sbar,
            q_last=q_last,
            filtered=True,
        )

    def one_step_correlation(self) -> np.ndarray:
        """Deterministic one-step-ahead copula correlation (DCC dynamics)."""
        if self.dynamics != "dcc":
            return self.R[0]
        a, b, _ = self._coefs()
        q1 = one_step_q(self.q_last, self.eta[-1], a, b, 0.0, self.Sbar, None)
        return cov2cor(q1)

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"<CopulaGARCHResult {self.copula}/{self.dynamics} "
            f"N={self.nassets} T={self.nobs} ll={self.loglikelihood:.2f}>"
        )


# -- model -----------------------------------------------------------------


class CopulaGARCH:
    """Gaussian or Student-t copula over arch marginals.

    copula: "gaussian" or "t".
    dynamics: "dcc" (scalar symmetric DCC on the copula shocks) or "static".
    margins: "parametric" (PIT through the fitted arch distributions; Normal
    and Student-t supported) or "empirical" (rank-based).

    Standard errors: computed for dynamics="dcc" with parametric margins via
    the same two-stage sandwich as DCC; for Student-t margins the PIT's
    dependence on the marginal shape parameter is held fixed (documented
    approximation). Empirical margins are rank-based, hence non-smooth in the
    marginal parameters, so compute_se is rejected there.
    """

    def __init__(
        self,
        copula: str = "gaussian",
        dynamics: str = "dcc",
        margins: str = "parametric",
    ):
        if copula not in ("gaussian", "t"):
            raise ValueError("copula must be 'gaussian' or 't'")
        if dynamics not in ("dcc", "static"):
            raise ValueError("dynamics must be 'dcc' or 'static'")
        if margins not in ("parametric", "empirical"):
            raise ValueError("margins must be 'parametric' or 'empirical'")
        self.copula = copula
        self.dynamics = dynamics
        self.margins = margins

    # psi layouts: dcc-gaussian (a,b); dcc-t (a,b,nu); static-gaussian ();
    # static-t (nu,)

    def fit(self, returns, marginals=None, compute_se: bool = False):
        mset, names, index = _build_marginals(returns, marginals)
        eps = mset.std_resid
        u = (
            pit_parametric(mset, eps)
            if self.margins == "parametric"
            else pit_empirical(eps)
        )
        T, _N = u.shape

        if self.dynamics == "static":
            return self._fit_static(mset, names, index, eps, u)

        studentt = self.copula == "t"

        def negll(x: np.ndarray) -> float:
            a_, b_ = float(x[0]), float(x[1])
            nu_ = float(x[2]) if studentt else None
            if a_ < 0 or b_ < 0 or a_ + b_ >= 1.0 or (studentt and nu_ <= 2.05):
                return _PENALTY
            eta_ = _eta_from_u(u, self.copula, nu_)
            llt = _copula_llt_dynamic(eta_, a_, b_, nu_)
            if llt is None or not np.all(np.isfinite(llt)):
                return _PENALTY
            return -float(np.sum(llt))

        bounds = [(0.0, 0.999), (0.0, 0.999)]
        x0s = [np.array([0.02, 0.95]), np.array([0.05, 0.90])]
        if studentt:
            bounds.append((2.1, 300.0))
            x0s = [np.append(x, 10.0) for x in x0s]
        best = None
        for x0 in x0s:
            r = minimize(
                negll,
                x0,
                method="SLSQP",
                bounds=bounds,
                constraints=[
                    {"type": "ineq", "fun": lambda x: 1.0 - 1e-6 - x[0] - x[1]}
                ],
                options={"maxiter": 500, "ftol": 1e-10},
            )
            if best is None or r.fun < best.fun:
                best = r
        a, b = float(best.x[0]), float(best.x[1])
        nu = float(best.x[2]) if studentt else None
        eta = _eta_from_u(u, self.copula, nu)
        Sbar = cov2cor(eta.T @ eta / T)
        path = dcc_path(eta, a, b, 0.0, Sbar, None)
        llt_c = _copula_llt_dynamic(eta, a, b, nu)
        ll1 = np.zeros(T)
        for i in range(mset.nassets):
            ll1 += mset.per_obs_loglik(i)
        llt = ll1 + llt_c
        psi = np.asarray(best.x, dtype=float)
        psi_names = ["alpha", "beta"] + (["nu"] if studentt else [])
        result = CopulaGARCHResult(
            copula=self.copula,
            dynamics="dcc",
            margins=self.margins,
            names=names,
            index=index,
            mset=mset,
            u=u,
            eta=eta,
            R=path.R,
            psi=psi,
            psi_names=psi_names,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            llt_copula=llt_c,
            Sbar=Sbar,
            q_last=path.q_last,
            converged=bool(best.success),
            message=str(best.message),
        )
        if compute_se:
            if self.margins != "parametric":
                raise ValueError(
                    "compute_se requires parametric margins: the empirical "
                    "PIT is rank-based and not differentiable in the "
                    "marginal parameters"
                )
            result_se = self._sandwich(mset, result)
            result.vcov = result_se["vcov"]
            result.se = result_se["se"]
            result.se_method = result_se["method"]
        return result

    def _fit_static(self, mset, names, index, eps, u):
        T, _N = u.shape
        if self.copula == "gaussian":
            eta = stats.norm.ppf(u)
            R = cov2cor(eta.T @ eta / T)
            nu = None
            psi = np.empty(0)
            psi_names: list[str] = []
        else:
            R = kendall_correlation(eps)

            def neg(nu_arr):
                nu_ = float(nu_arr[0])
                if nu_ <= 2.05:
                    return _PENALTY
                eta_ = _std_t_ppf(u, nu_)
                return -float(np.sum(_copula_llt_static(eta_, R, nu_)))

            r = minimize(
                neg, np.array([10.0]), method="L-BFGS-B", bounds=[(2.1, 300.0)]
            )
            nu = float(r.x[0])
            eta = _std_t_ppf(u, nu)
            psi = np.array([nu])
            psi_names = ["nu"]
        llt_c = _copula_llt_static(eta, R, nu)
        ll1 = np.zeros(T)
        for i in range(mset.nassets):
            ll1 += mset.per_obs_loglik(i)
        llt = ll1 + llt_c
        return CopulaGARCHResult(
            copula=self.copula,
            dynamics="static",
            margins=self.margins,
            names=names,
            index=index,
            mset=mset,
            u=u,
            eta=eta,
            R=np.tile(R[None, :, :], (T, 1, 1)),
            psi=psi,
            psi_names=psi_names,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            llt_copula=llt_c,
            Sbar=R,
        )

    def _sandwich(self, mset: MarginalSet, result: CopulaGARCHResult) -> dict:
        copula = self.copula
        margins_kind = self.margins

        def llt_fn(psi: np.ndarray, eps: np.ndarray):
            a_, b_ = float(psi[0]), float(psi[1])
            nu_ = float(psi[2]) if copula == "t" else None
            if a_ < 0 or b_ < 0 or a_ + b_ >= 1.0:
                return None
            u_ = (
                pit_parametric(mset, eps)
                if margins_kind == "parametric"
                else pit_empirical(eps)
            )
            eta_ = _eta_from_u(u_, copula, nu_)
            return _copula_llt_dynamic(eta_, a_, b_, nu_)

        layout = ParamLayout(asymmetric=False, studentt=copula == "t")
        fitlike = _FitShim(result.psi, layout, result.Sbar)
        return two_stage_vcov(mset, fitlike, llt_fn=llt_fn)


@dataclass
class _FitShim:
    """Duck-typed stand-in for Stage2Fit inside two_stage_vcov."""

    params: np.ndarray
    layout: ParamLayout
    Sbar: np.ndarray

    @property
    def Nbar(self) -> np.ndarray:
        return np.zeros_like(self.Sbar)
