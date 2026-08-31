"""User-facing model classes: DCC, ADCC, CCC, and a simulation DGP."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .correlation import correlation_targets, cov2cor, dcc_path
from .distributions import sample_standardized
from .estimation import ParamLayout, fit_stage2
from .inference import two_stage_vcov
from .marginals import MarginalSet, UnivariateSpec
from .results import MGARCHResult, _coerce_returns, _constant_corr_eval


def _normalize_dist(dist: str) -> str:
    d = dist.lower()
    if d in ("norm", "normal", "gaussian", "mvnorm"):
        return "norm"
    if d in ("t", "studentst", "student-t", "mvt"):
        return "t"
    raise ValueError(f"unknown distribution {dist!r}; use 'norm' or 't'")


def _build_marginals(returns, marginals) -> tuple[MarginalSet, list[str], pd.Index]:
    y, names, index = _coerce_returns(returns)
    if names is None:
        names = [f"y{i}" for i in range(y.shape[1])]
    frame = (
        returns
        if isinstance(returns, pd.DataFrame)
        else pd.DataFrame(y, columns=names, index=index)
    )
    if marginals is None:
        marginals = UnivariateSpec()
    if isinstance(marginals, UnivariateSpec):
        mset = MarginalSet.from_returns(frame, marginals)
    elif isinstance(marginals, (list, tuple)):
        mset = MarginalSet.from_results(list(marginals), names)
        if mset.nobs != y.shape[0]:
            raise ValueError(
                "fitted marginals do not match the return sample length "
                f"({mset.nobs} vs {y.shape[0]})"
            )
    else:
        raise TypeError(
            "marginals must be a UnivariateSpec, a list of fitted arch "
            f"results, or None; got {type(marginals).__name__}"
        )
    return mset, names, index


class _DCCBase:
    _asymmetric = False
    _model_name = "DCC"

    def __init__(self, dist: str = "norm"):
        self.dist = _normalize_dist(dist)

    def fit(self, returns, marginals=None, compute_se: bool = True) -> MGARCHResult:
        """Two-stage estimation: arch marginals, then correlation dynamics.

        returns : (T, N) DataFrame or ndarray of returns (percent scale
            recommended for optimizer health, matching arch's guidance).
        marginals : None (default GARCH(1,1)), a UnivariateSpec applied to
            every column, or a list of fitted arch ARCHModelResult objects.
        compute_se : compute Engle-Sheppard two-stage standard errors.
        """
        mset, names, index = _build_marginals(returns, marginals)
        eps = mset.std_resid
        layout = ParamLayout(asymmetric=self._asymmetric, studentt=self.dist == "t")
        fit2 = fit_stage2(eps, layout)
        a, b, g, _nu = layout.unpack(fit2.params)
        path = dcc_path(eps, a, b, g, fit2.Sbar, fit2.Nbar)
        if not path.ok:
            raise RuntimeError("correlation recursion failed at the optimum")
        llt = fit2.llt - np.sum(np.log(mset.sigma), axis=1)
        result = MGARCHResult(
            model=self._model_name,
            dist=self.dist,
            names=names,
            index=index,
            sigma=mset.sigma,
            eps=eps,
            R=path.R,
            logdet=path.logdet,
            quad=path.quad,
            Sbar=fit2.Sbar,
            Nbar=fit2.Nbar,
            psi=fit2.params,
            psi_names=layout.names,
            layout=layout,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            q_last=path.q_last,
            mset=mset,
            converged=fit2.converged,
            message=fit2.message,
        )
        if compute_se:
            vc = two_stage_vcov(mset, fit2)
            result.vcov = vc["vcov"]
            result.se = vc["se"]
            result.se_method = vc["method"]
        return result


class DCC(_DCCBase):
    """Engle (2002) scalar DCC(1,1) with correlation targeting."""

    _asymmetric = False
    _model_name = "DCC"


class ADCC(_DCCBase):
    """Cappiello-Engle-Sheppard (2006) scalar asymmetric DCC."""

    _asymmetric = True
    _model_name = "ADCC"


class CCC:
    """Bollerslev (1990) constant conditional correlation (Gaussian)."""

    def __init__(self, dist: str = "norm"):
        if _normalize_dist(dist) != "norm":
            raise NotImplementedError("CCC supports the Gaussian likelihood only")
        self.dist = "norm"

    def fit(self, returns, marginals=None) -> MGARCHResult:
        mset, names, index = _build_marginals(returns, marginals)
        eps = mset.std_resid
        Sbar, Nbar = correlation_targets(eps)
        llt2, logdet, quad, R, q_last = _constant_corr_eval(eps, Sbar, "norm", None)
        llt = llt2 - np.sum(np.log(mset.sigma), axis=1)
        return MGARCHResult(
            model="CCC",
            dist="norm",
            names=names,
            index=index,
            sigma=mset.sigma,
            eps=eps,
            R=R,
            logdet=logdet,
            quad=quad,
            Sbar=Sbar,
            Nbar=Nbar,
            psi=np.empty(0),
            psi_names=[],
            layout=None,
            loglikelihood=float(np.sum(llt)),
            llt=llt,
            q_last=q_last,
            mset=mset,
        )


def simulate_dcc(
    vol_params: list[dict],
    a: float,
    b: float,
    Sbar: np.ndarray,
    g: float = 0.0,
    Nbar: np.ndarray | None = None,
    mu: np.ndarray | None = None,
    dist: str = "norm",
    nu: float | None = None,
    nobs: int = 1000,
    burn: int = 500,
    seed: int | None = None,
) -> dict:
    """Simulate returns from a scalar (A)DCC with GARCH/GJR marginals.

    vol_params: one dict per asset with keys "omega" (float) and "alpha",
    "gamma", "beta" (sequences; "gamma" optional). The correlation intercept
    is (1-a-b)*Sbar - g*Nbar and must be PSD.
    """
    dist = _normalize_dist(dist)
    Sbar = np.asarray(Sbar, dtype=float)
    N = Sbar.shape[0]
    if len(vol_params) != N:
        raise ValueError("vol_params length must match Sbar dimension")
    if g > 0.0:
        if Nbar is None:
            raise ValueError("asymmetric simulation requires Nbar")
        Nbar = np.asarray(Nbar, dtype=float)
    omega_mat = (1.0 - a - b) * Sbar - (g * Nbar if g > 0.0 else 0.0)
    if np.linalg.eigvalsh(omega_mat)[0] < -1e-12:
        raise ValueError("correlation intercept (1-a-b)Sbar - g*Nbar is not PSD")
    mu = np.zeros(N) if mu is None else np.asarray(mu, dtype=float)
    rng = np.random.default_rng(seed)

    parsed = []
    for vp in vol_params:
        alpha = np.atleast_1d(np.asarray(vp.get("alpha", []), dtype=float))
        gamma = np.atleast_1d(np.asarray(vp.get("gamma", []), dtype=float))
        beta = np.atleast_1d(np.asarray(vp.get("beta", []), dtype=float))
        omega = float(vp["omega"])
        persistence = alpha.sum() + beta.sum() + 0.5 * gamma.sum()
        if persistence >= 1.0:
            raise ValueError("marginal persistence must be < 1 for simulation")
        parsed.append((omega, alpha, gamma, beta, omega / (1.0 - persistence)))

    total = nobs + burn
    ret = np.empty((total, N))
    sig = np.empty((total, N))
    epss = np.empty((total, N))
    Rs = np.empty((total, N, N))

    max_lag = max(max(len(p[1]), len(p[2]), 1) for p in parsed)
    max_q = max(max(len(p[3]), 1) for p in parsed)
    u_lags = np.zeros((max_lag, N))
    s2_lags = np.tile(np.array([p[4] for p in parsed]), (max_q, 1))
    Q = Sbar.copy()
    eps_prev = np.zeros(N)

    for t in range(total):
        s2 = np.empty(N)
        for i, (omega, alpha, gamma, beta, _) in enumerate(parsed):
            v = omega
            for lag, al in enumerate(alpha):
                v += al * u_lags[lag, i] ** 2
            for lag, gm in enumerate(gamma):
                if u_lags[lag, i] < 0.0:
                    v += gm * u_lags[lag, i] ** 2
            for lag, be in enumerate(beta):
                v += be * s2_lags[lag, i]
            s2[i] = v
        if t > 0:
            Q = omega_mat + a * np.outer(eps_prev, eps_prev) + b * Q
            if g > 0.0:
                n = np.minimum(eps_prev, 0.0)
                Q = Q + g * np.outer(n, n)
        R = cov2cor(Q)
        chol = np.linalg.cholesky(R)
        z = sample_standardized(rng, chol, 1, dist, nu)[0]
        u = np.sqrt(s2) * z
        ret[t] = mu + u
        sig[t] = np.sqrt(s2)
        epss[t] = z
        Rs[t] = R
        eps_prev = z
        u_lags = np.vstack([u[None, :], u_lags[:-1]])
        s2_lags = np.vstack([s2[None, :], s2_lags[:-1]])

    return {
        "returns": ret[burn:],
        "sigma": sig[burn:],
        "eps": epss[burn:],
        "R": Rs[burn:],
    }
