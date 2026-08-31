"""Stage-1 univariate marginals via the arch package.

pymgarch never re-implements univariate estimation: marginals are fit through
arch, or ingested as already-fitted ARCHModelResult objects. What this module
adds is the ability to re-evaluate a marginal's variance path and per-
observation log-likelihood at *perturbed* parameter values, which the
Engle-Sheppard two-stage covariance estimator requires. That re-evaluation
goes through arch's own VolatilityProcess.compute_variance so any volatility
family arch supports is supported here too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate import GARCH
from arch.univariate.base import ARCHModelResult


@dataclass(frozen=True)
class UnivariateSpec:
    """Thin declarative wrapper over arch_model for stage-1 fitting.

    vol="GARCH" with o > 0 is GJR-GARCH in arch's parameterization.
    """

    mean: str = "Constant"
    vol: str = "GARCH"
    p: int = 1
    o: int = 0
    q: int = 1
    power: float = 2.0
    dist: str = "normal"

    def fit(self, y) -> ARCHModelResult:
        # rescale=False keeps parameters comparable with R fits on the same
        # series; pass percent (not decimal) returns for optimizer health.
        model = arch_model(
            y,
            mean=self.mean,
            vol=self.vol,
            p=self.p,
            o=self.o,
            q=self.q,
            power=self.power,
            dist=self.dist,
            rescale=False,
        )
        return model.fit(disp="off", show_warning=False)


def _param_split(res: ARCHModelResult) -> tuple[int, int, int]:
    """(n_mean, n_vol, n_dist) parameter counts for a fitted arch result."""
    model = res.model
    n_mean = int(model.num_params)
    n_vol = int(model.volatility.num_params)
    n_dist = int(model.distribution.num_params)
    return n_mean, n_vol, n_dist


class MarginalSet:
    """A cross-section of fitted univariate marginals with aligned samples."""

    def __init__(self, results: list[ARCHModelResult], names: list[str]):
        if not results:
            raise ValueError("MarginalSet requires at least one marginal")
        nobs = {int(r.resid.shape[0]) for r in results}
        if len(nobs) != 1:
            raise ValueError(
                "all marginals must cover the same sample; got nobs "
                f"{sorted(nobs)}. Unbalanced panels are not supported in v0.1."
            )
        self.results = list(results)
        self.names = list(names)
        self.nobs = nobs.pop()
        self.nassets = len(results)
        resid = np.column_stack([np.asarray(r.resid, dtype=float) for r in results])
        sigma = np.column_stack(
            [np.asarray(r.conditional_volatility, dtype=float) for r in results]
        )
        if np.any(~np.isfinite(resid)) or np.any(~np.isfinite(sigma)):
            raise ValueError(
                "marginals produced non-finite residuals or volatilities; "
                "check for NaNs in the input returns"
            )
        self.resid = resid
        self.sigma = sigma
        self.std_resid = resid / sigma

    # -- construction ------------------------------------------------------

    @classmethod
    def from_returns(cls, returns: pd.DataFrame, spec: UnivariateSpec) -> MarginalSet:
        results = [spec.fit(returns[col]) for col in returns.columns]
        return cls(results, [str(c) for c in returns.columns])

    @classmethod
    def from_results(
        cls, results: list[ARCHModelResult], names: list[str] | None = None
    ) -> MarginalSet:
        for r in results:
            if not isinstance(r, ARCHModelResult):
                raise TypeError(
                    "marginals must be fitted arch ARCHModelResult objects; "
                    f"got {type(r).__name__}"
                )
        if names is None:
            names = [f"y{i}" for i in range(len(results))]
        return cls(list(results), list(names))

    # -- parameters --------------------------------------------------------

    @property
    def loglikelihood(self) -> float:
        return float(sum(r.loglikelihood for r in self.results))

    def n_params(self, i: int) -> int:
        return int(np.asarray(self.results[i].params).shape[0])

    @property
    def total_params(self) -> int:
        return sum(self.n_params(i) for i in range(self.nassets))

    def full_params(self, i: int) -> np.ndarray:
        return np.asarray(self.results[i].params, dtype=float).copy()

    def param_names(self, i: int) -> list[str]:
        return [f"{self.names[i]}.{n}" for n in self.results[i].params.index]

    # -- re-evaluation at perturbed parameters -----------------------------

    def _recompute(self, i: int, params: np.ndarray):
        """(resids, sigma2, per-obs loglik) for marginal i at `params`."""
        res = self.results[i]
        model = res.model
        params = np.asarray(params, dtype=float)
        n_mean, n_vol, _ = _param_split(res)
        mean_p = params[:n_mean]
        vol_p = params[n_mean : n_mean + n_vol]
        dist_p = params[n_mean + n_vol :]
        resids = np.asarray(model.resids(mean_p), dtype=float)
        vol = model.volatility
        sigma2 = np.empty_like(resids)
        # arch computes the backcast and variance bounds once per fit and
        # holds them fixed during optimization; reuse the stored values so
        # perturbed evaluations stay consistent with the objective arch
        # actually maximized (scores at the fitted params are then ~0).
        backcast = getattr(model, "_backcast", None)
        if backcast is None:
            backcast = vol.backcast(resids)
        var_bounds = getattr(model, "_var_bounds", None)
        if var_bounds is None:
            var_bounds = vol.variance_bounds(resids)
        vol.compute_variance(vol_p, resids, sigma2, backcast, var_bounds)
        llt = np.asarray(
            model.distribution.loglikelihood(dist_p, resids, sigma2, individual=True),
            dtype=float,
        )
        return resids, sigma2, llt

    def per_obs_loglik(self, i: int, params: np.ndarray | None = None) -> np.ndarray:
        if params is None:
            params = self.full_params(i)
        _, _, llt = self._recompute(i, params)
        return llt

    def std_resid_at(self, i: int, params: np.ndarray) -> np.ndarray:
        resids, sigma2, _ = self._recompute(i, params)
        return resids / np.sqrt(sigma2)

    # -- forward filtering on new data -------------------------------------

    def filter_new_data(self, i: int, y: np.ndarray):
        """(resids, sigma) on a new series using marginal i's fitted params.

        Only Constant and Zero conditional means are supported: for anything
        else the residual construction depends on arch internals tied to the
        estimation sample.
        """
        res = self.results[i]
        model = res.model
        n_mean, n_vol, _ = _param_split(res)
        params = self.full_params(i)
        y = np.asarray(y, dtype=float)
        if n_mean == 0:
            resids = y.copy()
        elif n_mean == 1 and type(model).__name__ in ("ConstantMean", "HARX"):
            resids = y - params[0]
        else:
            raise NotImplementedError(
                "filter() supports Constant or Zero mean marginals only"
            )
        vol = model.volatility
        vol_p = params[n_mean : n_mean + n_vol]
        sigma2 = np.empty_like(resids)
        # initialize the recursion from the fitted model's stored backcast
        # (part of the fitted state), so filtering the training sample
        # reproduces the fitted path exactly; bounds depend on sample length
        # and are recomputed for the new series.
        backcast = getattr(model, "_backcast", None)
        if backcast is None or np.ndim(backcast) > 0:
            backcast = vol.backcast(resids)
        var_bounds = vol.variance_bounds(resids)
        vol.compute_variance(vol_p, resids, sigma2, backcast, var_bounds)
        return resids, np.sqrt(sigma2)

    # -- simulation support -------------------------------------------------

    def garch_family(self, i: int):
        """(vol_params, p, o, q) if marginal i is squared-power GARCH/GJR, else None."""
        res = self.results[i]
        vol = res.model.volatility
        if not isinstance(vol, GARCH) or getattr(vol, "power", 2.0) != 2.0:
            return None
        n_mean, n_vol, _ = _param_split(res)
        vol_p = self.full_params(i)[n_mean : n_mean + n_vol]
        return vol_p, int(vol.p), int(vol.o), int(vol.q)

    def mean_offset(self, i: int) -> float:
        """Unconditional mean used when simulating returns (Constant/Zero only)."""
        res = self.results[i]
        n_mean, _, _ = _param_split(res)
        if n_mean == 0:
            return 0.0
        if n_mean == 1:
            return float(self.full_params(i)[0])
        raise NotImplementedError(
            "simulation supports Constant or Zero mean marginals only"
        )

    def sim_state(self, i: int):
        """Terminal lag state (u_lags, s2_lags) for forward simulation."""
        fam = self.garch_family(i)
        if fam is None:
            raise NotImplementedError(
                f"marginal {self.names[i]!r} is not GARCH/GJR (power 2); "
                "simulation-based methods support GARCH/GJR marginals only"
            )
        _, p, o, q = fam
        nlag = max(p, o, 1)
        u_lags = self.resid[-nlag:, i][::-1].copy()  # row 0 most recent
        s2_lags = (self.sigma[-max(q, 1):, i] ** 2)[::-1].copy()
        return u_lags, s2_lags
