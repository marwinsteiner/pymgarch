"""Stage-2 multivariate log-densities and samplers.

Both densities are for the standardized residual vector e_t with correlation
matrix R_t and unit marginal variances. The Student-t is parameterized so that
Cov(e_t) = R_t for nu > 2 (i.e. scale = R * (nu - 2) / nu).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

LOG_2PI = float(np.log(2.0 * np.pi))


def mvnorm_llt(logdet: np.ndarray, quad: np.ndarray, ndim: int) -> np.ndarray:
    """Per-observation multivariate normal log-likelihood of e_t."""
    return -0.5 * (ndim * LOG_2PI + logdet + quad)


def mvt_llt(logdet: np.ndarray, quad: np.ndarray, nu: float, ndim: int) -> np.ndarray:
    """Per-observation covariance-standardized multivariate t log-likelihood."""
    const = (
        gammaln(0.5 * (nu + ndim))
        - gammaln(0.5 * nu)
        - 0.5 * ndim * np.log(np.pi * (nu - 2.0))
    )
    return const - 0.5 * logdet - 0.5 * (nu + ndim) * np.log1p(quad / (nu - 2.0))


def sample_standardized(
    rng: np.random.Generator,
    chol: np.ndarray,
    nobs: int,
    dist: str,
    nu: float | None = None,
) -> np.ndarray:
    """Draw (nobs, N) shocks with correlation chol @ chol.T and unit variances."""
    n = chol.shape[0]
    z = rng.standard_normal((nobs, n)) @ chol.T
    if dist == "norm":
        return z
    if dist == "t":
        if nu is None or nu <= 2.0:
            raise ValueError("Student-t sampling requires nu > 2")
        w = rng.chisquare(nu, size=(nobs, 1))
        return z * np.sqrt((nu - 2.0) / w)
    raise ValueError(f"unknown distribution: {dist!r}")
