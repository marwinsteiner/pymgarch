import numpy as np
import pandas as pd
import pytest

from pymgarch import GOGARCH, UnivariateSpec
from pymgarch._kernels import garch_forward
from pymgarch.gogarch import _gogarch_llt
from pymgarch.ica import fastica

A_TRUE = np.array(
    [
        [1.0, 0.4, 0.2],
        [-0.3, 1.2, 0.5],
        [0.2, -0.6, 0.9],
    ]
)
# distinct GARCH dynamics per factor -> distinct excess kurtosis, which is
# what makes the ICA rotation identifiable
FACTOR_PARAMS = [
    np.array([0.05, 0.15, 0.80]),
    np.array([0.02, 0.05, 0.93]),
    np.array([0.10, 0.25, 0.65]),
]


@pytest.fixture(scope="module")
def gogarch_sim():
    rng = np.random.default_rng(3)
    T, burn = 3000, 500
    factors = np.empty((T + burn, 3))
    for i, vp in enumerate(FACTOR_PARAMS):
        uncond = vp[0] / (1.0 - vp[1] - vp[2])
        z = rng.standard_normal((T + burn, 1))
        u, _ = garch_forward(
            vp, 1, 0, 1, z, np.zeros((1, 1)), np.full((1, 1), uncond)
        )
        factors[:, i] = u[:, 0]
    returns = factors[burn:] @ A_TRUE.T
    return pd.DataFrame(returns, columns=["X", "Y", "Z"]), factors[burn:]


class TestFastICA:
    def test_recovers_mixing_up_to_sign_and_permutation(self, gogarch_sim):
        df, _ = gogarch_sim
        x = df.to_numpy() - df.to_numpy().mean(axis=0)
        ica = fastica(x, seed=0)
        assert ica.converged
        # U_est @ A_true should be a (scaled) signed permutation matrix:
        # each row dominated by a single entry
        P = ica.U @ A_TRUE
        for row in P:
            share = np.max(np.abs(row)) / np.linalg.norm(row)
            assert share > 0.85
        # exact inverse pair
        assert np.allclose(ica.U @ ica.A, np.eye(3), atol=1e-10)

    def test_deterministic_given_seed(self, gogarch_sim):
        df, _ = gogarch_sim
        x = df.to_numpy() - df.to_numpy().mean(axis=0)
        a = fastica(x, seed=0)
        b = fastica(x, seed=0)
        assert np.allclose(a.A, b.A)

    def test_rejects_singular_input(self):
        x = np.random.default_rng(0).standard_normal((200, 2))
        x = np.column_stack([x, x[:, 0]])  # collinear third column
        with pytest.raises(ValueError, match="singular"):
            fastica(x)


class TestGOGARCH:
    @pytest.fixture(scope="class")
    def fit(self, gogarch_sim):
        df, _ = gogarch_sim
        return GOGARCH().fit(df, seed=0)

    def test_loglik_matches_manual_decomposition(self, fit, gogarch_sim):
        df, _ = gogarch_sim
        factors = (df.to_numpy() - fit.mu) @ fit.U.T
        llt = _gogarch_llt(factors, fit.mset.sigma, fit.A)
        assert np.allclose(fit.llt, llt, atol=1e-8)
        assert fit.loglikelihood == pytest.approx(float(np.sum(llt)))

    def test_covariance_paths_are_psd(self, fit):
        H = fit.conditional_covariances
        assert H.shape == (fit.nobs, 3, 3)
        for t in (0, fit.nobs // 2, fit.nobs - 1):
            assert np.linalg.eigvalsh(H[t])[0] > 0
        R = fit.conditional_correlations
        assert np.allclose(np.einsum("tii->ti", R), 1.0, atol=1e-10)

    def test_unconditional_cov_close_to_sample(self, fit, gogarch_sim):
        df, _ = gogarch_sim
        Hbar = fit.conditional_covariances.mean(axis=0)
        sample = np.cov(df.to_numpy().T)
        assert np.abs(Hbar - sample).max() / np.abs(sample).max() < 0.15

    def test_forecast_analytic_and_simulation_agree_at_h1(self, fit):
        an = fit.forecast(horizon=1)
        sim = fit.forecast(horizon=1, method="simulation", n_paths=50, seed=1)
        # h=1 factor variance is deterministic
        assert np.allclose(sim.factor_variances[0], an.factor_variances[0], rtol=1e-8)
        assert np.allclose(sim.covariances[0], an.covariances[0], rtol=1e-8)

    def test_forecast_matches_garch_closed_form(self, fit):
        # E[s2_{T+h}] = uncond + rho^{h-1} (s2_{T+1} - uncond), rho = alpha+beta
        fc = fit.forecast(horizon=100)
        for i in range(fit.mset.nassets):
            vol_p, _p, _o, _q = fit.mset.garch_family(i)
            rho = float(vol_p[1:].sum())
            uncond = vol_p[0] / (1.0 - rho)
            s2_1 = fc.factor_variances[0, i]
            expected = uncond + rho**99 * (s2_1 - uncond)
            assert fc.factor_variances[-1, i] == pytest.approx(expected, rel=1e-6)

    def test_filter_on_training_data_reproduces_fit(self, fit, gogarch_sim):
        df, _ = gogarch_sim
        flt = fit.filter(df)
        assert flt.filtered
        assert np.allclose(flt.factor_sigma, fit.factor_sigma, rtol=1e-8)
        assert flt.loglikelihood == pytest.approx(fit.loglikelihood, rel=1e-8)

    def test_summary_renders(self, fit):
        text = fit.summary()
        assert "GO-GARCH" in text and "f0" in text and "AIC" in text

    def test_rejects_non_zero_mean_spec(self, gogarch_sim):
        df, _ = gogarch_sim
        with pytest.raises(ValueError, match="Zero"):
            GOGARCH().fit(df, spec=UnivariateSpec(mean="Constant"))
