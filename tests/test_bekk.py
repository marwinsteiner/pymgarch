import numpy as np
import pandas as pd
import pytest
from conftest import SBAR

from pymgarch import BEKK, simulate_bekk

A_TRUE, B_TRUE = 0.30, 0.92


@pytest.fixture(scope="module")
def bekk_returns():
    # SBAR doubles as the unconditional covariance target (unit variances)
    sim = simulate_bekk(A_TRUE, B_TRUE, SBAR, nobs=2000, seed=21)
    return pd.DataFrame(sim["returns"], columns=["A", "B", "C"])


@pytest.fixture(scope="module")
def scalar_fit(bekk_returns):
    return BEKK("scalar").fit(bekk_returns)


class TestScalarBEKK:
    def test_recovers_truth(self, scalar_fit):
        assert scalar_fit.params["a"] == pytest.approx(A_TRUE, abs=0.05)
        assert scalar_fit.params["b"] == pytest.approx(B_TRUE, abs=0.05)

    def test_ses_positive_and_plausible(self, scalar_fit):
        se = scalar_fit.std_errors
        assert se is not None
        assert 1e-4 < se["a"] < 0.1
        assert 1e-4 < se["b"] < 0.1
        assert scalar_fit.se_method == "qml-robust"

    def test_true_params_within_band(self, scalar_fit):
        se = scalar_fit.std_errors
        assert abs(scalar_fit.params["a"] - A_TRUE) < 5 * se["a"] + 0.01
        assert abs(scalar_fit.params["b"] - B_TRUE) < 5 * se["b"] + 0.01

    def test_covariance_paths_psd(self, scalar_fit):
        H = scalar_fit.conditional_covariances
        for t in (0, scalar_fit.nobs // 2, scalar_fit.nobs - 1):
            assert np.linalg.eigvalsh(H[t])[0] > 0
        R = scalar_fit.conditional_correlations
        assert np.allclose(np.einsum("tii->ti", R), 1.0, atol=1e-10)

    def test_summary_renders(self, scalar_fit):
        text = scalar_fit.summary()
        assert "BEKK" in text and "variance targeting" in text


class TestDiagonalBEKK:
    def test_nests_scalar_on_scalar_data(self, bekk_returns, scalar_fit):
        diag = BEKK("diagonal").fit(bekk_returns, compute_se=False)
        # per-asset coefficients should cluster near the common truth
        avals = [v for k, v in diag.params.items() if k.startswith("a.")]
        bvals = [v for k, v in diag.params.items() if k.startswith("b.")]
        assert np.allclose(avals, A_TRUE, atol=0.08)
        assert np.allclose(bvals, B_TRUE, atol=0.08)
        # and the richer model cannot fit worse than its restriction
        assert diag.loglikelihood >= scalar_fit.loglikelihood - 1e-6

    def test_recovers_heterogeneous_dynamics(self):
        avec = np.array([0.2, 0.35, 0.28])
        bvec = np.array([0.95, 0.88, 0.9])
        sim = simulate_bekk(avec, bvec, SBAR, nobs=3000, seed=33)
        df = pd.DataFrame(sim["returns"], columns=["A", "B", "C"])
        res = BEKK("diagonal").fit(df, compute_se=False)
        fitted_a = np.array([res.params[f"a.{c}"] for c in df.columns])
        fitted_b = np.array([res.params[f"b.{c}"] for c in df.columns])
        assert np.allclose(fitted_a, avec, atol=0.08)
        assert np.allclose(fitted_b, bvec, atol=0.08)


class TestForecastAndFilter:
    def test_one_step_matches_recursion(self, scalar_fit):
        fc = scalar_fit.forecast(horizon=1)
        a, b = scalar_fit.params["a"], scalar_fit.params["b"]
        u_last = scalar_fit._u_last
        C = scalar_fit.Sigma * (1.0 - a**2 - b**2)
        expected = C + a**2 * np.outer(u_last, u_last) + b**2 * scalar_fit.H[-1]
        assert np.allclose(fc["covariances"][0], expected, atol=1e-12)

    def test_long_horizon_converges_to_target(self, scalar_fit):
        fc = scalar_fit.forecast(horizon=400)
        # E[H] -> C / (1 - a^2 - b^2) elementwise = Sigma under targeting
        assert np.allclose(fc["covariances"][-1], scalar_fit.Sigma, rtol=0.02)

    def test_filter_on_training_data_reproduces_fit(self, scalar_fit, bekk_returns):
        flt = scalar_fit.filter(bekk_returns)
        assert flt.filtered
        assert np.allclose(flt.H, scalar_fit.H, rtol=1e-10)
        assert flt.loglikelihood == pytest.approx(
            scalar_fit.loglikelihood, rel=1e-10
        )


def test_simulate_rejects_nonstationary():
    with pytest.raises(ValueError, match="PSD"):
        simulate_bekk(0.8, 0.7, SBAR, nobs=10)
