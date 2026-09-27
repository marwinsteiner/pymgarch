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
        assert np.allclose(fc.covariances[0], expected, atol=1e-12)

    def test_forecast_is_family_shaped(self, scalar_fit):
        # BEKK forecasts return the same dataclass as DCC/GO-GARCH, so
        # generic consumers (risk metrics, rolling) work unchanged
        fc = scalar_fit.forecast(horizon=3)
        assert fc.covariances.shape == (3, 3, 3)
        assert fc.correlations.shape == (3, 3, 3)
        assert fc.variances.shape == (3, 3)
        assert fc.method == "analytic"
        with pytest.raises(ValueError, match="horizon"):
            scalar_fit.forecast(horizon=0)

    def test_filtered_result_forecasts_from_new_state(self, scalar_fit, bekk_returns):
        flt = scalar_fit.filter(bekk_returns.iloc[:800])
        fc_flt = flt.forecast(horizon=1)
        fc_fit = scalar_fit.forecast(horizon=1)
        assert np.all(np.isfinite(fc_flt.covariances))
        # different terminal states -> different one-step forecasts
        assert not np.allclose(fc_flt.covariances[0], fc_fit.covariances[0])

    def test_long_horizon_converges_to_target(self, scalar_fit):
        fc = scalar_fit.forecast(horizon=400)
        # E[H] -> C / (1 - a^2 - b^2) elementwise = Sigma under targeting
        assert np.allclose(fc.covariances[-1], scalar_fit.Sigma, rtol=0.02)

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


def test_simulate_validates_shapes_and_burn():
    with pytest.raises(ValueError, match="length-3"):
        simulate_bekk(np.array([0.2, 0.3]), 0.9, SBAR, nobs=10)
    with pytest.raises(ValueError, match="burn"):
        simulate_bekk(0.2, 0.9, SBAR, nobs=10, burn=-5)


class TestRobustness:
    """The review's confirmed crash paths must degrade, never kill a fit."""

    def test_no_arch_data_does_not_crash_ses(self):
        # iid noise can pin a at the 0 bound -> singular/near-singular bread
        # matrix; the contract is that the fit survives (pinv fallback or
        # NaN SEs with a warning), never crashes
        import warnings as _w

        rng = np.random.default_rng(0)
        df = pd.DataFrame(rng.standard_normal((600, 3)), columns=["A", "B", "C"])
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            res = BEKK("scalar").fit(df)
        assert np.isfinite(res.loglikelihood)
        assert res.se_method.startswith("qml-robust")

    def test_boundary_optimum_does_not_crash_ses(self):
        # near-integrated data can park the optimum on a^2+b^2 = 1-1e-6;
        # qml_vcov must degrade to NaN SEs, not raise out of fit()
        from pymgarch.bekk import _bekk_eval
        from pymgarch.inference import qml_vcov

        rng = np.random.default_rng(3)
        u = rng.standard_normal((400, 2))
        Sigma = np.cov(u.T)
        a = float(np.sqrt(1.0 - 1e-6 - 0.95**2))

        def llt_fn(x):
            return _bekk_eval(
                u, np.full(2, x[0]), np.full(2, x[1]), Sigma, check_psd=False
            )

        with pytest.warns(RuntimeWarning, match="boundary"):
            vc = qml_vcov(llt_fn, np.array([a, 0.95]))
        assert vc["method"] == "qml-robust (failed)"
        assert np.all(np.isnan(vc["se"]))

    def test_diagonal_ses_computed(self, bekk_returns):
        res = BEKK("diagonal").fit(bekk_returns)
        assert res.se is not None
        assert res.se_method.startswith("qml-robust")

    def test_small_sample_guard(self):
        rng = np.random.default_rng(1)
        df = pd.DataFrame(rng.standard_normal((3, 5)))
        with pytest.raises(ValueError, match="more observations than assets"):
            BEKK("scalar").fit(df)

    def test_filter_empty_rows_guard(self, scalar_fit, bekk_returns):
        with pytest.raises(ValueError, match="at least one observation"):
            scalar_fit.filter(bekk_returns.iloc[:0])


class TestMixedSignDiagonal:
    def test_recovers_mixed_sign_loadings(self):
        # negative cross news-impact is a valid diagonal BEKK; the fit must
        # be able to represent it (a-bounds are two-sided since the review)
        avec = np.array([0.30, -0.30, 0.30])
        bvec = np.array([0.90, 0.90, 0.90])
        Sigma = 0.25 * np.ones((3, 3)) + 0.75 * np.eye(3)
        sim = simulate_bekk(avec, bvec, Sigma, nobs=3000, seed=1)
        df = pd.DataFrame(sim["returns"], columns=["A", "B", "C"])
        res = BEKK("diagonal").fit(df, compute_se=False)
        fitted_a = np.array([res.params[f"a.{c}"] for c in df.columns])
        # identified up to a global sign flip; normalization pins the
        # largest-|a| element positive
        signs = np.sign(fitted_a) * np.sign(avec)
        assert np.allclose(np.abs(fitted_a), np.abs(avec), atol=0.08)
        assert len(set(signs[np.abs(avec) > 0.05])) == 1  # consistent pattern
