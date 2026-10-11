import numpy as np
import pytest

from pymgarch import (
    BEKK,
    CCC,
    DCC,
    GOGARCH,
    CopulaGARCH,
    expected_shortfall,
    roll,
    value_at_risk,
    var_coverage,
)

W = np.array([0.5, 0.3, 0.2])


class TestRoll:
    @pytest.fixture(scope="class")
    def rolled(self, dcc_returns):
        return roll(DCC(), dcc_returns, window=1000, refit_every=100)

    def test_shapes_and_bookkeeping(self, rolled, dcc_returns):
        T_oos = len(dcc_returns) - 1000
        assert rolled.nobs == T_oos
        assert rolled.covariances.shape == (T_oos, 3, 3)
        assert rolled.realized.shape == (T_oos, 3)
        assert len(rolled.params) == len(rolled.refit_points)
        assert all("alpha" in p for p in rolled.params)

    def test_forecasts_are_psd(self, rolled):
        for t in (0, rolled.nobs // 2, rolled.nobs - 1):
            assert np.linalg.eigvalsh(rolled.covariances[t])[0] > 0
        R = rolled.correlations
        assert np.allclose(np.einsum("tii->ti", R), 1.0, atol=1e-10)

    def test_single_refit_equals_filter_path(self, dcc_returns):
        # refit_every >= T-window -> one fit; OOS forecasts must equal the
        # filtered conditional covariances of that single fit
        window = 1200
        rolled = roll(DCC(), dcc_returns, window=window, refit_every=10_000)
        fitted = DCC().fit(dcc_returns.iloc[:window], compute_se=False)
        flt = fitted.filter(dcc_returns)
        expected = flt.conditional_covariances[window:]
        assert np.allclose(rolled.covariances, expected, rtol=1e-10)

    def test_expanding_mode(self, dcc_returns):
        rolled = roll(DCC(), dcc_returns, window=1200, refit_every=200, expanding=True)
        assert rolled.expanding
        assert rolled.nobs == len(dcc_returns) - 1200

    def test_other_models_roll(self, dcc_returns):
        for model in (CCC(), GOGARCH(), BEKK(), CopulaGARCH()):
            rolled = roll(model, dcc_returns, window=1300, refit_every=200)
            assert rolled.covariances.shape[0] == len(dcc_returns) - 1300
            assert np.all(np.isfinite(rolled.covariances))

    def test_parallel_matches_sequential(self, dcc_returns, rolled):
        par = roll(DCC(), dcc_returns, window=1000, refit_every=100, n_jobs=2)
        assert np.allclose(par.covariances, rolled.covariances, rtol=1e-10)

    def test_window_validation(self, dcc_returns):
        with pytest.raises(ValueError, match="window"):
            roll(DCC(), dcc_returns, window=len(dcc_returns))

    def test_coverage_workflow(self, rolled):
        out = rolled.coverage_test(W, alpha=0.05)
        assert 0 <= out["rate"] <= 1
        # a correctly specified model on its own DGP should pass coverage
        assert out["kupiec_pvalue"] > 0.01


class TestVaR:
    def test_analytic_vs_simulation_gaussian(self, dcc_fit):
        va = value_at_risk(dcc_fit, W, alpha=0.05, horizon=3, method="analytic")
        vs = value_at_risk(
            dcc_fit, W, alpha=0.05, horizon=3, method="simulation",
            n_paths=40000, seed=0,
        )
        assert va.shape == vs.shape == (3,)
        assert np.allclose(va, vs, rtol=0.08)

    def test_es_exceeds_var(self, dcc_fit):
        for method in ("analytic", "simulation"):
            v = value_at_risk(dcc_fit, W, method=method, n_paths=5000, seed=1)
            e = expected_shortfall(dcc_fit, W, method=method, n_paths=5000, seed=1)
            assert np.all(e > v)

    def test_t_model_var_exceeds_gaussian_far_tail(self, t_returns):
        rt = DCC(dist="t").fit(t_returns, compute_se=False)
        rn = DCC().fit(t_returns, compute_se=False)
        vt = value_at_risk(rt, W, alpha=0.01, method="analytic")
        vn = value_at_risk(rn, W, alpha=0.01, method="analytic")
        assert vt[0] > vn[0]

    def test_analytic_es_gaussian_formula(self, dcc_fit):
        from scipy import stats as sps

        fc = dcc_fit.forecast(horizon=1)
        sig = float(np.sqrt(W @ fc.covariances[0] @ W))
        expected = sig * sps.norm.pdf(sps.norm.ppf(0.05)) / 0.05
        got = expected_shortfall(dcc_fit, W, alpha=0.05, method="analytic")
        assert got[0] == pytest.approx(expected, rel=1e-10)

    def test_alpha_validation(self, dcc_fit):
        with pytest.raises(ValueError, match="alpha"):
            value_at_risk(dcc_fit, W, alpha=0.7)


class TestCoverage:
    def test_correct_rate_passes(self):
        rng = np.random.default_rng(0)
        losses = rng.standard_normal(2000)
        var = np.full(2000, 1.6449)  # 5% Gaussian VaR
        out = var_coverage(losses, var, alpha=0.05)
        assert out["kupiec_pvalue"] > 0.05
        assert out["conditional_pvalue"] > 0.05

    def test_wrong_rate_fails(self):
        rng = np.random.default_rng(1)
        losses = rng.standard_normal(2000)
        var = np.full(2000, 1.0)  # ~16% violations vs 5% claimed
        out = var_coverage(losses, var, alpha=0.05)
        assert out["kupiec_pvalue"] < 1e-6

    def test_clustered_violations_fail_independence(self):
        losses = np.zeros(1000)
        var = np.ones(1000)
        losses[100:150] = 2.0  # one long violation cluster
        out = var_coverage(losses, var, alpha=0.05)
        assert out["independence_pvalue"] < 0.01

    def test_length_mismatch(self):
        with pytest.raises(ValueError, match="equal length"):
            var_coverage(np.zeros(5), np.zeros(4))
