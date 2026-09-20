import numpy as np
import pandas as pd
import pytest
from conftest import GARCH11, TRUE_DCC

from pymgarch import ADCC, DCC, simulate_dcc
from pymgarch.estimation import composite_pairs


class TestPairSchemes:
    def test_contiguous(self):
        assert composite_pairs(4, "contiguous") == [(0, 1), (1, 2), (2, 3)]

    def test_all(self):
        assert composite_pairs(3, "all") == [(0, 1), (0, 2), (1, 2)]

    def test_rejects_unknown(self):
        with pytest.raises(ValueError):
            composite_pairs(3, "random")


class TestCompositeDCC:
    @pytest.fixture(scope="class")
    def comp_fit(self, dcc_returns):
        return DCC().fit(dcc_returns, method="composite", pairs="all")

    def test_close_to_full_likelihood_estimates(self, comp_fit, dcc_fit):
        assert comp_fit.params["alpha"] == pytest.approx(
            dcc_fit.params["alpha"], abs=0.02
        )
        assert comp_fit.params["beta"] == pytest.approx(
            dcc_fit.params["beta"], abs=0.05
        )

    def test_recovers_truth(self, comp_fit):
        assert comp_fit.params["alpha"] == pytest.approx(TRUE_DCC["a"], abs=0.03)
        assert comp_fit.params["beta"] == pytest.approx(TRUE_DCC["b"], abs=0.08)

    def test_full_ll_at_full_params_is_at_least_composite_params(
        self, comp_fit, dcc_fit
    ):
        # the reported joint ll is the full likelihood in both cases, and the
        # full-ML estimate maximizes it by construction
        assert dcc_fit.loglikelihood >= comp_fit.loglikelihood - 1e-6

    def test_composite_ses_positive(self, comp_fit):
        se = comp_fit.std_errors
        assert se is not None
        assert se["alpha"] > 0 and se["beta"] > 0
        assert comp_fit.extras["estimation_method"] == "composite"

    def test_contiguous_scheme_runs(self, dcc_returns):
        res = DCC().fit(
            dcc_returns, method="composite", pairs="contiguous", compute_se=False
        )
        assert np.isfinite(res.loglikelihood)


class TestCompositeLargerN:
    def test_ten_assets(self):
        # block-structured 10-asset target, composite contiguous pairs
        N = 10
        base = np.full((N, N), 0.3)
        np.fill_diagonal(base, 1.0)
        sim = simulate_dcc(
            [GARCH11] * N,
            a=TRUE_DCC["a"],
            b=TRUE_DCC["b"],
            Sbar=base,
            nobs=1200,
            seed=9,
        )
        df = pd.DataFrame(sim["returns"], columns=[f"a{i}" for i in range(N)])
        res = DCC().fit(df, method="composite", compute_se=False)
        assert res.params["alpha"] == pytest.approx(TRUE_DCC["a"], abs=0.03)
        assert res.params["beta"] == pytest.approx(TRUE_DCC["b"], abs=0.08)
        assert res.conditional_correlations.shape == (1200, N, N)


class TestCompositeADCC:
    def test_adcc_composite_recovers_asymmetry(self, adcc_returns):
        res = ADCC().fit(
            adcc_returns, method="composite", pairs="all", compute_se=False
        )
        assert res.params["gamma"] > 0.01
        assert res.params["beta"] == pytest.approx(0.90, abs=0.12)
