"""Replication tests against rmgarch fixtures.

These run only when tests/fixtures/rmgarch_fixtures.json exists; generate it
with `Rscript scripts/make_fixtures.R` (requires R with rmgarch installed).

Level 1 evaluates OUR likelihood at R's fitted parameters using R's own
sigma/residual paths and correlation targets, which isolates the correlation
math from optimizer and univariate-initialization differences. Level 2 runs
the full pipeline on the same CSV and compares loosely.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "rmgarch_fixtures.json"
CSV = Path(__file__).parent / "fixtures" / "dji30ret_5.csv"

pytestmark = pytest.mark.skipif(
    not FIXTURE.exists(), reason="run scripts/make_fixtures.R to enable"
)


@pytest.fixture(scope="module")
def fx():
    with open(FIXTURE) as fh:
        return json.load(fh)


def _eval_at_r_params(entry, asymmetric, studentt):
    from pymgarch.correlation import cov2cor, dcc_path, one_step_q
    from pymgarch.estimation import ParamLayout

    sigma = np.asarray(entry["sigma"], dtype=float)
    resid = np.asarray(entry["resid"], dtype=float)
    eps = resid / sigma
    Sbar = np.asarray(entry["Qbar"], dtype=float)
    Nbar = np.asarray(entry["Nbar"], dtype=float) if entry.get("Nbar") else None
    s2 = entry["stage2"]
    a, b = float(s2["dcca1"]), float(s2["dccb1"])
    g = float(s2.get("dccg1", 0.0)) if asymmetric else 0.0
    nu = float(s2["mshape"]) if studentt else None
    layout = ParamLayout(asymmetric=asymmetric, studentt=studentt)
    psi = layout.pack(a, b, g=g, nu=nu)

    from pymgarch.estimation import stage2_llt

    llt2 = stage2_llt(psi, eps, Sbar, Nbar if Nbar is not None else Sbar * 0, layout)
    assert llt2 is not None
    joint = float(np.sum(llt2)) - float(np.sum(np.log(sigma)))
    path = dcc_path(eps, a, b, g, Sbar, Nbar)
    q1 = one_step_q(path.q_last, eps[-1], a, b, g, Sbar, Nbar)
    return joint, path, cov2cor(q1)


class TestLevel1LikelihoodAtRParams:
    # Tolerance covers convention differences in Q_1 initialization only;
    # tighten after first successful fixture run if the gap is smaller.
    LL_ATOL = 1.0

    @pytest.mark.parametrize(
        "key,asym,st",
        [("dcc_norm", False, False), ("dcc_t", False, True), ("adcc_norm", True, False)],
    )
    def test_joint_loglik_matches(self, fx, key, asym, st):
        joint, _, _ = _eval_at_r_params(fx[key], asym, st)
        assert joint == pytest.approx(float(fx[key]["loglik"]), abs=self.LL_ATOL)

    @pytest.mark.parametrize(
        "key,asym,st",
        [("dcc_norm", False, False), ("adcc_norm", True, False)],
    )
    def test_terminal_correlation_matches(self, fx, key, asym, st):
        _, path, r1 = _eval_at_r_params(fx[key], asym, st)
        assert np.allclose(
            path.R[-1], np.asarray(fx[key]["Rlast"], dtype=float), atol=1e-3
        )
        assert np.allclose(
            r1, np.asarray(fx[key]["R1_forecast"], dtype=float), atol=1e-3
        )


@pytest.mark.slow
class TestLevel2FullPipeline:
    def test_dcc_norm_params_close(self, fx):
        from pymgarch import DCC

        returns = pd.read_csv(CSV)
        res = DCC().fit(returns, compute_se=False)
        s2 = fx["dcc_norm"]["stage2"]
        assert res.params["alpha"] == pytest.approx(float(s2["dcca1"]), abs=0.02)
        assert res.params["beta"] == pytest.approx(float(s2["dccb1"]), abs=0.05)
        # our optimum should not be materially worse than R's
        assert res.loglikelihood > float(fx["dcc_norm"]["loglik"]) - 5.0
