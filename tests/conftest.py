import numpy as np
import pandas as pd
import pytest

from pymgarch import DCC, simulate_dcc

SBAR = np.array(
    [
        [1.0, 0.5, 0.3],
        [0.5, 1.0, 0.4],
        [0.3, 0.4, 1.0],
    ]
)
GARCH11 = {"omega": 0.05, "alpha": [0.08], "beta": [0.90]}
TRUE_DCC = {"a": 0.04, "b": 0.93}


@pytest.fixture(scope="session")
def dcc_sim():
    return simulate_dcc(
        [GARCH11] * 3,
        a=TRUE_DCC["a"],
        b=TRUE_DCC["b"],
        Sbar=SBAR,
        nobs=1500,
        seed=42,
    )


@pytest.fixture(scope="session")
def dcc_returns(dcc_sim):
    return pd.DataFrame(dcc_sim["returns"], columns=["A", "B", "C"])


@pytest.fixture(scope="session")
def dcc_fit(dcc_returns):
    return DCC(dist="norm").fit(dcc_returns, compute_se=True)


@pytest.fixture(scope="session")
def adcc_sim():
    # Nbar = Sbar/2 has the exact Gaussian diagonal (E[n_i^2] = 1/2) and an
    # off-diagonal close enough for loose recovery tests.
    return simulate_dcc(
        [GARCH11] * 3,
        a=0.03,
        b=0.90,
        g=0.06,
        Sbar=SBAR,
        Nbar=SBAR / 2.0,
        nobs=2000,
        seed=7,
    )


@pytest.fixture(scope="session")
def adcc_returns(adcc_sim):
    return pd.DataFrame(adcc_sim["returns"], columns=["A", "B", "C"])


@pytest.fixture(scope="session")
def t_returns():
    sim = simulate_dcc(
        [GARCH11] * 3,
        a=TRUE_DCC["a"],
        b=TRUE_DCC["b"],
        Sbar=SBAR,
        dist="t",
        nu=8.0,
        nobs=2000,
        seed=11,
    )
    return pd.DataFrame(sim["returns"], columns=["A", "B", "C"])
