"""pymgarch: multivariate GARCH for Python on top of arch marginals."""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - before first build
    __version__ = "0.0.0"

from .bekk import BEKK, BEKKResult, simulate_bekk
from .copula import CopulaGARCH, CopulaGARCHResult
from .diagnostics import DCCTestResult, dcc_test
from .gogarch import GOGARCH, GOGARCHForecast, GOGARCHResult
from .marginals import MarginalSet, UnivariateSpec
from .models import ADCC, CCC, DCC, simulate_dcc
from .results import MGARCHForecast, MGARCHResult
from .risk import expected_shortfall, value_at_risk, var_coverage
from .rolling import RollResult, roll

__all__ = [
    "ADCC",
    "BEKK",
    "CCC",
    "DCC",
    "GOGARCH",
    "BEKKResult",
    "CopulaGARCH",
    "CopulaGARCHResult",
    "DCCTestResult",
    "GOGARCHForecast",
    "GOGARCHResult",
    "MGARCHForecast",
    "MGARCHResult",
    "MarginalSet",
    "RollResult",
    "UnivariateSpec",
    "__version__",
    "dcc_test",
    "expected_shortfall",
    "roll",
    "simulate_bekk",
    "simulate_dcc",
    "value_at_risk",
    "var_coverage",
]
