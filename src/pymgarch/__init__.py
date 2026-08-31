"""pymgarch: multivariate GARCH for Python on top of arch marginals."""

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - before first build
    __version__ = "0.0.0"

from .marginals import MarginalSet, UnivariateSpec
from .models import ADCC, CCC, DCC, simulate_dcc
from .results import MGARCHForecast, MGARCHResult

__all__ = [
    "ADCC",
    "CCC",
    "DCC",
    "MGARCHForecast",
    "MGARCHResult",
    "MarginalSet",
    "UnivariateSpec",
    "__version__",
    "simulate_dcc",
]
