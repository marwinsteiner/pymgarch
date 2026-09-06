"""Compare CCC, DCC, and ADCC on the same data by information criteria."""

from pathlib import Path

import pandas as pd

from pymgarch import ADCC, CCC, DCC

returns = pd.read_csv(Path(__file__).parent / "data" / "dji30ret_5.csv")

ccc = CCC().fit(returns)
dcc = DCC().fit(returns, compute_se=False)
adcc = ADCC().fit(returns, compute_se=False)

print(f"{'model':<6}{'loglik':>14}{'AIC':>12}{'BIC':>12}")
for res in (ccc, dcc, adcc):
    print(f"{res.model:<6}{res.loglikelihood:>14.2f}{res.aic:>12.2f}{res.bic:>12.2f}")

g = adcc.params["gamma"]
print(f"\nADCC asymmetry gamma = {g:.4f} "
      f"({'joint negative shocks raise correlations' if g > 0 else 'no asymmetry'})")

# ADCC has no analytic multi-step correlation forecast; simulate instead
fc = adcc.forecast(horizon=5, method="simulation", n_paths=2000, seed=0)
print(f"simulated h=5 corr(AA, AXP): {fc.correlations[4][0, 1]:.4f}")
