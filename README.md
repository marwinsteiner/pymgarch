# pymgarch

Multivariate GARCH for Python: DCC, ADCC, and CCC correlation dynamics built
on top of [arch](https://github.com/bashtage/arch) univariate marginals,
validated against R's rmgarch/tsmarch.

## Why

Python has no maintained general-purpose multivariate GARCH framework. The
existing packages cover Gaussian DCC(1,1) at most, while R users have had
DCC, ADCC, GO-GARCH and copula-GARCH in rmgarch (now tsmarch) for a decade.
pymgarch closes that gap incrementally, starting with the correlation layer:

- stage 1 (univariate volatility) is delegated to `arch`, the ecosystem's
  dominant, battle-tested GARCH package;
- stage 2 (correlation dynamics) is what this library implements, with
  correct two-stage Engle-Sheppard standard errors and replication tests
  against rmgarch's fitted parameters and likelihoods.

## Install

```
pip install pymgarch          # or: pip install pymgarch[numba]
```

The optional `numba` extra JIT-compiles the correlation recursions; without
it everything runs in pure NumPy.

## Quickstart

```python
import pymgarch as mg

# returns: (T, N) DataFrame, percent scale recommended
res = mg.DCC(dist="t").fit(returns)
print(res.summary())

res.conditional_correlations   # (T, N, N)
res.conditional_covariances    # (T, N, N)

fc = res.forecast(horizon=10)                    # analytic
fc = res.forecast(horizon=10, method="simulation", n_paths=2000)
flt = res.filter(new_returns)                    # fixed params, new data
```

Marginals default to constant-mean GARCH(1,1). Customize per-column via a
spec, or bring your own fitted arch results:

```python
spec = mg.UnivariateSpec(vol="GARCH", p=1, o=1, q=1, dist="t")  # GJR-t
res = mg.ADCC().fit(returns, marginals=spec)

from arch import arch_model
fitted = [arch_model(returns[c], rescale=False).fit(disp="off") for c in returns]
res = mg.DCC().fit(returns, marginals=fitted)
```

## Models (v0.1)

| Model | Distribution | Estimation |
| ----- | ------------ | ---------- |
| CCC (Bollerslev 1990) | Gaussian | closed form given marginals |
| DCC(1,1) (Engle 2002) | Gaussian, Student-t | two-stage QML, correlation targeting |
| ADCC (Cappiello-Engle-Sheppard 2006) | Gaussian, Student-t | two-stage QML, PSD-constrained targeting |

Standard errors are Engle-Sheppard (2001) two-stage sandwich estimates: the
marginal and correlation scores are stacked so stage-2 uncertainty reflects
stage-1 estimation error. Correlation targets are held fixed (same
approximation rmgarch makes). If the stacked system is singular the library
falls back to a stage-2-only sandwich and says so in `summary()`.

## Validation

`scripts/make_fixtures.R` fits rmgarch on a pinned dataset (percent returns
of five dji30ret constituents) and exports parameters, likelihoods, sigma and
residual paths, and forecasts to JSON. The test suite then evaluates
pymgarch's likelihood at R's fitted parameters, on R's own volatility paths,
and requires agreement, isolating the correlation math from optimizer and
initialization differences. Run the script once with R installed to enable
these tests locally; they skip otherwise.

## Roadmap

- v0.2: GO-GARCH via ICA
- v0.3: copula-GARCH (Gaussian and Student-t copulas)
- v0.4: composite likelihood for large cross-sections, scalar/diagonal BEKK

## Scope notes

- No look-ahead anywhere: all filtered and fitted quantities at time t use
  information through t-1 in the recursions, and forecasts condition on the
  sample end.
- Unbalanced panels and non-Constant/Zero mean specs for filtering and
  simulation are not supported in v0.1.
- Forecasting from a filtered result lands in v0.2.

## License

MIT
