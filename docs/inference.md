# Inference

## Two-stage standard errors

Two-stage QML is consistent but the stage-2 information matrix alone
understates uncertainty: the correlation parameters are estimated on
residuals that were themselves standardized by estimated marginals. pymgarch
implements the Engle and Sheppard (2001) correction by stacking the
per-observation scores of every marginal likelihood and the correlation
likelihood into one M-estimator moment vector
$g_t(\theta)$, $\theta = (\phi_1, \dots, \phi_N, \psi)$, and computing

$$
\widehat{\mathrm{Avar}}(\hat\theta) = \hat A^{-1} \hat B \hat A^{-\top}/T,
\qquad
\hat A = \frac{1}{T}\sum_t \frac{\partial g_t}{\partial \theta'},
\qquad
\hat B = \frac{1}{T}\sum_t g_t g_t'.
$$

$\hat A$ is block lower-triangular because marginal scores do not depend on
$\psi$. All derivatives are central finite differences; marginal likelihoods
are re-evaluated through arch's own variance recursions with the backcast and
variance bounds arch used during fitting held fixed, so the perturbed
evaluations are consistent with the objective arch actually maximized.

`summary()` labels the estimates `two-stage-robust`. If the stacked bread
matrix is numerically singular, the library falls back to a stage-2-only
sandwich (labelled `stage2-robust`) and warns; those standard errors ignore
marginal estimation error.

## Known approximations

- Correlation targets $\bar S$ and $\bar N$ are held fixed at their point
  estimates, so targeting uncertainty is ignored. rmgarch makes the same
  approximation.
- Set `compute_se=False` on `fit()` to skip the sandwich entirely (useful in
  simulation studies where only point estimates matter).

## Replication against rmgarch

`scripts/make_fixtures.R` fits rmgarch on a pinned dataset (percent returns
of five dji30ret constituents) and exports parameters, likelihoods, sigma and
residual paths, correlation targets, and one-step forecasts to JSON. The
test suite then:

1. evaluates pymgarch's likelihood at R's fitted parameters on R's own
   volatility paths and correlation targets, isolating the correlation math
   from optimizer and univariate-initialization differences; and
2. runs the full pipeline on the same CSV and requires parameter agreement
   and a log-likelihood no worse than R's optimum.

The tests skip automatically when the fixture file is absent.
