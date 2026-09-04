# Generate replication fixtures from rmgarch for pymgarch's test suite.
#
# Usage (from the repository root, requires R with rmgarch + jsonlite):
#   Rscript scripts/make_fixtures.R
#
# Writes:
#   tests/fixtures/dji30ret_5.csv        the exact data both languages fit
#   tests/fixtures/rmgarch_fixtures.json parameters, likelihoods, paths
#
# The JSON deliberately includes rmgarch's fitted sigma and residual matrices
# so the Python tests can evaluate the stage-2 correlation likelihood at R's
# parameters without any univariate-initialization confounds.

library(rmgarch)
library(jsonlite)

data(dji30ret)
X <- dji30ret[1:1000, 1:5] * 100  # percent returns, first 5 Dow constituents

uspec <- multispec(replicate(
  5,
  ugarchspec(
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    mean.model = list(armaOrder = c(0, 0), include.mean = TRUE),
    distribution.model = "norm"
  )
))

fit_one <- function(spec) {
  fit <- dccfit(spec, data = X)
  # n.ahead = 1: rmgarch does not support multi-step aDCC forecasts, and the
  # Python tests only consume the (deterministic) one-step correlation anyway
  fc <- dccforecast(fit, n.ahead = 1)
  rc <- rcor(fit)
  list(
    stage2 = as.list(coef(fit, type = "dcc")),
    marginal_coefs = lapply(fit@mfit$matcoef[, 1], identity),
    loglik = likelihood(fit),
    sigma = unname(as.matrix(sigma(fit))),
    resid = unname(as.matrix(residuals(fit))),
    Qbar = unname(fit@mfit$Qbar),
    Nbar = tryCatch(unname(fit@mfit$Nbar), error = function(e) NULL),
    Rlast = unname(rc[, , dim(rc)[3]]),
    R1_forecast = unname(rcor(fc)[[1]][, , 1]),
    H1_forecast = unname(rcov(fc)[[1]][, , 1])
  )
}

fit_gogarch <- function() {
  gspec <- gogarchspec(
    mean.model = list(model = "constant"),
    variance.model = list(model = "sGARCH", garchOrder = c(1, 1)),
    distribution.model = "mvnorm",
    ica = "fastica"
  )
  set.seed(42)  # rmgarch's fastICA has random initialization
  gfit <- gogarchfit(gspec, data = X)
  rc <- rcov(gfit)
  # factor sigmas live on the internal multifit; access defensively so a
  # future rmgarch refactor degrades the fixture rather than killing the run
  fsig <- tryCatch(
    unname(as.matrix(sigma(gfit@mfit$ufit))),
    error = function(e) NULL
  )
  list(
    A = unname(as.matrix(gfit@mfit$A)),
    resid = unname(as.matrix(residuals(gfit))),
    factor_sigma = fsig,
    factor_coefs = unname(coef(gfit)),
    loglik = likelihood(gfit),
    Hlast = unname(rc[, , dim(rc)[3]])
  )
}

fit_cgarch <- function(time_varying) {
  cspec <- cgarchspec(
    uspec,
    dccOrder = c(1, 1),
    distribution.model = list(
      copula = "mvt",
      method = "Kendall",
      time.varying = time_varying,
      transformation = "parametric"
    )
  )
  cfit <- cgarchfit(cspec, data = X)
  rc <- tryCatch(rcor(cfit), error = function(e) NULL)
  list(
    coefs = as.list(coef(cfit)),
    loglik = likelihood(cfit),
    sigma = unname(as.matrix(sigma(cfit))),
    resid = unname(as.matrix(residuals(cfit))),
    Qbar = tryCatch(unname(cfit@mfit$Qbar), error = function(e) NULL),
    Rstatic = tryCatch(unname(cfit@mfit$Rt), error = function(e) NULL),
    Rlast = if (!is.null(rc) && length(dim(rc)) == 3)
      unname(rc[, , dim(rc)[3]]) else NULL
  )
}

out <- list(
  meta = list(
    package = "rmgarch",
    version = as.character(packageVersion("rmgarch")),
    data = "dji30ret[1:1000, 1:5] * 100",
    T = nrow(X),
    N = ncol(X),
    columns = colnames(X)
  ),
  dcc_norm = fit_one(dccspec(uspec, dccOrder = c(1, 1), distribution = "mvnorm")),
  dcc_t = fit_one(dccspec(uspec, dccOrder = c(1, 1), distribution = "mvt")),
  adcc_norm = fit_one(dccspec(uspec, dccOrder = c(1, 1), model = "aDCC",
                              distribution = "mvnorm")),
  gogarch_norm = fit_gogarch(),
  cgarch_t_dcc = fit_cgarch(TRUE),
  cgarch_t_static = fit_cgarch(FALSE)
)

dir.create("tests/fixtures", recursive = TRUE, showWarnings = FALSE)
write.csv(data.frame(X), "tests/fixtures/dji30ret_5.csv", row.names = FALSE)
write_json(out, "tests/fixtures/rmgarch_fixtures.json",
           digits = NA, auto_unbox = TRUE)
cat("fixtures written to tests/fixtures/\n")
