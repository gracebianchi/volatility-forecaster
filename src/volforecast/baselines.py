"""The three classical benchmarks: EWMA, GARCH(1,1) and HAR-RV."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch import arch_model

from .config import EWMA_LAMBDA, HAR_LAGS, TRADING_DAYS


def ewma_vol(returns: pd.Series, lam: float = EWMA_LAMBDA) -> pd.Series:
    """RiskMetrics EWMA: variance as a decay-weighted average of squared returns.

    Causal by construction — the value at ``t`` uses only returns up to ``t`` —
    so it can be computed once over the full sample and sliced afterwards.
    """
    var = returns.pow(2).ewm(alpha=1 - lam).mean()
    return (np.sqrt(var * TRADING_DAYS) * 100).rename("ewma_vol")


def garch_conditional_vol(returns: pd.Series) -> pd.Series:
    """In-sample GARCH(1,1) conditional volatility, annualized in percent."""
    res = arch_model(returns * 100, vol="Garch", p=1, q=1, dist="Normal").fit(disp="off")
    return (res.conditional_volatility * np.sqrt(TRADING_DAYS)).rename("garch_vol")


def garch_rolling_forecast(
    returns: pd.Series, split_date: str, refit_every: int = 21
) -> pd.Series:
    """Out-of-sample GARCH forecasts, refit every ``refit_every`` trading days.

    Each refit uses only returns strictly before the block it forecasts, so no
    test-period observation ever informs the parameters used to predict it.
    """
    test_returns = returns[returns.index >= split_date]
    forecasts: list[float] = []

    for i in range(0, len(test_returns), refit_every):
        train_slice = returns[returns.index < test_returns.index[i]]
        res = arch_model(
            train_slice * 100, vol="Garch", p=1, q=1, dist="Normal"
        ).fit(disp="off")

        horizon = min(refit_every, len(test_returns) - i)
        f = res.forecast(horizon=horizon, reindex=False)
        forecasts.extend(np.sqrt(f.variance.values[-1] * TRADING_DAYS))

    return pd.Series(
        forecasts[: len(test_returns)], index=test_returns.index, name="garch_forecast"
    )


def fit_har(har_df: pd.DataFrame, lags: tuple[str, ...] = HAR_LAGS):
    """OLS of realized vol on its daily / weekly / monthly lagged averages."""
    X = sm.add_constant(har_df[list(lags)])
    return sm.OLS(har_df["rv"], X).fit()


def har_predict(model, har_df: pd.DataFrame, lags: tuple[str, ...] = HAR_LAGS) -> pd.Series:
    return model.predict(sm.add_constant(har_df[list(lags)]))


def rmse(actual, predicted) -> float:
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))
