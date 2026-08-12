"""Feature construction.

Every predictor is built from ``.shift(1)`` before any rolling window is
applied, so the value on row ``t`` uses information available at the close of
``t-1`` only. Getting this wrong is the single easiest way to manufacture a
volatility forecaster that looks brilliant and is worthless; ``tests/`` asserts
it explicitly.
"""

from __future__ import annotations

import pandas as pd


def har_design(rv: pd.Series) -> pd.DataFrame:
    """Corsi (2009) HAR-RV design matrix: daily, weekly and monthly lags.

    Returns columns ``rv`` (the target) plus ``rv_lag1``, ``rv_lag5``,
    ``rv_lag22``.
    """
    rv = rv.dropna()
    return pd.DataFrame(
        {
            "rv": rv,
            "rv_lag1": rv.shift(1),
            "rv_lag5": rv.shift(1).rolling(5).mean(),
            "rv_lag22": rv.shift(1).rolling(22).mean(),
        }
    ).dropna()


def next_day_lags(rv: pd.Series) -> pd.DataFrame:
    """The HAR design row for the *next* trading day.

    ``har_design`` produces rows whose lags predict the same day's realized
    vol, so its last row is already spent. To forecast tomorrow the lags have
    to be rebuilt from the observations through today.

    Returned as a one-row frame indexed by the last observed date (the "as of"
    date), not by the day being forecast — the calendar date of the next
    session depends on holidays and is resolved by the caller.
    """
    rv = rv.dropna()
    if len(rv) < 22:
        raise ValueError(f"need >= 22 observations to build lags, got {len(rv)}")

    return pd.DataFrame(
        {
            "rv_lag1": [rv.iloc[-1]],
            "rv_lag5": [rv.iloc[-5:].mean()],
            "rv_lag22": [rv.iloc[-22:].mean()],
        },
        index=[rv.index[-1]],
    )


def build_features(spy: pd.DataFrame, returns: pd.Series) -> pd.DataFrame:
    """Extended feature set used by the machine-learning comparison.

    The three HAR lags plus four candidate additions: vol-of-vol, and the
    rolling skewness, kurtosis and day-of-week that a tree model might
    plausibly exploit.
    """
    rv = spy["realized_vol"]

    features = pd.DataFrame(
        {
            "rv": rv,
            "rv_lag1": rv.shift(1),
            "rv_lag5": rv.shift(1).rolling(5).mean(),
            "rv_lag22": rv.shift(1).rolling(22).mean(),
            "rv_vol_of_vol": rv.shift(1).rolling(22).std(),
            "ret_skew_22": returns.shift(1).rolling(22).skew(),
            "ret_kurt_22": returns.shift(1).rolling(22).kurt(),
            "day_of_week": rv.index.dayofweek,
        }
    )

    return features.dropna()
