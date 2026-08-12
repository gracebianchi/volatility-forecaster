"""Lookahead is the failure mode that matters most here, so it gets its own file."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.data import garman_klass_vol
from volforecast.features import build_features, har_design, next_day_lags


def test_garman_klass_is_nonnegative_and_annualized(ohlc):
    rv = garman_klass_vol(ohlc)
    assert (rv.dropna() >= 0).all()
    # Annualized percent: a broad-market series should sit in single/double digits.
    assert 1 < rv.mean() < 200


def test_garman_klass_scales_with_range(ohlc):
    """Doubling every intraday range must raise the estimator."""
    wider = ohlc.copy()
    mid = (wider["High"] + wider["Low"]) / 2
    wider["High"] = mid + (wider["High"] - mid) * 2
    wider["Low"] = mid - (mid - wider["Low"]) * 2
    assert garman_klass_vol(wider).mean() > garman_klass_vol(ohlc).mean()


def test_har_lags_use_only_past_information():
    rv = pd.Series(
        np.arange(1.0, 101.0), index=pd.bdate_range("2020-01-01", periods=100)
    )
    har = har_design(rv)

    row = har.iloc[10]
    date = har.index[10]
    pos = rv.index.get_loc(date)

    assert row["rv_lag1"] == pytest.approx(rv.iloc[pos - 1])
    assert row["rv_lag5"] == pytest.approx(rv.iloc[pos - 5 : pos].mean())
    assert row["rv_lag22"] == pytest.approx(rv.iloc[pos - 22 : pos].mean())


def test_har_target_is_never_in_its_own_lags():
    """Perturbing rv at date t must not change the lags on date t."""
    rv = pd.Series(
        np.linspace(10, 30, 200), index=pd.bdate_range("2020-01-01", periods=200)
    )
    base = har_design(rv)

    bumped = rv.copy()
    target = base.index[50]
    bumped.loc[target] += 100

    perturbed = har_design(bumped)
    lags = ["rv_lag1", "rv_lag5", "rv_lag22"]
    pd.testing.assert_series_equal(base.loc[target, lags], perturbed.loc[target, lags])
    assert perturbed.loc[target, "rv"] != base.loc[target, "rv"]


def test_extended_features_have_no_lookahead(ohlc):
    """Every non-target column on day t must be unchanged by day t's own data."""
    spy = ohlc.copy()
    spy["realized_vol"] = garman_klass_vol(spy)
    returns = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()

    base = build_features(spy, returns)
    target = base.index[100]

    tampered = spy.copy()
    tampered.loc[target, "realized_vol"] *= 5
    tampered_returns = returns.copy()
    tampered_returns.loc[target] *= 5

    after = build_features(tampered, tampered_returns)
    predictors = [c for c in base.columns if c != "rv"]
    pd.testing.assert_series_equal(
        base.loc[target, predictors], after.loc[target, predictors]
    )


def test_next_day_lags_continue_the_series():
    rv = pd.Series(
        np.arange(1.0, 51.0), index=pd.bdate_range("2020-01-01", periods=50)
    )
    lags = next_day_lags(rv)

    assert lags.index[-1] == rv.index[-1]
    assert lags.iloc[0]["rv_lag1"] == pytest.approx(rv.iloc[-1])
    assert lags.iloc[0]["rv_lag5"] == pytest.approx(rv.iloc[-5:].mean())
    assert lags.iloc[0]["rv_lag22"] == pytest.approx(rv.iloc[-22:].mean())


def test_next_day_lags_requires_enough_history():
    rv = pd.Series(np.arange(10.0), index=pd.bdate_range("2020-01-01", periods=10))
    with pytest.raises(ValueError, match="22"):
        next_day_lags(rv)
