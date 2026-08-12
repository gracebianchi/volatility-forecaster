"""End-to-end checks on synthetic data, plus the daily job's file handling."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from volforecast.daily import (
    HISTORY_COLUMNS,
    backfill_outcomes,
    live_scorecard,
    load_history,
    normalize,
)
from volforecast.data import garman_klass_vol
from volforecast.pipeline import fit_forecaster, forecast_next_day


@pytest.fixture(scope="module")
def synthetic_market(regime_series):
    """An OHLC frame whose realized vol *is* the known regime-switching series."""
    rv = regime_series
    rng = np.random.default_rng(99)

    daily_sigma = rv.values / 100 / np.sqrt(252)
    ret = rng.normal(0, daily_sigma)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close / np.exp(ret)
    span = daily_sigma * 1.5
    spy = pd.DataFrame(
        {
            "Open": open_,
            "High": np.maximum(open_, close) * (1 + span),
            "Low": np.minimum(open_, close) * (1 - span),
            "Close": close,
        },
        index=rv.index,
    )
    spy["realized_vol"] = rv.values  # use the known series, not the estimator
    returns = pd.Series(ret, index=rv.index).iloc[1:]
    return spy, returns


@pytest.fixture(scope="module")
def forecaster(synthetic_market):
    spy, returns = synthetic_market
    return fit_forecaster(spy, returns, n_states=2, labels=("Calm", "Panic"))


def test_fit_produces_a_complete_bundle(forecaster):
    assert forecaster.regime_model.n_states == 2
    assert forecaster.ms_har.coefs.shape[0] == 4
    assert np.isfinite(forecaster.tail.xi)
    assert forecaster.tail.n_exceed > 20


def test_train_end_excludes_later_data(synthetic_market):
    """Nothing after train_end may touch a parameter."""
    spy, returns = synthetic_market
    cutoff = spy.index[1000]
    fitted = fit_forecaster(
        spy, returns, train_end=str(cutoff.date()), n_states=2, labels=("Calm", "Panic")
    )
    assert fitted.trained_through < cutoff


def test_forecast_has_the_expected_structure(forecaster, synthetic_market):
    spy, _ = synthetic_market
    fc = forecast_next_day(forecaster, spy, horizon=5, n_paths=200)

    assert fc["as_of"] == spy.index[-1].strftime("%Y-%m-%d")
    assert fc["forecast_rv"] > 0
    assert 0 <= fc["stress_index"] <= 1
    assert fc["regime_today"] in forecaster.regime_model.labels
    assert len(fc["fan"]["days_ahead"]) == 5
    assert sum(fc["regime_probs_tomorrow"].values()) == pytest.approx(1.0)
    for level in ("99", "95"):
        assert fc["risk"][level]["es"] > fc["risk"][level]["var"] > 0


def test_forecast_is_json_serializable(forecaster, synthetic_market):
    """The daily job writes this straight to disk; numpy scalars would break it."""
    spy, _ = synthetic_market
    fc = forecast_next_day(forecaster, spy, horizon=3, n_paths=100)
    json.loads(json.dumps(fc))


def test_99_percent_var_exceeds_95_percent(forecaster, synthetic_market):
    spy, _ = synthetic_market
    fc = forecast_next_day(forecaster, spy, horizon=3, n_paths=100)
    assert fc["risk"]["99"]["var"] > fc["risk"]["95"]["var"]


# --- daily job plumbing ----------------------------------------------------


def _history_row(as_of: str, **overrides) -> dict:
    row = {c: np.nan for c in HISTORY_COLUMNS}
    row.update(
        {
            "as_of": as_of,
            "forecast_rv": 15.0,
            "forecast_rv_single_har": 16.0,
            "daily_sigma": 0.0094,
            "stress_index": 0.3,
            "regime": "Normal",
            "var99": 0.03,
            "var95": 0.02,
        }
    )
    row.update(overrides)
    return row


def test_backfill_fills_the_following_session(synthetic_market):
    spy, returns = synthetic_market
    as_of = spy.index[100]
    target = spy.index[101]

    history = pd.DataFrame([_history_row(as_of.strftime("%Y-%m-%d"))])
    filled = backfill_outcomes(history, spy, returns)

    assert filled.loc[0, "realized_rv"] == pytest.approx(spy["realized_vol"].loc[target])
    assert filled.loc[0, "realized_return"] == pytest.approx(returns.loc[target])


def test_backfill_leaves_the_newest_forecast_unresolved(synthetic_market):
    """The most recent forecast has no outcome yet and must stay blank."""
    spy, returns = synthetic_market
    history = pd.DataFrame([_history_row(spy.index[-1].strftime("%Y-%m-%d"))])
    filled = backfill_outcomes(history, spy, returns)
    assert pd.isna(filled.loc[0, "realized_rv"])


def test_backfill_is_idempotent(synthetic_market):
    spy, returns = synthetic_market
    history = pd.DataFrame([_history_row(spy.index[100].strftime("%Y-%m-%d"))])

    once = backfill_outcomes(history, spy, returns)
    twice = backfill_outcomes(once, spy, returns)
    pd.testing.assert_frame_equal(once, twice)


def test_backfill_flags_a_var_breach(synthetic_market):
    """A loss deeper than the stated VaR must be recorded as a breach."""
    spy, returns = synthetic_market

    # Pick a genuine down day, then set VaR inside the loss it produced.
    down_day = returns[returns < 0].index[10]
    as_of = spy.index[spy.index.get_loc(down_day) - 1]
    loss = -returns.loc[down_day]

    history = pd.DataFrame(
        [_history_row(as_of.strftime("%Y-%m-%d"), var99=loss / 2, var95=loss / 2)]
    )
    filled = backfill_outcomes(history, spy, returns)
    assert filled.loc[0, "breach99"] == 1


def test_backfill_does_not_flag_a_loss_inside_var(synthetic_market):
    spy, returns = synthetic_market
    down_day = returns[returns < 0].index[10]
    as_of = spy.index[spy.index.get_loc(down_day) - 1]
    loss = -returns.loc[down_day]

    history = pd.DataFrame(
        [_history_row(as_of.strftime("%Y-%m-%d"), var99=loss * 2, var95=loss * 2)]
    )
    filled = backfill_outcomes(history, spy, returns)
    assert filled.loc[0, "breach99"] == 0


def test_written_history_is_byte_stable_across_a_read_write_cycle(tmp_path):
    """A rewrite with no new data must produce an identical file.

    Regression test for float-format churn: a freshly appended row lives in an
    object-dtype column and is written with 17 significant digits, but after a
    read it is float64 and written with 16 — so the scheduled job committed a
    diff of trailing digits every single day.
    """
    path = tmp_path / "history.csv"
    fresh = pd.DataFrame([_history_row("2026-08-11")])

    normalize(fresh).to_csv(path, index=False)
    first = path.read_text()

    normalize(load_history(path)).to_csv(path, index=False)
    assert path.read_text() == first

    normalize(load_history(path)).to_csv(path, index=False)
    assert path.read_text() == first


def test_normalize_preserves_text_columns_and_values():
    row = _history_row("2026-08-11", realized_rv=14.25)
    out = normalize(pd.DataFrame([row]))

    assert out.loc[0, "as_of"] == "2026-08-11"
    assert out.loc[0, "regime"] == "Normal"
    assert out.loc[0, "realized_rv"] == pytest.approx(14.25)
    assert out["forecast_rv"].dtype == np.float64


def test_scorecard_ignores_unresolved_forecasts():
    history = pd.DataFrame(
        [
            _history_row("2026-01-05", realized_rv=14.0, breach99=0.0),
            _history_row("2026-01-06", realized_rv=18.0, breach99=1.0),
            _history_row("2026-01-07"),  # not yet resolved
        ]
    )
    score = live_scorecard(history)
    assert score["n"] == 2
    assert score["breaches_99"] == 1
    assert score["rmse_ms_har"] == pytest.approx(np.sqrt(((14 - 15) ** 2 + (18 - 15) ** 2) / 2))


def test_scorecard_handles_an_empty_history():
    assert live_scorecard(pd.DataFrame(columns=HISTORY_COLUMNS))["n"] == 0


def test_load_history_returns_an_empty_frame_when_absent(tmp_path):
    history = load_history(tmp_path / "nope.csv")
    assert history.empty
    assert list(history.columns) == HISTORY_COLUMNS


def test_garman_klass_matches_the_module_export(ohlc):
    """Guards against the estimator drifting between data.py and the notebook."""
    rv = garman_klass_vol(ohlc)
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    log_co = np.log(ohlc["Close"] / ohlc["Open"])
    expected = np.sqrt(
        (0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2) * 252
    ) * 100
    pd.testing.assert_series_equal(rv, expected)
