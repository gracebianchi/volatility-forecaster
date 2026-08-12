"""End-to-end wiring: the research evaluation and the live forecasting model.

Both entry points share the same estimation code, which is the point of the
package — the daily forecaster cannot drift away from the model the notebook
documents.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import statsmodels.api as sm

from .baselines import fit_har, rmse
from .config import (
    GPD_THRESHOLD_Q,
    HAR_LAGS,
    MODELS_DIR,
    N_STATES,
    REGIME_LABELS,
    SPLIT_DATE,
    VAR_LEVELS,
)
from .data import to_daily_sigma
from .evt import GPDTail, backtest_var, fit_gpd_tail
from .features import har_design, next_day_lags
from .ms_har import MSHAR, diebold_mariano, qlike, qlike_loss
from .regimes import RegimeModel
from .simulate import path_quantiles, simulate_paths


@dataclass
class VolForecaster:
    """Everything needed to produce tomorrow's forecast, in one saveable object."""

    regime_model: RegimeModel
    ms_har: MSHAR
    single_har_params: pd.Series
    tail: GPDTail
    trained_through: pd.Timestamp
    labels: tuple[str, ...] = REGIME_LABELS

    def save(self, path=None):
        import joblib

        path = path or MODELS_DIR / "forecaster.joblib"
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path=None) -> VolForecaster:
        import joblib

        return joblib.load(path or MODELS_DIR / "forecaster.joblib")


def fit_forecaster(
    spy: pd.DataFrame,
    returns: pd.Series,
    train_end: str | None = None,
    n_states: int = N_STATES,
    threshold_q: float = GPD_THRESHOLD_Q,
    labels: tuple[str, ...] | None = None,
) -> VolForecaster:
    """Estimate every component on data up to ``train_end`` (default: everything).

    The HMM, the regime-specific HARs and the GPD tail are all fit on the same
    window. When ``train_end`` is set, nothing after it touches any parameter —
    that is what makes the out-of-sample evaluation honest.
    """
    rv = spy["realized_vol"].dropna()
    if train_end is not None:
        rv = rv[rv.index < train_end]

    har_df = har_design(spy["realized_vol"])
    har_train = har_df[har_df.index <= rv.index[-1]]

    regime_model = RegimeModel.fit(rv, n_states=n_states, labels=labels)
    regimes = regime_model.hard_states(rv)
    ms_har = MSHAR.fit(har_train, regimes, labels=regime_model.labels)

    single_har = fit_har(har_train)

    # Standardize training returns by the model's own volatility forecast, then
    # fit the tail to what is left. Using the regime-blended forecast here (not
    # a hard regime assignment) keeps the tail consistent with how the forecast
    # is actually produced out of sample.
    probs_train = regime_model.predicted_probs(rv)
    fc_train = ms_har.forecast(har_train, probs_train)
    idx = fc_train.index.intersection(returns.index)
    z_train = returns.loc[idx].values / to_daily_sigma(fc_train.loc[idx].values)
    tail = fit_gpd_tail(z_train, threshold_q)

    return VolForecaster(
        regime_model=regime_model,
        ms_har=ms_har,
        single_har_params=single_har.params,
        tail=tail,
        trained_through=rv.index[-1],
        labels=regime_model.labels,
    )


def forecast_next_day(
    forecaster: VolForecaster,
    spy: pd.DataFrame,
    horizon: int = 10,
    n_paths: int = 3000,
    levels: tuple[float, ...] = VAR_LEVELS,
    seed: int = 0,
) -> dict:
    """Produce tomorrow's regime, point forecast, fan chart and risk numbers."""
    rv = spy["realized_vol"].dropna()
    as_of = rv.index[-1]

    filtered = forecaster.regime_model.filtered_probs(rv)
    pi_today = filtered.iloc[-1]
    pi_tomorrow = forecaster.regime_model.next_regime_probs(rv)

    lags = next_day_lags(rv)
    regime_preds = forecaster.ms_har.regime_predictions(lags).iloc[0]
    point = float((regime_preds.values * pi_tomorrow.values).sum())

    single = float(
        forecaster.single_har_params["const"]
        + sum(forecaster.single_har_params[c] * lags.iloc[0][c] for c in HAR_LAGS)
    )

    paths = simulate_paths(
        rv_history=rv.values,
        pi0=pi_today.values,
        transmat=forecaster.regime_model.transition_matrix.values,
        coefs=forecaster.ms_har.coefs.values,
        resid_std=forecaster.ms_har.resid_std.values,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
    )
    fan = path_quantiles(paths)

    sigma = float(to_daily_sigma(point))
    risk = {
        f"{int(p * 100)}": {
            "var": forecaster.tail.var(p) * sigma,
            "es": forecaster.tail.es(p) * sigma,
        }
        for p in levels
    }

    stress = float(
        forecaster.regime_model.stress_index(pi_tomorrow.to_frame().T).iloc[0]
    )

    return {
        "as_of": as_of.strftime("%Y-%m-%d"),
        "generated_at": pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "trained_through": forecaster.trained_through.strftime("%Y-%m-%d"),
        "realized_vol": float(rv.iloc[-1]),
        "forecast_rv": point,
        "forecast_rv_single_har": single,
        "daily_sigma": sigma,
        "stress_index": stress,
        "regime_today": str(pi_today.idxmax()),
        "regime_probs_today": {k: float(v) for k, v in pi_today.items()},
        "regime_probs_tomorrow": {k: float(v) for k, v in pi_tomorrow.items()},
        "fan": {
            "days_ahead": fan.index.tolist(),
            **{c: [float(v) for v in fan[c]] for c in fan.columns},
        },
        "risk": risk,
    }


# --- research evaluation ---------------------------------------------------


def evaluate_out_of_sample(
    spy: pd.DataFrame,
    returns: pd.Series,
    split_date: str = SPLIT_DATE,
    n_states: int = N_STATES,
    horizon_levels: tuple[float, ...] = VAR_LEVELS,
) -> dict:
    """Full out-of-sample comparison of single-regime HAR against MS-HAR.

    Everything — HMM parameters, regime HAR coefficients, the GPD tail — is
    estimated on data before ``split_date`` only.
    """
    forecaster = fit_forecaster(spy, returns, train_end=split_date, n_states=n_states)
    rv = spy["realized_vol"].dropna()
    har_df = har_design(spy["realized_vol"])

    test_idx = har_df.index[har_df.index >= split_date]
    har_test = har_df.loc[test_idx]
    y_true = har_test["rv"]

    # Regime beliefs are filtered over the *full* series so that the recursion
    # carries its state into the test period, then read off on test dates only.
    predicted_probs = forecaster.regime_model.predicted_probs(rv)

    ms_pred = forecaster.ms_har.forecast(har_test, predicted_probs)
    single_pred = pd.Series(
        sm.add_constant(har_test[list(HAR_LAGS)]).values @ forecaster.single_har_params.values,
        index=har_test.index,
    )

    idx = ms_pred.index
    y, ms, single = y_true.loc[idx].values, ms_pred.values, single_pred.loc[idx].values

    loss_single, loss_ms = qlike_loss(y, single), qlike_loss(y, ms)
    dm_stat, dm_p = diebold_mariano(loss_single, loss_ms)

    # --- risk backtest ----------------------------------------------------
    har_train = har_df[har_df.index < split_date]
    single_train = pd.Series(
        sm.add_constant(har_train[list(HAR_LAGS)]).values @ forecaster.single_har_params.values,
        index=har_train.index,
    )
    train_idx = single_train.index.intersection(returns.index)
    z_single_train = returns.loc[train_idx].values / to_daily_sigma(
        single_train.loc[train_idx].values
    )
    tail_single = fit_gpd_tail(z_single_train)

    r_test = returns.loc[idx].values
    sig_ms, sig_single = to_daily_sigma(ms), to_daily_sigma(single)

    backtests = []
    for level in horizon_levels:
        backtests.append(
            backtest_var(r_test, sig_single, tail_single, level, "Single-regime HAR")
        )
        backtests.append(backtest_var(r_test, sig_ms, forecaster.tail, level, "MS-HAR"))

    return {
        "forecaster": forecaster,
        "test_dates": idx,
        "y_true": y,
        "ms_pred": ms,
        "single_pred": single,
        "rmse_ms": rmse(y, ms),
        "rmse_single": rmse(y, single),
        "qlike_ms": qlike(y, ms),
        "qlike_single": qlike(y, single),
        "dm_stat": dm_stat,
        "dm_p": dm_p,
        "tail_ms": forecaster.tail,
        "tail_single": tail_single,
        "backtests": backtests,
    }
