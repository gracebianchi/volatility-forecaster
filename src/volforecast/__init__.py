"""Markov regime-switching volatility forecasting for the S&P 500."""

from .config import N_STATES, REGIME_LABELS, SPLIT_DATE
from .data import garman_klass_vol, load_spy_data, to_daily_sigma
from .evt import GPDTail, backtest_var, fit_gpd_tail
from .features import build_features, har_design, next_day_lags
from .ms_har import MSHAR, diebold_mariano, qlike
from .pipeline import (
    VolForecaster,
    evaluate_out_of_sample,
    fit_forecaster,
    forecast_next_day,
)
from .regimes import RegimeModel
from .simulate import path_quantiles, simulate_from_model, simulate_paths

__version__ = "0.2.0"

__all__ = [
    "GPDTail",
    "MSHAR",
    "N_STATES",
    "REGIME_LABELS",
    "RegimeModel",
    "SPLIT_DATE",
    "VolForecaster",
    "backtest_var",
    "build_features",
    "diebold_mariano",
    "evaluate_out_of_sample",
    "fit_forecaster",
    "fit_gpd_tail",
    "forecast_next_day",
    "garman_klass_vol",
    "har_design",
    "load_spy_data",
    "next_day_lags",
    "path_quantiles",
    "qlike",
    "simulate_from_model",
    "simulate_paths",
    "to_daily_sigma",
]
