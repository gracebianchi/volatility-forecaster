"""SPY price data and the Garman-Klass realized-volatility estimator."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import yfinance as yf

from .config import START, TICKER, TRADING_DAYS

# Garman-Klass uses the intraday high/low range in addition to open/close, so
# it extracts far more information from one day's bar than a squared close-to-
# close return does. Constant below is (2 * ln 2 - 1).
_GK_CO_COEF = 2 * np.log(2) - 1


def garman_klass_vol(ohlc: pd.DataFrame) -> pd.Series:
    """Annualized Garman-Klass realized volatility, in percent.

    Parameters
    ----------
    ohlc
        Frame with ``Open``, ``High``, ``Low``, ``Close`` columns.
    """
    log_hl = np.log(ohlc["High"] / ohlc["Low"])
    log_co = np.log(ohlc["Close"] / ohlc["Open"])
    gk_var = 0.5 * log_hl**2 - _GK_CO_COEF * log_co**2
    return np.sqrt(gk_var * TRADING_DAYS) * 100


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance returns a MultiIndex when given a ticker list; drop that level."""
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


def _download(ticker: str, start: str, end: str | None, retries: int, pause: float) -> pd.DataFrame:
    """Download with retries.

    The daily job runs unattended against a free endpoint that intermittently
    returns an empty frame under load; one transient failure should not break a
    day's forecast.
    """
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
            if not df.empty:
                return df
            last_error = RuntimeError("empty frame")
        except Exception as exc:  # network/parsing failures are both transient
            last_error = exc

        if attempt < retries - 1:
            time.sleep(pause * (2**attempt))

    raise RuntimeError(
        f"yfinance returned no usable data for {ticker} after {retries} attempts: {last_error}"
    )


def load_spy_data(
    start: str = START,
    end: str | None = None,
    ticker: str = TICKER,
    retries: int = 3,
    pause: float = 2.0,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download OHLC data and derive realized volatility and log returns.

    ``end`` defaults to ``None``, which means "through the latest available
    bar". The notebook passes ``NOTEBOOK_DATA_END`` to pin itself to a frozen
    sample; the daily pipeline leaves it unset.

    Returns
    -------
    (spy, returns)
        ``spy`` has an added ``realized_vol`` column (annualized %); ``returns``
        is the daily log-return series with the first NaN dropped.
    """
    spy = _flatten_columns(_download(ticker, start, end, retries, pause))
    spy["realized_vol"] = garman_klass_vol(spy)
    returns = np.log(spy["Close"] / spy["Close"].shift(1)).dropna()
    returns.name = "return"

    return spy, returns


def to_daily_sigma(annual_vol_pct: np.ndarray | pd.Series) -> np.ndarray | pd.Series:
    """Convert annualized volatility in percent to a daily return standard deviation."""
    return annual_vol_pct / 100 / np.sqrt(TRADING_DAYS)
