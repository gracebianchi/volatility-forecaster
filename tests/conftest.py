"""Synthetic fixtures.

The tests deliberately do not hit the network. A regime-switching generator
gives us data whose true structure is known, so the model can be checked
against ground truth rather than against whatever the market did.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def rng():
    return np.random.default_rng(20240101)


@pytest.fixture(scope="session")
def ohlc(rng) -> pd.DataFrame:
    """A plausible OHLC frame with a volatility level shift partway through."""
    n = 800
    dates = pd.bdate_range("2018-01-01", periods=n)

    sigma = np.where(np.arange(n) < 500, 0.008, 0.025)
    ret = rng.normal(0, sigma)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = close / np.exp(ret)

    span = np.abs(rng.normal(0, sigma)) + np.abs(ret)
    high = np.maximum(open_, close) * (1 + span)
    low = np.minimum(open_, close) * (1 - span)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close}, index=dates
    )


@pytest.fixture(scope="session")
def regime_series(rng) -> pd.Series:
    """Realized vol from a known 2-state Markov chain with well-separated means."""
    n = 1500
    A = np.array([[0.98, 0.02], [0.06, 0.94]])
    mu = np.log(np.array([10.0, 35.0]))

    states = np.zeros(n, dtype=int)
    for t in range(1, n):
        states[t] = rng.choice(2, p=A[states[t - 1]])

    values = np.exp(rng.normal(mu[states], 0.25))
    return pd.Series(values, index=pd.bdate_range("2015-01-01", periods=n), name="realized_vol")
