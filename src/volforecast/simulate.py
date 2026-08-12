"""Monte Carlo simulation of future volatility paths.

A point forecast throws away what the regime model uniquely knows: the *shape*
of the distribution of future volatility, which depends on the state the market
is in today. Simulating regime paths and volatility jointly recovers it.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def simulate_paths(
    rv_history: np.ndarray,
    pi0: np.ndarray,
    transmat: np.ndarray,
    coefs: np.ndarray,
    resid_std: np.ndarray,
    horizon: int = 10,
    n_paths: int = 3000,
    seed: int = 0,
    floor: float = 0.1,
) -> np.ndarray:
    """Simulate ``n_paths`` realized-volatility paths ``horizon`` days ahead.

    Each path draws a starting regime from ``pi0``, then at every step
    transitions the regime, applies that regime's HAR coefficients to the lags
    implied by the path so far, and adds regime-specific Gaussian noise. The
    feedback from simulated values into subsequent lags is what makes the
    resulting distribution right-skewed rather than symmetric.

    Parameters
    ----------
    rv_history
        At least the last 22 realized-vol observations, oldest first.
    pi0
        Regime distribution today, in severity order.
    transmat
        (K, K) transition matrix in severity order.
    coefs
        (4, K) HAR coefficients: rows ``[const, lag1, lag5, lag22]``.
    resid_std
        (K,) residual standard deviation per regime.
    floor
        Volatility cannot be negative; simulated values are clipped here.

    Returns
    -------
    ndarray of shape ``(n_paths, horizon)``.
    """
    if len(rv_history) < 22:
        raise ValueError(f"need >= 22 history points, got {len(rv_history)}")

    rng = np.random.default_rng(seed)
    n_states = len(pi0)
    buf0 = list(np.asarray(rv_history, dtype=float)[-22:])
    paths = np.zeros((n_paths, horizon))

    for i in range(n_paths):
        buf = buf0.copy()
        state = rng.choice(n_states, p=pi0)
        for h in range(horizon):
            state = rng.choice(n_states, p=transmat[state])
            lag1, lag5, lag22 = buf[-1], np.mean(buf[-5:]), np.mean(buf[-22:])
            pred = (
                coefs[0, state]
                + coefs[1, state] * lag1
                + coefs[2, state] * lag5
                + coefs[3, state] * lag22
            )
            value = max(pred + rng.normal(0, resid_std[state]), floor)
            paths[i, h] = value
            buf.append(value)

    return paths


def path_quantiles(
    paths: np.ndarray, quantiles: tuple[float, ...] = (5, 25, 50, 75, 95)
) -> pd.DataFrame:
    """Fan-chart bands: one row per day ahead, one column per percentile."""
    qs = np.percentile(paths, quantiles, axis=0)
    return pd.DataFrame(
        qs.T,
        index=pd.RangeIndex(1, paths.shape[1] + 1, name="days_ahead"),
        columns=[f"p{int(q)}" for q in quantiles],
    )


def simulate_from_model(
    rv: pd.Series,
    regime_model,
    ms_har,
    horizon: int = 10,
    n_paths: int = 3000,
    seed: int = 0,
    as_of=None,
) -> np.ndarray:
    """Convenience wrapper: launch a simulation from a date in ``rv``.

    ``as_of`` defaults to the last observation. The launch regime distribution
    is the filtered posterior on that day, so the cone is conditioned on where
    the market actually is rather than on its unconditional average.
    """
    rv = rv.dropna()
    as_of = rv.index[-1] if as_of is None else pd.Timestamp(as_of)
    history = rv.loc[:as_of]

    pi0 = regime_model.filtered_probs(history).iloc[-1].values
    return simulate_paths(
        rv_history=history.values,
        pi0=pi0,
        transmat=regime_model.transition_matrix.values,
        coefs=ms_har.coefs.values,
        resid_std=ms_har.resid_std.values,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
    )
