from __future__ import annotations

import numpy as np
import pytest

from volforecast.simulate import path_quantiles, simulate_paths

# A two-regime toy world: a quiet state anchored near 10, a violent one near 40.
TRANSMAT = np.array([[0.95, 0.05], [0.10, 0.90]])
COEFS = np.array(
    [
        [2.0, 8.0],  # const
        [0.4, 0.5],  # rv_lag1
        [0.2, 0.2],  # rv_lag5
        [0.2, 0.1],  # rv_lag22
    ]
)
RESID_STD = np.array([1.0, 6.0])
HISTORY = np.full(30, 12.0)


def _simulate(pi0, **kwargs):
    return simulate_paths(
        rv_history=HISTORY,
        pi0=np.asarray(pi0, dtype=float),
        transmat=TRANSMAT,
        coefs=COEFS,
        resid_std=RESID_STD,
        **kwargs,
    )


def test_output_shape():
    paths = _simulate([1.0, 0.0], horizon=7, n_paths=200, seed=1)
    assert paths.shape == (200, 7)


def test_paths_are_strictly_positive():
    """Volatility has no negative branch; the floor must hold even in the tail."""
    paths = _simulate([0.0, 1.0], horizon=15, n_paths=500, seed=2)
    assert (paths > 0).all()


def test_simulation_is_reproducible():
    a = _simulate([0.5, 0.5], horizon=5, n_paths=100, seed=7)
    b = _simulate([0.5, 0.5], horizon=5, n_paths=100, seed=7)
    np.testing.assert_array_equal(a, b)


def test_different_seeds_give_different_paths():
    a = _simulate([0.5, 0.5], horizon=5, n_paths=100, seed=7)
    b = _simulate([0.5, 0.5], horizon=5, n_paths=100, seed=8)
    assert not np.array_equal(a, b)


def test_launching_from_a_stressed_state_raises_the_forecast():
    """The whole point of the model: the cone depends on where you start."""
    calm = _simulate([1.0, 0.0], horizon=10, n_paths=1500, seed=3)
    stressed = _simulate([0.0, 1.0], horizon=10, n_paths=1500, seed=3)
    assert np.median(stressed[:, 0]) > np.median(calm[:, 0])


def test_stressed_launch_widens_the_distribution():
    """Uncertainty is regime-dependent, not a fixed band."""
    calm = _simulate([1.0, 0.0], horizon=10, n_paths=1500, seed=4)
    stressed = _simulate([0.0, 1.0], horizon=10, n_paths=1500, seed=4)

    calm_width = np.percentile(calm[:, -1], 95) - np.percentile(calm[:, -1], 5)
    stress_width = np.percentile(stressed[:, -1], 95) - np.percentile(stressed[:, -1], 5)
    assert stress_width > calm_width


def test_predictive_distribution_is_right_skewed():
    """Regime switching produces an asymmetric fan a single-regime model cannot."""
    paths = _simulate([1.0, 0.0], horizon=10, n_paths=4000, seed=5)
    final = paths[:, -1]

    median = np.median(final)
    upper = np.percentile(final, 95) - median
    lower = median - np.percentile(final, 5)
    assert upper > lower


def test_history_shorter_than_the_monthly_lag_is_rejected():
    with pytest.raises(ValueError, match="22"):
        simulate_paths(
            rv_history=np.full(10, 12.0),
            pi0=np.array([1.0, 0.0]),
            transmat=TRANSMAT,
            coefs=COEFS,
            resid_std=RESID_STD,
        )


def test_quantile_bands_are_ordered():
    paths = _simulate([0.5, 0.5], horizon=10, n_paths=1000, seed=6)
    fan = path_quantiles(paths)

    assert list(fan.columns) == ["p5", "p25", "p50", "p75", "p95"]
    assert len(fan) == 10
    values = fan.values
    assert (np.diff(values, axis=1) >= 0).all()
