from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import genpareto

from volforecast.evt import (
    GPDTail,
    backtest_var,
    christoffersen_cc,
    fit_gpd_tail,
    kupiec_pof,
)


def test_recovers_known_gpd_parameters(rng):
    """Returns whose lower tail is GPD by construction should be recovered."""
    xi_true, beta_true, u_true = 0.2, 1.0, 2.0
    n_body, n_tail = 9000, 1000

    body = rng.uniform(0, u_true, n_body)
    tail = u_true + genpareto.rvs(xi_true, loc=0, scale=beta_true, size=n_tail, random_state=1)
    losses = np.concatenate([body, tail])

    fitted = fit_gpd_tail(-losses, threshold_q=0.90)
    assert fitted.xi == pytest.approx(xi_true, abs=0.1)
    assert fitted.beta == pytest.approx(beta_true, rel=0.25)


def test_heavier_tail_gives_a_larger_var_multiplier():
    light = GPDTail(threshold=1.6, xi=0.05, beta=0.85, n=2000, n_exceed=200)
    heavy = GPDTail(threshold=1.6, xi=0.35, beta=0.85, n=2000, n_exceed=200)
    assert heavy.var(0.99) > light.var(0.99)


def test_var_increases_with_confidence():
    tail = GPDTail(threshold=1.6, xi=0.15, beta=0.85, n=2000, n_exceed=200)
    assert tail.var(0.999) > tail.var(0.99) > tail.var(0.95)


def test_es_exceeds_var():
    """Expected shortfall is a conditional mean beyond VaR, so it must be larger."""
    tail = GPDTail(threshold=1.6, xi=0.15, beta=0.85, n=2000, n_exceed=200)
    for p in (0.95, 0.99):
        assert tail.es(p) > tail.var(p)


def test_zero_shape_uses_the_exponential_limit():
    """xi -> 0 is a 0/0 in the general formula; the limit must stay finite."""
    tail = GPDTail(threshold=1.6, xi=0.0, beta=0.85, n=2000, n_exceed=200)
    nearly = GPDTail(threshold=1.6, xi=1e-9, beta=0.85, n=2000, n_exceed=200)
    assert np.isfinite(tail.var(0.99))
    assert tail.var(0.99) == pytest.approx(nearly.var(0.99), rel=1e-4)


def test_infinite_mean_tail_reports_infinite_es():
    assert np.isinf(GPDTail(threshold=1.0, xi=1.2, beta=1.0, n=1000, n_exceed=100).es(0.99))


def test_fit_rejects_too_few_exceedances(rng):
    with pytest.raises(ValueError, match="exceedances"):
        fit_gpd_tail(rng.normal(size=50), threshold_q=0.90)


def test_kupiec_accepts_a_correctly_calibrated_model(rng):
    """1% violations out of 1000 days is exactly on target."""
    violations = np.zeros(1000, dtype=int)
    violations[rng.choice(1000, 10, replace=False)] = 1
    _, p = kupiec_pof(violations, 0.01)
    assert p > 0.10


def test_kupiec_rejects_a_badly_calibrated_model():
    violations = np.zeros(1000, dtype=int)
    violations[:80] = 1  # 8% observed against a 1% target
    _, p = kupiec_pof(violations, 0.01)
    assert p < 0.01


def test_christoffersen_rejects_clustered_violations():
    """Right count, wrong timing: ten breaches all in a row."""
    clustered = np.zeros(1000, dtype=int)
    clustered[500:510] = 1

    _, p_uc = kupiec_pof(clustered, 0.01)
    _, p_cc = christoffersen_cc(clustered, 0.01)

    assert p_uc > 0.10  # the rate alone looks fine
    assert p_cc < 0.05  # conditional coverage catches the clustering


def test_backtest_counts_violations_correctly():
    returns = np.array([-0.05, 0.01, -0.001, -0.04, 0.02])
    sigma = np.full(5, 0.01)
    tail = GPDTail(threshold=1.6, xi=0.1, beta=0.9, n=1000, n_exceed=100)

    result = backtest_var(returns, sigma, tail, 0.99, "test")
    breached = returns < -tail.var(0.99) * sigma

    assert result["violations"] == int(breached.sum())
    assert result["realized_es"] == pytest.approx(-returns[breached].mean())
