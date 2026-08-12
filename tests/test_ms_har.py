from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from volforecast.features import har_design
from volforecast.ms_har import MSHAR, diebold_mariano, qlike, qlike_loss


@pytest.fixture(scope="module")
def har_and_regimes(regime_series):
    har = har_design(regime_series)
    regimes = pd.Series(
        np.where(regime_series.loc[har.index] > 20, "Panic", "Calm"),
        index=har.index,
        name="regime",
    )
    return har, regimes


@pytest.fixture(scope="module")
def fitted(har_and_regimes):
    har, regimes = har_and_regimes
    return MSHAR.fit(har, regimes, labels=("Calm", "Panic"))


def test_coefficients_have_the_expected_shape(fitted):
    assert list(fitted.coefs.index) == ["const", "rv_lag1", "rv_lag5", "rv_lag22"]
    assert list(fitted.coefs.columns) == ["Calm", "Panic"]


def test_residual_scale_grows_with_regime_severity(fitted):
    """Panic days are noisier in level terms; the simulator depends on this."""
    assert fitted.resid_std["Panic"] > fitted.resid_std["Calm"]


def test_fit_rejects_a_sparsely_populated_regime(har_and_regimes):
    har, regimes = har_and_regimes
    lopsided = regimes.copy()
    lopsided.iloc[:] = "Calm"
    lopsided.iloc[:5] = "Panic"

    with pytest.raises(ValueError, match="Panic"):
        MSHAR.fit(har, lopsided, labels=("Calm", "Panic"))


def test_blend_lies_between_the_regime_predictions(fitted, har_and_regimes):
    har, _ = har_and_regimes
    probs = pd.DataFrame(
        np.tile([0.5, 0.5], (len(har), 1)), index=har.index, columns=["Calm", "Panic"]
    )

    blended = fitted.forecast(har, probs)
    per_regime = fitted.regime_predictions(har)

    lo, hi = per_regime.min(axis=1), per_regime.max(axis=1)
    assert (blended >= lo - 1e-9).all()
    assert (blended <= hi + 1e-9).all()


def test_degenerate_probabilities_reproduce_a_single_regime(fitted, har_and_regimes):
    har, _ = har_and_regimes
    certain = pd.DataFrame(
        np.tile([1.0, 0.0], (len(har), 1)), index=har.index, columns=["Calm", "Panic"]
    )
    np.testing.assert_allclose(
        fitted.forecast(har, certain).values,
        fitted.regime_predictions(har)["Calm"].values,
        rtol=1e-9,
    )


def test_forecast_is_never_negative(fitted, har_and_regimes):
    har, _ = har_and_regimes
    probs = pd.DataFrame(
        np.tile([0.5, 0.5], (len(har), 1)), index=har.index, columns=["Calm", "Panic"]
    )
    assert (fitted.forecast(har, probs) > 0).all()


def test_qlike_is_zero_for_a_perfect_forecast():
    actual = np.array([10.0, 20.0, 30.0])
    assert qlike(actual, actual) == pytest.approx(0.0, abs=1e-12)


def test_qlike_penalizes_under_prediction_more_than_over_prediction():
    """The asymmetry is the whole reason QLIKE is used for volatility."""
    actual = np.full(50, 20.0)
    under = qlike(actual, np.full(50, 10.0))
    over = qlike(actual, np.full(50, 40.0))
    assert under > over


def test_diebold_mariano_finds_no_difference_between_identical_losses(rng):
    loss = rng.normal(1.0, 0.2, 500)
    stat, p = diebold_mariano(loss, loss.copy())
    assert stat == pytest.approx(0.0)
    assert p == pytest.approx(1.0)


def test_diebold_mariano_detects_a_genuinely_better_model(rng):
    worse = rng.normal(1.0, 0.2, 500)
    better = worse - 0.1
    stat, p = diebold_mariano(worse, better)
    assert stat > 0  # positive means the second argument wins
    assert p < 0.01


def test_qlike_loss_is_elementwise():
    actual = np.array([10.0, 20.0])
    pred = np.array([12.0, 18.0])
    assert qlike_loss(actual, pred).shape == (2,)
    assert qlike(actual, pred) == pytest.approx(qlike_loss(actual, pred).mean())
