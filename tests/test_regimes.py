from __future__ import annotations

import numpy as np
import pytest

from volforecast.regimes import RegimeModel, as_observations, forward_filter


@pytest.fixture(scope="module")
def fitted(regime_series):
    return RegimeModel.fit(regime_series, n_states=2, n_restarts=5, labels=("Calm", "Panic"))


def test_states_are_ordered_by_severity(fitted):
    """Regime i must always be less volatile than regime i+1, whatever hmmlearn numbered them."""
    vols = fitted.mean_vols.values
    assert np.all(np.diff(vols) > 0)


def test_recovers_the_generating_means(fitted):
    """The synthetic chain was built with means of 10 and 35."""
    calm, panic = fitted.mean_vols
    assert calm == pytest.approx(10, rel=0.25)
    assert panic == pytest.approx(35, rel=0.25)


def test_transition_matrix_rows_sum_to_one(fitted):
    assert np.allclose(fitted.transition_matrix.values.sum(axis=1), 1.0)


def test_regimes_are_persistent(fitted):
    """Volatility regimes cluster; the diagonal should dominate."""
    A = fitted.transition_matrix.values
    assert np.all(np.diag(A) > 0.8)
    assert (fitted.expected_duration > 5).all()


def test_forward_filter_matches_expanding_predict_proba(fitted, regime_series):
    """The O(n) recursion must equal the O(n^2) expanding-window loop it replaces.

    The last row of a smoothed posterior over data[:t+1] has no future to smooth
    over, so it is the filtered probability at t. This pins that equivalence.
    """
    rv = regime_series.iloc[:150]
    X = as_observations(rv)

    reference = np.array(
        [fitted.hmm.predict_proba(X[: i + 1])[-1] for i in range(len(X))]
    )[:, fitted.order]

    np.testing.assert_allclose(fitted.filtered_probs(rv).values, reference, atol=1e-8)


def test_forward_filter_normalizes():
    log_b = np.log(np.array([[0.6, 0.4], [0.2, 0.8], [0.5, 0.5]]))
    A = np.array([[0.9, 0.1], [0.3, 0.7]])
    alpha = forward_filter(log_b, np.array([0.5, 0.5]), A)

    assert np.allclose(alpha.sum(axis=1), 1.0)
    assert (alpha >= 0).all()


def test_filtered_probs_are_causal(fitted, regime_series):
    """Changing a future observation must not move today's filtered belief."""
    rv = regime_series.iloc[:300]
    base = fitted.filtered_probs(rv)

    tampered = rv.copy()
    tampered.iloc[250:] *= 4
    after = fitted.filtered_probs(tampered)

    np.testing.assert_allclose(base.iloc[:250].values, after.iloc[:250].values, atol=1e-10)


def test_smoothed_probs_are_not_causal(fitted, regime_series):
    """The contrast that motivates using filtered probabilities for forecasting."""
    rv = regime_series.iloc[:300]
    base = fitted.smoothed_probs(rv)

    tampered = rv.copy()
    tampered.iloc[250:] *= 4
    after = fitted.smoothed_probs(tampered)

    assert not np.allclose(base.iloc[:250].values, after.iloc[:250].values, atol=1e-6)


def test_predicted_probs_are_filtered_propagated_one_step(fitted, regime_series):
    rv = regime_series.iloc[:200]
    filt = fitted.filtered_probs(rv)
    A = fitted.transition_matrix.values

    predicted = fitted.predicted_probs(rv)
    np.testing.assert_allclose(predicted.values, filt.values[:-1] @ A, atol=1e-12)


def test_predicted_probs_are_indexed_by_the_day_being_forecast(fitted, regime_series):
    """Regression test for an off-by-one that leaks day t into its own forecast.

    Row d must be the belief formed on d-1, so that aligning it with a design
    matrix whose row d targets day d is causally correct.
    """
    rv = regime_series.iloc[:200]
    filt = fitted.filtered_probs(rv)
    predicted = fitted.predicted_probs(rv)

    # The first day cannot be forecast: there is no prior day to forecast from.
    assert predicted.index[0] == rv.index[1]
    assert predicted.index[-1] == rv.index[-1]
    assert len(predicted) == len(filt) - 1

    A = fitted.transition_matrix.values
    for d in (5, 50, 199):
        target = rv.index[d]
        np.testing.assert_allclose(
            predicted.loc[target].values, filt.iloc[d - 1].values @ A, atol=1e-12
        )


def test_predicted_probs_do_not_see_the_day_they_forecast(fitted, regime_series):
    """Perturbing day d must leave the forecast *for* day d untouched."""
    rv = regime_series.iloc[:200]
    base = fitted.predicted_probs(rv)

    target = rv.index[150]
    tampered = rv.copy()
    tampered.loc[target] *= 6

    after = fitted.predicted_probs(tampered)
    np.testing.assert_allclose(
        base.loc[:target].values, after.loc[:target].values, atol=1e-10
    )
    # ...but the day *after* the perturbation must react to it.
    assert not np.allclose(
        base.iloc[151].values, after.iloc[151].values, atol=1e-6
    )


def test_next_regime_probs_matches_a_one_step_extension(fitted, regime_series):
    """The live path and the historical path must agree on the same quantity."""
    rv = regime_series.iloc[:200]
    live = fitted.next_regime_probs(rv.iloc[:-1])
    historical = fitted.predicted_probs(rv).iloc[-1]
    np.testing.assert_allclose(live.values, historical.values, atol=1e-12)


def test_stress_index_spans_the_unit_interval(fitted, regime_series):
    probs = fitted.filtered_probs(regime_series)
    stress = fitted.stress_index(probs)

    assert stress.between(0, 1).all()
    # Calm days should score low and panic days high.
    calm_days = probs["Calm"] > 0.9
    panic_days = probs["Panic"] > 0.9
    assert stress[calm_days].mean() < 0.1
    assert stress[panic_days].mean() > 0.9


def test_label_count_must_match_state_count(regime_series):
    with pytest.raises(ValueError, match="labels"):
        RegimeModel.fit(regime_series, n_states=3, n_restarts=2, labels=("A", "B"))


def test_roundtrip_save_load(fitted, tmp_path, regime_series):
    path = tmp_path / "model.joblib"
    fitted.save(path)
    restored = RegimeModel.load(path)

    np.testing.assert_allclose(restored.mean_vols.values, fitted.mean_vols.values)
    np.testing.assert_allclose(
        restored.filtered_probs(regime_series.iloc[:50]).values,
        fitted.filtered_probs(regime_series.iloc[:50]).values,
    )
