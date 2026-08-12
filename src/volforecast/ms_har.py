"""Markov-switching HAR: one HAR regression per regime, blended by regime odds.

A single-regime HAR imposes one fixed relationship between lagged and current
volatility across every market state. The premise here is that the relationship
itself changes: in calm markets volatility is anchored to its monthly average,
while in a panic yesterday dominates and mean reversion is much weaker.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from .config import HAR_COLS, HAR_LAGS, REGIME_LABELS


@dataclass
class MSHAR:
    """Regime-specific HAR coefficients and residual scales.

    ``coefs`` is indexed by ``HAR_COLS`` with one column per regime;
    ``resid_std`` gives each regime's residual standard deviation, which the
    Monte Carlo simulator needs to inject the right amount of within-regime
    noise.
    """

    coefs: pd.DataFrame
    resid_std: pd.Series
    n_obs: pd.Series
    labels: tuple[str, ...] = REGIME_LABELS

    @classmethod
    def fit(
        cls,
        har_df: pd.DataFrame,
        regimes: pd.Series,
        labels: tuple[str, ...] = REGIME_LABELS,
        lags: tuple[str, ...] = HAR_LAGS,
        min_obs: int = 30,
    ) -> MSHAR:
        """Fit one OLS per regime on the rows assigned to it.

        ``har_df`` and ``regimes`` must both already be restricted to the
        training period — this function does no splitting of its own.
        """
        idx = har_df.index.intersection(regimes.index)
        har_df, regimes = har_df.loc[idx], regimes.loc[idx]

        coefs, resid_std, n_obs = {}, {}, {}
        for label in labels:
            sub = har_df[regimes == label]
            if len(sub) < min_obs:
                raise ValueError(
                    f"regime {label!r} has only {len(sub)} training rows "
                    f"(minimum {min_obs}); refit the HMM or merge states"
                )
            fit = sm.OLS(sub["rv"], sm.add_constant(sub[list(lags)])).fit()
            coefs[label] = fit.params
            resid_std[label] = fit.resid.std()
            n_obs[label] = len(sub)

        return cls(
            coefs=pd.DataFrame(coefs)[list(labels)].reindex(list(HAR_COLS)),
            resid_std=pd.Series(resid_std)[list(labels)],
            n_obs=pd.Series(n_obs)[list(labels)],
            labels=labels,
        )

    def regime_predictions(
        self, har_df: pd.DataFrame, lags: tuple[str, ...] = HAR_LAGS
    ) -> pd.DataFrame:
        """What each regime's HAR would predict on each day, ignoring regime odds."""
        X = np.column_stack([np.ones(len(har_df)), har_df[list(lags)].values])
        return pd.DataFrame(
            X @ self.coefs.values, index=har_df.index, columns=list(self.labels)
        )

    def forecast(
        self,
        har_df: pd.DataFrame,
        probs: pd.DataFrame,
        lags: tuple[str, ...] = HAR_LAGS,
        floor: float = 1e-6,
    ) -> pd.Series:
        """Probability-weighted blend of the regime-specific predictions.

        ``probs`` should be the *one-step-ahead* regime distribution
        (:meth:`RegimeModel.predicted_probs`) aligned to the same dates, so the
        forecast for day ``t`` uses only information through ``t-1``.
        """
        idx = har_df.index.intersection(probs.index)
        preds = self.regime_predictions(har_df.loc[idx], lags=lags)
        blended = (preds.values * probs.loc[idx, list(self.labels)].values).sum(axis=1)
        return pd.Series(np.clip(blended, floor, None), index=idx, name="ms_har")


# --- forecast evaluation ---------------------------------------------------


def qlike_loss(actual, predicted) -> np.ndarray:
    """Per-observation QLIKE loss on the variance scale.

    QLIKE is asymmetric: it punishes under-prediction of volatility far more
    than over-prediction, which is the right asymmetry for risk work and the
    reason it is preferred over squared error for volatility forecasts.
    """
    a = np.asarray(actual, dtype=float) ** 2
    p = np.asarray(predicted, dtype=float) ** 2
    r = a / p
    return r - np.log(r) - 1


def qlike(actual, predicted) -> float:
    return float(np.mean(qlike_loss(actual, predicted)))


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """Diebold-Mariano test on a loss differential, with the HLN correction.

    A positive statistic means model B (the second argument) has the lower
    average loss. Returns ``(statistic, two-sided p-value)``.
    """
    d = np.asarray(loss_a) - np.asarray(loss_b)
    n = len(d)

    lrv = np.var(d, ddof=0)
    for k in range(1, h):  # Newey-West correction; a no-op at h=1
        lrv += 2 * (1 - k / h) * np.cov(d[:-k], d[k:])[0, 1]

    if lrv <= 0:
        # Identical (or constant-differential) loss series: the test statistic
        # is 0/0. There is no evidence of a difference, so report exactly that
        # rather than propagating a NaN into the results table.
        return 0.0, 1.0

    dm = d.mean() / np.sqrt(lrv / n)
    # Harvey-Leybourne-Newbold small-sample adjustment.
    dm *= np.sqrt((n + 1 - 2 * h + h * (h - 1) / n) / n)
    p = 2 * (1 - stats.t.cdf(abs(dm), df=n - 1))
    return float(dm), float(p)
