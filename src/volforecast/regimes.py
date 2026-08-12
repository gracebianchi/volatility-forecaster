"""Gaussian hidden Markov model for latent volatility regimes.

Two things in here matter more than the rest:

1. :meth:`RegimeModel.fit` is meant to be called on the *training* sample only.
   Fitting the HMM on the full sample and then evaluating a regime-switching
   forecast out-of-sample leaks the test period into the transition matrix and
   the regime means. The filtered probabilities are causal given the
   parameters, but the parameters themselves would not be.

2. :meth:`RegimeModel.filtered_probs` runs a proper forward recursion instead
   of calling ``predict_proba`` on every expanding window. The two are
   mathematically identical (the last row of a smoothed posterior has no future
   to smooth over, so it *is* the filtered probability), but the recursion is
   O(n) rather than O(n^2) — which is what makes a daily refit cheap.
"""

from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM

from .config import HMM_MAX_ITER, HMM_RESTARTS, N_STATES, REGIME_LABELS


def default_labels(n_states: int) -> tuple[str, ...]:
    """Names for the regimes, in ascending-volatility order.

    The project's four-state severity ladder is the configured default; any
    other state count gets neutral names rather than a misleading subset of it
    (a two-state model's high regime is a crisis, not a "Normal" one).
    """
    if n_states == len(REGIME_LABELS):
        return REGIME_LABELS
    return tuple(f"State {i + 1}" for i in range(n_states))


def as_observations(rv: pd.Series) -> np.ndarray:
    """Realized vol -> the (n, 1) log-scale observation matrix the HMM expects.

    Volatility is right-skewed and strictly positive; on the log scale the
    regime-conditional distributions are much closer to Gaussian.
    """
    return np.log(rv.dropna().values).reshape(-1, 1)


def fit_hmm(
    X: np.ndarray,
    n_states: int,
    n_restarts: int = HMM_RESTARTS,
    max_iter: int = HMM_MAX_ITER,
) -> GaussianHMM:
    """Best-of-``n_restarts`` Baum-Welch fit, selected by log-likelihood.

    A single fit is unreliable: EM converges to a local optimum that depends on
    the random initialization, and a one-shot fit can land on a solution no
    better than a model with fewer states.
    """
    best, best_ll = None, -np.inf
    for seed in range(n_restarts):
        model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            n_iter=max_iter,
            random_state=seed,
        )
        try:
            model.fit(X)
            ll = model.score(X)
        except Exception:  # a degenerate start can collapse a state's variance
            continue
        if ll > best_ll:
            best, best_ll = model, ll

    if best is None:
        raise RuntimeError(f"no HMM with {n_states} states converged in {n_restarts} restarts")
    return best


def select_n_states(
    X: np.ndarray,
    candidates: tuple[int, ...] = (2, 3, 4, 5, 6),
    n_restarts: int = HMM_RESTARTS,
) -> pd.DataFrame:
    """Information criteria across candidate state counts, for the elbow plot."""
    rows = []
    for k in candidates:
        model = fit_hmm(X, k, n_restarts=n_restarts)
        rows.append(
            {
                "n_states": k,
                "logL": model.score(X),
                "AIC": model.aic(X),
                "BIC": model.bic(X),
                "n_params": k * k + 2 * k - 1,
            }
        )
    return pd.DataFrame(rows).set_index("n_states")


def _emission_logpdf(model: GaussianHMM, X: np.ndarray) -> np.ndarray:
    """(n_obs, n_states) Gaussian log-densities, computed without private APIs."""
    means = model.means_.ravel()
    variances = model.covars_.reshape(model.n_components)
    z = (X - means) ** 2 / variances
    return -0.5 * (np.log(2 * np.pi * variances) + z)


def forward_filter(
    log_b: np.ndarray, startprob: np.ndarray, transmat: np.ndarray
) -> np.ndarray:
    """Scaled forward recursion returning P(state_t | observations 1..t).

    ``log_b`` is the (n_obs, n_states) emission log-density matrix. Each step is
    renormalized, which keeps the recursion numerically stable without needing
    log-sum-exp.
    """
    n_obs, n_states = log_b.shape
    alpha = np.zeros((n_obs, n_states))

    # Subtract the row max before exponentiating: the scale cancels in the
    # normalization, so this is exact, not an approximation.
    b = np.exp(log_b - log_b.max(axis=1, keepdims=True))

    a = startprob * b[0]
    alpha[0] = a / a.sum()
    for t in range(1, n_obs):
        a = (alpha[t - 1] @ transmat) * b[t]
        alpha[t] = a / a.sum()

    return alpha


@dataclass
class RegimeModel:
    """A fitted HMM plus the severity ordering that makes its states meaningful.

    hmmlearn numbers its states arbitrarily. Everything this class exposes is
    reindexed into ascending-volatility order, so column ``i`` is always
    ``REGIME_LABELS[i]`` and the internal numbering never escapes.
    """

    hmm: GaussianHMM
    order: np.ndarray
    labels: tuple[str, ...]
    fitted_through: pd.Timestamp | None = None

    # -- construction -------------------------------------------------------
    @classmethod
    def fit(
        cls,
        rv: pd.Series,
        n_states: int = N_STATES,
        n_restarts: int = HMM_RESTARTS,
        labels: tuple[str, ...] | None = None,
    ) -> RegimeModel:
        """Fit on ``rv``. Pass the *training* slice only when evaluating OOS."""
        labels = labels if labels is not None else default_labels(n_states)
        if len(labels) != n_states:
            raise ValueError(f"{n_states} states but {len(labels)} labels")
        rv = rv.dropna()
        model = fit_hmm(as_observations(rv), n_states, n_restarts=n_restarts)
        order = np.argsort(model.means_.ravel())
        return cls(hmm=model, order=order, labels=labels, fitted_through=rv.index[-1])

    # -- fitted parameters, in severity order -------------------------------
    @property
    def n_states(self) -> int:
        return self.hmm.n_components

    @property
    def mean_vols(self) -> pd.Series:
        """Each regime's mean annualized volatility, back-transformed from log."""
        return pd.Series(
            np.exp(self.hmm.means_.ravel()[self.order]), index=list(self.labels)
        )

    @property
    def transition_matrix(self) -> pd.DataFrame:
        """Rows = regime today, columns = regime tomorrow."""
        A = self.hmm.transmat_[np.ix_(self.order, self.order)]
        return pd.DataFrame(A, index=list(self.labels), columns=list(self.labels))

    @property
    def expected_duration(self) -> pd.Series:
        """1 / (1 - p_ii): mean number of consecutive days spent in each regime."""
        p_stay = np.diag(self.transition_matrix.values)
        return pd.Series(1.0 / (1.0 - p_stay), index=list(self.labels))

    # -- inference ----------------------------------------------------------
    def _frame(self, probs: np.ndarray, index: pd.Index) -> pd.DataFrame:
        return pd.DataFrame(probs[:, self.order], index=index, columns=list(self.labels))

    def smoothed_probs(self, rv: pd.Series) -> pd.DataFrame:
        """P(state_t | the whole sample). Descriptive use only — not causal."""
        rv = rv.dropna()
        return self._frame(self.hmm.predict_proba(as_observations(rv)), rv.index)

    def filtered_probs(self, rv: pd.Series) -> pd.DataFrame:
        """P(state_t | observations up to and including t). Safe for forecasting."""
        rv = rv.dropna()
        X = as_observations(rv)
        alpha = forward_filter(
            _emission_logpdf(self.hmm, X), self.hmm.startprob_, self.hmm.transmat_
        )
        return self._frame(alpha, rv.index)

    def predicted_probs(self, rv: pd.Series) -> pd.DataFrame:
        """One-step-ahead regime distribution, indexed by the day being forecast.

        Row ``d`` is P(regime on day ``d`` | observations through ``d-1``): the
        filtered belief formed on the previous day, propagated forward through
        the transition matrix.

        The indexing convention is deliberate and load-bearing. Labelling these
        rows by the day the belief was *formed* rather than the day being
        *forecast* makes them align, one day too late, with any design matrix
        whose row ``d`` targets day ``d`` — silently leaking day ``d`` into its
        own forecast. Indexing by the target day makes ``.loc``-alignment
        correct by construction.
        """
        filt = self.filtered_probs(rv)
        A = self.hmm.transmat_[np.ix_(self.order, self.order)]
        return pd.DataFrame(
            filt.values[:-1] @ A, index=filt.index[1:], columns=list(self.labels)
        )

    def next_regime_probs(self, rv: pd.Series) -> pd.Series:
        """P(regime on the next session | everything observed so far).

        The live counterpart to :meth:`predicted_probs`, whose target day has
        no calendar date yet.
        """
        filt = self.filtered_probs(rv).iloc[-1].values
        A = self.hmm.transmat_[np.ix_(self.order, self.order)]
        return pd.Series(filt @ A, index=list(self.labels), name="next_regime_probs")

    def hard_states(self, rv: pd.Series, causal: bool = False) -> pd.Series:
        """Most likely regime per day, as labels."""
        probs = self.filtered_probs(rv) if causal else self.smoothed_probs(rv)
        return pd.Series(probs.values.argmax(axis=1), index=probs.index).map(
            dict(enumerate(self.labels))
        ).rename("regime")

    def stress_index(self, probs: pd.DataFrame) -> pd.Series:
        """Severity-weighted collapse of the regime distribution onto [0, 1].

        Calm = 0, Panic = 1, evenly spaced in between. A continuous read on
        where the market sits on the calm-to-panic spectrum, which the discrete
        state label throws away.
        """
        weights = np.linspace(0.0, 1.0, self.n_states)
        return pd.Series(probs.values @ weights, index=probs.index, name="stress_index")

    # -- persistence --------------------------------------------------------
    def save(self, path) -> None:
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> RegimeModel:
        return joblib.load(path)
