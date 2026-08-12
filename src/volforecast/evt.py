"""Conditional EVT: a generalized Pareto tail on volatility-standardized returns.

The two-step logic (McNeil & Frey, 2000): divide returns by a volatility
forecast to strip out the time-varying scale, then fit a generalized Pareto
distribution to the exceedances of the resulting standardized losses over a
high threshold. Extreme value theory says the GPD is the limiting distribution
of those exceedances regardless of the parent distribution — which is why it
succeeds on crash-day tails where a normal or Student-t assumption on raw
returns does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, genpareto

from .config import GPD_THRESHOLD_Q


@dataclass
class GPDTail:
    """A fitted peaks-over-threshold tail on standardized losses."""

    threshold: float
    xi: float  # shape: > 0 means a genuinely heavy (power-law) tail
    beta: float  # scale
    n: int  # total observations
    n_exceed: int  # observations above the threshold

    def var(self, p: float) -> float:
        """Standardized VaR at confidence ``p`` — a multiple of daily volatility."""
        ratio = (self.n / self.n_exceed) * (1 - p)
        if abs(self.xi) < 1e-8:  # Gumbel limit; the general formula is 0/0 here
            return self.threshold + self.beta * (-np.log(ratio))
        return self.threshold + (self.beta / self.xi) * (ratio ** (-self.xi) - 1)

    def es(self, p: float) -> float:
        """Standardized expected shortfall: the mean loss *given* a VaR breach."""
        if self.xi >= 1:
            return float("inf")  # infinite-mean tail; ES is undefined
        v = self.var(p)
        return v / (1 - self.xi) + (self.beta - self.xi * self.threshold) / (1 - self.xi)

    def multipliers(self, levels=(0.99, 0.95)) -> dict[float, tuple[float, float]]:
        return {p: (self.var(p), self.es(p)) for p in levels}


def fit_gpd_tail(z: np.ndarray, threshold_q: float = GPD_THRESHOLD_Q) -> GPDTail:
    """Fit a GPD to the lower tail of standardized returns ``z``.

    Sign convention: losses are ``-z``, so the *lower* tail of returns becomes
    the *upper* tail of losses, which is what peaks-over-threshold models.
    ``floc=0`` pins the GPD location at the threshold, as the theory requires.
    """
    losses = -np.asarray(z, dtype=float)
    losses = losses[np.isfinite(losses)]
    u = float(np.quantile(losses, threshold_q))
    excess = losses[losses > u] - u
    if len(excess) < 20:
        raise ValueError(f"only {len(excess)} exceedances above the threshold; need >= 20")

    xi, _, beta = genpareto.fit(excess, floc=0)
    return GPDTail(threshold=u, xi=float(xi), beta=float(beta), n=len(losses), n_exceed=len(excess))


# --- coverage tests --------------------------------------------------------


def kupiec_pof(violations: np.ndarray, p: float) -> tuple[float, float]:
    """Kupiec unconditional-coverage test: is the violation *rate* right?"""
    v = np.asarray(violations).astype(int)
    n, x = len(v), int(v.sum())
    pi_hat = x / n

    ll_null = (n - x) * np.log(1 - p) + x * np.log(p)
    ll_alt = (n - x) * np.log(1 - pi_hat) if pi_hat < 1 else 0.0
    if x > 0:
        ll_alt += x * np.log(pi_hat)

    lr = -2 * (ll_null - ll_alt)
    return float(lr), float(1 - chi2.cdf(lr, 1))


def christoffersen_cc(violations: np.ndarray, p: float) -> tuple[float, float]:
    """Christoffersen conditional coverage: right rate *and* no clustering.

    A model can post a perfect violation count and still be useless if all the
    breaches arrive in one week — that is exactly the failure a static tail
    assumption produces during a crisis.
    """
    v = np.asarray(violations).astype(int)
    counts = {"00": 0, "01": 0, "10": 0, "11": 0}
    for prev, cur in zip(v[:-1], v[1:], strict=True):
        counts[f"{prev}{cur}"] += 1

    n00, n01, n10, n11 = counts["00"], counts["01"], counts["10"], counts["11"]
    pi01 = n01 / (n00 + n01) if (n00 + n01) else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) else 0.0
    pi = (n01 + n11) / sum(counts.values()) if sum(counts.values()) else 0.0

    def loglik(prob: float, n_zero: int, n_one: int) -> float:
        total = 0.0
        if prob < 1 and n_zero > 0:
            total += n_zero * np.log(1 - prob)
        if prob > 0 and n_one > 0:
            total += n_one * np.log(prob)
        return total

    lr_ind = -2 * (
        loglik(pi, n00 + n10, n01 + n11)
        - loglik(pi01, n00, n01)
        - loglik(pi11, n10, n11)
    )
    lr_uc, _ = kupiec_pof(v, p)
    lr_cc = lr_uc + lr_ind
    return float(lr_cc), float(1 - chi2.cdf(lr_cc, 2))


def backtest_var(
    returns: np.ndarray,
    sigma: np.ndarray,
    tail: GPDTail,
    level: float,
    name: str = "",
) -> dict:
    """Backtest one model at one confidence level.

    ``sigma`` is the daily volatility forecast; the standardized VaR multiplier
    from the fitted tail scales it into a per-day VaR.
    """
    returns, sigma = np.asarray(returns, dtype=float), np.asarray(sigma, dtype=float)
    var_mult, es_mult = tail.var(level), tail.es(level)
    var, es = var_mult * sigma, es_mult * sigma

    tail_prob = 1 - level
    violations = returns < -var
    n_viol = int(violations.sum())

    _, p_uc = kupiec_pof(violations, tail_prob)
    _, p_cc = christoffersen_cc(violations, tail_prob)

    return {
        "model": name,
        "level": level,
        "violations": n_viol,
        "expected": round(len(returns) * tail_prob, 1),
        "rate": n_viol / len(returns),
        "kupiec_p": p_uc,
        "christoffersen_p": p_cc,
        # ES is only checkable on breach days: predicted average loss given a
        # breach, against what actually happened on those days.
        "predicted_es": float(es[violations].mean()) if n_viol else np.nan,
        "realized_es": float(-returns[violations].mean()) if n_viol else np.nan,
        "var_multiplier": var_mult,
        "es_multiplier": es_mult,
    }


def backtest_table(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["rate"] = (df["rate"] * 100).round(2).astype(str) + "%"
    return df.round(4)
