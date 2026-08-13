# SPY Volatility Forecasting — Markov Regime-Switching

Does modelling the market as switching between distinct volatility regimes improve
on standard single-regime volatility forecasts?

This repository answers that question twice: once as a **research notebook** pinned
to a frozen sample, and once as a **live forecaster** that reruns every weekday after
the US close and publishes its results.

📈 **[Live dashboard](https://gracebianchi.github.io/volatility-forecaster/)** — current
regime, stress index, 10-day forecast cone, and EVT tail risk.

📓 **[Research notebook](notebooks/vol_forecaster.ipynb)** — the full study.

---

## What it does

**Stage 1 — baselines.** EWMA, GARCH(1,1) and HAR-RV are compared out-of-sample on a
chronological split (train 2015–2022, test 2023 onward). HAR-RV wins. A machine-learning
extension — linear regression, random forest, gradient boosting on an expanded feature
set — fails to beat it, and a cross-validated stepwise selection independently recovers
the plain HAR specification from the expanded features.

**Stage 2 — regimes.** A Gaussian hidden Markov model on log realized volatility
recovers four regimes (Calm / Normal / Stressed / Panic) with an ordered-ladder
transition structure: the market escalates and de-escalates one rung at a time, so acute
panic is always bracketed by a stressed phase. From that:

- a **regime-switching HAR** that blends four regime-specific regressions by the
  model's one-step-ahead regime probabilities;
- a continuous **stress index** on [0, 1] as an early-warning signal;
- **Monte Carlo** predictive distributions whose width depends on the launch regime;
- **conditional EVT** — a generalized Pareto tail on volatility-standardized returns —
  producing VaR and expected shortfall that pass Kupiec and Christoffersen backtests.

The headline result is deliberately not a victory lap: on point-forecast accuracy the
regime model is statistically indistinguishable from a plain HAR. Its contribution is
the conditional *distribution*, not a sharper central number.

## Layout

```
src/volforecast/       the model, as an importable package
  data.py              SPY download + Garman-Klass realized volatility
  features.py          HAR design matrix and the extended ML feature set
  baselines.py         EWMA, GARCH(1,1), HAR-RV
  regimes.py           HMM fitting, state ordering, causal forward filter
  ms_har.py            regime-specific HAR, QLIKE, Diebold-Mariano
  simulate.py          Monte Carlo path simulation
  evt.py               GPD tail, VaR/ES, Kupiec & Christoffersen tests
  pipeline.py          fit everything / evaluate out-of-sample
  daily.py             the scheduled forecast job
notebooks/             the research write-up (frozen sample)
tests/                 offline test suite on synthetic data
docs/                  GitHub Pages dashboard
data/forecasts/        rolling history of live forecasts and their outcomes
```

The notebook imports from the package rather than defining models inline, so the
research write-up and the live forecaster cannot drift apart.

## Running it

```bash
python -m venv venv && source venv/bin/activate
pip install -e ".[dev,notebook]"
```

Produce a forecast for the next session:

```bash
vol-forecast
```

This writes `data/forecasts/latest.json`, appends a row to
`data/forecasts/history.csv`, backfills the outcome of every earlier forecast, and
mirrors both into `docs/data/` for the dashboard. It is safe to run repeatedly —
on a day with no new bar it only backfills.

Force a refit (otherwise the stored model is reused until it is 30 days old):

```bash
vol-forecast --refit
```

Run the tests:

```bash
pytest
```

The suite is offline by design — it runs against synthetic data with a known
regime structure, so CI never depends on Yahoo Finance being up.

## Automation

- `.github/workflows/ci.yml` — lint and test on every push and pull request.
- `.github/workflows/daily-forecast.yml` — runs at 22:30 UTC on weekdays (after the
  US close in both EST and EDT), commits the new forecast, and publishes the dashboard.
- `.github/workflows/pages.yml` — republishes the dashboard when a human edits `docs/`.

Pages is served by **Settings → Pages → Source: GitHub Actions**, not from a branch.
That distinction matters: with branch-based Pages, the site rebuild is itself a workflow
triggered by the push, and a push made with `GITHUB_TOKEN` deliberately does not trigger
workflows — so the daily commit would land in the repo without the site ever updating.
The daily job therefore publishes inline, in the same run that produced the data.

Two GitHub behaviours are worth knowing about:

- **Scheduled runs are not punctual.** GitHub queues `schedule` events on shared
  capacity and can delay them by tens of minutes, or drop one entirely under load. A
  missed day leaves a gap in the history rather than corrupting it, and the next run
  backfills the outcome of every earlier forecast regardless.
- **Scheduled workflows are disabled after 60 days without repository activity.**
  GitHub emails first. Any commit resets the clock.

## Methodology notes

A few decisions that a reader should not have to reverse-engineer:

**Realized volatility** uses the Garman-Klass estimator, which exploits the intraday
high/low range and so extracts more information per day than squared close-to-close
returns. It omits overnight moves — a known limitation.

**Regime parameters are estimated on the training window only.** Fitting the HMM on the
full sample and then evaluating a regime-switching forecast out-of-sample leaks the test
period into the transition matrix and the regime means. The notebook keeps a full-sample
fit for *descriptive* work (the regime map, the stress index) and refits on the training
window for everything used to forecast.

**Regime probabilities are filtered, not smoothed.** A smoothed posterior at day *t*
conditions on the whole sample, including the future. The forecast weights come from a
forward recursion — P(regime *t* | data through *t*) — propagated one step through the
transition matrix, and are indexed by the day being forecast so that aligning them with
the design matrix is causally correct by construction.

**The EVT tail is fit and applied symmetrically.** Both models standardize training
returns by the same kind of forecast they are scored with out of sample.

**A generalized Pareto tail, not normal or Student-t.** Extreme value theory makes the
GPD the limiting distribution of threshold exceedances regardless of the parent
distribution, which is why it calibrates on crash days where a parametric assumption on
raw returns does not.

## Limitations

The out-of-sample window contains one major crisis (the April 2025 tariff selloff),
which limits the statistical power of exactly the comparisons where a regime model
should matter most. The regime-switching HAR is a two-step, regime-conditional model
rather than a jointly estimated one. And the whole thing is single-asset: a multivariate
detector fed cross-asset correlations and credit spreads could separate systemic from
idiosyncratic stress in a way this cannot.


