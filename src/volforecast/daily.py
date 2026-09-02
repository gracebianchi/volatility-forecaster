"""Daily forecast job.

Pulls the latest SPY bar, produces a one-day-ahead forecast, appends it to a
running history, and backfills the outcome of every past forecast so the model
is scored on live data rather than only on the 2023-2026 backtest.

Designed to be safe to run repeatedly: if there is no new trading day since the
last recorded forecast the job exits without touching anything, so a cron that
fires on holidays produces no commit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DOCS_DIR, FORECAST_DIR, MODELS_DIR, REGIME_LABELS
from .data import load_spy_data
from .pipeline import VolForecaster, fit_forecaster, forecast_next_day

HISTORY_COLUMNS = [
    "as_of",
    "target_date",
    "forecast_rv",
    "forecast_rv_single_har",
    "daily_sigma",
    "stress_index",
    "regime",
    *[f"p_{label.lower()}" for label in REGIME_LABELS],
    "var99",
    "es99",
    "var95",
    "es95",
    "trained_through",
    "realized_rv",
    "realized_return",
    "breach99",
    "breach95",
]


# Everything in the history except these is a number.
_TEXT_COLUMNS = ("as_of", "regime", "trained_through")


def normalize(history: pd.DataFrame) -> pd.DataFrame:
    """Coerce numeric columns to float64 so repeated writes are byte-identical.

    A row appended in-process arrives in an object-dtype column and ``to_csv``
    formats it with ``repr`` (17 significant digits); read back, the column is
    float64 and ``to_csv`` uses 16. The value then churns in the last digit on
    the next write, producing a daily commit whose entire diff is trailing
    noise. Normalizing before every write makes the formatter agree with
    itself, and 16-digit formatting is idempotent once applied.
    """
    history = history.copy()
    for column in history.columns:
        if column not in _TEXT_COLUMNS:
            history[column] = pd.to_numeric(history[column], errors="coerce")
    return history


def _model_is_stale(path: Path, max_age_days: int) -> bool:
    if not path.exists():
        return True
    age = datetime.now(UTC) - datetime.fromtimestamp(
        path.stat().st_mtime, tz=UTC
    )
    return age > timedelta(days=max_age_days)


def load_or_fit(
    spy: pd.DataFrame,
    returns: pd.Series,
    model_path: Path,
    max_age_days: int,
    force_refit: bool,
) -> tuple[VolForecaster, bool]:
    """Reuse the stored model unless it is missing, stale, or a refit is forced.

    Refitting every single day would make the forecast history incomparable
    over time — the regime definitions themselves would drift underneath it.
    Refitting on a slow cadence keeps the series interpretable while still
    letting the model absorb new data.
    """
    if force_refit or _model_is_stale(model_path, max_age_days):
        forecaster = fit_forecaster(spy, returns)
        forecaster.save(model_path)
        return forecaster, True

    return VolForecaster.load(model_path), False


def _row_from_forecast(fc: dict) -> dict:
    row = {
        "as_of": fc["as_of"],
        "forecast_rv": fc["forecast_rv"],
        "forecast_rv_single_har": fc["forecast_rv_single_har"],
        "daily_sigma": fc["daily_sigma"],
        "stress_index": fc["stress_index"],
        "regime": fc["regime_today"],
        "var99": fc["risk"]["99"]["var"],
        "es99": fc["risk"]["99"]["es"],
        "var95": fc["risk"]["95"]["var"],
        "es95": fc["risk"]["95"]["es"],
        "trained_through": fc["trained_through"],
        "realized_rv": np.nan,
        "realized_return": np.nan,
        "breach99": np.nan,
        "breach95": np.nan,
    }
    for label in REGIME_LABELS:
        row[f"p_{label.lower()}"] = fc["regime_probs_tomorrow"][label]
    return row


def backfill_outcomes(
    history: pd.DataFrame, spy: pd.DataFrame, returns: pd.Series
) -> pd.DataFrame:
    """Fill in what actually happened on the day after each stored forecast.

    A forecast made with data through ``as_of`` predicts the *next* trading
    session, so the outcome is the first available bar strictly after that date.
    """
    rv = spy["realized_vol"].dropna()
    history = history.copy()

    for i, row in history.iterrows():
        if not pd.isna(row.get("realized_rv")):
            continue

        as_of = pd.Timestamp(row["as_of"])
        future = rv.index[rv.index > as_of]
        if len(future) == 0:
            continue  # the forecast day has not happened yet

        target = future[0]
        history.at[i, "target_date"] = target.strftime("%Y-%m-%d")
        history.at[i, "realized_rv"] = float(rv.loc[target])
        if target in returns.index:
            r = float(returns.loc[target])
            history.at[i, "realized_return"] = r
            history.at[i, "breach99"] = int(r < -row["var99"])
            history.at[i, "breach95"] = int(r < -row["var95"])

    return history


def load_history(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=HISTORY_COLUMNS)
    return pd.read_csv(path)


def live_scorecard(history: pd.DataFrame) -> dict:
    """Out-of-sample performance on forecasts actually made in production."""
    scored = history.dropna(subset=["realized_rv"])
    if scored.empty:
        return {"n": 0}

    err_ms = scored["realized_rv"] - scored["forecast_rv"]
    err_single = scored["realized_rv"] - scored["forecast_rv_single_har"]
    breaches = scored["breach99"].dropna()

    return {
        "n": int(len(scored)),
        "rmse_ms_har": float(np.sqrt((err_ms**2).mean())),
        "rmse_single_har": float(np.sqrt((err_single**2).mean())),
        "bias_ms_har": float(err_ms.mean()),
        "breaches_99": int(breaches.sum()) if len(breaches) else 0,
        "expected_breaches_99": round(len(breaches) * 0.01, 2),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Produce the daily volatility forecast")
    parser.add_argument("--refit", action="store_true", help="refit the model before forecasting")
    parser.add_argument(
        "--max-model-age-days",
        type=int,
        default=30,
        help="refit automatically once the stored model is older than this",
    )
    parser.add_argument("--horizon", type=int, default=10, help="simulation horizon in days")
    parser.add_argument("--n-paths", type=int, default=3000)
    parser.add_argument(
        "--force", action="store_true", help="rewrite today's row even if it already exists"
    )
    parser.add_argument("--output-dir", type=Path, default=FORECAST_DIR)
    args = parser.parse_args(argv)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history_path = args.output_dir / "history.csv"
    latest_path = args.output_dir / "latest.json"

    spy, returns = load_spy_data()
    history = load_history(history_path)

    as_of = spy["realized_vol"].dropna().index[-1].strftime("%Y-%m-%d")
    already_recorded = as_of in set(history.get("as_of", pd.Series(dtype=str)).astype(str))

    if already_recorded and not args.force:
        # Still worth writing: yesterday's outcome may only now be observable.
        history = normalize(backfill_outcomes(history, spy, returns))
        history.to_csv(history_path, index=False)
        _write_dashboard_data(latest_path, history)
        print(f"no new trading day since {as_of}; backfilled outcomes only")
        return 0

    forecaster, refit = load_or_fit(
        spy, returns, MODELS_DIR / "forecaster.joblib", args.max_model_age_days, args.refit
    )
    if refit:
        print(f"refit model on data through {forecaster.trained_through.date()}")

    fc = forecast_next_day(
        forecaster, spy, horizon=args.horizon, n_paths=args.n_paths
    )

    new_row = pd.DataFrame([_row_from_forecast(fc)])
    if len(history):
        history = history[history["as_of"].astype(str) != as_of]
        history = pd.concat([history, new_row], ignore_index=True)
    else:
        history = new_row
    history = history.reindex(columns=HISTORY_COLUMNS).sort_values("as_of", ignore_index=True)
    history = normalize(backfill_outcomes(history, spy, returns))
    history.to_csv(history_path, index=False)

    fc["scorecard"] = live_scorecard(history)
    latest_path.write_text(json.dumps(fc, indent=2))
    _write_dashboard_data(latest_path, history, fc)

    print(
        f"{fc['as_of']}: regime={fc['regime_today']} "
        f"stress={fc['stress_index']:.3f} forecast_rv={fc['forecast_rv']:.2f}% "
        f"VaR99={fc['risk']['99']['var']:.4f}"
    )
    return 0


def _write_dashboard_data(
    latest_path: Path, history: pd.DataFrame, fc: dict | None = None
) -> None:
    """Mirror the outputs into docs/ so GitHub Pages serves them statically."""
    data_dir = DOCS_DIR / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    if fc is None and latest_path.exists():
        fc = json.loads(latest_path.read_text())
    if fc is None:
        return

    fc = {**fc, "scorecard": live_scorecard(history)}
    (data_dir / "latest.json").write_text(json.dumps(fc, indent=2))

    trimmed = history.tail(400).replace({np.nan: None})
    (data_dir / "history.json").write_text(
        json.dumps(trimmed.to_dict(orient="records"), indent=2)
    )


if __name__ == "__main__":
    sys.exit(main())
