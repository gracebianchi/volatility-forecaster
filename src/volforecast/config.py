"""Central configuration for the volatility forecaster.

Every magic constant that appears in more than one place lives here, so the
research notebook and the live daily pipeline cannot silently disagree about
the sample window, the train/test split, or the regime labels.
"""

from __future__ import annotations

from pathlib import Path

# --- paths -----------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = REPO_ROOT / "models"
DATA_DIR = REPO_ROOT / "data"
FORECAST_DIR = DATA_DIR / "forecasts"
DOCS_DIR = REPO_ROOT / "docs"

# --- sample ----------------------------------------------------------------
TICKER = "SPY"
START = "2015-01-01"

# The research notebook is a frozen artifact: it is pinned to this end date so
# its narrative numbers stay true and it stays reproducible. The live pipeline
# deliberately ignores this and pulls through today.
NOTEBOOK_DATA_END = "2026-08-02"

# Chronological train/test split used throughout the notebook.
SPLIT_DATE = "2023-01-01"

# --- model -----------------------------------------------------------------
N_STATES = 4
# Ordered low -> high volatility. Index i of this tuple is regime i everywhere
# in the codebase; the HMM's own internal state numbering is never exposed.
REGIME_LABELS = ("Calm", "Normal", "Stressed", "Panic")

HAR_LAGS = ("rv_lag1", "rv_lag5", "rv_lag22")
HAR_COLS = ("const",) + HAR_LAGS

# Restarts for Baum-Welch. EM converges only to a local optimum and the
# solution depends on initialization, so every fit takes the best of N seeds.
HMM_RESTARTS = 25
HMM_MAX_ITER = 1000

EWMA_LAMBDA = 0.94

# --- EVT / risk ------------------------------------------------------------
GPD_THRESHOLD_Q = 0.90
VAR_LEVELS = (0.99, 0.95)

TRADING_DAYS = 252
