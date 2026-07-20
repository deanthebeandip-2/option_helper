"""
Stock Monte Carlo Simulator — State-Conditional Forecasting Engine
====================================================================
Replaces the old "Monday regime" model with a general state-based
conditional Monte Carlo engine that works on any trading day.

Core idea:
  - The trading week is a repeating 5-transition cycle:
        Fri->Mon (weekend gap), Mon->Tue, Tue->Wed, Wed->Thu, Thu->Fri
  - Each transition is classified Bear/Flat/Bull using thresholds that
    are optimized INDEPENDENTLY per transition type (Fri->Mon behaves
    differently than Tue->Wed, etc.)
  - The "market state" = (regime of the 2nd-most-recent transition,
    regime of the most recent transition) -> 9 possible states.
  - Monte Carlo paths are simulated as a genuine Markov chain: each
    simulated day, every path samples its next move from historical
    observations that occurred in ITS current state, then the path's
    state updates based on what was sampled. Paths diverge over time.
  - Supports rolling forecast horizons +1 through +5 trading days,
    labeled with actual calendar dates, from whatever the most recent
    close is, on any day of the week.

Install deps:
    pip install yfinance pandas numpy matplotlib scipy

Usage:
    python stock_monte_carlo.py
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from collections import defaultdict
import json
import os
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS        = ["SOFI"]
#TICKERS        = ["SOFI", "CRWV", "NVDA", "WULF", "HOOD"]
LOOKBACK_YEARS = 10
N_SIMULATIONS  = 1_000_000
#N_SIMULATIONS  = 10_000
CONFIDENCE     = [0.05, 0.25, 0.50, 0.75, 0.95]
FORECAST_HORIZONS = [1, 2, 3, 4, 5]     # rolling forecast horizons, in trading days


# ─────────────────────────────────────────────
# REPORT CONFIGURATION
# ─────────────────────────────────────────────

REPORT = {
    # Console reports
    "spread_table": 0,          # don't need
    "weekly_summary": 1,        # useful, still Mon open -> Fri close (unconditional)
    "simulation_summary": True, # unconditional weekly Monte Carlo
    "options_grid": True,       # used by both weekly and state-forecast sections
    "state_forecast": True,     # state-conditional rolling forecast engine
    "state_diagnostics": True,  # print the 9-state occurrence table

    # Graphics
    "charts": False,            # unconditional weekly chart only

    # Future modules
    "backtesting": False,
    "live_analysis": False,

    # Calibration backtest (runs after the normal per-ticker report)
    "calibration_backtest": True,
}

# The 5-transition weekly cycle. Order matters — it defines both the
# unconditional weekly chain AND the state-machine cycle.
DAY_PAIRS = [
    ("Friday",    "Monday"),    # weekend gap (crosses weeks)
    ("Monday",    "Tuesday"),
    ("Tuesday",   "Wednesday"),
    ("Wednesday", "Thursday"),
    ("Thursday",  "Friday"),
]
DAY_NAMES = ["Fri*", "Mon", "Tue", "Wed", "Thu", "Fri"]   # for the weekly chain (6 points)

TRANSITION_TYPE_LABELS = [f"{f[:3]}->{t[:3]}" for f, t in DAY_PAIRS]   # Fri->Mon, Mon->Tue, ...
TRANSITION_TYPE_INDEX  = {pair: i for i, pair in enumerate(DAY_PAIRS)}
N_TYPES = len(DAY_PAIRS)

# ── State-Conditional Engine settings ────────────────────────────────────
REGIME_LABELS       = ["bear", "flat", "bull"]
DEFAULT_BOUNDARIES  = [-1.0, 1.0]      # fallback if a type can't be optimized

OPTIMIZE_THRESHOLDS      = True
SAVE_OPTIMIZED_THRESHOLDS = True
THRESHOLDS_FILE          = "state_regime_thresholds.json"

MIN_BUCKET_SIZE   = 15   # min occurrences per bear/flat/bull bucket for a threshold candidate to be valid
MIN_STATE_SAMPLES = 20   # min (state, next_type) pool size before falling back to the type's unconditional pool

# Threshold search grid (per transition type, searched independently)
BEAR_GRID = np.arange(-5.0, -0.24, 0.25)   # -5.00 ... -0.25
BULL_GRID = np.arange(0.25, 5.01, 0.25)    # +0.25 ... +5.00

# ── Threshold Optimizer scoring (evaluator-driven, not raw-error-driven) ──
# Philosophy change: thresholds are no longer picked by whichever (lo, hi)
# minimizes prediction error. Each candidate (lo, hi) is scored by
# evaluate_threshold_candidate() on three axes — calibration, directional
# assignment accuracy, and 1-day forecast error — combined into one Overall
# Score, and optimize_type_thresholds() keeps whichever candidate maximizes
# that score. See evaluate_threshold_candidate()'s docstring for the exact
# method and its cost tradeoff vs. a full Monte-Carlo walk-forward per
# candidate.
SCORE_WEIGHTS = {
    "calibration": 0.50,
    "assignment":  0.30,
    "mae":         0.20,
}
CALIBRATION_TARGET_CI      = 0.90   # the CI level the calibration term is scored against
CALIBRATION_TRAIN_FRACTION = 0.6    # chronological train/test split within each bucket

# ── Optimizer mode ────────────────────────────────────────────────────────
# "full": THE redesign. The Threshold Optimizer calls the real Monte Carlo
#         engine hundreds of times (once per candidate, walked forward through
#         history) — thresholds are chosen because they make the ENGINE's
#         forecasts best, exactly per the "Threshold Optimizer -> calls ->
#         Evaluator" architecture. Slow (minutes), faithful to the design.
# "fast": the lightweight proxy — scores each candidate against the type's
#         historical (pct, next_pct) pool directly, no Monte Carlo re-run.
#         Seconds, not minutes. Useful as a quick sanity check or fallback.
OPTIMIZER_MODE = "full"

# Cost/precision knobs for "full" mode. Total evaluator calls ~=
# coord_ascent_passes x N_TYPES x len(bear_grid) x len(bull_grid), and each
# call itself walks forward through roughly
# (n_trading_days - min_history_days) / stride historical points, running a
# real (small) Monte Carlo simulation at each one. Tune these down for a
# quick pass, up for a more precise (much slower) one.
FULL_EVALUATOR_CONFIG = {
    "n_sims":              400,                          # sims per walk-forward day inside the search (<< live N_SIMULATIONS)
    "stride":              25,                            # historical days skipped between walk-forward evals
    "min_history_days":    400,                           # need this much history before the first eval point
    "horizons":            [1, 3, 5],                     # subset of FORECAST_HORIZONS scored inside the search (calibration)
    "confidence_levels":   [0.50, 0.75, 0.90, 0.95],       # unused directly by scoring (single CALIBRATION_TARGET_CI drives
                                                            # the score) but recorded for future use / diagnostics
    "coord_ascent_passes": 1,                              # full passes over all 5 types; raise to 2-3 to let types converge
    "bear_grid":           np.arange(-4.0, -0.49, 0.75),   # coarser than the fast-mode grid — each point is expensive here
    "bull_grid":           np.arange(0.5, 4.01, 0.75),
}

# ── Options grid display settings ────────────────────────────────────────
MIN_DISPLAY_PROB = 1.99   # rows with prob <= this (in %) are cut off
MAX_STRIKE_PCT   = 60      # safety cap so a degenerate distribution can't loop forever

# ── Calibration backtest settings ────────────────────────────────────────
# Walks back through history, pretends each sampled day is "today", rebuilds
# the model from ONLY the data available as of that day, runs the normal
# +1..+5 state-conditional forecast, and checks whether the actual future
# close fell inside the predicted confidence interval. This never touches
# the live forecasting logic above — it just calls the same functions with
# a truncated dataframe.
BACKTEST_CONFIG = {
    "n_sims":           2_000,   # smaller than live N_SIMULATIONS — this runs hundreds of times
    "stride":           5,       # evaluate every Nth trading day (1 = every day, slow)
    "min_history_days": 500,     # need enough history before the first eval day
    "confidence_levels": [0.50, 0.75, 0.90, 0.95],
    "assessment_ci":    0.90,    # which CI drives the pass/fail "Overall Assessment" line
    "tolerance_pts":    3.0,     # +/- percentage points considered "well calibrated"
}

# Palette
C_BEAR   = "#d73027"
C_BULL   = "#313695"
C_MEDIAN = "#1a1a2e"
C_MEAN   = "#e63946"
C_BLUE   = "#4C9BE8"
C_GREEN  = "#2A9D8F"
C_ORANGE = "#F4A261"


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────
def fetch_data(ticker: str, years: int = LOOKBACK_YEARS) -> pd.DataFrame:
    end   = datetime.today()
    start = end - timedelta(days=365 * years)
    df    = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df[["Open", "Close"]].copy()
    df["Open"]  = df["Open"].squeeze()
    df["Close"] = df["Close"].squeeze()
    df.index    = pd.to_datetime(df.index)
    df["DayName"] = df.index.day_name()
    return df.dropna().sort_index()


# ─────────────────────────────────────────────
# TRADING-DATE HELPER  (for forecast labels)
# ─────────────────────────────────────────────
def future_trading_date(last_date, n_days: int):
    """
    Returns the calendar date N *trading* days after last_date, skipping
    weekends via numpy's business-day calendar. No holiday calendar is
    applied (same limitation as the simulation's Mon-Tue-Wed-Thu-Fri
    cycle itself, which also doesn't account for market holidays) — so
    around holidays the label may be off by a day or two versus the
    exchange's actual calendar.
    """
    if n_days == 0:
        return pd.Timestamp(last_date).date()
    base = np.datetime64(pd.Timestamp(last_date).date(), 'D')
    result = np.busday_offset(base, n_days, roll='forward')
    return pd.Timestamp(result).date()


def format_date_short(d) -> str:
    return f"{d.month}/{d.day}"


# ─────────────────────────────────────────────
# WEEKEND GAP SPREAD  (Fri Close -> Mon Open, crosses weeks)
# — used only by the unconditional weekly chain below
# ─────────────────────────────────────────────
def compute_weekend_spread(df: pd.DataFrame) -> np.ndarray:
    df_sorted  = df.sort_index()
    prev_close = df_sorted["Close"].shift(1)
    prev_day   = df_sorted["DayName"].shift(1)
    mask = (df_sorted["DayName"] == "Monday") & (prev_day == "Friday")
    mon_open  = df_sorted.loc[mask, "Open"]
    fri_close = prev_close[mask]
    pct_chg = ((mon_open - fri_close) / fri_close) * 100
    return np.array(pct_chg).flatten()


# ─────────────────────────────────────────────
# DAY-TO-DAY SPREAD DISTRIBUTIONS (unconditional, week-keyed)
# ─────────────────────────────────────────────
def compute_dow_spreads(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ISOWeek"] = df.index.isocalendar().week.values
    df["ISOYear"] = df.index.isocalendar().year.values
    df["WeekKey"] = df["ISOYear"].astype(str) + "-" + df["ISOWeek"].astype(str)

    spreads = {}
    for from_day, to_day in DAY_PAIRS:
        if from_day == "Friday" and to_day == "Monday":
            spreads[(from_day, to_day)] = compute_weekend_spread(df)
            continue
        if from_day == "Monday":
            from_rows = (df[df["DayName"] == from_day][["Open", "WeekKey"]]
                         .rename(columns={"Open": "From"}))
        else:
            from_rows = (df[df["DayName"] == from_day][["Close", "WeekKey"]]
                         .rename(columns={"Close": "From"}))
        to_rows = (df[df["DayName"] == to_day][["Close", "WeekKey"]]
                   .rename(columns={"Close": "To"}))
        merged  = pd.merge(from_rows, to_rows, on="WeekKey")
        pct_chg = ((merged["To"] - merged["From"]) / merged["From"]) * 100
        spreads[(from_day, to_day)] = np.array(pct_chg).flatten()
    return spreads


# ─────────────────────────────────────────────
# TRANSITION SEQUENCE  (continuous, chronological, holiday-safe)
# — this is the backbone of the state-conditional engine
# ─────────────────────────────────────────────
def build_transition_sequence(df: pd.DataFrame) -> list:
    """
    Walks the dataframe day by day and emits one entry per valid transition
    (i.e. where consecutive trading days match one of the 5 canonical weekly
    pairs). Holiday-shortened weeks that break a pair (e.g. Monday directly
    followed by Wednesday because Tuesday was a holiday) are silently
    skipped for that specific transition — the sequence just has a gap
    there, which is fine since the state machine only looks at the most
    recent two entries in the list, not the calendar.
    """
    df = df.sort_index()
    n  = len(df)
    transitions = []
    for i in range(1, n):
        prev_day = df["DayName"].iloc[i - 1]
        curr_day = df["DayName"].iloc[i]
        pair = (prev_day, curr_day)
        if pair not in TRANSITION_TYPE_INDEX:
            continue
        type_idx = TRANSITION_TYPE_INDEX[pair]
        from_price = df["Open"].iloc[i - 1] if prev_day == "Monday" else df["Close"].iloc[i - 1]
        to_price   = df["Open"].iloc[i]     if curr_day == "Monday" else df["Close"].iloc[i]
        pct = float((to_price - from_price) / from_price * 100)
        transitions.append({
            "type_idx":   type_idx,
            "type_label": TRANSITION_TYPE_LABELS[type_idx],
            "pct":        pct,
            "date":       df.index[i],
        })
    return transitions


# ─────────────────────────────────────────────
# REGIME CLASSIFICATION
# ─────────────────────────────────────────────
def classify_regime(pct: float, lo: float, hi: float) -> str:
    if pct < lo:
        return "bear"
    elif pct > hi:
        return "bull"
    return "flat"


# ─────────────────────────────────────────────
# PER-TYPE THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────
def build_type_occurrences(transitions: list, type_idx: int) -> pd.DataFrame:
    """
    For a given transition type, returns a DataFrame of (pct, next_pct)
    pairs — next_pct being whatever transition immediately follows it in
    the sequence (any type). This is the training data used to test
    whether a candidate regime split on THIS type's pct is predictive of
    what happens right after.
    """
    rows = []
    for i, t in enumerate(transitions):
        if t["type_idx"] != type_idx:
            continue
        if i + 1 >= len(transitions):
            continue
        rows.append({"pct": t["pct"], "next_pct": transitions[i + 1]["pct"]})
    return pd.DataFrame(rows)


def evaluate_threshold_candidate(occ_df: pd.DataFrame, lo: float, hi: float) -> dict | None:
    """
    THE EVALUATOR. Given a candidate (lo, hi) threshold pair, returns a
    dict with an "overall_score" (0-100). optimize_type_thresholds() picks
    the candidate that maximizes this score — thresholds are chosen because
    they make the forecasting engine perform best, not because the split
    itself "looks like" a good bear/bull boundary.

    Three sub-scores (each 0-1), combined via SCORE_WEIGHTS:

      - calibration_score: buckets each row bear/flat/bull under (lo, hi),
        then for each bucket does a chronological train/test split
        (CALIBRATION_TRAIN_FRACTION) — builds the CALIBRATION_TARGET_CI
        interval (e.g. 90% -> 5th/95th percentile) from the train slice
        only, and checks how often the test slice's actual next_pct falls
        inside it. Score = 1 - |observed - target| / target.

      - assignment_score: does the bucket's out-of-sample (leave-one-out)
        predicted direction match the actual next-day direction? This is a
        proxy for "did the regime call point the right way" — not a full
        state-machine assignment accuracy (that would require re-running
        the 9-state Monte Carlo per candidate; see note below).

      - mae_score: 1-day leave-one-out MAE, normalized against a naive
        "always predict the bucket-free overall mean" baseline, so 1.0 =
        perfect, 0.0 = no better than guessing the mean.

    COST NOTE: this scores each candidate against the per-type historical
    pool directly (same O(n) cost class as the old RMSE-only backtest —
    fast enough for a ~400-point grid x 5 transition types). It does NOT
    re-run the full state-conditional Monte Carlo engine per candidate,
    which the literal "run 1,200 historical MC backtests per candidate"
    design would require — that's roughly 2,000 full walk-forward runs and
    would take hours rather than seconds. If you want that heavier, more
    literal version later, run_calibration_backtest()/evaluate_calibration_for_ticker()
    (added earlier) is the piece to call from inside this function instead.
    """
    if occ_df.empty:
        return None
    df2 = occ_df.copy()
    df2["bucket"] = df2["pct"].apply(lambda p: classify_regime(p, lo, hi))

    counts = df2["bucket"].value_counts()
    for lbl in REGIME_LABELS:
        if counts.get(lbl, 0) < MIN_BUCKET_SIZE:
            return None

    # ── Leave-one-out MAE/RMSE (unchanged from before) ──────────────────
    grp        = df2.groupby("bucket")["next_pct"]
    bucket_sum = grp.transform("sum")
    bucket_n   = grp.transform("count")
    loo_pred   = (bucket_sum - df2["next_pct"]) / (bucket_n - 1)
    err        = df2["next_pct"] - loo_pred
    mae        = float(err.abs().mean())
    rmse       = float(np.sqrt((err ** 2).mean()))

    baseline_mae = float((df2["next_pct"] - df2["next_pct"].mean()).abs().mean())
    mae_score    = 0.0 if baseline_mae <= 0 else max(0.0, min(1.0, 1 - mae / baseline_mae))

    # ── Assignment score: LOO-predicted direction vs. actual direction ──
    pred_sign         = np.sign(loo_pred)
    actual_sign        = np.sign(df2["next_pct"])
    assignment_score   = float((pred_sign == actual_sign).mean())

    # ── Calibration score: chronological train/test split per bucket ───
    tail = (1 - CALIBRATION_TARGET_CI) / 2 * 100
    hits, total = 0, 0
    for _, group in df2.groupby("bucket", sort=False):
        vals  = group["next_pct"].values
        n     = len(vals)
        split = int(n * CALIBRATION_TRAIN_FRACTION)
        if split < 2 or split >= n:
            continue
        train, test = vals[:split], vals[split:]
        lo_p = np.percentile(train, tail)
        hi_p = np.percentile(train, 100 - tail)
        hits  += int(np.sum((test >= lo_p) & (test <= hi_p)))
        total += len(test)

    if total == 0:
        calibration_score, observed_calibration = 0.0, None
    else:
        observed_calibration = hits / total
        calibration_score = max(0.0, 1 - abs(observed_calibration - CALIBRATION_TARGET_CI) / CALIBRATION_TARGET_CI)

    overall_score = (
        SCORE_WEIGHTS["calibration"] * calibration_score +
        SCORE_WEIGHTS["assignment"]  * assignment_score +
        SCORE_WEIGHTS["mae"]         * mae_score
    ) * 100

    return {
        "mae":  mae,
        "rmse": rmse,
        "n":    int(len(df2)),
        "calibration_pct":   None if observed_calibration is None else round(observed_calibration * 100, 2),
        "assignment_pct":    round(assignment_score * 100, 2),
        "calibration_score": round(calibration_score, 4),
        "assignment_score":  round(assignment_score, 4),
        "mae_score":         round(mae_score, 4),
        "overall_score":     round(overall_score, 2),
    }


def optimize_type_thresholds(transitions: list, type_idx: int, verbose: bool = True) -> dict:
    occ_df = build_type_occurrences(transitions, type_idx)
    best = None
    for lo in BEAR_GRID:
        for hi in BULL_GRID:
            metrics = evaluate_threshold_candidate(occ_df, lo, hi)
            if metrics is None:
                continue
            if best is None or metrics["overall_score"] > best["overall_score"]:
                best = {"boundaries": [round(float(lo), 3), round(float(hi), 3)], **metrics}

    label = TRANSITION_TYPE_LABELS[type_idx]
    if best is None:
        best = {
            "boundaries": list(DEFAULT_BOUNDARIES), "mae": None, "rmse": None, "n": len(occ_df),
            "calibration_pct": None, "assignment_pct": None,
            "calibration_score": None, "assignment_score": None, "mae_score": None,
            "overall_score": None,
        }
        if verbose:
            print(f"    {label:<10}  insufficient data (n={len(occ_df)}) — using default {DEFAULT_BOUNDARIES}")
    elif verbose:
        print(f"    {label:<10}  boundaries={best['boundaries']}  "
              f"Score={best['overall_score']:.1f}  "
              f"(calib={best['calibration_pct']}%  assign={best['assignment_pct']}%  "
              f"MAE={best['mae']:.3f}%)  (n={best['n']})")
    return best


def optimize_all_thresholds(transitions: list, verbose: bool = True) -> dict:
    return {i: optimize_type_thresholds(transitions, i, verbose) for i in range(N_TYPES)}


def load_thresholds() -> dict:
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_thresholds(all_thresholds: dict):
    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(all_thresholds, f, indent=2)


def get_thresholds_by_type(ticker: str, transitions: list, stored: dict) -> dict:
    """Returns {type_idx: [lo, hi]} — optimized fresh, or loaded from cache/default."""
    if OPTIMIZE_THRESHOLDS:
        print(f"\n  Optimizing per-transition-type regime thresholds for {ticker}...")
        type_results = optimize_all_thresholds(transitions)
        if SAVE_OPTIMIZED_THRESHOLDS:
            stored[ticker] = {
                TRANSITION_TYPE_LABELS[i]: {
                    "boundaries":         type_results[i]["boundaries"],
                    "mae":                type_results[i]["mae"],
                    "rmse":               type_results[i]["rmse"],
                    "n":                  type_results[i]["n"],
                    "overall_score":      type_results[i].get("overall_score"),
                    "calibration_pct":    type_results[i].get("calibration_pct"),
                    "assignment_pct":     type_results[i].get("assignment_pct"),
                } for i in range(N_TYPES)
            }
            save_thresholds(stored)
        return {i: type_results[i]["boundaries"] for i in range(N_TYPES)}

    cached = stored.get(ticker, {})
    out = {}
    for i in range(N_TYPES):
        entry = cached.get(TRANSITION_TYPE_LABELS[i])
        out[i] = entry["boundaries"] if entry else list(DEFAULT_BOUNDARIES)
    print(f"\n  Using cached/default thresholds for {ticker}: "
          f"{{ {', '.join(f'{TRANSITION_TYPE_LABELS[i]}: {out[i]}' for i in range(N_TYPES))} }}")
    return out


# ─────────────────────────────────────────────
# STATE INDEX  (9 states x 5 next-transition types)
# ─────────────────────────────────────────────
def build_state_index(transitions: list, thresholds_by_type: dict) -> tuple:
    """
    Classifies every transition into bear/flat/bull using its own type's
    thresholds, then builds:
      - state_map[(state_tuple, next_type_idx)] -> np.array of historical
        % moves that occurred immediately after that state, for that type
      - unconditional_by_type[type_idx] -> np.array of ALL historical %
        moves of that type (fallback pool when a state is too thin)
      - current_state = regimes of the two most recent transitions
      - current_next_type_idx = type of the transition that comes next
    """
    regimes = [classify_regime(t["pct"], *thresholds_by_type[t["type_idx"]]) for t in transitions]

    unconditional_by_type = defaultdict(list)
    for t in transitions:
        unconditional_by_type[t["type_idx"]].append(t["pct"])
    unconditional_by_type = {k: np.array(v) for k, v in unconditional_by_type.items()}

    state_map = defaultdict(list)
    for i in range(1, len(transitions) - 1):
        state_tuple   = (regimes[i - 1], regimes[i])
        next_type_idx = transitions[i + 1]["type_idx"]
        next_pct      = transitions[i + 1]["pct"]
        state_map[(state_tuple, next_type_idx)].append(next_pct)
    state_map = {k: np.array(v) for k, v in state_map.items()}

    current_state         = (regimes[-2], regimes[-1])
    current_next_type_idx = (transitions[-1]["type_idx"] + 1) % N_TYPES

    return state_map, unconditional_by_type, regimes, current_state, current_next_type_idx


def print_state_diagnostics(transitions: list, regimes: list):
    counts    = defaultdict(int)
    next_pcts = defaultdict(list)
    for i in range(1, len(transitions) - 1):
        st = (regimes[i - 1], regimes[i])
        counts[st] += 1
        next_pcts[st].append(transitions[i + 1]["pct"])

    print(f"\n  {'-'*56}")
    print(f"  9-State Occurrence Table  (prev regime, curr regime)")
    print(f"  {'-'*56}")
    print(f"  {'State':<22} {'N':>6}  {'Avg next move':>15}")
    for prev in REGIME_LABELS:
        for curr in REGIME_LABELS:
            st = (prev, curr)
            n  = counts.get(st, 0)
            label = f"({prev}, {curr})"
            if n > 0:
                avg = np.mean(next_pcts[st])
                print(f"  {label:<22} {n:>6}  {avg:>+14.2f}%")
            else:
                print(f"  {label:<22} {n:>6}  {'--':>15}")


# ─────────────────────────────────────────────
# STATE-CONDITIONAL MONTE CARLO
# ─────────────────────────────────────────────
def run_state_conditional_mc(
    start_price: float,
    current_state: tuple,
    next_type_idx: int,
    state_map: dict,
    unconditional_by_type: dict,
    thresholds_by_type: dict,
    horizon: int,
    n_sims: int = N_SIMULATIONS,
    seed: int = 7,
) -> np.ndarray:
    """
    Vectorized Markov-chain simulation. Every sim path tracks its own
    (prev_regime, curr_regime) state. At each simulated day, paths are
    grouped by their current state (<=9 groups), each group samples from
    the historical pool matching (state, this day's transition type) —
    falling back to the type's unconditional pool if that pool is too
    thin (< MIN_STATE_SAMPLES). Each path's state is then updated from
    its own sampled outcome, so paths diverge across the simulation.
    """
    rng = np.random.default_rng(seed)
    regime_to_idx = {"bear": 0, "flat": 1, "bull": 2}
    idx_to_regime = {v: k for k, v in regime_to_idx.items()}

    paths = np.zeros((n_sims, horizon + 1))
    paths[:, 0] = start_price

    prev_regime_idx = np.full(n_sims, regime_to_idx[current_state[0]], dtype=np.int8)
    curr_regime_idx = np.full(n_sims, regime_to_idx[current_state[1]], dtype=np.int8)

    type_idx = next_type_idx
    for day in range(1, horizon + 1):
        sampled_pct  = np.empty(n_sims)
        state_codes  = prev_regime_idx * 3 + curr_regime_idx

        for code in np.unique(state_codes):
            mask = state_codes == code
            p_idx, c_idx = divmod(int(code), 3)
            state_tuple = (idx_to_regime[p_idx], idx_to_regime[c_idx])

            pool = state_map.get((state_tuple, type_idx))
            if pool is None or len(pool) < MIN_STATE_SAMPLES:
                pool = unconditional_by_type[type_idx]

            sampled_pct[mask] = rng.choice(pool, size=int(mask.sum()), replace=True)

        paths[:, day] = paths[:, day - 1] * (1 + sampled_pct / 100)

        lo, hi = thresholds_by_type[type_idx]
        new_regime_idx = np.where(sampled_pct < lo, 0, np.where(sampled_pct > hi, 2, 1)).astype(np.int8)
        prev_regime_idx = curr_regime_idx
        curr_regime_idx = new_regime_idx

        type_idx = (type_idx + 1) % N_TYPES

    return paths


def summarize_forecast(paths: np.ndarray, horizon: int) -> pd.DataFrame:
    rows = []
    for d in range(horizon + 1):
        prices = paths[:, d]
        row = {"Day": d}
        for p in CONFIDENCE:
            row[f"p{int(p*100)}"] = np.percentile(prices, p * 100)
        row["mean"] = np.mean(prices)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Day")


# ─────────────────────────────────────────────
# FULL EVALUATOR-DRIVEN THRESHOLD OPTIMIZER
# ─────────────────────────────────────────────
# This is the redesigned pipeline:
#
#     Historical Data -> Threshold Optimizer -> [calls] Evaluator
#                                                    |
#                                          runs the REAL Monte Carlo
#                                          engine, walked forward
#                                          through history
#                                                    |
#                                             Overall Score
#                                                    |
#                              Threshold Optimizer keeps the winner
#
# Unlike evaluate_threshold_candidate() (the fast proxy above, which never
# touches run_state_conditional_mc), evaluate_thresholds_walkforward() is
# the literal evaluator: for a given FIXED thresholds_by_type, it walks
# forward through df, and at each sampled day rebuilds the state map from
# ONLY the data available as of that day (no lookahead) and calls the real
# run_state_conditional_mc engine, then scores calibration + directional
# assignment + 1-day MAE against the actual future closes. Same three-part
# scoring formula and SCORE_WEIGHTS as the fast evaluator, so scores from
# both are on the same 0-100 scale and comparable.
#
# One consequence of using the real 9-state engine: all 5 transition types'
# thresholds affect the state machine jointly, so a true joint grid search
# would be a 10-dimensional optimization — computationally impossible.
# optimize_thresholds_full() instead does coordinate ascent: hold 4 types
# fixed, grid-search the 5th against the evaluator, move to the next type,
# repeat for FULL_EVALUATOR_CONFIG["coord_ascent_passes"] passes.
def evaluate_thresholds_walkforward(
    df: pd.DataFrame,
    thresholds_by_type: dict,
    config: dict = None,
) -> dict | None:
    config      = config or FULL_EVALUATOR_CONFIG
    n_sims      = config["n_sims"]
    stride      = config["stride"]
    min_hist    = config["min_history_days"]
    horizons    = config["horizons"]
    target_ci   = CALIBRATION_TARGET_CI

    df = df.sort_index()
    n  = len(df)
    max_h = max(horizons)
    last_valid = n - max_h - 1
    if last_valid <= min_hist:
        return None

    calib_hits, calib_total   = 0, 0
    assign_hits, assign_total = 0, 0
    abs_errs, baseline_abs_errs = [], []

    for i in range(min_hist, last_valid + 1, stride):
        df_hist = df.iloc[: i + 1]
        transitions = build_transition_sequence(df_hist)
        if len(transitions) < 50:
            continue

        state_map, unconditional_by_type, regimes, current_state, current_next_type_idx = \
            build_state_index(transitions, thresholds_by_type)

        start_price = float(df_hist["Close"].iloc[-1])

        paths = run_state_conditional_mc(
            start_price, current_state, current_next_type_idx,
            state_map, unconditional_by_type, thresholds_by_type,
            max_h, n_sims=n_sims, seed=2_000_000 + i,
        )

        # calibration, at CALIBRATION_TARGET_CI, across the configured horizons
        for h in horizons:
            future_idx = i + h
            if future_idx >= n:
                continue
            actual = float(df["Close"].iloc[future_idx])
            sim    = paths[:, h]
            tail   = (1 - target_ci) / 2 * 100
            lo_p, hi_p = np.percentile(sim, tail), np.percentile(sim, 100 - tail)
            calib_total += 1
            if lo_p <= actual <= hi_p:
                calib_hits += 1

        # directional assignment + MAE, at the +1 day horizon
        future_idx1 = i + 1
        if future_idx1 < n:
            actual1 = float(df["Close"].iloc[future_idx1])
            pred1   = float(np.mean(paths[:, 1]))
            assign_total += 1
            if np.sign(pred1 - start_price) == np.sign(actual1 - start_price):
                assign_hits += 1
            abs_errs.append(abs(actual1 - pred1))
            baseline_abs_errs.append(abs(actual1 - start_price))   # naive random-walk baseline

    if calib_total == 0 or assign_total == 0 or not abs_errs:
        return None

    observed_calibration = calib_hits / calib_total
    calibration_score    = max(0.0, 1 - abs(observed_calibration - target_ci) / target_ci)

    assignment_score = assign_hits / assign_total

    mae          = float(np.mean(abs_errs))
    baseline_mae = float(np.mean(baseline_abs_errs))
    mae_score    = 0.0 if baseline_mae <= 0 else max(0.0, min(1.0, 1 - mae / baseline_mae))

    overall_score = (
        SCORE_WEIGHTS["calibration"] * calibration_score +
        SCORE_WEIGHTS["assignment"]  * assignment_score +
        SCORE_WEIGHTS["mae"]         * mae_score
    ) * 100

    return {
        "mae": mae,
        "n":   calib_total,
        "calibration_pct":   round(observed_calibration * 100, 2),
        "assignment_pct":    round(assignment_score * 100, 2),
        "calibration_score": round(calibration_score, 4),
        "assignment_score":  round(assignment_score, 4),
        "mae_score":         round(mae_score, 4),
        "overall_score":     round(overall_score, 2),
    }


def optimize_thresholds_full(df: pd.DataFrame, config: dict = None, verbose: bool = True) -> tuple:
    """
    THE (full) Threshold Optimizer. Coordinate-ascent grid search, each
    candidate scored by evaluate_thresholds_walkforward() (which runs the
    real Monte Carlo engine). Returns (thresholds_by_type, score_log).

    WARNING: this calls the real Monte Carlo engine on real historical
    walk-forward points hundreds of times over. Expect minutes, not
    seconds — tune FULL_EVALUATOR_CONFIG to trade precision for speed.
    """
    config = config or FULL_EVALUATOR_CONFIG
    thresholds_by_type = {i: list(DEFAULT_BOUNDARIES) for i in range(N_TYPES)}
    score_log           = {i: None for i in range(N_TYPES)}

    t_start = time.time()
    n_candidates_per_type = len(config["bear_grid"]) * len(config["bull_grid"])
    total_calls = n_candidates_per_type * N_TYPES * config["coord_ascent_passes"]
    if verbose:
        print(f"    (grid: {len(config['bear_grid'])}x{len(config['bull_grid'])} per type, "
              f"{config['coord_ascent_passes']} pass(es) -> {total_calls} Monte Carlo walk-forward evaluations total)")

    for p in range(config["coord_ascent_passes"]):
        if verbose:
            print(f"\n  -- coordinate-ascent pass {p + 1}/{config['coord_ascent_passes']} --")
        for type_idx in range(N_TYPES):
            label = TRANSITION_TYPE_LABELS[type_idx]
            best = None
            for lo in config["bear_grid"]:
                for hi in config["bull_grid"]:
                    candidate = dict(thresholds_by_type)
                    candidate[type_idx] = [round(float(lo), 3), round(float(hi), 3)]
                    metrics = evaluate_thresholds_walkforward(df, candidate, config)
                    if metrics is None:
                        continue
                    if best is None or metrics["overall_score"] > best["overall_score"]:
                        best = {"boundaries": candidate[type_idx], **metrics}

            if best is not None:
                thresholds_by_type[type_idx] = best["boundaries"]
                score_log[type_idx] = best
                if verbose:
                    elapsed = time.time() - t_start
                    print(f"    {label:<10}  boundaries={best['boundaries']}  "
                          f"Score={best['overall_score']:.1f}  "
                          f"(calib={best['calibration_pct']}%  assign={best['assignment_pct']}%  "
                          f"MAE={best['mae']:.3f})  [{elapsed:,.0f}s elapsed]")
            elif verbose:
                print(f"    {label:<10}  insufficient walk-forward data — keeping default {thresholds_by_type[type_idx]}")

    if verbose:
        print(f"\n  Full evaluator-driven optimization complete in {time.time() - t_start:,.0f}s")

    return thresholds_by_type, score_log


def get_thresholds_full(ticker: str, df: pd.DataFrame, stored: dict) -> dict:
    print(f"\n  [FULL evaluator] Optimizing thresholds for {ticker} — baking the Monte Carlo "
          f"walk-forward evaluator directly into the threshold search...")
    thresholds_by_type, score_log = optimize_thresholds_full(df)
    if SAVE_OPTIMIZED_THRESHOLDS:
        stored[ticker] = {
            TRANSITION_TYPE_LABELS[i]: {
                "boundaries": thresholds_by_type[i],
                **({k: v for k, v in score_log[i].items() if k != "boundaries"} if score_log[i] else {}),
                "optimizer_mode": "full",
            } for i in range(N_TYPES)
        }
        save_thresholds(stored)
    return thresholds_by_type


def resolve_thresholds(ticker: str, df: pd.DataFrame, transitions: list, stored: dict) -> dict:
    """
    Single entry point main() calls to get {type_idx: [lo, hi]}. Dispatches
    to the full evaluator-in-the-loop optimizer, the fast proxy optimizer,
    or the cache/default, based on OPTIMIZE_THRESHOLDS / OPTIMIZER_MODE.
    """
    if not OPTIMIZE_THRESHOLDS:
        cached = stored.get(ticker, {})
        out = {}
        for i in range(N_TYPES):
            entry = cached.get(TRANSITION_TYPE_LABELS[i])
            out[i] = entry["boundaries"] if entry else list(DEFAULT_BOUNDARIES)
        print(f"\n  Using cached/default thresholds for {ticker}: "
              f"{{ {', '.join(f'{TRANSITION_TYPE_LABELS[i]}: {out[i]}' for i in range(N_TYPES))} }}")
        return out

    if OPTIMIZER_MODE == "full":
        return get_thresholds_full(ticker, df, stored)
    else:
        return get_thresholds_by_type(ticker, transitions, stored)


# ─────────────────────────────────────────────
# MON OPEN -> FRI CLOSE WEEKLY SPREAD  (unconditional baseline, unchanged)
# ─────────────────────────────────────────────
def compute_weekly_spread(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ISOWeek"] = df.index.isocalendar().week.values
    df["ISOYear"] = df.index.isocalendar().year.values
    df["WeekKey"] = df["ISOYear"].astype(str) + "-" + df["ISOWeek"].astype(str)

    df_reset = df.reset_index()
    date_col = df_reset.columns[0]

    mon = (df_reset[df_reset["DayName"] == "Monday"]
           [[date_col, "Open", "WeekKey"]]
           .rename(columns={date_col: "Date_Mon", "Open": "Mon_Open"}))
    fri = (df_reset[df_reset["DayName"] == "Friday"]
           [[date_col, "Close", "WeekKey"]]
           .rename(columns={date_col: "Date_Fri", "Close": "Fri_Close"}))

    merged = pd.merge(mon, fri, on="WeekKey").dropna()
    merged["PctChange"] = (merged["Fri_Close"] / merged["Mon_Open"] - 1) * 100
    merged = merged.sort_values("Date_Mon").reset_index(drop=True)
    return merged


def weekly_spread_stats(ws: pd.DataFrame) -> dict:
    arr = np.array(ws["PctChange"]).flatten()
    return {
        "n":            len(arr),
        "mean":         np.mean(arr),
        "median":       np.median(arr),
        "std":          np.std(arr),
        "min":          np.min(arr),
        "max":          np.max(arr),
        "p5":           np.percentile(arr,  5),
        "p10":          np.percentile(arr, 10),
        "p25":          np.percentile(arr, 25),
        "p75":          np.percentile(arr, 75),
        "p90":          np.percentile(arr, 90),
        "p95":          np.percentile(arr, 95),
        "pct_positive": np.mean(arr > 0) * 100,
        "pct_up2":      np.mean(arr >  2) * 100,
        "pct_up5":      np.mean(arr >  5) * 100,
        "pct_down2":    np.mean(arr < -2) * 100,
        "pct_down5":    np.mean(arr < -5) * 100,
    }


def spread_stats(arr: np.ndarray) -> dict:
    return {
        "mean":   np.mean(arr),
        "median": np.median(arr),
        "std":    np.std(arr),
        "min":    np.min(arr),
        "max":    np.max(arr),
        "p5":     np.percentile(arr,  5),
        "p25":    np.percentile(arr, 25),
        "p75":    np.percentile(arr, 75),
        "p95":    np.percentile(arr, 95),
        "n":      len(arr),
    }


def run_monte_carlo(start_price: float, spreads: dict, n_sims: int = N_SIMULATIONS, seed: int = 42) -> np.ndarray:
    rng    = np.random.default_rng(seed)
    n_cols = len(DAY_PAIRS) + 1
    paths  = np.zeros((n_sims, n_cols))
    paths[:, 0] = start_price
    for col_idx, (from_day, to_day) in enumerate(DAY_PAIRS):
        historical_moves = np.array(spreads.get((from_day, to_day), np.array([0.0]))).flatten()
        sampled_pct = rng.choice(historical_moves, size=n_sims, replace=True)
        paths[:, col_idx + 1] = paths[:, col_idx] * (1 + sampled_pct / 100)
    return paths


def summarize_simulations(paths: np.ndarray) -> pd.DataFrame:
    rows = []
    for day_idx, day in enumerate(DAY_NAMES):
        prices = paths[:, day_idx]
        row = {"Day": day}
        for p in CONFIDENCE:
            row[f"p{int(p*100)}"] = np.percentile(prices, p * 100)
        row["mean"] = np.mean(prices)
        rows.append(row)
    return pd.DataFrame(rows).set_index("Day")


# ─────────────────────────────────────────────
# VISUALIZATION  (unconditional weekly chart, unchanged)
# ─────────────────────────────────────────────
def plot_ticker(ticker, start_price, df, spreads, paths, sim_summary, weekly_spread):
    ws_s   = weekly_spread_stats(weekly_spread)
    ws_arr = np.array(weekly_spread["PctChange"]).flatten()

    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor("#f9f9f9")
    fig.suptitle(
        f"{ticker}  --  Monte Carlo Week Simulation  (prior Fri close: ${start_price:.2f})",
        fontsize=17, fontweight="bold", y=0.99,
    )
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.35)

    ax0 = fig.add_subplot(gs[0, :2])
    ax0.plot(df.index, df["Close"], linewidth=1, color=C_BLUE, zorder=2)
    ax0.fill_between(df.index, df["Close"], alpha=0.08, color=C_BLUE)
    ax0.set_title("Historical Close Price", fontsize=11, fontweight="bold")
    ax0.set_ylabel("Price ($)")
    ax0.grid(alpha=0.25)
    ax0.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.2f}"))

    ax1 = fig.add_subplot(gs[0, 2])
    labels   = [f"{f[:3]}->{t[:3]}" for f, t in DAY_PAIRS]
    box_data = [spreads[pair] for pair in DAY_PAIRS]
    bp = ax1.boxplot(box_data, labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=2),
                     flierprops=dict(marker="o", markersize=3, alpha=0.4))
    box_colors = [C_ORANGE, C_BLUE, C_ORANGE, C_GREEN, "#E76F51"]
    for patch, color in zip(bp["boxes"], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_title("Day-to-Day % Move Distributions\n(Fri->Mon = weekend gap, Mon uses Open price)",
                  fontsize=11, fontweight="bold")
    ax1.set_ylabel("% Change")
    ax1.grid(axis="y", alpha=0.25)

    ax2 = fig.add_subplot(gs[1, :2])
    pct        = weekly_spread["PctChange"].values
    dates_mon  = pd.to_datetime(weekly_spread["Date_Mon"])
    bar_colors = [C_BULL if v >= 0 else C_BEAR for v in pct]
    ax2.bar(dates_mon, pct, color=bar_colors, alpha=0.72, width=4, zorder=2)
    ax2.axhline(0,            color="black",  linewidth=0.8, zorder=3)
    ax2.axhline(ws_s["mean"], color=C_MEAN,   linewidth=1.5, linestyle="--",
                label=f"Mean: {ws_s['mean']:+.2f}%", zorder=3)
    ax2.axhline(ws_s["p25"],  color=C_ORANGE, linewidth=1.2, linestyle=":",
                label=f"P25:  {ws_s['p25']:+.2f}%",  zorder=3)
    ax2.axhline(ws_s["p75"],  color=C_GREEN,  linewidth=1.2, linestyle=":",
                label=f"P75:  {ws_s['p75']:+.2f}%",  zorder=3)
    ax2.set_title(
        f"Historical Mon Open -> Fri Close Weekly % Change  "
        f"(blue = up, red = down  |  {ws_s['pct_positive']:.0f}% of weeks closed higher)",
        fontsize=11, fontweight="bold",
    )
    ax2.set_ylabel("% Change  (Mon open -> Fri close)")
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.25)
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.1f}%"))

    ax3 = fig.add_subplot(gs[1, 2])
    n_bins = min(50, max(20, len(ws_arr) // 5))
    ax3.hist(ws_arr, bins=n_bins, color=C_BLUE, alpha=0.75, edgecolor="none", zorder=2)
    for val, col, lbl in [
        (ws_s["p5"],     C_BEAR,   f"P5   {ws_s['p5']:+.1f}%"),
        (ws_s["p25"],    C_ORANGE, f"P25  {ws_s['p25']:+.1f}%"),
        (ws_s["median"], C_MEDIAN, f"P50  {ws_s['median']:+.1f}%"),
        (ws_s["p75"],    C_GREEN,  f"P75  {ws_s['p75']:+.1f}%"),
        (ws_s["p95"],    C_BULL,   f"P95  {ws_s['p95']:+.1f}%"),
    ]:
        ax3.axvline(val, color=col, linewidth=1.8, label=lbl)
    ax3.axvline(0, color="black", linewidth=1, linestyle="--", alpha=0.5)
    ax3.set_title("Mon Open -> Fri Close\n% Change Distribution", fontsize=11, fontweight="bold")
    ax3.set_xlabel("% Change (Mon open -> Fri close)")
    ax3.set_ylabel("# of Weeks")
    ax3.legend(fontsize=7, loc="upper left")
    ax3.grid(axis="y", alpha=0.25)

    ax4 = fig.add_subplot(gs[2, :2])
    x = range(len(DAY_NAMES))
    sample_idx = np.random.choice(len(paths), size=min(400, len(paths)), replace=False)
    for i in sample_idx:
        ax4.plot(x, paths[i], color="gray", alpha=0.04, linewidth=0.7)
    for (lo, hi), fc in [((5, 95), "#f4a58a"), ((25, 75), "#92b4d9")]:
        ax4.fill_between(x, sim_summary[f"p{lo}"], sim_summary[f"p{hi}"],
                         alpha=0.4, color=fc, label=f"P{lo}-P{hi}")
    ax4.plot(x, sim_summary["p50"],  color=C_MEDIAN, linewidth=2.5, label="Median",     zorder=5)
    ax4.plot(x, sim_summary["mean"], color=C_MEAN,   linewidth=2,   label="Mean",
             linestyle="--", zorder=5)
    ax4.plot(x, sim_summary["p5"],   color=C_BEAR,   linewidth=1.5, label="P5 (bear)",  linestyle=":")
    ax4.plot(x, sim_summary["p95"],  color=C_BULL,   linewidth=1.5, label="P95 (bull)", linestyle=":")
    ax4.set_xticks(x)
    ax4.set_xticklabels(["Fri*\n(prior close)", "Mon\n(open)", "Tue\n(close)",
                          "Wed\n(close)", "Thu\n(close)", "Fri\n(close)"])
    ax4.set_title(f"Monte Carlo Simulation ({N_SIMULATIONS:,} paths)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Simulated Price ($)")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(alpha=0.25)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.2f}"))

    ax5 = fig.add_subplot(gs[2, 2])
    friday_prices = paths[:, -1]
    ax5.hist(friday_prices, bins=80, color=C_BLUE, alpha=0.75, edgecolor="none")
    for p, color in [(5, C_BEAR), (25, C_ORANGE), (50, C_MEDIAN), (75, C_GREEN), (95, C_BULL)]:
        val = np.percentile(friday_prices, p)
        ax5.axvline(val, color=color, linewidth=1.5, label=f"P{p}: ${val:.2f}")
    ax5.axvline(start_price, color="green", linewidth=2, linestyle="--",
                label=f"Prior Fri close: ${start_price:.2f}")
    ax5.set_title("Simulated Friday Close Price", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Price ($)")
    ax5.set_ylabel("Frequency")
    ax5.legend(fontsize=7)
    ax5.grid(alpha=0.25)
    ax5.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.2f}"))

    plt.savefig(f"{ticker}_monte_carlo.png", dpi=150, bbox_inches="tight")
    print(f"  Saved: {ticker}_monte_carlo.png")
    plt.close()


# ─────────────────────────────────────────────
# CONSOLE OUTPUT
# ─────────────────────────────────────────────
def print_spread_table(ticker: str, spreads: dict):
    print(f"\n{'='*66}")
    print(f"  {ticker} -- Day-to-Day Historical Spread Summary")
    print(f"  (Fri->Mon is the weekend gap; Mon->Tue uses Monday OPEN as entry price)")
    print(f"{'='*66}")
    print(f"  {'Transition':<16} {'N':>5} {'Mean%':>7} {'Std%':>7} {'P5%':>7} {'P25%':>7} {'P75%':>7} {'P95%':>7}")
    print(f"  {'-'*62}")
    for (from_day, to_day), arr in spreads.items():
        s     = spread_stats(arr)
        label = f"{from_day[:3]}->{to_day[:3]}"
        note  = " *open" if from_day == "Monday" else (" *gap" if from_day == "Friday" else "")
        print(f"  {label+note:<16} {s['n']:>5} {s['mean']:>7.2f} {s['std']:>7.2f} "
              f"{s['p5']:>7.2f} {s['p25']:>7.2f} {s['p75']:>7.2f} {s['p95']:>7.2f}")


def print_weekly_spread_table(ticker: str, weekly_spread: pd.DataFrame):
    ws = weekly_spread_stats(weekly_spread)
    print(f"\n{'='*66}")
    print(f"  {ticker} -- Monday OPEN -> Friday CLOSE  (full-week spread)")
    print(f"{'='*66}")
    print(f"  Weeks analysed  : {ws['n']}")
    print(f"  Weeks positive  : {ws['pct_positive']:.1f}%  (Fri close > Mon open)")
    print(f"  Mean change     : {ws['mean']:+.2f}%")
    print(f"  Median change   : {ws['median']:+.2f}%")
    print(f"  Std dev         : {ws['std']:.2f}%")
    print(f"\n  Percentile breakdown:")
    print(f"    Bear  P5  : {ws['p5']:+.2f}%")
    print(f"          P10 : {ws['p10']:+.2f}%")
    print(f"          P25 : {ws['p25']:+.2f}%")
    print(f"    Flat  P50 : {ws['median']:+.2f}%")
    print(f"    Bull  P75 : {ws['p75']:+.2f}%")
    print(f"          P90 : {ws['p90']:+.2f}%")
    print(f"          P95 : {ws['p95']:+.2f}%")
    print(f"\n  Tail probabilities (historical base rates):")
    print(f"    Finished week DOWN < -5% : {ws['pct_down5']:.1f}%  of weeks")
    print(f"    Finished week DOWN < -2% : {ws['pct_down2']:.1f}%  of weeks")
    print(f"    Finished week UP   > +2% : {ws['pct_up2']:.1f}%  of weeks")
    print(f"    Finished week UP   > +5% : {ws['pct_up5']:.1f}%  of weeks")
    print(f"\n  Last 5 weeks:")
    for _, row in weekly_spread.tail(5).iterrows():
        arrow = "^" if row["PctChange"] >= 0 else "v"
        print(f"    {str(row['Date_Mon'])[:10]}  open ${float(row['Mon_Open']):>8.2f}  ->  "
              f"{str(row['Date_Fri'])[:10]}  close ${float(row['Fri_Close']):>8.2f}  "
              f"{arrow} {float(row['PctChange']):+.2f}%")


def print_options_grid(ticker: str, start_price: float, fri: np.ndarray, label: str = ""):
    """
    Dynamic-range options grid. Instead of a fixed +-12% window, walks
    outward one percent at a time on each side and keeps going as long
    as probability stays above MIN_DISPLAY_PROB — so a calm stock won't
    show a wall of 0.0% rows, and a volatile stock will show its full
    tail instead of getting clipped at +-12%.
    """
    tag = f"  [{label}]" if label else ""
    print(f"\n  Options insights{tag}  (start ${start_price:.2f}):")
    print(f"  {'Strike':>10}  {'Move':>6}  {'Prob':>7}  Direction")
    print(f"  {'-'*42}")

    # Upside (covered call zone) — walk +1%, +2%, ... until prob drops out
    call_rows = []
    pct = 1
    while pct < MAX_STRIKE_PCT:
        strike = start_price * (1 + pct / 100)
        prob   = np.mean(fri > strike) * 100
        if prob <= MIN_DISPLAY_PROB:
            break
        call_rows.append((pct, strike, prob))
        pct += 1

    for pct, strike, prob in reversed(call_rows):
        bar = "█" * int(prob / 2)
        print(f"  ${strike:>9.2f}  {f'+{pct}%':>6}  {prob:>6.1f}%  {bar}  <- covered call")

    print()

    # Downside (secured put zone) — walk -1%, -2%, ... until prob drops out
    put_rows = []
    pct = 1
    while pct < MAX_STRIKE_PCT:
        strike = start_price * (1 - pct / 100)
        if strike <= 0:
            break
        prob = np.mean(fri < strike) * 100
        if prob <= MIN_DISPLAY_PROB:
            break
        put_rows.append((pct, strike, prob))
        pct += 1

    for pct, strike, prob in put_rows:
        bar = "█" * int(prob / 2)
        print(f"  ${strike:>9.2f}  {f'-{pct}%':>6}  {prob:>6.1f}%  {bar}  <- secured put")

    p5_fri  = np.percentile(fri,  5)
    p95_fri = np.percentile(fri, 95)
    print(f"\n    Simulated 90% confidence range: ${p5_fri:.2f} - ${p95_fri:.2f}")


def print_simulation_summary(ticker: str, start_price: float, sim_summary: pd.DataFrame, paths: np.ndarray):
    print(f"\n{'='*66}")
    print(f"  {ticker} -- Unconditional Weekly Monte Carlo  (prior Fri close ${start_price:.2f})")
    print(f"{'='*66}")
    print(f"  {'Day':<16} {'P5 (bear)':>11} {'P25':>11} {'Median':>11} {'P75':>11} {'P95 (bull)':>11} {'Mean':>11}")
    print(f"  {'-'*66}")
    labels = ["Fri* (prior close)", "Mon (open)", "Tue (close)", "Wed (close)", "Thu (close)", "Fri (close)"]
    for (day, row), lbl in zip(sim_summary.iterrows(), labels):
        print(f"  {lbl:<16}"
              f"  ${row['p5']:>9.2f}  ${row['p25']:>9.2f}  ${row['p50']:>9.2f}"
              f"  ${row['p75']:>9.2f}  ${row['p95']:>9.2f}  ${row['mean']:>9.2f}")
    if REPORT["options_grid"]:
        print_options_grid(ticker, start_price, paths[:, -1], label="Unconditional weekly")


def print_forecast_summary(ticker: str, start_price: float, horizon: int,
                            summary_df: pd.DataFrame, paths: np.ndarray, last_date):
    print(f"\n{'='*66}")
    print(f"  {ticker} -- State-Conditional Forecast  (+{horizon} trading day{'s' if horizon != 1 else ''})")
    print(f"{'='*66}")
    print(f"  {'Date':<8} {'P5':>10} {'P25':>10} {'Median':>10} {'P75':>10} {'P95':>10} {'Mean':>10}")
    print(f"  {'-'*66}")
    for day, row in summary_df.iterrows():
        if day == 0:
            tag = "now"
        else:
            tag = format_date_short(future_trading_date(last_date, day))
        print(f"  {tag:<8} ${row['p5']:>8.2f} ${row['p25']:>8.2f} ${row['p50']:>8.2f} "
              f"${row['p75']:>8.2f} ${row['p95']:>8.2f} ${row['mean']:>8.2f}")
    target_date = format_date_short(future_trading_date(last_date, horizon))
    if REPORT["options_grid"]:
        print_options_grid(ticker, start_price, paths[:, -1], label=f"{target_date} state-conditional (+{horizon}d)")


# ─────────────────────────────────────────────
# MONTE CARLO CALIBRATION BACKTEST
# ─────────────────────────────────────────────
# Purpose: measure whether the state-conditional MC engine's predicted
# confidence intervals are statistically well calibrated, by replaying
# history day by day. This section only READS df / calls the existing
# forecasting functions with a truncated ("as of that day") dataframe —
# it does not alter run_state_conditional_mc, build_state_index, or any
# other forecasting logic.
def evaluate_calibration_for_ticker(
    ticker: str,
    df: pd.DataFrame,
    horizons: list = FORECAST_HORIZONS,
    confidence_levels: list = None,
    n_sims: int = None,
    stride: int = None,
    min_history_days: int = None,
) -> dict:
    """
    Walks df day by day (every `stride`-th day, starting once
    `min_history_days` of history exist and at least max(horizons)
    trading days remain after it). For each such day:
      - truncates df to ONLY data up to and including that day
      - rebuilds transitions + optimizes thresholds from that truncated
        history only (no lookahead)
      - runs the normal state-conditional MC out to the longest horizon
      - for every horizon and every confidence level, checks whether the
        REAL future close (read from the full df, used only for scoring,
        never for building the model) fell inside the simulated interval

    Returns: {horizon: {conf_level: {"hits": int, "total": int}}}
    """
    confidence_levels = confidence_levels or BACKTEST_CONFIG["confidence_levels"]
    n_sims            = n_sims           or BACKTEST_CONFIG["n_sims"]
    stride            = stride           or BACKTEST_CONFIG["stride"]
    min_history_days  = min_history_days or BACKTEST_CONFIG["min_history_days"]

    df = df.sort_index()
    n  = len(df)
    max_horizon = max(horizons)

    results = {h: {c: {"hits": 0, "total": 0} for c in confidence_levels} for h in horizons}

    last_valid_idx = n - max_horizon - 1
    if last_valid_idx <= min_history_days:
        print(f"  [{ticker}] Not enough history for calibration backtest — skipping.")
        return results

    eval_indices = range(min_history_days, last_valid_idx + 1, stride)

    for i in eval_indices:
        df_hist = df.iloc[: i + 1]   # everything known "as of" this simulated day

        transitions = build_transition_sequence(df_hist)
        if len(transitions) < 50:
            continue

        # Thresholds re-optimized from history-to-date only (bypasses the
        # live threshold cache file entirely — this is a scratch model).
        type_results       = optimize_all_thresholds(transitions, verbose=False)
        thresholds_by_type = {k: v["boundaries"] for k, v in type_results.items()}

        state_map, unconditional_by_type, regimes, current_state, current_next_type_idx = \
            build_state_index(transitions, thresholds_by_type)

        start_price = float(df_hist["Close"].iloc[-1])

        paths = run_state_conditional_mc(
            start_price, current_state, current_next_type_idx,
            state_map, unconditional_by_type, thresholds_by_type,
            max_horizon, n_sims=n_sims, seed=1_000_000 + i,
        )

        for h in horizons:
            future_idx = i + h
            if future_idx >= n:
                continue
            actual_price = float(df["Close"].iloc[future_idx])
            sim_prices   = paths[:, h]

            for c in confidence_levels:
                tail    = (1 - c) / 2 * 100
                lower   = np.percentile(sim_prices, tail)
                upper   = np.percentile(sim_prices, 100 - tail)
                results[h][c]["total"] += 1
                if lower <= actual_price <= upper:
                    results[h][c]["hits"] += 1

    return results


def assess_calibration(observed_pct: float, target_pct: float, tolerance: float = None) -> tuple:
    """Returns (verdict_line, explanation_line) for the Overall Assessment block."""
    tolerance = tolerance if tolerance is not None else BACKTEST_CONFIG["tolerance_pts"]
    diff = observed_pct - target_pct
    if abs(diff) <= tolerance:
        return ("✓ Excellent calibration",
                "The model's predicted confidence intervals closely match historical outcomes.")
    elif diff < 0:
        return ("⚠ Confidence intervals appear too narrow.",
                "The model is underestimating uncertainty and producing overly confident forecasts.")
    else:
        return ("⚠ Confidence intervals appear too wide.",
                "The model is overestimating uncertainty and producing overly conservative forecasts.")


def print_calibration_reports(all_results: dict, horizons: list = None, confidence_levels: list = None):
    """
    all_results: {ticker: {horizon: {conf_level: {"hits", "total"}}}}
    Prints one table per horizon (rows = tickers, cols = confidence levels),
    followed by an Overall Assessment for that horizon.
    """
    horizons          = horizons or FORECAST_HORIZONS
    confidence_levels = confidence_levels or BACKTEST_CONFIG["confidence_levels"]
    assessment_ci      = BACKTEST_CONFIG["assessment_ci"]

    tickers = [t for t in all_results if any(
        all_results[t][h][c]["total"] > 0 for h in horizons for c in confidence_levels
    )]

    print(f"\n{'='*66}")
    print("MONTE CARLO CALIBRATION REPORT")
    print(f"{'='*66}")

    if not tickers:
        print("\n  No tickers had enough history to run the calibration backtest.")
        return

    for h in horizons:
        day_label = f"+{h} Trading Day" + ("s" if h != 1 else "")
        print(f"\nForecast Horizon: {day_label}")
        print(f"\n{'-'*61}")
        header = f"{'Ticker':<12}" + "".join(f"{str(int(c*100)) + '%':>9}" for c in confidence_levels)
        print(header)
        print(f"{'-'*61}")

        ci_calibrations = {c: [] for c in confidence_levels}

        for t in tickers:
            row = f"{t:<12}"
            for c in confidence_levels:
                bucket = all_results[t][h][c]
                if bucket["total"] > 0:
                    pct = bucket["hits"] / bucket["total"] * 100
                    ci_calibrations[c].append(pct)
                    row += f"{pct:>8.1f}%"
                else:
                    row += f"{'--':>9}"
            print(row)

        print(f"{'-'*61}")

        # Overall Assessment for this horizon, driven by the configured CI
        assess_vals = ci_calibrations.get(assessment_ci, [])
        print("\nOverall Assessment")
        print(f"\nAverage {int(assessment_ci*100)}% Calibration\n")
        if assess_vals:
            avg = sum(assess_vals) / len(assess_vals)
            print(f"{avg:.1f}%\n")
            verdict, explanation = assess_calibration(avg, assessment_ci * 100)
            print(verdict)
            print(f"\n{explanation}")
        else:
            print("  (no data)")


def run_calibration_backtest(tickers: list, dataframes: dict):
    """
    dataframes: {ticker: df} — the already-fetched full-history dataframes
    from the normal run, reused here so we don't re-download anything.
    """
    print(f"\n{'#'*66}")
    print("  Running Monte Carlo calibration backtest ...")
    print(f"  (n_sims={BACKTEST_CONFIG['n_sims']:,} per eval day, "
          f"stride={BACKTEST_CONFIG['stride']}, "
          f"min_history_days={BACKTEST_CONFIG['min_history_days']})")
    print(f"{'#'*66}")

    all_results = {}
    for ticker in tickers:
        df = dataframes.get(ticker)
        if df is None or df.empty:
            continue
        print(f"\n  Backtesting {ticker} ...")
        all_results[ticker] = evaluate_calibration_for_ticker(ticker, df)

    print_calibration_reports(all_results)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    custom_start_prices = {"NVDA": None, "WULF": None, "SOFI": None}
    fetched_dataframes  = {}   # reused by the calibration backtest below, avoids re-downloading

    for ticker in TICKERS:
        print(f"\n{'#'*66}")
        print(f"  Processing {ticker} ...")
        print(f"{'#'*66}")

        df = fetch_data(ticker)
        if df.empty:
            print(f"  ERROR: No data for {ticker}, skipping.")
            continue
        fetched_dataframes[ticker] = df

        # ── Unconditional weekly chain (prior Fri close -> next Fri close) ──
        if custom_start_prices.get(ticker) is not None:
            weekly_start_price = float(custom_start_prices[ticker])
        else:
            mondays = df[df["DayName"] == "Monday"]
            weekly_start_price = None
            if not mondays.empty:
                last_monday_date = mondays.index[-1]
                pos = df.index.get_loc(last_monday_date)
                if pos > 0 and df["DayName"].iloc[pos - 1] == "Friday":
                    weekly_start_price = float(df["Close"].iloc[pos - 1])
                else:
                    weekly_start_price = float(mondays["Open"].iloc[-1])
            if weekly_start_price is None:
                weekly_start_price = float(df["Open"].iloc[-1])

        print(f"  Data range        : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} trading days)")

        spreads = compute_dow_spreads(df)
        if REPORT["spread_table"]:
            print_spread_table(ticker, spreads)

        weekly_spread = compute_weekly_spread(df)
        if REPORT["weekly_summary"]:
            print_weekly_spread_table(ticker, weekly_spread)

        weekly_paths = run_monte_carlo(weekly_start_price, spreads)
        sim_summary  = summarize_simulations(weekly_paths)
        if REPORT["simulation_summary"]:
            print_simulation_summary(ticker, weekly_start_price, sim_summary, weekly_paths)
        if REPORT["charts"]:
            plot_ticker(ticker, weekly_start_price, df, spreads, weekly_paths, sim_summary, weekly_spread)

        # ── State-conditional rolling forecast engine ────────────────────
        if REPORT["state_forecast"]:
            transitions = build_transition_sequence(df)
            stored = load_thresholds()
            thresholds_by_type = resolve_thresholds(ticker, df, transitions, stored)

            state_map, unconditional_by_type, regimes, current_state, current_next_type_idx = \
                build_state_index(transitions, thresholds_by_type)

            last_date = df.index[-1]

            print(f"\n{'='*66}")
            print(f"  {ticker} -- Current Market State")
            print(f"{'='*66}")
            print(f"  As of              : {transitions[-1]['date'].date()}")
            print(f"  Current state       : (prev={current_state[0].upper()}, curr={current_state[1].upper()})")
            print(f"  Next transition type: {TRANSITION_TYPE_LABELS[current_next_type_idx]}")
            print(f"  Optimized thresholds:")
            for i in range(N_TYPES):
                lo, hi = thresholds_by_type[i]
                print(f"    {TRANSITION_TYPE_LABELS[i]:<10}  Bear < {lo:+.2f}%   Bull > {hi:+.2f}%")

            if REPORT["state_diagnostics"]:
                print_state_diagnostics(transitions, regimes)

            state_start_price = float(df["Close"].iloc[-1])
            for horizon in FORECAST_HORIZONS:
                paths = run_state_conditional_mc(
                    state_start_price, current_state, current_next_type_idx,
                    state_map, unconditional_by_type, thresholds_by_type,
                    horizon, n_sims=N_SIMULATIONS,
                )
                summary_df = summarize_forecast(paths, horizon)
                print_forecast_summary(ticker, state_start_price, horizon, summary_df, paths, last_date)

            print(f"\n\n\n\n{'#'*66}")

    if REPORT["calibration_backtest"]:
        run_calibration_backtest(TICKERS, fetched_dataframes)

    print(f"\nDone!\n")


if __name__ == "__main__":
    main()