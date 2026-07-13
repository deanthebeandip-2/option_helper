"""
Stock Monte Carlo Simulator — Day-of-Week Spread Analysis
=========================================================
Uses historical closing prices to model day-to-day % moves.
  - Fri->Mon  : Friday CLOSE (prior week) -> Monday OPEN   [weekend gap]
  - Mon->Tue  : Monday OPEN  -> Tuesday CLOSE
  - Tue->Wed  : Tuesday CLOSE -> Wednesday CLOSE
  - Wed->Thu  : Wednesday CLOSE -> Thursday CLOSE
  - Thu->Fri  : Thursday CLOSE -> Friday CLOSE
  - Weekly    : Monday OPEN  -> Friday CLOSE

Conditional Sampling: sub-buckets the post-Monday distributions based on
how strong/weak Monday's open->close move was (the "regime"), capturing
momentum and mean-reversion effects. Regime boundaries are NOT hardcoded —
they're chosen automatically per-ticker by backtesting a grid of candidate
thresholds against actual historical weeks (leave-one-week-out validation)
and picking whichever split minimizes prediction error. Results are cached
in regime_thresholds.json so you don't have to re-optimize every run.

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
import json
import os
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS        = ["NVDA"]
#TICKERS        = ["NVDA", "WULF", "SOFI"]
LOOKBACK_YEARS = 6
N_SIMULATIONS  = 1_000_000
#N_SIMULATIONS  = 10_000
CONFIDENCE     = [0.05, 0.25, 0.50, 0.75, 0.95]


# ─────────────────────────────────────────────
# REPORT CONFIGURATION
# ─────────────────────────────────────────────

REPORT = {
    # Console reports
    "spread_table": 0, #don't need
    "weekly_summary": 1, #useful
    "simulation_summary": True, # option grid nested inside
    "options_grid": True,
    "conditional_sampling": 1, #now with dynamic conditional

    # Graphics
    "charts": 1,

    # Future modules
    "backtesting": False,
    "live_analysis": False,
}

# Order matters: Fri->Mon (weekend gap) now leads the weekday chain.
DAY_PAIRS = [
    ("Friday",    "Monday"),    # weekend gap (crosses weeks)
    ("Monday",    "Tuesday"),
    ("Tuesday",   "Wednesday"),
    ("Wednesday", "Thursday"),
    ("Thursday",  "Friday"),
]
# 6 labels for 6 path columns: start (prior Fri close) + 5 transitions
DAY_NAMES = ["Fri*", "Mon", "Tue", "Wed", "Thu", "Fri"]

# ── Dynamic Conditional Sampling ─────────────────────────────────────────
N_REGIMES              = 7      # 3, 5, or 7 buckets
                                 #   3 -> bear / flat / bull
                                 #   5 -> bear / slight_bear / flat / slight_bull / bull
                                 #   7 -> strong_bear ... strong_bull  (BigDown..BigUp)
OPTIMIZE_THRESHOLDS     = True  # search for best regime boundaries per ticker each run
SAVE_OPTIMIZED_THRESHOLDS = True
THRESHOLDS_FILE        = "regime_thresholds.json"
MIN_BUCKET_SIZE         = 8     # min weeks per regime bucket to be considered valid
MIN_BACKTEST_WEEKS      = 10    # min usable backtested weeks to trust a scheme

# Search grids
BEAR_GRID  = np.arange(-5.0, -0.4, 0.5)   # used when N_REGIMES == 3: -5.0 ... -0.5
BULL_GRID  = np.arange(0.5, 5.1, 0.5)     # used when N_REGIMES == 3: +0.5 ... +5.0
SCALE_GRID = np.arange(0.3, 2.05, 0.1)    # used when N_REGIMES in (5, 7)

# Relative boundary shapes (in %), scaled by SCALE_GRID for 5/7-bucket search
REGIME_TEMPLATES = {
    3: [-1.0, 1.0],
    5: [-2.0, -1.0, 1.0, 2.0],
    7: [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0],
}
REGIME_LABELS = {
    3: ["bear", "flat", "bull"],
    5: ["bear", "slight_bear", "flat", "slight_bull", "bull"],
    7: ["strong_bear", "bear", "slight_bear", "flat", "slight_bull", "bull", "strong_bull"],
}
REGIME_DISPLAY = {
    "strong_bear": "STRONG BEAR (BigDown)", "bear": "BEAR",
    "slight_bear": "SLIGHT BEAR",           "flat": "FLAT",
    "slight_bull": "SLIGHT BULL",           "bull": "BULL",
    "strong_bull": "STRONG BULL (BigUp)",
}

# Legacy fallback if optimization is off and no saved thresholds exist yet
MON_REGIME_THRESHOLDS_DEFAULT = REGIME_TEMPLATES[3]  # (-1.0, +1.0)

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
    return df.dropna()


# ─────────────────────────────────────────────
# WEEKEND GAP SPREAD  (Fri Close -> Mon Open, crosses weeks)
# ─────────────────────────────────────────────
def compute_weekend_spread(df: pd.DataFrame) -> np.ndarray:
    """
    Pairs each Monday's Open with the CLOSE of the immediately preceding
    trading day, keeping only cases where that prior trading day was
    actually a Friday (skips holiday-shortened weeks where Monday follows
    a Tuesday-Thursday close instead).
    """
    df_sorted = df.sort_index()
    prev_close = df_sorted["Close"].shift(1)
    prev_day   = df_sorted["DayName"].shift(1)

    mask = (df_sorted["DayName"] == "Monday") & (prev_day == "Friday")

    mon_open  = df_sorted.loc[mask, "Open"]
    fri_close = prev_close[mask]

    pct_chg = ((mon_open - fri_close) / fri_close) * 100
    return np.array(pct_chg).flatten()


# ─────────────────────────────────────────────
# DAY-TO-DAY SPREAD DISTRIBUTIONS
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
# REGIME CLASSIFICATION (generic N-bucket)
# ─────────────────────────────────────────────
def classify_regime(pct: float, boundaries: list, labels: list) -> str:
    """
    boundaries has len(labels)-1 entries, sorted ascending.
    e.g. boundaries=[-1,1], labels=["bear","flat","bull"]
         pct < -1        -> "bear"
         -1 <= pct <= 1  -> "flat"
         pct > 1         -> "bull"
    """
    for edge, lbl in zip(boundaries, labels[:-1]):
        if pct < edge:
            return lbl
    return labels[-1]


def format_regime_range(boundaries: list, labels: list, idx: int) -> str:
    if idx == 0:
        return f"< {boundaries[0]:+.2f}%"
    if idx == len(labels) - 1:
        return f"> {boundaries[-1]:+.2f}%"
    return f"{boundaries[idx-1]:+.2f}% to {boundaries[idx]:+.2f}%"


# ─────────────────────────────────────────────
# CONDITIONAL SAMPLING  (Mon regime -> rest-of-week dist)
# ─────────────────────────────────────────────
def compute_conditional_spreads(df: pd.DataFrame, boundaries: list, labels: list) -> tuple:
    """
    Splits the Mon->Tue, Tue->Wed, Wed->Thu, Thu->Fri distributions based on
    how Monday's open->close move (the "regime") behaved, using the given
    boundaries/labels (either the fixed default or optimizer output).

    The Fri->Mon weekend leg happens BEFORE the regime is known, so it uses
    the same unconditional weekend distribution for every regime bucket.

    Returns (conditional_spreads dict, week_regime dataframe).
    """
    df = df.copy()
    df["ISOWeek"] = df.index.isocalendar().week.values
    df["ISOYear"] = df.index.isocalendar().year.values
    df["WeekKey"] = df["ISOYear"].astype(str) + "-" + df["ISOWeek"].astype(str)

    mon_days = df[df["DayName"] == "Monday"][["Open", "Close", "WeekKey"]].copy()
    mon_days["MonRegimePct"] = ((mon_days["Close"] - mon_days["Open"]) / mon_days["Open"]) * 100
    mon_days["Regime"] = mon_days["MonRegimePct"].apply(lambda p: classify_regime(p, boundaries, labels))
    week_regime = mon_days[["WeekKey", "Regime", "MonRegimePct"]].copy()

    conditional = {}

    weekend_arr = compute_weekend_spread(df)
    for regime in labels:
        conditional[("Friday", "Monday", regime)] = weekend_arr

    tue_days = df[df["DayName"] == "Tuesday"][["Close", "WeekKey"]].rename(columns={"Close": "TueClose"})
    mon_open = mon_days[["Open", "WeekKey", "Regime", "MonRegimePct"]].rename(columns={"Open": "MonOpen"})
    mon_tue  = pd.merge(mon_open, tue_days, on="WeekKey").dropna()
    mon_tue["PctChg"] = ((mon_tue["TueClose"] - mon_tue["MonOpen"]) / mon_tue["MonOpen"]) * 100

    for regime in labels:
        subset = mon_tue[mon_tue["Regime"] == regime]["PctChg"].values
        conditional[("Monday", "Tuesday", regime)] = np.array(subset).flatten()

    for from_day, to_day in DAY_PAIRS[2:]:  # Tue->Wed, Wed->Thu, Thu->Fri
        from_rows = (df[df["DayName"] == from_day][["Close", "WeekKey"]]
                     .rename(columns={"Close": "From"}))
        to_rows   = (df[df["DayName"] == to_day][["Close", "WeekKey"]]
                     .rename(columns={"Close": "To"}))
        merged    = pd.merge(from_rows, to_rows, on="WeekKey")
        merged    = pd.merge(merged, week_regime, on="WeekKey")
        merged["PctChg"] = ((merged["To"] - merged["From"]) / merged["From"]) * 100

        for regime in labels:
            subset = merged[merged["Regime"] == regime]["PctChg"].values
            conditional[(from_day, to_day, regime)] = np.array(subset).flatten()

    return conditional, week_regime


def run_monte_carlo_conditional(
    start_price: float,
    conditional_spreads: dict,
    mon_regime: str,
    n_sims: int = N_SIMULATIONS,
    seed: int = 99,
) -> np.ndarray:
    """
    Same structure as run_monte_carlo but samples from the regime-specific
    sub-buckets rather than the full unconditional distributions.
    """
    rng    = np.random.default_rng(seed)
    n_cols = len(DAY_PAIRS) + 1
    paths  = np.zeros((n_sims, n_cols))
    paths[:, 0] = start_price

    for col_idx, (from_day, to_day) in enumerate(DAY_PAIRS):
        key   = (from_day, to_day, mon_regime)
        moves = np.array(conditional_spreads.get(key, np.array([0.0]))).flatten()
        if len(moves) < 5:
            moves = np.array([0.0])  # too few data points — fall back to neutral
        sampled_pct = rng.choice(moves, size=n_sims, replace=True)
        paths[:, col_idx + 1] = paths[:, col_idx] * (1 + sampled_pct / 100)

    return paths


# ─────────────────────────────────────────────
# DYNAMIC THRESHOLD OPTIMIZATION
# ─────────────────────────────────────────────
def backtest_thresholds(df: pd.DataFrame, boundaries: list, labels: list) -> dict | None:
    """
    Leave-one-week-out backtest of a regime classification scheme.

    For every historical week, the regime is classified from Monday's
    open->close move, then that week's Tue/Wed/Thu/Fri outcome is predicted
    using ONLY the *other* weeks in the same regime bucket (never itself —
    no data leakage). Prediction error is scored via:
      - MAE / RMSE of the median-predicted vs actual Friday close (%)
      - Percentile calibration (does the P25/P50/P75/P95 band actually
        contain the right fraction of real outcomes?)
      - Brier score on simple OTM strike hit/miss (covered-call / secured-put
        style probability calibration)

    Returns a metrics dict, or None if any regime bucket has too few weeks
    to be trustworthy.
    """
    df2 = df.copy()
    df2["ISOWeek"] = df2.index.isocalendar().week.values
    df2["ISOYear"] = df2.index.isocalendar().year.values
    df2["WeekKey"] = df2["ISOYear"].astype(str) + "-" + df2["ISOWeek"].astype(str)

    mon_days = df2[df2["DayName"] == "Monday"][["Open", "Close", "WeekKey"]].copy()
    mon_days["MonRegimePct"] = ((mon_days["Close"] - mon_days["Open"]) / mon_days["Open"]) * 100
    mon_days["Regime"] = mon_days["MonRegimePct"].apply(lambda p: classify_regime(p, boundaries, labels))

    counts = mon_days["Regime"].value_counts()
    for lbl in labels:
        if counts.get(lbl, 0) < MIN_BUCKET_SIZE:
            return None  # this split starves a bucket of data — reject it

    wr_map = mon_days.set_index("WeekKey")["Regime"].to_dict()

    weekday_pairs = DAY_PAIRS[1:]  # Mon->Tue ... Thu->Fri (weekend gap excluded — regime-independent)

    # Per-week actual % move for each weekday transition
    week_actual = {}
    for from_day, to_day in weekday_pairs:
        if from_day == "Monday":
            from_rows = df2[df2["DayName"] == from_day][["Open", "WeekKey"]].rename(columns={"Open": "From"})
        else:
            from_rows = df2[df2["DayName"] == from_day][["Close", "WeekKey"]].rename(columns={"Close": "From"})
        to_rows = df2[df2["DayName"] == to_day][["Close", "WeekKey"]].rename(columns={"Close": "To"})
        merged  = pd.merge(from_rows, to_rows, on="WeekKey")
        merged["Pct"] = (merged["To"] - merged["From"]) / merged["From"] * 100
        for _, r in merged.iterrows():
            week_actual.setdefault(r["WeekKey"], {})[(from_day, to_day)] = r["Pct"]

    abs_errors, sq_errors = [], []
    calib_hits  = {0.05: 0, 0.25: 0, 0.50: 0, 0.75: 0, 0.95: 0}
    calib_total = 0
    brier_scores = []
    STRIKE_PCTS  = [0.02, 0.05, 0.10]
    N_BOOT = 2000

    for wk, actual_chain in week_actual.items():
        if wk not in wr_map or len(actual_chain) < len(weekday_pairs):
            continue
        regime = wr_map[wk]

        train_moves = {t: [] for t in weekday_pairs}
        for other_wk, other_regime in wr_map.items():
            if other_wk == wk or other_regime != regime:
                continue
            other_chain = week_actual.get(other_wk, {})
            for t in weekday_pairs:
                if t in other_chain:
                    train_moves[t].append(other_chain[t])

        if any(len(v) < 5 for v in train_moves.values()):
            continue  # not enough leave-one-out training data for this week

        pred_price   = 1.0
        actual_price = 1.0
        rng = np.random.default_rng(abs(hash(wk)) % (2**32))
        sim_multiplier = np.ones(N_BOOT)

        for t in weekday_pairs:
            med_move = np.median(train_moves[t])
            pred_price   *= (1 + med_move / 100)
            actual_price *= (1 + actual_chain[t] / 100)
            sampled = rng.choice(train_moves[t], size=N_BOOT, replace=True)
            sim_multiplier *= (1 + sampled / 100)

        err = actual_price - pred_price
        abs_errors.append(abs(err))
        sq_errors.append(err ** 2)

        rank = np.mean(sim_multiplier < actual_price)
        calib_total += 1
        for q in calib_hits:
            if rank <= q:
                calib_hits[q] += 1

        for pct in STRIKE_PCTS:
            strike_up, strike_dn = 1 + pct, 1 - pct
            p_up = np.mean(sim_multiplier > strike_up)
            p_dn = np.mean(sim_multiplier < strike_dn)
            brier_scores.append((p_up - float(actual_price > strike_up)) ** 2)
            brier_scores.append((p_dn - float(actual_price < strike_dn)) ** 2)

    if calib_total < MIN_BACKTEST_WEEKS:
        return None

    mae  = float(np.mean(abs_errors)) * 100
    rmse = float(np.sqrt(np.mean(sq_errors))) * 100
    calibration_error = float(np.mean([abs(calib_hits[q] / calib_total - q) for q in calib_hits]))
    brier = float(np.mean(brier_scores))

    return {"mae": mae, "rmse": rmse, "calibration_error": calibration_error,
            "brier": brier, "n_weeks": calib_total}


def optimize_regime_thresholds(df: pd.DataFrame, n_regimes: int = N_REGIMES, verbose: bool = True) -> dict | None:
    """
    Searches candidate regime boundary schemes and returns whichever
    minimizes a composite backtest error score (lower = better):

        score = RMSE(%) + 20 * calibration_error + 20 * brier

    RMSE is already in % terms; calibration_error and brier are both in
    [0,1], so the 20x weighting brings them onto a comparable scale to a
    few-percent RMSE. Tune the weights if you want to favor one metric.
    """
    labels = REGIME_LABELS[n_regimes]

    if n_regimes == 3:
        candidates = [([b, u], {"bear": round(float(b), 2), "bull": round(float(u), 2)})
                      for b in BEAR_GRID for u in BULL_GRID]
    else:
        template = np.array(REGIME_TEMPLATES[n_regimes])
        candidates = [((template * s).round(3).tolist(), {"scale": round(float(s), 3)})
                      for s in SCALE_GRID]

    results = []
    for boundaries, meta in candidates:
        metrics = backtest_thresholds(df, boundaries, labels)
        if metrics is None:
            continue
        score = metrics["rmse"] + 20 * metrics["calibration_error"] + 20 * metrics["brier"]
        results.append({"boundaries": boundaries, "meta": meta, "score": score, **metrics})

    if not results:
        if verbose:
            print("  WARNING: no threshold combination had enough data — falling back to defaults.")
        return None

    results.sort(key=lambda r: r["score"])
    best = results[0]

    if verbose:
        print(f"  Tested {len(results)} valid threshold combinations (of {len(candidates)} candidates).")
        print(f"  Best boundaries : {[round(b, 2) for b in best['boundaries']]}  ({', '.join(labels)})")
        print(f"  RMSE            : {best['rmse']:.3f}%   MAE: {best['mae']:.3f}%")
        print(f"  Calibration err : {best['calibration_error']:.3f}   Brier: {best['brier']:.3f}")
        print(f"  Composite score : {best['score']:.3f}  (n={best['n_weeks']} backtested weeks)")

    return best


def load_optimized_thresholds() -> dict:
    if os.path.exists(THRESHOLDS_FILE):
        with open(THRESHOLDS_FILE, "r") as f:
            return json.load(f)
    return {}


def save_optimized_thresholds(all_thresholds: dict):
    with open(THRESHOLDS_FILE, "w") as f:
        json.dump(all_thresholds, f, indent=2)


def get_ticker_boundaries(ticker: str, n_regimes: int, stored: dict) -> tuple:
    """
    Returns (boundaries, labels) for a ticker — uses cached optimized values
    if present, otherwise falls back to the template default.
    """
    labels = REGIME_LABELS[n_regimes]
    entry  = stored.get(ticker, {}).get(str(n_regimes))
    if entry and "boundaries" in entry:
        return entry["boundaries"], labels
    return list(REGIME_TEMPLATES[n_regimes]), labels


# ─────────────────────────────────────────────
# MON OPEN -> FRI CLOSE WEEKLY SPREAD
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


# ─────────────────────────────────────────────
# MONTE CARLO (unconditional)
# ─────────────────────────────────────────────
def run_monte_carlo(
    start_price: float,
    spreads: dict,
    n_sims: int = N_SIMULATIONS,
    seed: int = 42,
) -> np.ndarray:
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
# VISUALIZATION  (3 rows x 3 cols)
# ─────────────────────────────────────────────
def plot_ticker(
    ticker: str,
    start_price: float,
    df: pd.DataFrame,
    spreads: dict,
    paths: np.ndarray,
    sim_summary: pd.DataFrame,
    weekly_spread: pd.DataFrame,
):
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
    print(f"    Finished week UP   > +2% : {ws['pct_up2']:.1f}%  of weeks")
    print(f"    Finished week UP   > +5% : {ws['pct_up5']:.1f}%  of weeks")
    print(f"    Finished week DOWN < -2% : {ws['pct_down2']:.1f}%  of weeks")
    print(f"    Finished week DOWN < -5% : {ws['pct_down5']:.1f}%  of weeks")

    print(f"\n  Last 5 weeks:")
    for _, row in weekly_spread.tail(5).iterrows():
        arrow = "^" if row["PctChange"] >= 0 else "v"
        print(f"    {str(row['Date_Mon'])[:10]}  open ${float(row['Mon_Open']):>8.2f}  ->  "
              f"{str(row['Date_Fri'])[:10]}  close ${float(row['Fri_Close']):>8.2f}  "
              f"{arrow} {float(row['PctChange']):+.2f}%")


def print_options_grid(ticker: str, start_price: float, fri: np.ndarray, label: str = ""):
    """
    Prints a granular +1% to +10% / -1% to -10% probability grid.
    """
    tag = f"  [{label}]" if label else ""
    print(f"\n  Options insights for Friday expiry{tag}  (start ${start_price:.2f}):")
    print(f"  {'Strike':>10}  {'Move':>6}  {'Prob':>7}  Direction")
    print(f"  {'-'*42}")

    for pct in range(12, 0, -1):
        strike = start_price * (1 + pct / 100)
        prob   = np.mean(fri > strike) * 100
        bar    = "█" * int(prob / 2)
        print(f"  ${strike:>9.2f}  {f'+{pct}%':>6}  {prob:>6.1f}%  {bar}  <- covered call")

    print()

    for pct in range(1, 13):
        strike = start_price * (1 - pct / 100)
        prob   = np.mean(fri < strike) * 100
        bar    = "█" * int(prob / 2)
        print(f"  ${strike:>9.2f}  {f'-{pct}%':>6}  {prob:>6.1f}%  {bar}  <- secured put")

    p5_fri  = np.percentile(fri,  5)
    p95_fri = np.percentile(fri, 95)
    print(f"\n    Simulated 90% confidence range: ${p5_fri:.2f} - ${p95_fri:.2f}")


def print_simulation_summary(ticker: str, start_price: float,
                              sim_summary: pd.DataFrame, paths: np.ndarray):
    print(f"\n{'='*66}")
    print(f"  {ticker} -- Monte Carlo Simulation  (prior Fri close ${start_price:.2f})")
    print(f"{'='*66}")
    print(f"  {'Day':<16} {'P5 (bear)':>11} {'P25':>11} {'Median':>11} {'P75':>11} {'P95 (bull)':>11} {'Mean':>11}")
    print(f"  {'-'*66}")
    labels = ["Fri* (prior close)", "Mon (open)", "Tue (close)",
              "Wed (close)", "Thu (close)", "Fri (close)"]
    for (day, row), lbl in zip(sim_summary.iterrows(), labels):
        print(f"  {lbl:<16}"
              f"  ${row['p5']:>9.2f}"
              f"  ${row['p25']:>9.2f}"
              f"  ${row['p50']:>9.2f}"
              f"  ${row['p75']:>9.2f}"
              f"  ${row['p95']:>9.2f}"
              f"  ${row['mean']:>9.2f}")

    fri = paths[:, -1]
    if REPORT["options_grid"]:
        print_options_grid(ticker, start_price, fri, label="Unconditional")


def print_conditional_summary(
    ticker: str,
    start_price: float,
    conditional_spreads: dict,
    week_regime: pd.DataFrame,
    boundaries: list,
    labels: list,
):
    print(f"\n{'='*66}")
    print(f"  {ticker} -- Conditional Sampling Analysis  ({len(labels)}-bucket regime)")
    print(f"  Monday regime defined by Mon open->close % change:")
    for i, regime in enumerate(labels):
        print(f"    {REGIME_DISPLAY.get(regime, regime.upper()):<24} : {format_regime_range(boundaries, labels, i)}")
    print(f"  (Fri->Mon weekend leg is unconditional — regime isn't known until Monday)")
    print(f"{'='*66}")

    for regime in labels:
        regime_weeks = week_regime[week_regime["Regime"] == regime]
        n_weeks = len(regime_weeks)
        if n_weeks == 0:
            continue

        avg_mon_move = regime_weeks["MonRegimePct"].mean()
        print(f"\n  {REGIME_DISPLAY.get(regime, regime.upper())}  "
              f"({n_weeks} weeks, avg Mon move: {avg_mon_move:+.2f}%)")
        print(f"  {'-'*62}")
        print(f"  {'Transition':<16} {'N':>5} {'Mean%':>7} {'Std%':>7} "
              f"{'P5%':>7} {'P25%':>7} {'P75%':>7} {'P95%':>7}")
        print(f"  {'-'*62}")

        for from_day, to_day in DAY_PAIRS:
            key = (from_day, to_day, regime)
            arr = conditional_spreads.get(key, np.array([]))
            if len(arr) < 3:
                print(f"  {from_day[:3]}->{to_day[:3]:<12}  (insufficient data: {len(arr)} weeks)")
                continue
            s = spread_stats(arr)
            label = f"{from_day[:3]}->{to_day[:3]}"
            note  = " *open" if from_day == "Monday" else (" *gap" if from_day == "Friday" else "")
            print(f"  {label+note:<16} {s['n']:>5} {s['mean']:>7.2f} {s['std']:>7.2f} "
                  f"{s['p5']:>7.2f} {s['p25']:>7.2f} {s['p75']:>7.2f} {s['p95']:>7.2f}")

        if REPORT["options_grid"]:
            cond_paths = run_monte_carlo_conditional(
                start_price, conditional_spreads, regime, n_sims=N_SIMULATIONS
            )
            cond_fri = cond_paths[:, -1]
            print_options_grid(ticker, start_price, cond_fri, label=REGIME_DISPLAY.get(regime, regime))


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    custom_start_prices = {
        "NVDA": None,
        "WULF": None,
        "SOFI": None,
    }

    for ticker in TICKERS:
        print(f"\n{'#'*66}")
        print(f"  Processing {ticker} ...")
        print(f"{'#'*66}")

        df = fetch_data(ticker)
        if df.empty:
            print(f"  ERROR: No data for {ticker}, skipping.")
            continue

        if custom_start_prices.get(ticker) is not None:
            start_price = float(custom_start_prices[ticker])
        else:
            mondays = df[df["DayName"] == "Monday"]
            start_price = None
            if not mondays.empty:
                last_monday_date = mondays.index[-1]
                pos = df.index.get_loc(last_monday_date)
                if pos > 0 and df["DayName"].iloc[pos - 1] == "Friday":
                    start_price = float(df["Close"].iloc[pos - 1])
                else:
                    start_price = float(mondays["Open"].iloc[-1])
            if start_price is None:
                start_price = float(df["Open"].iloc[-1])

        print(f"  Using start price : ${start_price:.2f}  (prior Friday close)")
        print(f"  Data range        : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} trading days)")

        # ── Unconditional analysis ───────────────────────────────────────
        spreads = compute_dow_spreads(df)
        if REPORT["spread_table"]:
            print_spread_table(ticker, spreads)

        weekly_spread = compute_weekly_spread(df)
        if REPORT["weekly_summary"]:
            print_weekly_spread_table(ticker, weekly_spread)

        paths       = run_monte_carlo(start_price, spreads)
        sim_summary = summarize_simulations(paths)

        if REPORT["simulation_summary"]:
            print_simulation_summary(ticker, start_price, sim_summary, paths)
        if REPORT["charts"]:
            plot_ticker(ticker, start_price, df, spreads, paths, sim_summary, weekly_spread)

        # ── Conditional sampling analysis ────────────────────────────────
        if REPORT["conditional_sampling"]:
            stored = load_optimized_thresholds()

            if OPTIMIZE_THRESHOLDS:
                print(f"\n  Optimizing regime thresholds for {ticker} ({N_REGIMES}-bucket)...")
                best = optimize_regime_thresholds(df, N_REGIMES)
                if best is not None:
                    boundaries, labels = best["boundaries"], REGIME_LABELS[N_REGIMES]
                    if SAVE_OPTIMIZED_THRESHOLDS:
                        stored.setdefault(ticker, {})[str(N_REGIMES)] = {
                            "boundaries": boundaries,
                            "labels": labels,
                            "metrics": {k: best[k] for k in
                                        ("mae", "rmse", "calibration_error", "brier", "n_weeks", "score")},
                        }
                        save_optimized_thresholds(stored)
                else:
                    boundaries, labels = get_ticker_boundaries(ticker, N_REGIMES, stored)
            else:
                boundaries, labels = get_ticker_boundaries(ticker, N_REGIMES, stored)
                print(f"\n  Using cached/default thresholds for {ticker}: "
                      f"{[round(b, 2) for b in boundaries]}")

            conditional_spreads, week_regime = compute_conditional_spreads(df, boundaries, labels)
            print_conditional_summary(ticker, start_price, conditional_spreads, week_regime, boundaries, labels)
            print(f"\n\n\n\n{'#'*66}")

    print(f"\nDone! Charts saved as PNG files in the current directory.\n")


if __name__ == "__main__":
    main()