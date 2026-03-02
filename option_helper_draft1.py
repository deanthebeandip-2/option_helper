"""
Stock Monte Carlo Simulator — Day-of-Week Spread Analysis
=========================================================
Uses historical closing prices to model day-to-day % moves.
  - Mon->Tue  : Monday OPEN  -> Tuesday CLOSE
  - Tue->Wed  : Tuesday CLOSE -> Wednesday CLOSE
  - Wed->Thu  : Wednesday CLOSE -> Thursday CLOSE
  - Thu->Fri  : Thursday CLOSE -> Friday CLOSE
  - Weekly    : Monday OPEN  -> Friday CLOSE

Also includes Conditional Sampling analysis: sub-buckets the Mon->Tue
distribution based on how strong/weak Monday's open-to-close move was,
capturing momentum and mean-reversion effects.

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
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS        = ["NVDA", "WULF", "SOFI"]
LOOKBACK_YEARS = 3
N_SIMULATIONS  = 10_000_000
#N_SIMULATIONS  = 10_000
CONFIDENCE     = [0.05, 0.25, 0.50, 0.75, 0.95]

DAY_PAIRS = [
    ("Monday",    "Tuesday"),
    ("Tuesday",   "Wednesday"),
    ("Wednesday", "Thursday"),
    ("Thursday",  "Friday"),
]
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri"]

# Conditional sampling thresholds for Monday open->close move
# Defines 3 regimes: strong down, flat, strong up
MON_REGIME_THRESHOLDS = (-1.0, +1.0)  # % change Mon open->close

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
# DAY-TO-DAY SPREAD DISTRIBUTIONS
# ─────────────────────────────────────────────
def compute_dow_spreads(df: pd.DataFrame) -> dict:
    df = df.copy()
    df["ISOWeek"] = df.index.isocalendar().week.values
    df["ISOYear"] = df.index.isocalendar().year.values
    df["WeekKey"] = df["ISOYear"].astype(str) + "-" + df["ISOWeek"].astype(str)

    spreads = {}
    for from_day, to_day in DAY_PAIRS:
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
# CONDITIONAL SAMPLING  (Mon regime -> Tue dist)
# ─────────────────────────────────────────────
def compute_conditional_spreads(df: pd.DataFrame) -> dict:
    """
    Splits the Tue->Wed, Wed->Thu, Thu->Fri distributions based on how
    Monday's open-to-close move (the "regime") behaved.

    Regime buckets (Mon open->close %):
        'bear'  : < MON_REGIME_THRESHOLDS[0]   (e.g. < -1%)
        'flat'  : between thresholds            (e.g. -1% to +1%)
        'bull'  : > MON_REGIME_THRESHOLDS[1]   (e.g. > +1%)

    Returns dict keyed by (from_day, to_day, regime) with arrays of % changes.
    Also returns the Mon open->close % changes per week for regime labeling.
    """
    df = df.copy()
    df["ISOWeek"] = df.index.isocalendar().week.values
    df["ISOYear"] = df.index.isocalendar().year.values
    df["WeekKey"] = df["ISOYear"].astype(str) + "-" + df["ISOWeek"].astype(str)

    lo, hi = MON_REGIME_THRESHOLDS

    # Monday open->close % change (the regime signal)
    mon_days = df[df["DayName"] == "Monday"][["Open", "Close", "WeekKey"]].copy()
    mon_days["MonRegimePct"] = ((mon_days["Close"] - mon_days["Open"]) / mon_days["Open"]) * 100

    def regime_label(pct):
        if pct < lo:   return "bear"
        elif pct > hi: return "bull"
        else:          return "flat"

    mon_days["Regime"] = mon_days["MonRegimePct"].apply(regime_label)
    week_regime = mon_days[["WeekKey", "Regime", "MonRegimePct"]].copy()

    conditional = {}

    # For Mon->Tue: From = Mon Open, To = Tue Close, conditioned on Mon regime
    tue_days = df[df["DayName"] == "Tuesday"][["Close", "WeekKey"]].rename(columns={"Close": "TueClose"})
    mon_open = mon_days[["Open", "WeekKey", "Regime", "MonRegimePct"]].rename(columns={"Open": "MonOpen"})
    mon_tue  = pd.merge(mon_open, tue_days, on="WeekKey").dropna()
    mon_tue["PctChg"] = ((mon_tue["TueClose"] - mon_tue["MonOpen"]) / mon_tue["MonOpen"]) * 100

    for regime in ["bear", "flat", "bull"]:
        subset = mon_tue[mon_tue["Regime"] == regime]["PctChg"].values
        conditional[("Monday", "Tuesday", regime)] = np.array(subset).flatten()

    # For remaining pairs: conditioned on what Monday's regime was that week
    for from_day, to_day in DAY_PAIRS[1:]:  # Tue->Wed, Wed->Thu, Thu->Fri
        from_rows = (df[df["DayName"] == from_day][["Close", "WeekKey"]]
                     .rename(columns={"Close": "From"}))
        to_rows   = (df[df["DayName"] == to_day][["Close", "WeekKey"]]
                     .rename(columns={"Close": "To"}))
        merged    = pd.merge(from_rows, to_rows, on="WeekKey")
        merged    = pd.merge(merged, week_regime, on="WeekKey")
        merged["PctChg"] = ((merged["To"] - merged["From"]) / merged["From"]) * 100

        for regime in ["bear", "flat", "bull"]:
            subset = merged[merged["Regime"] == regime]["PctChg"].values
            conditional[(from_day, to_day, regime)] = np.array(subset).flatten()

    return conditional, week_regime


def run_monte_carlo_conditional(
    start_price: float,
    conditional_spreads: dict,
    mon_regime: str,          # "bear", "flat", or "bull"
    n_sims: int = N_SIMULATIONS,
    seed: int = 99,
) -> np.ndarray:
    """
    Same structure as run_monte_carlo but samples from the regime-specific
    sub-buckets rather than the full unconditional distributions.
    """
    rng   = np.random.default_rng(seed)
    paths = np.zeros((n_sims, 5))
    paths[:, 0] = start_price

    for col_idx, (from_day, to_day) in enumerate(DAY_PAIRS):
        key   = (from_day, to_day, mon_regime)
        moves = np.array(conditional_spreads.get(key, np.array([0.0]))).flatten()
        if len(moves) < 5:
            # Too few data points — fall back to a neutral 0% move
            moves = np.array([0.0])
        sampled_pct = rng.choice(moves, size=n_sims, replace=True)
        paths[:, col_idx + 1] = paths[:, col_idx] * (1 + sampled_pct / 100)

    return paths


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
    rng   = np.random.default_rng(seed)
    paths = np.zeros((n_sims, 5))
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
        f"{ticker}  --  Monte Carlo Week Simulation  (Mon open: ${start_price:.2f})",
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
    for patch, color in zip(bp["boxes"], [C_BLUE, C_ORANGE, C_GREEN, "#E76F51"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.axhline(0, color="red", linestyle="--", linewidth=1, alpha=0.6)
    ax1.set_title("Day-to-Day % Move Distributions\n(Mon uses Open price)",
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
    x = range(5)
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
    ax4.set_xticklabels(["Mon\n(open)", "Tue\n(close)", "Wed\n(close)", "Thu\n(close)", "Fri\n(close)"])
    ax4.set_title(f"Monte Carlo Simulation ({N_SIMULATIONS:,} paths)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("Simulated Price ($)")
    ax4.legend(fontsize=8, loc="upper left")
    ax4.grid(alpha=0.25)
    ax4.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"${x:,.2f}"))

    ax5 = fig.add_subplot(gs[2, 2])
    friday_prices = paths[:, 4]
    ax5.hist(friday_prices, bins=80, color=C_BLUE, alpha=0.75, edgecolor="none")
    for p, color in [(5, C_BEAR), (25, C_ORANGE), (50, C_MEDIAN), (75, C_GREEN), (95, C_BULL)]:
        val = np.percentile(friday_prices, p)
        ax5.axvline(val, color=color, linewidth=1.5, label=f"P{p}: ${val:.2f}")
    ax5.axvline(start_price, color="green", linewidth=2, linestyle="--",
                label=f"Mon open: ${start_price:.2f}")
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
    print(f"  (Mon->Tue uses Monday OPEN as entry price)")
    print(f"{'='*66}")
    print(f"  {'Transition':<16} {'N':>5} {'Mean%':>7} {'Std%':>7} {'P5%':>7} {'P25%':>7} {'P75%':>7} {'P95%':>7}")
    print(f"  {'-'*62}")
    for (from_day, to_day), arr in spreads.items():
        s     = spread_stats(arr)
        label = f"{from_day[:3]}->{to_day[:3]}"
        note  = " *open" if from_day == "Monday" else ""
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
    print(f"\n  Options insights for Friday expiry{tag}  (Mon open ${start_price:.2f}):")
    print(f"  {'Strike':>10}  {'Move':>6}  {'Prob':>7}  Direction")
    print(f"  {'-'*42}")

    # Upside (covered call zone) — +12 down to +1
    for pct in range(12, 0, -1):
        strike = start_price * (1 + pct / 100)
        prob   = np.mean(fri > strike) * 100
        bar    = "█" * int(prob / 2)
        print(f"  ${strike:>9.2f}  {f'+{pct}%':>6}  {prob:>6.1f}%  {bar}  <- covered call")

    print()

    # Downside (secured put zone) — -1 down to -12
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
    print(f"  {ticker} -- Monte Carlo Simulation  (Mon open ${start_price:.2f})")
    print(f"{'='*66}")
    print(f"  {'Day':<12} {'P5 (bear)':>11} {'P25':>11} {'Median':>11} {'P75':>11} {'P95 (bull)':>11} {'Mean':>11}")
    print(f"  {'-'*66}")
    labels = ["Mon (open)", "Tue (close)", "Wed (close)", "Thu (close)", "Fri (close)"]
    for (day, row), lbl in zip(sim_summary.iterrows(), labels):
        print(f"  {lbl:<12}"
              f"  ${row['p5']:>9.2f}"
              f"  ${row['p25']:>9.2f}"
              f"  ${row['p50']:>9.2f}"
              f"  ${row['p75']:>9.2f}"
              f"  ${row['p95']:>9.2f}"
              f"  ${row['mean']:>9.2f}")

    fri = paths[:, 4]
    print_options_grid(ticker, start_price, fri, label="Unconditional")


def print_conditional_summary(
    ticker: str,
    start_price: float,
    conditional_spreads: dict,
    week_regime: pd.DataFrame,
):
    lo, hi = MON_REGIME_THRESHOLDS
    print(f"\n{'='*66}")
    print(f"  {ticker} -- Conditional Sampling Analysis")
    print(f"  Monday regime defined by Mon open->close % change:")
    print(f"    BEAR : < {lo:+.1f}%   |   FLAT : {lo:+.1f}% to {hi:+.1f}%   |   BULL : > {hi:+.1f}%")
    print(f"{'='*66}")

    for regime in ["bear", "flat", "bull"]:
        regime_weeks = week_regime[week_regime["Regime"] == regime]
        n_weeks = len(regime_weeks)
        if n_weeks == 0:
            continue

        avg_mon_move = regime_weeks["MonRegimePct"].mean()
        label_map = {"bear": "BEAR Monday", "flat": "FLAT Monday", "bull": "BULL Monday"}
        emoji_map = {"bear": "v", "flat": "~", "bull": "^"}

        print(f"\n  {emoji_map[regime]} {label_map[regime]}  "
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
            note  = " *open" if from_day == "Monday" else ""
            print(f"  {label+note:<16} {s['n']:>5} {s['mean']:>7.2f} {s['std']:>7.2f} "
                  f"{s['p5']:>7.2f} {s['p25']:>7.2f} {s['p75']:>7.2f} {s['p95']:>7.2f}")

        # Run a mini Monte Carlo for this regime and print the options grid
        cond_paths = run_monte_carlo_conditional(
            start_price, conditional_spreads, regime, n_sims=N_SIMULATIONS
        )
        cond_fri = cond_paths[:, 4]
        print_options_grid(ticker, start_price, cond_fri, label=f"{label_map[regime]}")


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
            start_price = float(mondays["Open"].iloc[-1]) if not mondays.empty else float(df["Open"].iloc[-1])

        print(f"  Using Mon open    : ${start_price:.2f}")
        print(f"  Data range        : {df.index[0].date()} -> {df.index[-1].date()}  ({len(df)} trading days)")

        # ── Unconditional analysis ───────────────────────────────────────
        spreads = compute_dow_spreads(df)
        print_spread_table(ticker, spreads)

        weekly_spread = compute_weekly_spread(df)
        print_weekly_spread_table(ticker, weekly_spread)

        paths       = run_monte_carlo(start_price, spreads)
        sim_summary = summarize_simulations(paths)
        print_simulation_summary(ticker, start_price, sim_summary, paths)

        plot_ticker(ticker, start_price, df, spreads, paths, sim_summary, weekly_spread)

        # ── Conditional sampling analysis ────────────────────────────────
        conditional_spreads, week_regime = compute_conditional_spreads(df)
        print_conditional_summary(ticker, start_price, conditional_spreads, week_regime)

    print(f"\nDone! Charts saved as PNG files in the current directory.\n")


if __name__ == "__main__":
    main()