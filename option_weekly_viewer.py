"""
Option Chain Premium Table — Strike vs. Expiration
====================================================
For each ticker, downloads the full option chain (calls), builds a
strike x expiration grid of bid premiums, and prints it with:
  - a "current price" marker row so you can see where spot sits relative
    to the strike grid
  - a premium-per-week helper column next to every expiration, so you can
    compare normalized weekly cost across different-dated contracts
  - an optional %-of-stock-price-per-week column, so you can compare
    weekly premium yield across tickers at different price points

Structure:
    main() -> download_option_chain() -> build_dataframe() -> print_table()

Install deps:
    pip install yfinance pandas openpyxl

Usage:
    python option_weekly_viewer.py
"""

import csv
import os
import pickle
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS = ["SOFI", "CRWV"]

# How many strike rows to show above and below the current price. Sorted
# descending by strike, this keeps N rows above spot (higher strikes) and
# M rows below spot (lower strikes) — replaces the old %-window approach.
ROWS_ABOVE_SPOT = 15   # n
ROWS_BELOW_SPOT = 3   # m

# Only include expirations on or before this date (format "M/D/YYYY"),
# to cut down on how many columns show up. Set to None for no cutoff.
FINAL_EXPIRATION_DATE = "8/14/2027"

# Which columns to show per expiration — set any combination to True:
SHOW_PREMIUM       = False    # "M/D"      raw bid premium
SHOW_PER_WEEK      = False    # "M/D /wk"  bid normalized per week-to-expiration
SHOW_PCT_PER_WEEK  = False     # "M/D %/wk" ROI/week:  100 * (bid/weeks) / current_price
SHOW_PCT_PER_YEAR  = True    # "M/D %/yr" ROI/year:  pct_week * 52
# Whenever SHOW_PCT_PER_WEEK / SHOW_PCT_PER_YEAR are on, a matching "x/wk"
# / "x/yr" multiple column is added automatically (1 + pct/100), e.g. a
# +100% ROI shows as 2.00x, +150% shows as 2.50x — every dollar invested
# grows to that multiple over that period.

PREMIUM_DECIMALS  = 2   # decimals for raw premium columns
PERWEEK_DECIMALS  = 4   # decimals for /wk and %/wk / %/yr columns
MULTIPLE_DECIMALS = 2   # decimals for x/wk and x/yr columns

# When only SHOW_PCT_PER_WEEK and/or SHOW_PCT_PER_YEAR are on (i.e. no raw
# $ premium columns), main() switches to "stacked" mode: every ticker in
# TICKERS is combined into ONE table (a "Ticker" column labels each row),
# columns are week-buckets (Wk1, Wk2, ...) rather than specific dates so
# tickers with different expiration calendars still line up, and the
# result is saved to disk for you to color-code in Excel.
REPORTS_DIR     = "reports"                # xlsx/csv exports are saved here
EXPORT_FORMAT   = "xlsx"                   # "xlsx" or "csv"
EXPORT_FILENAME = "option_roi_stacked"     # timestamp + extension added automatically

# Cache directory — every successful live download is saved here per
# ticker. When the market's closed (nights/weekends/holidays*), the
# script reuses the last cached snapshot instead of hitting yfinance.
# (*No holiday calendar — a market holiday during regular hours will
# still be treated as "open" and attempt a live fetch.)
CACHE_DIR = "option_chain_cache"

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


# ─────────────────────────────────────────────
# MARKET HOURS / CACHE
# ─────────────────────────────────────────────
def is_market_open(now: datetime = None) -> bool:
    """
    Regular NYSE/Nasdaq hours: weekdays 9:30am-4:00pm America/New_York.
    Does not account for market holidays (e.g. Thanksgiving, Good Friday)
    — those will be misreported as "open" since they fall on a weekday.
    """
    if now is None:
        now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:   # Sat=5, Sun=6
        return False
    open_time  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return open_time <= now <= close_time


def _cache_path(ticker_symbol: str) -> str:
    return os.path.join(CACHE_DIR, f"{ticker_symbol}.pkl")


def save_cache(ticker_symbol: str, current_price: float, chains: dict, as_of: datetime):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(ticker_symbol), "wb") as f:
        pickle.dump({"current_price": current_price, "chains": chains, "as_of": as_of}, f)


def load_cache(ticker_symbol: str) -> dict:
    """Returns {"current_price", "chains", "as_of"} or None if no cache exists yet."""
    path = _cache_path(ticker_symbol)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
def download_option_chain(ticker_symbol: str) -> tuple:
    """
    Downloads the current price and the full set of call chains (one
    DataFrame per expiration) for a ticker, but only actually hits
    yfinance while the market's open. Outside market hours (evenings,
    weekends), it reuses the last cached live snapshot instead — so
    running this on a Saturday still works and shows you what the chain
    looked like as of the last close, rather than failing or fetching a
    quote that's already just as stale from the source itself.

    Returns (current_price, {expiration_str: calls_df}, as_of, from_cache)
    """
    if not is_market_open():
        cached = load_cache(ticker_symbol)
        if cached is not None:
            return cached["current_price"], cached["chains"], cached["as_of"], True
        print(f"  [{ticker_symbol}] Market closed and no cache found yet — "
              f"fetching live anyway (data will reflect the last close).")

    tk = yf.Ticker(ticker_symbol)
    current_price = float(tk.history(period="1d")["Close"].iloc[-1])

    cutoff = (datetime.strptime(FINAL_EXPIRATION_DATE, "%m/%d/%Y")
              if FINAL_EXPIRATION_DATE else None)

    chains = {}
    for expiration in tk.options:
        if cutoff is not None and datetime.strptime(expiration, "%Y-%m-%d") > cutoff:
            continue
        calls = tk.option_chain(expiration).calls
        chains[expiration] = calls[["strike", "bid"]].copy()

    as_of = datetime.now()
    save_cache(ticker_symbol, current_price, chains, as_of)
    return current_price, chains, as_of, False


# ─────────────────────────────────────────────
# BUILD DATAFRAME
# ─────────────────────────────────────────────
def _week_reference_date(now: datetime = None):
    """
    The date "today" counts as for week-bucket purposes. Normally just
    today — but once today's trading session has closed (after 4pm ET on
    a weekday, or any time on a weekend), the reference rolls forward to
    tomorrow, since today's own expiration (if any) has already settled
    and no longer counts as "still available this week".
    """
    if now is None:
        now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:   # weekend — the prior Friday's close has already passed
        return (now + timedelta(days=1)).date()
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now >= close_time:
        return (now + timedelta(days=1)).date()
    return now.date()


def weeks_away(expiration_str: str) -> int:
    """
    How many weeks away the expiration is. Counts from today's date,
    unless today's trading session has already closed (after 4pm ET on a
    weekday, or any time on a weekend) — in which case it counts from
    tomorrow instead, since today's own expiration has already settled.
    E.g. if today is Friday 8/7:
      Before the close: 8/7 -> week 1, 8/14 -> week 2
      After the close:  8/14 -> week 1, 8/21 -> week 2
    """
    exp_date  = datetime.strptime(expiration_str, "%Y-%m-%d").date()
    reference = _week_reference_date()
    days = (exp_date - reference).days
    days = max(days, 0)
    return days // 7 + 1


def build_dataframe(current_price: float, chains: dict) -> tuple:
    """
    Builds a strike-indexed grid. For each expiration, adds whichever
    columns are toggled on:
      SHOW_PREMIUM      -> "M/D"       raw bid premium
      SHOW_PER_WEEK     -> "M/D /wk"   bid normalized per week-to-expiration
      SHOW_PCT_PER_WEEK -> "M/D %/wk"  ROI/week, plus "M/D x/wk" multiple
      SHOW_PCT_PER_YEAR -> "M/D %/yr"  ROI/year, plus "M/D x/yr" multiple
    Any combination may be True; at least one must be.

    Also adds a "<== spot" marker column flagging the strike row closest
    to current_price, and keeps ROWS_ABOVE_SPOT rows above + ROWS_BELOW_SPOT
    rows below that strike.

    Returns (master_df, col_kind, col_weeks) where col_kind maps each data
    column name to "premium" / "per_week" / "pct_week" / "pct_year" /
    "multiple" for print-time formatting, and col_weeks maps each data
    column name to its weeks-to-expiration bucket for the "weeks" header row.
    """
    if not (SHOW_PREMIUM or SHOW_PER_WEEK or SHOW_PCT_PER_WEEK or SHOW_PCT_PER_YEAR):
        raise ValueError("At least one of SHOW_PREMIUM / SHOW_PER_WEEK / "
                          "SHOW_PCT_PER_WEEK / SHOW_PCT_PER_YEAR must be True")

    master_df = pd.DataFrame()
    col_kind  = {}
    col_weeks = {}

    for expiration, calls in chains.items():
        exp_date  = datetime.strptime(expiration, "%Y-%m-%d")
        col_label = f"{exp_date.month}/{exp_date.day:02d}"
        wk        = weeks_away(expiration)
        per_week  = calls["bid"] / wk
        pct_week  = 100 * per_week / current_price

        temp = calls[["strike"]].copy()

        if SHOW_PREMIUM:
            temp[col_label] = calls["bid"]
            col_kind[col_label] = "premium"
            col_weeks[col_label] = wk

        if SHOW_PER_WEEK:
            perwk_label = f"{col_label} /wk"
            temp[perwk_label] = per_week
            col_kind[perwk_label] = "per_week"
            col_weeks[perwk_label] = wk

        if SHOW_PCT_PER_WEEK:
            pct_label = f"{col_label} %/wk"
            temp[pct_label] = pct_week
            col_kind[pct_label] = "pct_week"
            col_weeks[pct_label] = wk

            multwk_label = f"{col_label} x/wk"
            temp[multwk_label] = 1 + pct_week / 100
            col_kind[multwk_label] = "multiple"
            col_weeks[multwk_label] = wk

        if SHOW_PCT_PER_YEAR:
            pct_year = pct_week * 52
            pctyr_label = f"{col_label} %/yr"
            temp[pctyr_label] = pct_year
            col_kind[pctyr_label] = "pct_year"
            col_weeks[pctyr_label] = wk

            multyr_label = f"{col_label} x/yr"
            temp[multyr_label] = 1 + pct_year / 100
            col_kind[multyr_label] = "multiple"
            col_weeks[multyr_label] = wk

        if master_df.empty:
            master_df = temp
        else:
            master_df = master_df.merge(temp, on="strike", how="outer")

    master_df = master_df.sort_values(by="strike", ascending=False).reset_index(drop=True)

    if master_df.empty:
        return master_df, col_kind, col_weeks

    # Trim to ROWS_ABOVE_SPOT rows above + ROWS_BELOW_SPOT rows below the
    # strike closest to current_price
    closest_pos = (master_df["strike"] - current_price).abs().idxmin()
    start = max(0, closest_pos - ROWS_ABOVE_SPOT)
    end   = min(len(master_df) - 1, closest_pos + ROWS_BELOW_SPOT)
    master_df = master_df.iloc[start:end + 1].reset_index(drop=True)

    # Mark the strike closest to the current price (recompute — position
    # may have shifted after trimming)
    closest_idx = (master_df["strike"] - current_price).abs().idxmin()
    master_df.insert(1, "", "")
    master_df.loc[closest_idx, ""] = "<== spot"
    col_kind[""] = "marker"

    master_df = master_df.set_index("strike")
    return master_df, col_kind, col_weeks


def build_stacked_table(ticker_data: list) -> tuple:
    """
    ticker_data: list of (ticker_symbol, current_price, chains) tuples.

    Builds ONE combined table across every ticker — a "Ticker" column
    labels each row so multiple tickers stack vertically instead of
    spreading out into side-by-side wide tables.

    Only includes the multiple columns ("x/wk", "x/yr") — the plain %
    columns are dropped, since 1+pct/100 at a glance is what matters here
    (raw $ premiums also stay excluded, since they aren't comparable
    across tickers at different share prices). Where both are enabled,
    x/wk sits immediately left of x/yr for a given week.

    Keeps ROWS_ABOVE_SPOT rows above + ROWS_BELOW_SPOT rows below the
    strike closest to each ticker's current price. Missing values (e.g. a
    ticker with no expiration in a given week bucket) are filled with 0
    rather than NaN.

    Assumes at most one expiration per week-bucket per ticker. If a
    ticker has two expirations landing in the same bucket (e.g. a weekly
    and a monthly overlapping), the later one processed wins.

    Returns (combined_df, date_row, week_row):
      combined_df — flat-column data (Ticker, strike, "", then one column
        per week/metric, e.g. "Wk1 x/wk").
      date_row/week_row — display header lists aligned to combined_df's
        columns: date_row holds the expiration date per week bucket (e.g.
        "8/07", "" for the leading columns); week_row holds the week
        number for data columns and "Ticker"/"strike"/"" for the leading
        ones. The date for a given week bucket is taken from whichever
        ticker hits it first — this assumes tickers share the same weekly
        expiration calendar (true for most optionable equities); a ticker
        missing weeklies could show a mismatched date for that bucket.
    """
    all_frames = []
    week_dates = {}   # wk -> "M/D" date string, first ticker to hit it wins

    for ticker_symbol, current_price, chains in ticker_data:
        rows = {}
        for expiration, calls in chains.items():
            exp_date  = datetime.strptime(expiration, "%Y-%m-%d")
            date_str  = f"{exp_date.month}/{exp_date.day:02d}"
            wk        = weeks_away(expiration)
            week_dates.setdefault(wk, date_str)
            for _, r in calls.iterrows():
                pct_week = 100 * (r["bid"] / wk) / current_price
                entry = rows.setdefault(r["strike"], {})
                if SHOW_PCT_PER_WEEK:
                    entry[f"Wk{wk} x/wk"] = 1 + pct_week / 100
                if SHOW_PCT_PER_YEAR:
                    entry[f"Wk{wk} x/yr"] = 1 + (pct_week * 52) / 100

        if not rows:
            continue

        df = pd.DataFrame.from_dict(rows, orient="index")
        df.index.name = "strike"
        df = df.reset_index()
        df = df.sort_values(by="strike", ascending=False).reset_index(drop=True)
        if df.empty:
            continue

        # Trim to ROWS_ABOVE_SPOT rows above + ROWS_BELOW_SPOT rows below spot
        closest_pos = (df["strike"] - current_price).abs().idxmin()
        start = max(0, closest_pos - ROWS_ABOVE_SPOT)
        end   = min(len(df) - 1, closest_pos + ROWS_BELOW_SPOT)
        df = df.iloc[start:end + 1].reset_index(drop=True)

        closest_idx = (df["strike"] - current_price).abs().idxmin()
        df.insert(0, "", "")
        df.loc[closest_idx, ""] = "<== spot"
        df.insert(0, "Ticker", ticker_symbol)

        all_frames.append(df)

    if not all_frames:
        return pd.DataFrame(), [], []

    combined = pd.concat(all_frames, ignore_index=True, sort=False)

    metric_cols = sorted(
        (c for c in combined.columns if c not in ("Ticker", "strike", "")),
        key=lambda c: (int(c.split(" ")[0][2:]), c.endswith("x/yr"))   # week #, then x/wk before x/yr
    )
    combined = combined[["Ticker", "strike", ""] + metric_cols]
    combined[metric_cols] = combined[metric_cols].fillna(0).round(MULTIPLE_DECIMALS)

    date_row = ["", "", ""] + [week_dates.get(int(c.split(" ")[0][2:]), "") for c in metric_cols]
    week_row = ["Ticker", "strike", ""] + [c.split(" ")[0][2:] for c in metric_cols]

    return combined, date_row, week_row


def export_stacked_table(df: pd.DataFrame, date_row: list, week_row: list):
    """
    Writes date_row and week_row as two manual header rows above the data
    (no pandas column header), so the sheet/CSV shows exactly:
      row 1: expiration dates (e.g. "8/07")
      row 2: "Ticker", "strike", "", week numbers (e.g. "1")
      row 3+: data
    """
    if df.empty:
        print("No data to export.")
        return
    os.makedirs(REPORTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%H%M%S")
    base = f"{EXPORT_FILENAME}_{timestamp}"

    if EXPORT_FORMAT == "xlsx":
        path = os.path.join(REPORTS_DIR, base + ".xlsx")
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, header=False, startrow=2)
            ws = writer.sheets["Sheet1"]
            for col_idx, (d, w) in enumerate(zip(date_row, week_row), start=1):
                ws.cell(row=1, column=col_idx, value=d)
                ws.cell(row=2, column=col_idx, value=w)
    elif EXPORT_FORMAT == "csv":
        path = os.path.join(REPORTS_DIR, base + ".csv")
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(date_row)
            writer.writerow(week_row)
            for row in df.itertuples(index=False, name=None):
                writer.writerow(row)
    else:
        raise ValueError(f"EXPORT_FORMAT must be 'xlsx' or 'csv', got {EXPORT_FORMAT!r}")
    print(f"Saved stacked ROI table -> {path}")


def print_stacked_table(df: pd.DataFrame, date_row: list, week_row: list):
    """Console version of the same 2-row header (date, week) + data rows."""
    cols = list(df.columns)
    data_rows = [list(row) for row in
                 df.astype(str).itertuples(index=False, name=None)]

    widths = [max(len(str(date_row[i])), len(str(week_row[i])),
                  *(len(r[i]) for r in data_rows)) if data_rows
              else max(len(str(date_row[i])), len(str(week_row[i])))
              for i in range(len(cols))]

    def fmt_row(cells):
        return " | ".join(str(c).rjust(w) for c, w in zip(cells, widths))

    print(fmt_row(date_row))
    print(fmt_row(week_row))
    print("-+-".join("-" * w for w in widths))
    for r in data_rows:
        print(fmt_row(r))


# ─────────────────────────────────────────────
# PRINT
# ─────────────────────────────────────────────
def format_cell(value, kind: str) -> str:
    if pd.isna(value):
        return ""
    if kind == "marker":
        return str(value)
    if kind == "premium":
        return f"{value:.{PREMIUM_DECIMALS}f}"
    if kind == "multiple":
        return f"{value:.{MULTIPLE_DECIMALS}f}x"
    # per_week / pct_week / pct_year
    return f"{value:.{PERWEEK_DECIMALS}f}"


def print_table(ticker_symbol: str, current_price: float, master_df: pd.DataFrame,
                 col_kind: dict, col_weeks: dict, as_of: datetime, from_cache: bool):
    print("=" * 120)
    print(f"{ticker_symbol} Option Chain — Strike vs. Expiration (Call Bid Premium, incl. $/week)")
    print(f"Current Price: ${current_price:.2f}")
    status = "CACHED — market closed" if from_cache else "LIVE"
    print(f"As of: {as_of:%Y-%m-%d %I:%M %p}  [{status}]")
    print("=" * 120)

    headers     = ["strike"] + list(master_df.columns)
    weeks_row   = ["weeks"] + [str(col_weeks[col]) if col in col_weeks else ""
                                for col in master_df.columns]
    rows = [[f"{idx:.2f}"] + [format_cell(v, col_kind.get(col, "premium"))
                               for col, v in zip(master_df.columns, row)]
            for idx, row in zip(master_df.index, master_df.itertuples(index=False, name=None))]

    widths = [max(len(str(headers[i])), len(str(weeks_row[i])),
                  *(len(r[i]) for r in rows)) if rows else max(len(str(headers[i])), len(str(weeks_row[i])))
              for i in range(len(headers))]

    def fmt_row(cells):
        return " | ".join(str(c).rjust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print(fmt_row(weeks_row))
    print("-+-".join("-" * w for w in widths))

    marker_col_idx = headers.index("") if "" in headers else None
    full_width     = len(fmt_row(headers))

    for r in rows:
        is_spot = marker_col_idx is not None and r[marker_col_idx] == "<== spot"
        if is_spot:
            print("=" * full_width)
        print(fmt_row(r))
        if is_spot:
            print("=" * full_width)
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # If the only active metrics are the ROI-style % columns (no raw $
    # premium), combine every ticker into one stacked table and export it
    # instead of printing separate per-ticker date-column grids.
    stacked_mode = (SHOW_PCT_PER_WEEK or SHOW_PCT_PER_YEAR) and not SHOW_PREMIUM and not SHOW_PER_WEEK

    if stacked_mode:
        ticker_data = []
        for ticker_symbol in TICKERS:
            current_price, chains, as_of, from_cache = download_option_chain(ticker_symbol)
            if not chains:
                print(f"  No option chain data for {ticker_symbol}, skipping.\n")
                continue
            ticker_data.append((ticker_symbol, current_price, chains))

        combined, date_row, week_row = build_stacked_table(ticker_data)
        if combined.empty:
            print("No data available to build the stacked ROI table.")
            return

        print_stacked_table(combined, date_row, week_row)
        print()
        export_stacked_table(combined, date_row, week_row)
        return

    for ticker_symbol in TICKERS:
        current_price, chains, as_of, from_cache = download_option_chain(ticker_symbol)
        if not chains:
            print(f"  No option chain data for {ticker_symbol}, skipping.\n")
            continue
        master_df, col_kind, col_weeks = build_dataframe(current_price, chains)
        if master_df.empty:
            print(f"  No strikes for {ticker_symbol} in the configured "
                  f"ROWS_ABOVE_SPOT/ROWS_BELOW_SPOT window, skipping.\n")
            continue
        print_table(ticker_symbol, current_price, master_df, col_kind, col_weeks, as_of, from_cache)


if __name__ == "__main__":
    main()


# ─────────────────────────────────────────────
# ROADMAP / NOTES (not yet implemented)
# ─────────────────────────────────────────────
'''
Have a lookup table, where for each ticker I can add my "Cost Basis" price,
just to show how much I also lose if I sell at a loss. This way it
completely balances out, direct profit no matter what.
So for example if I bought at 100, and I select a 101 strike price, then it's
101's Premium + (101-100)*100 shares = pure profit
If I bought at 100 and I select 98 strike price, it's
98's Premium + (98-100)*100 shares = pure profit
So: Total Premium + (strike price - cost basis)*100 shares = P/L for this option.

Down the line, combine these 2 babies to make my MC+2MKV model help assess
risk, then I can combine the risk + pure profit to create a predictive
hedging tool. Each strike price will have a "probability" attached, just to
easily visualize how possible this price is 1 week away.
'''