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
    pip install yfinance pandas

Usage:
    python option_chain_table.py
"""

import math
import os
import pickle
import yfinance as yf
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS = ["SOFI", "CRWV"]

# Only show strikes between current_price*(1+MIN_STRIKE_PCT/100) and
# current_price*(1+MAX_STRIKE_PCT/100) — i.e. spot up to +50% by default.
MIN_STRIKE_PCT = 0
MAX_STRIKE_PCT = 50

# Only include expirations on or before this date (format "M/D/YYYY"),
# to cut down on how many columns show up. Set to None for no cutoff.
FINAL_EXPIRATION_DATE = "8/14/2027"

# Which columns to show per expiration — set any combination to True:
SHOW_PREMIUM       = False    # "M/D"      raw bid premium
SHOW_PER_WEEK      = False    # "M/D /wk"  bid normalized per week-to-expiration
SHOW_PCT_PER_WEEK  = True    # "M/D %/wk" 100 * (bid/weeks) / current_price

PREMIUM_DECIMALS  = 2   # decimals for raw premium columns
PERWEEK_DECIMALS  = 4   # decimals for /wk and %/wk columns

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
def weeks_away(expiration_str: str) -> int:
    """
    Which "week bucket" the expiration falls into, counting today's date
    as the start of week 1. E.g. if today is 8/5:
      8/7  -> 2 days out  -> still within week 1 -> divide by 1
      8/14 -> 9 days out  -> into week 2         -> divide by 2
    """
    exp_date = datetime.strptime(expiration_str, "%Y-%m-%d")
    days = (exp_date - datetime.now()).days
    days = max(days, 1)   # avoid divide-by-zero / negative on same-day expiry
    return max(math.ceil(days / 7), 1)


def build_dataframe(current_price: float, chains: dict) -> tuple:
    """
    Builds a strike-indexed grid. For each expiration, adds whichever
    columns are toggled on:
      SHOW_PREMIUM      -> "M/D"       raw bid premium
      SHOW_PER_WEEK     -> "M/D /wk"   bid normalized per week-to-expiration
      SHOW_PCT_PER_WEEK -> "M/D %/wk"  100 * (bid/weeks) / current_price
    Any combination of the three may be True; at least one must be.

    Also adds a "<== spot" marker column flagging the strike row closest
    to current_price, and limits strikes to the configured +MIN%/+MAX%
    window around current_price.

    Returns (master_df, col_kind, col_weeks) where col_kind maps each data
    column name to "premium" / "per_week" / "pct_week" for print-time
    formatting, and col_weeks maps each data column name to its
    weeks-to-expiration bucket for the "weeks" header row.
    """
    if not (SHOW_PREMIUM or SHOW_PER_WEEK or SHOW_PCT_PER_WEEK):
        raise ValueError("At least one of SHOW_PREMIUM / SHOW_PER_WEEK / SHOW_PCT_PER_WEEK must be True")

    master_df = pd.DataFrame()
    col_kind  = {}
    col_weeks = {}

    for expiration, calls in chains.items():
        exp_date  = datetime.strptime(expiration, "%Y-%m-%d")
        col_label = f"{exp_date.month}/{exp_date.day:02d}"
        wk        = weeks_away(expiration)
        per_week  = calls["bid"] / wk

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
            temp[pct_label] = 100 * per_week / current_price
            col_kind[pct_label] = "pct_week"
            col_weeks[pct_label] = wk

        if master_df.empty:
            master_df = temp
        else:
            master_df = master_df.merge(temp, on="strike", how="outer")

    # Limit to the strike window: [current_price*(1+MIN%), current_price*(1+MAX%)]
    lo_bound = current_price * (1 + MIN_STRIKE_PCT / 100)
    hi_bound = current_price * (1 + MAX_STRIKE_PCT / 100)
    master_df = master_df[(master_df["strike"] >= lo_bound) & (master_df["strike"] <= hi_bound)]

    master_df = master_df.sort_values(by="strike", ascending=False).reset_index(drop=True)

    if master_df.empty:
        return master_df, col_kind, col_weeks

    # Mark the strike closest to the current price
    closest_idx = (master_df["strike"] - current_price).abs().idxmin()
    master_df.insert(1, "", "")
    master_df.loc[closest_idx, ""] = "<== spot"
    col_kind[""] = "marker"

    master_df = master_df.set_index("strike")
    return master_df, col_kind, col_weeks


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
    # per_week / pct_week
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
    for ticker_symbol in TICKERS:
        current_price, chains, as_of, from_cache = download_option_chain(ticker_symbol)
        if not chains:
            print(f"  No option chain data for {ticker_symbol}, skipping.\n")
            continue
        master_df, col_kind, col_weeks = build_dataframe(current_price, chains)
        if master_df.empty:
            print(f"  No strikes for {ticker_symbol} in the "
                  f"+{MIN_STRIKE_PCT}%/+{MAX_STRIKE_PCT}% window, skipping.\n")
            continue
        print_table(ticker_symbol, current_price, master_df, col_kind, col_weeks, as_of, from_cache)


if __name__ == "__main__":
    main()




'''
PWD
1) Premium/StckPrice/Weeks = premium per dollar invested per week
1) make P/SP/wk in dollar amounts. So $10/$50/2wk would mean: how much premium per 
dollar invested did I get back every week? In short: return per week.
make it in $ amount just for my understanding
2) make it uniform across all my tickers
-> final table, have the premium/week/stock price,
stack the tickers like first display all Sofi, but have a column to show it's SOFI
then put CRWV right under it, just have a ticker column to show this is all CRWV. 
make an excel file, so I can color code in Excel
3) have a lookup table, where for each one I can add my "bought at" price, just to show
how much I also lose if I sell at a loss. This way it completely balances out, direct profit
no matter what.
So for example if I bought at 100, and i select a 101 strike price, then it's
Premium + (101-100)*100 shares = pure profit

If I bought at 100 and I select 98 strike price, it's
Premium + (98-100)*100 shares = pure profit

So:
Total Premium


4) down the line, combine these 2 babies to make my MC+2MKV model help assess risk, 
then I can combine the risk + pure profit to create a predictive hedging tool
'''