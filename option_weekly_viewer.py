

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
import yfinance as yf
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
TICKERS = ["SOFI", "CRWV"]

# Only show strikes between current_price*(1+MIN_STRIKE_PCT/100) and
# current_price*(1+MAX_STRIKE_PCT/100) — i.e. spot up to +50% by default.
MIN_STRIKE_PCT = 0
MAX_STRIKE_PCT = 40

# Which premium columns to display per expiration:
#   "premium"   -> just the raw bid
#   "per_week"  -> just the bid normalized per week-to-expiration
#   "both"      -> raw bid AND per-week, side by side
DISPLAY_MODE = "per_week"

# Extra column: 100 * (premium/week) / current_price, i.e. weekly premium
# yield as a % of every dollar invested in the stock. Independent of
# DISPLAY_MODE — can be layered on top of any of the three modes above.
SHOW_PCT_PER_WEEK = True

PREMIUM_DECIMALS  = 2   # decimals for raw premium columns
PERWEEK_DECIMALS  = 4   # decimals for /wk and %/wk columns

pd.set_option("display.max_rows", None)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 250)


# ─────────────────────────────────────────────
# DOWNLOAD
# ─────────────────────────────────────────────
def download_option_chain(ticker_symbol: str) -> tuple:
    """
    Downloads the current price and the full set of call chains (one
    DataFrame per expiration) for a ticker.

    Returns (current_price, {expiration_str: calls_df})
    """
    tk = yf.Ticker(ticker_symbol)
    current_price = float(tk.history(period="1d")["Close"].iloc[-1])

    chains = {}
    for expiration in tk.options:
        calls = tk.option_chain(expiration).calls
        chains[expiration] = calls[["strike", "bid"]].copy()

    return current_price, chains


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


def build_dataframe(current_price: float, chains: dict, display_mode: str = DISPLAY_MODE) -> tuple:
    """
    Builds a strike-indexed grid. For each expiration, adds columns per
    display_mode:
      "premium"  -> "M/D"          raw bid premium
      "per_week" -> "M/D /wk"      bid normalized per week-to-expiration
      "both"     -> both of the above, side by side
    If SHOW_PCT_PER_WEEK is True, also adds "M/D %/wk" = 100 * (bid/weeks)
    / current_price, regardless of display_mode.

    Also adds a "<== spot" marker column flagging the strike row closest
    to current_price, and limits strikes to the configured +MIN%/+MAX%
    window around current_price.

    Returns (master_df, col_kind) where col_kind maps each data column
    name to "premium" / "per_week" / "pct_week" for print-time formatting.
    """
    if display_mode not in ("premium", "per_week", "both"):
        raise ValueError(f"DISPLAY_MODE must be 'premium', 'per_week', or 'both', got {display_mode!r}")

    master_df = pd.DataFrame()
    col_kind  = {}

    for expiration, calls in chains.items():
        exp_date  = datetime.strptime(expiration, "%Y-%m-%d")
        col_label = f"{exp_date.month}/{exp_date.day:02d}"
        wk        = weeks_away(expiration)
        per_week  = calls["bid"] / wk

        temp = calls[["strike"]].copy()

        if display_mode in ("premium", "both"):
            temp[col_label] = calls["bid"]
            col_kind[col_label] = "premium"

        if display_mode in ("per_week", "both"):
            perwk_label = f"{col_label} /wk" if display_mode == "both" else col_label
            temp[perwk_label] = per_week
            col_kind[perwk_label] = "per_week"

        if SHOW_PCT_PER_WEEK:
            pct_label = f"{col_label} %/wk"
            temp[pct_label] = 100 * per_week / current_price
            col_kind[pct_label] = "pct_week"

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
        return master_df, col_kind

    # Mark the strike closest to the current price
    closest_idx = (master_df["strike"] - current_price).abs().idxmin()
    master_df.insert(1, "", "")
    master_df.loc[closest_idx, ""] = "<== spot"
    col_kind[""] = "marker"

    master_df = master_df.set_index("strike")
    return master_df, col_kind


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


def print_table(ticker_symbol: str, current_price: float, master_df: pd.DataFrame, col_kind: dict):
    print("=" * 120)
    print(f"{ticker_symbol} Option Chain — Strike vs. Expiration (Call Bid Premium, incl. $/week)")
    print(f"Current Price: ${current_price:.2f}")
    print("=" * 120)

    headers = ["strike"] + list(master_df.columns)
    rows = [[f"{idx:.2f}"] + [format_cell(v, col_kind.get(col, "premium"))
                               for col, v in zip(master_df.columns, row)]
            for idx, row in zip(master_df.index, master_df.itertuples(index=False, name=None))]

    widths = [max(len(str(headers[i])), *(len(r[i]) for r in rows)) if rows else len(str(headers[i]))
              for i in range(len(headers))]

    def fmt_row(cells):
        return " | ".join(str(c).rjust(w) for c, w in zip(cells, widths))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for r in rows:
        print(fmt_row(r))
    print()


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    for ticker_symbol in TICKERS:
        current_price, chains = download_option_chain(ticker_symbol)
        if not chains:
            print(f"  No option chain data for {ticker_symbol}, skipping.\n")
            continue
        master_df, col_kind = build_dataframe(current_price, chains)
        if master_df.empty:
            print(f"  No strikes for {ticker_symbol} in the "
                  f"+{MIN_STRIKE_PCT}%/+{MAX_STRIKE_PCT}% window, skipping.\n")
            continue
        print_table(ticker_symbol, current_price, master_df, col_kind)


if __name__ == "__main__":
    main()





'''
just 3 quick changes:
1) could you add a ---- or a ==== row around the current price, just so it's easier for me to see where the strike price row is?
2) have a "final date" so that if I want to put in "8/14/2027" as the final expiration date I could put it in config? right now there's too many columns
3) make the 100*premium/week/stock price its own toggle, so there are 3 possible columns, and only show the columns where I mark as True
'''