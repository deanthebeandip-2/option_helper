Stock Monte Carlo Simulator

Uses 3 years of historical price data to simulate where a stock might close by Friday — built for options traders placing covered calls and cash-secured puts.


What it does
You give it a stock ticker and a Monday opening price. It analyzes 3 years of historical day-to-day moves and runs 10 million simulated weeks to show you the probability distribution of where the stock lands each day through Friday close.
The output answers questions like:

"What's the probability NVDA closes above $185 by Friday?" → helps you pick a covered call strike
"What's the probability SOFI drops below $12 by Friday?" → helps you pick a cash-secured put strike
"Monday was a strong up day — does that change the rest of the week?" → yes, and the conditional analysis shows you how


How it works
Step 1 — Build historical distributions
For each day transition (Mon→Tue, Tue→Wed, etc.), it collects every historical instance of that move over the past 3 years. Monday uses the open price as the entry point since that's when you'd place your trade.
Step 2 — Monte Carlo simulation
It randomly samples from those real historical distributions 10 million times, chaining Mon→Tue→Wed→Thu→Fri to generate 10 million simulated weeks. No assumptions about bell curves — it uses the actual shape of each distribution including fat tails and skew.
Step 3 — Conditional sampling
Weeks are bucketed by how Monday behaved (bear / flat / bull). Each regime gets its own separate simulation, capturing momentum and mean-reversion effects.

 Options insights for Friday expiry  [Unconditional]  (Mon open $174.80):
      Strike    Move     Prob  Direction
  ------------------------------------------
  $   195.78    +12%     5.7%  ██  <- covered call 
  
  $   194.03    +11%     7.1%  ███  <- covered call 
  
  $   192.28    +10%     8.9%  ████  <- covered call 
  
  $   190.53     +9%    11.2%  █████  <- covered call 
  
  $   162.56     -7%     7.2%  ███  <- secured put 
  
  $   160.82     -8%     5.1%  ██  <- secured put 
  
  $   159.07     -9%     3.5%  █  <- secured put 
  
  $   157.32    -10%     2.4%  █  <- secured put  
  

  Simulated 90% confidence range: $14.82 - $19.91

Setup
bashpip install yfinance pandas numpy matplotlib scipy
python stock_monte_carlo.py

Configuration
At the top of stock_monte_carlo.py:
VariableDefaultDescriptionTICKERS["NVDA", "WULF", "SOFI"]Stocks to analyzeLOOKBACK_YEARS3Years of historical data to useN_SIMULATIONS10,000,000Monte Carlo paths (reduce to 10_000 for quick runs)MON_REGIME_THRESHOLDS(-1.0, +1.0)% thresholds defining bear/flat/bull Monday
To set a specific Monday open price instead of pulling the latest:
pythoncustom_start_prices = {
    "NVDA": 182.00,   # pin a specific price
    "WULF": None,     # None = use latest Monday open from yfinance
}

Output per ticker

PNG chart — 6-panel visualization saved as {TICKER}_monte_carlo.png
Console report — spread tables, weekly Mon→Fri history, simulation summary, and the full +1% to +12% / -1% to -12% options probability grid for each regime


Disclaimer
This is a personal research tool, not financial advice. Past distributions don't guarantee future returns. Always do your own due diligence before placing any options trade.
