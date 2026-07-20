# Stock Monte Carlo Simulator — State-Conditional Options Forecasting Engine

A Python Monte Carlo simulator for covered call / secured put decisions. Forecasts
+1 through +5 trading day price distributions using a state-conditional Markov
chain model, and includes tooling to measure and optimize how well-calibrated
those forecasts actually are.

## Status

Actively developed, working end-to-end. Current core pieces:

- **State-conditional Monte Carlo engine** — the trading week is modeled as a
  5-transition cycle (Fri→Mon weekend gap, Mon→Tue, Tue→Wed, Wed→Thu, Thu→Fri).
  Each transition is classified bear/flat/bull, and the **market state** (regime
  of the last two transitions) drives which historical distribution each
  simulated path samples from next. Paths diverge day-by-day as a genuine
  Markov chain — this replaced an earlier, more narrow "Monday regime" model.
- **Dynamic regime thresholds** — bear/bull cutoffs are optimized per
  transition type rather than hardcoded. Two optimizer modes:
  - `fast` — scores candidate thresholds against each type's historical
    (move, next-move) pool directly (seconds).
  - `full` — bakes the real Monte Carlo engine into the threshold search
    itself: each candidate is scored by walking it forward through history
    and running the actual simulation engine, then optimized via
    coordinate ascent across the 5 transition types (minutes — the more
    faithful, more expensive option).
  Both modes score candidates on the same 0–100 scale, combining
  calibration, directional assignment accuracy, and 1-day MAE.
- **Options grid** — dynamically walks outward from the current price to find
  covered-call and secured-put strikes, stopping once probability drops below
  a display threshold, for both the unconditional weekly forecast and every
  state-conditional forecast horizon.
- **Calibration backtest** — a separate historical walk-forward evaluator that
  checks whether the engine's predicted 50/75/90/95% confidence intervals
  actually contain future closes at the rate they claim to, across all 5
  forecast horizons and all configured tickers. Runs automatically after the
  normal report and prints a calibration table + pass/fail assessment per
  horizon.

## Known tradeoffs / open items

- The `full` optimizer mode uses coordinate ascent (one transition type at a
  time) rather than a true joint search across all 5 types — a joint search
  is combinatorially infeasible given the 9-state machine mixes all types
  together. This converges to a good answer, not a provably optimal one.
- Runtime for `full` mode scales with `FULL_EVALUATOR_CONFIG` (grid
  coarseness, walk-forward stride, sim count) — roughly ~20 minutes on 10
  years of daily data at current defaults. `fast` mode is available as a
  cheap fallback/sanity check.
- Directional "assignment accuracy" (does the model call the right direction
  at +1 day) is inherently close to a coin flip for most equities — this is
  expected, not a bug, and worth keeping in mind when reading Overall Score.

## Requirements

```
pip install yfinance pandas numpy matplotlib scipy
```

## Usage

```
python option_helper.py
```

Configure tickers, lookback window, forecast horizons, optimizer mode, and
report sections at the top of the script (`TICKERS`, `LOOKBACK_YEARS`,
`FORECAST_HORIZONS`, `OPTIMIZER_MODE`, `REPORT`).
