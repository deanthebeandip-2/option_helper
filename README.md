# Stock State-Based Monte Carlo Forecasting Engine

A probabilistic forecasting engine that uses historical market behavior and state-aware Monte Carlo simulation to estimate future stock price distributions. Designed primarily for selecting optimal **covered call** and **cash-secured put** strikes.

---

# Overview

Instead of assuming stock prices follow a normal distribution, this project learns directly from historical trading behavior.

Given a stock ticker and the current market price, it estimates the probability distribution of future prices over the next **1–5 trading days** using millions of Monte Carlo simulations conditioned on similar historical market states.

The goal is not to predict a single future price.

The goal is to estimate the **entire probability distribution** of possible outcomes.

This allows traders to answer questions such as:

* What is the probability NVDA finishes above my covered call strike?
* What is the probability SOFI finishes below my cash-secured put strike?
* How likely is the stock to remain within my desired range?
* How much uncertainty exists over the next trading week?

---

# Key Features

* State-aware Monte Carlo simulation
* Dynamic historical sampling
* Second-order market state model
* Automatic regime optimization
* Multi-day probability forecasts (+1 through +5 trading days)
* Covered call and cash-secured put probability analysis
* Historical model evaluation and calibration (planned)

---

# How It Works

## Step 1 — Collect Historical Transitions

Download approximately three years of daily market data using Yahoo Finance.

Historical returns are separated into individual trading-day transitions:

```text
Friday Close  → Monday Open
Monday Open   → Tuesday Close
Tuesday Close → Wednesday Close
Wednesday Close → Thursday Close
Thursday Close → Friday Close
```

Each transition maintains its own empirical return distribution.

No assumptions are made about return shape.

Historical skew, fat tails, and volatility clustering are preserved.

---

## Step 2 — Learn Market States

Rather than treating every historical week equally, the model identifies the current market environment.

The current implementation uses a **second-order state model**.

Each completed trading-day transition is classified into one of three regimes:

* Bear
* Flat
* Bull

The previous two completed transitions determine the current market state.

Example:

```text
Previous Transition : Bear

Current Transition  : Bull

↓

Current State = (Bear, Bull)
```

This produces nine possible market states:

```text
Bear Bear
Bear Flat
Bear Bull

Flat Bear
Flat Flat
Flat Bull

Bull Bear
Bull Flat
Bull Bull
```

Unlike earlier versions, the market state is **not tied to Monday**.

The simulator works from any trading day by evaluating the two most recent completed transitions.

---

## Step 3 — Dynamic Regime Optimization

Bear/Flat/Bull thresholds are **not hardcoded**.

For every transition type, the model automatically searches thousands of candidate threshold combinations and evaluates them using historical leave-one-week-out backtesting.

The thresholds producing the lowest prediction error are selected and cached.

This allows every stock to develop its own optimal regime definitions.

Example:

```text
NVDA

Bear < -2.7%

Bull > +2.1%
```

```text
SOFI

Bear < -1.3%

Bull > +1.4%
```

---

## Step 4 — State-Aware Monte Carlo Simulation

The simulation begins from the current trading day.

Instead of sampling randomly from all historical observations, it only samples from historical periods matching the current market state.

Each simulated transition updates the market state before generating the next transition.

This creates a path-dependent Monte Carlo process that captures momentum and mean-reversion more realistically than unconditional sampling.

Millions of simulated price paths are generated to estimate the full probability distribution.

---

## Step 5 — Multi-Day Forecasts

Forecasts are generated for the next five trading days.

Each forecast includes:

* Expected price
* Median
* Standard deviation
* Confidence intervals
* Full probability distribution

Forecast horizons automatically adjust based on the current day.

For example:

If today is Wednesday:

```text
Thursday

Friday

Monday

Tuesday

Wednesday
```

If today is Friday:

```text
Monday

Tuesday

Wednesday

Thursday

Friday
```

The model is designed to produce useful forecasts regardless of when it is executed.

---

# Option Analysis

Using the simulated distributions, the engine estimates assignment probabilities for common option strikes.

Example:

```text
Covered Call

Strike

$195

Probability of Assignment

7.4%
```

```text
Cash-Secured Put

Strike

$165

Probability of Assignment

5.1%
```

Rather than estimating option prices, the model estimates the probability of finishing beyond each strike.

This helps identify attractive risk/reward opportunities for premium selling strategies.

---

# Planned Features

## Model Evaluation Framework

Every model revision will be evaluated using historical walk-forward testing.

Metrics include:

* +1 day forecast accuracy
* +3 day forecast accuracy
* +5 day forecast accuracy
* Calibration of confidence intervals
* Covered-call assignment probability accuracy
* Cash-secured put assignment probability accuracy

The evaluation framework will determine whether new features genuinely improve predictive performance before they are adopted.

---

## Future Research

Potential future enhancements include:

* Hidden Markov Models (HMMs)
* Regime-switching models
* Volatility-aware state variables
* SPY trend conditioning
* VIX conditioning
* Trading volume features
* ATR and momentum indicators

These features will only be incorporated if they demonstrate measurable improvements during historical backtesting.

---

# Installation

```bash
pip install yfinance pandas numpy matplotlib scipy
```

Run:

```bash
python option_helper_draft1.py
```

---

# Configuration

Primary configuration options include:

* Stock tickers
* Historical lookback period
* Number of Monte Carlo simulations
* Custom starting prices
* Report modules
* Forecast horizon

The simulator is designed to support both quick exploratory runs and large-scale research experiments.

---

# Philosophy

This project is built around one guiding principle:

> Historical market behavior contains useful probabilistic information.

Rather than attempting to predict a single future price, the objective is to estimate a realistic probability distribution conditioned on similar historical market environments.

Every new feature is evaluated quantitatively through historical backtesting.

If a feature does not improve forecasting performance, it is removed.

---

# Disclaimer

This software is intended for educational and research purposes only.

It does not provide financial advice or guarantee future investment performance.

Historical market behavior does not guarantee future results, and all investment decisions should be made independently after appropriate due diligence.
