# Interview Prep: alpha-signal-framework

## Elevator Pitch

Backtesting framework with walk-forward validation and lookahead prevention. Every retail trader has curve-fitted a strategy on historical data and lost money live. This enforces strict temporal ordering to prevent that mistake.

## The Core Issue

**Lookahead bias**: Using future data accidentally in backtest. Example: "If I know the price tomorrow, I can trade perfectly today."

## The Solution

Walk-forward validation:
1. Split data into periods
2. Train on period 1, test on period 2
3. Train on periods 1+2, test on period 3
4. Repeat, never using future data

## Why This Matters

If strategy beats walk-forward test (where you can't peek forward), it might beat live markets. If it only beats simple backtest, it's likely overfitted.

## Interview Points

- Understand survivorship bias (markets remove bad traders)
- Understand overfitting (curve-fitting on historical data)
- Know walk-forward methodology (used by institutions)

