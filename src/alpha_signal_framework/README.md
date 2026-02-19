# alpha_signal_framework Module

Walk-forward backtesting with lookahead bias prevention.

## Module Structure

- **backtest.py**: Main walk-forward backtesting engine
  - Handles expanding/rolling windows
  - Enforces strict temporal ordering
  - Computes Sharpe, Sortino, max drawdown metrics

- **portfolio.py**: Portfolio construction and optimization
  - Equal-weight, risk-parity, min-variance methods
  - Position size constraints and leverage control
  - Transaction cost modeling

- **signal.py**: Factor and signal definitions
  - Mean reversion, momentum, value signals
  - Custom signal subclassing interface
  - Ensemble/composite signal generation

- **execution.py**: Realistic trade execution
  - Bid-ask spread modeling
  - Partial fills and slippage
  - Market impact calculation

## Design Notes

The core insight: most backtests leak future information accidentally. This framework enforces discipline:

- Features computed using only data available at decision time
- Test periods never touch training data
- Rebalancing happens at explicitly specified frequencies

The window expansion pattern (train 2020-2021, test 2021-Q1; then train 2020-2022, test 2022-Q1) mimics how strategies would perform if deployed continuously.
