"""Backtesting with walk-forward and lookahead prevention."""
from .backtest import BacktestEngine
from .signals import (
    SignalResult,
    SignalType,
    RiskMetrics,
    momentum_signal,
    mean_reversion_signal,
    volatility_signal,
    combine_signals,
    compute_risk_metrics,
)

__version__ = "1.0.0"
__all__ = [
    "BacktestEngine",
    "SignalResult",
    "SignalType",
    "RiskMetrics",
    "momentum_signal",
    "mean_reversion_signal",
    "volatility_signal",
    "combine_signals",
    "compute_risk_metrics",
]
