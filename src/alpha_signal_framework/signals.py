"""Alpha signal generation and portfolio construction.

Provides signal generators (momentum, mean-reversion, volatility), a signal
combiner for multi-factor models, and risk metrics for portfolio analysis.
All signals are designed to prevent lookahead bias by using only past data.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Minimum data points required for signal computation
MIN_LOOKBACK = 5

# Default annualization factor (trading days)
TRADING_DAYS_PER_YEAR = 252

# Signal clipping bounds to prevent extreme positions
DEFAULT_SIGNAL_CLIP = 3.0


class SignalType(Enum):
    """Classification of signal types."""

    MOMENTUM = "MOMENTUM"
    MEAN_REVERSION = "MEAN_REVERSION"
    VOLATILITY = "VOLATILITY"
    CUSTOM = "CUSTOM"


@dataclass(frozen=True)
class SignalResult:
    """Output from a signal computation.

    Attributes:
        name: Signal identifier.
        signal_type: Classification of the signal.
        values: Series of signal values indexed by date.
        metadata: Extra information about the computation.
    """

    name: str
    signal_type: SignalType
    values: pd.Series
    metadata: Dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RiskMetrics:
    """Portfolio risk metrics.

    Attributes:
        annualized_return: Annualized return as decimal.
        annualized_volatility: Annualized volatility.
        sharpe_ratio: Sharpe ratio (excess return / volatility).
        max_drawdown: Maximum peak-to-trough drawdown as decimal.
        sortino_ratio: Downside risk-adjusted return.
        calmar_ratio: Return / max drawdown.
        win_rate: Fraction of positive return periods.
        profit_factor: Gross profit / gross loss.
    """

    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    sortino_ratio: float
    calmar_ratio: float
    win_rate: float
    profit_factor: float


def momentum_signal(
    prices: pd.Series,
    lookback: int = 20,
    normalize: bool = True,
) -> SignalResult:
    """Compute momentum signal from price returns over lookback period.

    Uses the rate of change (ROC) of prices to generate a momentum signal.
    Positive values indicate upward momentum, negative for downward.

    Args:
        prices: Time series of prices, indexed by date.
        lookback: Number of periods for return calculation.
        normalize: Whether to z-score normalize the signal.

    Returns:
        SignalResult with momentum values.

    Raises:
        ValueError: If insufficient data for lookback.
    """
    _validate_price_series(prices, lookback)

    returns = prices.pct_change(lookback)
    signal = returns.dropna()

    if normalize and len(signal) >= MIN_LOOKBACK:
        signal = _z_score_normalize(signal)

    logger.debug(
        "Momentum signal: lookback=%d, mean=%.4f, std=%.4f",
        lookback, signal.mean(), signal.std(),
    )

    return SignalResult(
        name=f"momentum_{lookback}",
        signal_type=SignalType.MOMENTUM,
        values=signal,
        metadata={"lookback": lookback, "mean": signal.mean()},
    )


def mean_reversion_signal(
    prices: pd.Series,
    lookback: int = 20,
    normalize: bool = True,
) -> SignalResult:
    """Compute mean-reversion signal using z-score of price vs moving average.

    Measures how far price has deviated from its rolling mean, normalized
    by rolling standard deviation. Extreme positive values suggest overbought
    (signal to sell); extreme negative values suggest oversold (signal to buy).

    Args:
        prices: Time series of prices.
        lookback: Rolling window size.
        normalize: Whether to clip to +/- DEFAULT_SIGNAL_CLIP.

    Returns:
        SignalResult with mean-reversion values (negated: high = buy signal).

    Raises:
        ValueError: If insufficient data.
    """
    _validate_price_series(prices, lookback)

    rolling_mean = prices.rolling(window=lookback).mean()
    rolling_std = prices.rolling(window=lookback).std()

    # Z-score: how far price is from its mean, in std units
    z_score = (prices - rolling_mean) / rolling_std
    # Negate: oversold -> positive signal (buy), overbought -> negative
    signal = -z_score.dropna()

    if normalize:
        signal = signal.clip(-DEFAULT_SIGNAL_CLIP, DEFAULT_SIGNAL_CLIP)

    logger.debug(
        "Mean-reversion signal: lookback=%d, mean=%.4f",
        lookback, signal.mean(),
    )

    return SignalResult(
        name=f"mean_reversion_{lookback}",
        signal_type=SignalType.MEAN_REVERSION,
        values=signal,
        metadata={"lookback": lookback, "mean": signal.mean()},
    )


def volatility_signal(
    prices: pd.Series,
    lookback: int = 20,
    target_vol: float = 0.15,
) -> SignalResult:
    """Compute volatility-targeting signal.

    Scales position size inversely with realized volatility. When volatility
    is high, reduce exposure; when low, increase it.

    Args:
        prices: Time series of prices.
        lookback: Window for volatility estimation.
        target_vol: Target annualized volatility (e.g., 0.15 = 15%).

    Returns:
        SignalResult with volatility-scaled signal values.

    Raises:
        ValueError: If insufficient data or invalid target_vol.
    """
    _validate_price_series(prices, lookback)
    if target_vol <= 0:
        raise ValueError(f"target_vol must be positive, got {target_vol}")

    daily_returns = prices.pct_change().dropna()
    rolling_vol = daily_returns.rolling(window=lookback).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    )

    # Position scale: target_vol / realized_vol (inverse vol weighting)
    signal = (target_vol / rolling_vol).dropna()
    signal = signal.clip(0, DEFAULT_SIGNAL_CLIP)

    logger.debug(
        "Volatility signal: lookback=%d, target=%.2f, mean_scale=%.2f",
        lookback, target_vol, signal.mean(),
    )

    return SignalResult(
        name=f"vol_target_{lookback}",
        signal_type=SignalType.VOLATILITY,
        values=signal,
        metadata={
            "lookback": lookback,
            "target_vol": target_vol,
            "mean_scale": signal.mean(),
        },
    )


def combine_signals(
    signals: List[SignalResult],
    weights: Optional[List[float]] = None,
    normalize: bool = True,
) -> pd.Series:
    """Combine multiple signals into a single composite signal.

    Aligns signals by index, applies weights, and sums. Missing values
    are forward-filled to prevent gaps.

    Args:
        signals: List of SignalResult to combine.
        weights: Per-signal weights (equal weight if None).
        normalize: Whether to z-score the combined signal.

    Returns:
        Combined signal as pd.Series.

    Raises:
        ValueError: If signals list is empty or weights length mismatches.
    """
    _validate_signal_list(signals, weights)

    if weights is None:
        weights = [1.0 / len(signals)] * len(signals)

    combined = _weighted_sum(signals, weights)

    if normalize and len(combined) >= MIN_LOOKBACK:
        combined = _z_score_normalize(combined)

    logger.info(
        "Combined %d signals: mean=%.4f, std=%.4f",
        len(signals), combined.mean(), combined.std(),
    )
    return combined


def compute_risk_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.02,
) -> RiskMetrics:
    """Compute comprehensive risk metrics from a return series.

    Args:
        returns: Daily return series.
        risk_free_rate: Annual risk-free rate for Sharpe/Sortino.

    Returns:
        RiskMetrics with all computed values.

    Raises:
        ValueError: If returns series is too short.
    """
    if len(returns) < MIN_LOOKBACK:
        raise ValueError(
            f"Need at least {MIN_LOOKBACK} returns, got {len(returns)}"
        )

    returns_array = returns.values.astype(float)
    daily_rf = risk_free_rate / TRADING_DAYS_PER_YEAR

    ann_ret = _annualized_return(returns_array)
    ann_vol = _annualized_volatility(returns_array)
    sharpe = _sharpe_ratio(returns_array, daily_rf)
    max_dd = _max_drawdown(returns_array)
    sortino = _sortino_ratio(returns_array, daily_rf)
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0.0
    win = _win_rate(returns_array)
    pf = _profit_factor(returns_array)

    return RiskMetrics(
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        sortino_ratio=sortino,
        calmar_ratio=calmar,
        win_rate=win,
        profit_factor=pf,
    )


def _validate_price_series(prices: pd.Series, lookback: int) -> None:
    """Validate a price series has enough data.

    Args:
        prices: Price series.
        lookback: Required lookback period.

    Raises:
        ValueError: If series is too short or contains non-positive prices.
    """
    if len(prices) < lookback + 1:
        raise ValueError(
            f"Need at least {lookback + 1} data points, got {len(prices)}"
        )
    if (prices <= 0).any():
        raise ValueError("Prices must be strictly positive")


def _validate_signal_list(
    signals: List[SignalResult],
    weights: Optional[List[float]],
) -> None:
    """Validate signal combination inputs.

    Args:
        signals: Signal list.
        weights: Optional weights.

    Raises:
        ValueError: If inputs are invalid.
    """
    if not signals:
        raise ValueError("Signals list must not be empty")
    if weights is not None and len(weights) != len(signals):
        raise ValueError(
            f"Weights length ({len(weights)}) must match signals ({len(signals)})"
        )


def _z_score_normalize(series: pd.Series) -> pd.Series:
    """Z-score normalize a series.

    Args:
        series: Input series.

    Returns:
        Normalized series with mean ~0 and std ~1.
    """
    std = series.std()
    if std == 0 or np.isnan(std):
        return series * 0.0
    return (series - series.mean()) / std


def _weighted_sum(
    signals: List[SignalResult], weights: List[float]
) -> pd.Series:
    """Compute weighted sum of signals, aligned by index.

    Args:
        signals: Signal results.
        weights: Per-signal weights.

    Returns:
        Weighted sum series.
    """
    aligned = pd.DataFrame(
        {s.name: s.values for s in signals}
    ).ffill().fillna(0)

    result = pd.Series(0.0, index=aligned.index)
    for signal, weight in zip(signals, weights):
        if signal.name in aligned.columns:
            result += aligned[signal.name] * weight

    return result


def _annualized_return(returns: np.ndarray) -> float:
    """Compute annualized return from daily returns.

    Args:
        returns: Array of daily returns.

    Returns:
        Annualized return as decimal.
    """
    cumulative = np.prod(1 + returns) - 1
    n_years = len(returns) / TRADING_DAYS_PER_YEAR
    if n_years <= 0:
        return 0.0
    return (1 + cumulative) ** (1 / n_years) - 1


def _annualized_volatility(returns: np.ndarray) -> float:
    """Compute annualized volatility.

    Args:
        returns: Array of daily returns.

    Returns:
        Annualized volatility.
    """
    return float(np.std(returns) * np.sqrt(TRADING_DAYS_PER_YEAR))


def _sharpe_ratio(returns: np.ndarray, daily_rf: float) -> float:
    """Compute annualized Sharpe ratio.

    Args:
        returns: Array of daily returns.
        daily_rf: Daily risk-free rate.

    Returns:
        Sharpe ratio.
    """
    excess = returns - daily_rf
    vol = np.std(excess)
    if vol == 0:
        return 0.0
    return float(np.mean(excess) / vol * np.sqrt(TRADING_DAYS_PER_YEAR))


def _max_drawdown(returns: np.ndarray) -> float:
    """Compute maximum drawdown from peak.

    Args:
        returns: Array of daily returns.

    Returns:
        Max drawdown as negative decimal (e.g., -0.15 = 15% loss).
    """
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    return float(np.min(drawdown))


def _sortino_ratio(returns: np.ndarray, daily_rf: float) -> float:
    """Compute annualized Sortino ratio (downside deviation).

    Args:
        returns: Array of daily returns.
        daily_rf: Daily risk-free rate.

    Returns:
        Sortino ratio.
    """
    excess = returns - daily_rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return 0.0
    downside_std = np.std(downside)
    if downside_std == 0:
        return 0.0
    return float(
        np.mean(excess) / downside_std * np.sqrt(TRADING_DAYS_PER_YEAR)
    )


def _win_rate(returns: np.ndarray) -> float:
    """Compute fraction of positive returns.

    Args:
        returns: Array of daily returns.

    Returns:
        Win rate as decimal [0, 1].
    """
    if len(returns) == 0:
        return 0.0
    return float(np.sum(returns > 0) / len(returns))


def _profit_factor(returns: np.ndarray) -> float:
    """Compute profit factor: gross profit / gross loss.

    Args:
        returns: Array of daily returns.

    Returns:
        Profit factor (>1 means profitable, 0 if no losses).
    """
    gains = np.sum(returns[returns > 0])
    losses = abs(np.sum(returns[returns < 0]))
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)
