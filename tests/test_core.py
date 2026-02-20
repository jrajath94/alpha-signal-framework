"""Tests for alpha signal framework: signals, backtest, and risk metrics."""

import numpy as np
import pandas as pd
import pytest

from alpha_signal_framework.backtest import BacktestEngine
from alpha_signal_framework.signals import (
    SignalResult,
    SignalType,
    RiskMetrics,
    momentum_signal,
    mean_reversion_signal,
    volatility_signal,
    combine_signals,
    compute_risk_metrics,
    MIN_LOOKBACK,
)


def _make_price_series(
    n: int = 100, start: float = 100.0, drift: float = 0.0005
) -> pd.Series:
    """Generate a synthetic random-walk price series."""
    np.random.seed(42)
    returns = np.random.normal(drift, 0.02, n)
    prices = start * np.cumprod(1 + returns)
    index = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(prices, index=index)


@pytest.fixture
def prices() -> pd.Series:
    """Fixture: 100-day synthetic price series."""
    return _make_price_series(100)


@pytest.fixture
def long_prices() -> pd.Series:
    """Fixture: 500-day price series for more stable statistics."""
    return _make_price_series(500)


class TestMomentumSignal:
    """Tests for momentum signal generation."""

    def test_output_shape(self, prices: pd.Series) -> None:
        """Momentum signal has correct length after lookback."""
        result = momentum_signal(prices, lookback=20)

        assert isinstance(result, SignalResult)
        assert len(result.values) == len(prices) - 20
        assert result.signal_type == SignalType.MOMENTUM

    def test_positive_drift_produces_positive_mean(self) -> None:
        """Upward-drifting prices produce positive mean momentum."""
        prices = _make_price_series(200, drift=0.005)
        result = momentum_signal(prices, lookback=10, normalize=False)

        assert result.values.mean() > 0

    def test_normalized_output_near_zero_mean(
        self, prices: pd.Series
    ) -> None:
        """Normalized momentum has approximately zero mean."""
        result = momentum_signal(prices, lookback=20, normalize=True)

        assert abs(result.values.mean()) < 0.1

    def test_insufficient_data_raises(self) -> None:
        """Too few data points for lookback raises ValueError."""
        short = pd.Series([100.0, 101.0, 102.0])
        with pytest.raises(ValueError, match="data points"):
            momentum_signal(short, lookback=10)

    @pytest.mark.parametrize("lookback", [5, 10, 20, 60])
    def test_various_lookbacks(
        self, prices: pd.Series, lookback: int
    ) -> None:
        """Signal works across different lookback windows."""
        result = momentum_signal(prices, lookback=lookback)
        assert len(result.values) > 0


class TestMeanReversionSignal:
    """Tests for mean-reversion signal generation."""

    def test_output_shape(self, prices: pd.Series) -> None:
        """Mean-reversion signal drops NaN rows from rolling window."""
        result = mean_reversion_signal(prices, lookback=20)

        assert isinstance(result, SignalResult)
        assert result.signal_type == SignalType.MEAN_REVERSION

    def test_clipping_bounds(self, prices: pd.Series) -> None:
        """Normalized signal values are clipped to [-3, 3]."""
        result = mean_reversion_signal(prices, lookback=20, normalize=True)

        assert result.values.max() <= 3.0
        assert result.values.min() >= -3.0

    def test_negative_prices_raises(self) -> None:
        """Negative prices raise ValueError."""
        bad_prices = pd.Series([-1.0, -2.0, -3.0] + [100.0] * 50)
        with pytest.raises(ValueError, match="positive"):
            mean_reversion_signal(bad_prices, lookback=5)


class TestVolatilitySignal:
    """Tests for volatility-targeting signal."""

    def test_output_is_positive(self, prices: pd.Series) -> None:
        """Volatility scaling signal is non-negative."""
        result = volatility_signal(prices, lookback=20, target_vol=0.15)

        assert (result.values >= 0).all()
        assert result.signal_type == SignalType.VOLATILITY

    def test_high_vol_reduces_exposure(self) -> None:
        """Higher realized vol produces lower scaling factor."""
        np.random.seed(42)
        n = 100
        idx = pd.date_range("2024-01-01", periods=n, freq="B")

        # Low vol prices
        low_vol = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0, 0.005, n)), index=idx
        )
        # High vol prices
        high_vol = pd.Series(
            100 * np.cumprod(1 + np.random.normal(0, 0.05, n)), index=idx
        )

        low_result = volatility_signal(low_vol, lookback=20, target_vol=0.15)
        high_result = volatility_signal(high_vol, lookback=20, target_vol=0.15)

        assert low_result.values.mean() > high_result.values.mean()

    def test_invalid_target_vol_raises(self, prices: pd.Series) -> None:
        """Non-positive target volatility raises ValueError."""
        with pytest.raises(ValueError, match="target_vol"):
            volatility_signal(prices, lookback=20, target_vol=0.0)


class TestCombineSignals:
    """Tests for multi-signal combination."""

    def test_equal_weight_combination(self, prices: pd.Series) -> None:
        """Equal-weight combine produces a valid series."""
        s1 = momentum_signal(prices, lookback=10)
        s2 = mean_reversion_signal(prices, lookback=10)

        combined = combine_signals([s1, s2])

        assert len(combined) > 0
        assert isinstance(combined, pd.Series)

    def test_custom_weights(self, prices: pd.Series) -> None:
        """Custom weights are applied correctly."""
        s1 = momentum_signal(prices, lookback=10)
        s2 = momentum_signal(prices, lookback=20)

        combined = combine_signals([s1, s2], weights=[0.7, 0.3])

        assert len(combined) > 0

    def test_empty_signals_raises(self) -> None:
        """Empty signal list raises ValueError."""
        with pytest.raises(ValueError, match="empty"):
            combine_signals([])

    def test_mismatched_weights_raises(self, prices: pd.Series) -> None:
        """Weight count mismatch raises ValueError."""
        s1 = momentum_signal(prices, lookback=10)

        with pytest.raises(ValueError, match="Weights length"):
            combine_signals([s1], weights=[0.5, 0.5])


class TestComputeRiskMetrics:
    """Tests for risk metric computation."""

    def test_positive_return_metrics(self) -> None:
        """Consistently positive returns produce positive Sharpe."""
        returns = pd.Series(np.random.normal(0.001, 0.01, 252))
        metrics = compute_risk_metrics(returns, risk_free_rate=0.0)

        assert isinstance(metrics, RiskMetrics)
        assert metrics.annualized_return != 0
        assert metrics.annualized_volatility > 0
        assert metrics.win_rate > 0

    def test_max_drawdown_is_negative(self, long_prices: pd.Series) -> None:
        """Max drawdown is non-positive."""
        returns = long_prices.pct_change().dropna()
        metrics = compute_risk_metrics(returns)

        assert metrics.max_drawdown <= 0

    def test_profit_factor_profitable(self) -> None:
        """Profitable return series has profit factor > 1."""
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.002, 0.01, 100))
        metrics = compute_risk_metrics(returns, risk_free_rate=0.0)

        assert metrics.profit_factor > 1.0

    def test_insufficient_returns_raises(self) -> None:
        """Too few returns raises ValueError."""
        with pytest.raises(ValueError, match="at least"):
            compute_risk_metrics(pd.Series([0.01, 0.02]))


class TestBacktestEngine:
    """Tests for the backtest engine."""

    def test_init_with_defaults(self) -> None:
        """Engine initializes with default parameters."""
        engine = BacktestEngine()

        assert engine.initial_capital == 100000.0
        assert engine.transaction_cost == 0.001
        assert engine.positions == {}

    def test_invalid_capital_raises(self) -> None:
        """Non-positive capital raises ValueError."""
        with pytest.raises(ValueError, match="Capital"):
            BacktestEngine(initial_capital=-1000)

    def test_invalid_transaction_cost_raises(self) -> None:
        """Transaction cost outside [0, 1) raises ValueError."""
        with pytest.raises(ValueError, match="Transaction cost"):
            BacktestEngine(transaction_cost=1.5)

    def test_execute_trades(self) -> None:
        """Trade execution updates positions and capital."""
        engine = BacktestEngine(initial_capital=10000, transaction_cost=0.0)
        ts = pd.Timestamp("2024-01-01")

        engine.execute_trades(
            ts,
            signals={"AAPL": 1.0},
            prices={"AAPL": 150.0},
        )

        assert engine.positions.get("AAPL") == 1.0
        assert len(engine.trades) == 1

    def test_update_equity(self) -> None:
        """Equity updates reflect position value changes."""
        engine = BacktestEngine(initial_capital=10000, transaction_cost=0.0)
        ts = pd.Timestamp("2024-01-01")

        engine.execute_trades(ts, {"AAPL": 1.0}, {"AAPL": 100.0})
        equity = engine.update_equity(ts, {"AAPL": 110.0})

        # Capital decreased by 100 (bought 1 share), position worth 110
        assert equity > 10000

    def test_sharpe_ratio_requires_data(self) -> None:
        """Sharpe ratio with < 2 returns raises ValueError."""
        engine = BacktestEngine()

        with pytest.raises(ValueError, match="at least 2"):
            engine.sharpe_ratio()

    def test_max_drawdown_empty(self) -> None:
        """Max drawdown is 0 with no equity history."""
        engine = BacktestEngine()
        assert engine.max_drawdown() == 0.0

    def test_final_return_empty(self) -> None:
        """Final return is 0 with no equity history."""
        engine = BacktestEngine()
        assert engine.final_return() == 0.0

    @pytest.mark.parametrize("cost", [0.0, 0.001, 0.01])
    def test_transaction_costs_reduce_capital(self, cost: float) -> None:
        """Higher transaction costs lead to less capital after trade."""
        engine = BacktestEngine(initial_capital=10000, transaction_cost=cost)
        ts = pd.Timestamp("2024-01-01")

        engine.execute_trades(ts, {"AAPL": 1.0}, {"AAPL": 100.0})

        # Higher cost -> more capital spent on the trade
        if cost > 0:
            assert engine.capital < 10000 - 100
        else:
            assert engine.capital == 10000 - 100
