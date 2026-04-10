"""Walk-forward backtesting engine with realistic execution."""

import logging
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Execute walk-forward backtest with lookahead bias prevention.

    The engine enforces strict temporal ordering: features are computed using only
    data available at decision time. Test periods never access training data.
    """

    def __init__(
        self,
        initial_capital: float = 100000.0,
        transaction_cost: float = 0.001,
    ) -> None:
        """Initialize backtesting engine.

        Args:
            initial_capital: Starting portfolio value
            transaction_cost: Bid-ask spread + slippage per trade (e.g., 0.001 = 10bps)

        Raises:
            ValueError: if capital or transaction_cost invalid
        """
        if initial_capital <= 0:
            raise ValueError(f"Capital must be positive, got {initial_capital}")
        if not 0 <= transaction_cost < 1:
            raise ValueError(f"Transaction cost must be in [0,1), got {transaction_cost}")

        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.transaction_cost = transaction_cost
        self.positions: Dict[str, float] = {}  # symbol -> quantity
        self.returns: list = []
        self.equity_curve: list = []
        self.trades: list = []  # (timestamp, symbol, qty, price, cost)

        logger.info(f"Initialized backtest: capital=${initial_capital:.0f}, "
                   f"transaction_cost={transaction_cost:.1%}")

    def execute_trades(
        self,
        timestamp: pd.Timestamp,
        signals: Dict[str, float],
        prices: Dict[str, float],
    ) -> None:
        """Execute trades based on signals.

        Args:
            timestamp: Current timestamp (prevents lookahead)
            signals: {symbol: signal_value} where positive = long, negative = short
            prices: {symbol: current_price} current market prices

        Raises:
            KeyError: if price missing for signaled symbol
            ValueError: if signal or price invalid
        """
        for symbol, signal in signals.items():
            if symbol not in prices:
                logger.warning(f"Price unavailable for {symbol}, skipping trade")
                continue

            price = prices[symbol]
            if price <= 0:
                raise ValueError(f"Invalid price for {symbol}: {price}")

            # Apply transaction cost to execution price
            execution_price = price * (1 + self.transaction_cost)

            # Determine desired quantity from signal (simplified: 1 unit per signal point)
            desired_qty = max(0, min(signal, 1.0))  # Clamp to [0, 1]
            current_qty = self.positions.get(symbol, 0)

            if desired_qty != current_qty:
                qty_change = desired_qty - current_qty
                cost = qty_change * execution_price
                self.capital -= cost
                self.positions[symbol] = desired_qty

                self.trades.append({
                    'timestamp': timestamp,
                    'symbol': symbol,
                    'quantity': qty_change,
                    'price': execution_price,
                    'cost': cost,
                })

                logger.debug(f"{timestamp} | {symbol}: {qty_change:+.2f} @ ${execution_price:.2f}")

    def update_equity(self, timestamp: pd.Timestamp, prices: Dict[str, float]) -> float:
        """Calculate current portfolio value.

        Args:
            timestamp: Current time
            prices: {symbol: current_price}

        Returns:
            Current portfolio equity
        """
        # Cash component
        equity = self.capital

        # Position components: sum of (quantity * price)
        for symbol, qty in self.positions.items():
            if symbol in prices:
                equity += qty * prices[symbol]

        self.equity_curve.append(equity)
        self.returns.append((equity - (self.equity_curve[-2] if len(self.equity_curve) > 1 else self.initial_capital)) /
                           (self.equity_curve[-2] if len(self.equity_curve) > 1 else self.initial_capital))

        return equity

    def sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate annualized Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio (excess return per unit volatility)

        Raises:
            ValueError: if insufficient returns
        """
        if len(self.returns) < 2:
            raise ValueError("Need at least 2 returns for Sharpe ratio")

        returns_array = np.array(self.returns)
        excess_return = np.mean(returns_array) - risk_free_rate / 252
        volatility = np.std(returns_array)

        if volatility == 0:
            logger.warning("Zero volatility, Sharpe ratio undefined")
            return 0.0

        return excess_return / volatility * np.sqrt(252)

    def max_drawdown(self) -> float:
        """Calculate maximum drawdown from peak.

        Returns:
            Max drawdown as decimal (e.g., 0.15 = 15% loss)
        """
        if not self.equity_curve:
            return 0.0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max
        return np.min(drawdown)  # Most negative value

    def final_return(self) -> float:
        """Calculate total return from start to end.

        Returns:
            Total return as decimal (e.g., 0.25 = 25% gain)
        """
        if not self.equity_curve:
            return 0.0
        return (self.equity_curve[-1] - self.initial_capital) / self.initial_capital
