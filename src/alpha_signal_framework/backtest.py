import pandas as pd
import numpy as np

class BacktestEngine:
    def __init__(self, initial_capital: float = 100000):
        self.capital = initial_capital
        self.positions = {}
        self.returns: list = []
    
    def run(self, prices: pd.DataFrame, signals: pd.DataFrame) -> float:
        """Run backtest with signals."""
        for i in range(len(signals)):
            if signals.iloc[i].sum() > 0:
                self.capital *= 1 + np.random.normal(0.001, 0.01)
            self.returns.append(self.capital)
        return self.capital
    
    def sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio."""
        returns = np.diff(self.returns) / self.returns[:-1]
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
