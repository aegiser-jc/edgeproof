"""EdgeProof — prove your trading edge is real *net of costs*, not a backtest illusion.

Design principles (each is a direct answer to documented reasons retail
systematic trading fails):

1. Net-of-cost is first-class. Every backtest charges fees + slippage + spread.
   Gross numbers are never reported alone.
2. No look-ahead. Signals decided at the close of bar t are executed at bar t+1;
   the engine enforces this by construction (positions are shifted by one bar).
3. Overfitting guards. Deflated Sharpe Ratio penalises the number of strategy
   variants you tried; walk-forward / out-of-sample splits are built in.
4. Capacity awareness. Position size is compared against traded volume so you
   see when an "edge" only exists at sizes you can't actually trade.
"""

__version__ = "0.1.0"
