"""
Bracket Order Manager — simultaneous Buy + Stop-Loss + Take-Profit.

Creates a complete bracket order using the RiskManager for stop-loss
calculation and a configurable ATR multiplier for the take-profit target.

Usage:
    mgr = BracketOrderManager(broker, guard)
    result = mgr.create_bracket("ASII", entry=5000, atr=150, regime="BULL", capital=100_000_000)
"""

from __future__ import annotations

import logging

from config.settings import (
    BRACKET_ORDER_TP_ATR_MULTIPLIER,
    MAX_RISK_PER_TRADE_PCT,
)
from core.broker import BracketOrderResult, BrokerAdapter
from core.failsafes import FailsafeGuard
from core.risk import RiskManager

logger = logging.getLogger(__name__)


class BracketOrderManager:
    """
    Creates and submits bracket orders (Entry + SL + TP).

    Integrates with the RiskManager for position sizing and
    stop-loss calculation, and with the FailsafeGuard for
    pre-submission safety checks.
    """

    def __init__(
        self,
        broker: BrokerAdapter,
        guard: FailsafeGuard | None = None,
        risk_mgr: RiskManager | None = None,
    ):
        self._broker = broker
        self._guard = guard or FailsafeGuard()
        self._risk_mgr = risk_mgr or RiskManager()

    def create_bracket(
        self,
        ticker: str,
        entry_price: float,
        atr_value: float,
        regime: str = "BULL",
        capital: float = 100_000_000,
        tp_atr_multiplier: float = BRACKET_ORDER_TP_ATR_MULTIPLIER,
    ) -> BracketOrderResult:
        """
        Create and submit a bracket order.

        1. Calculates stop-loss via RiskManager (1.5x ATR below entry).
        2. Calculates take-profit at tp_atr_multiplier x ATR above entry.
        3. Calculates position size based on regime-adjusted risk.
        4. Runs all failsafe checks.
        5. Submits Entry + SL + TP orders simultaneously.

        Parameters
        ----------
        ticker : str
            Ticker code (without .JK).
        entry_price : float
            Planned entry price.
        atr_value : float
            Current ATR(14) value.
        regime : str
            Market regime (BULL/CAUTION/BEAR).
        capital : float
            Available capital for position sizing.
        tp_atr_multiplier : float
            ATR multiplier for take-profit (default: 3.0).

        Returns
        -------
        BracketOrderResult
            Contains results for all three orders.
        """
        # Calculate stop-loss
        stop_loss = self._risk_mgr.calculate_stop_loss(
            entry_price, atr_value
        )

        # Calculate take-profit
        take_profit = round(
            entry_price + (atr_value * tp_atr_multiplier), 2
        )

        # Calculate position size
        adj_risk = self._risk_mgr.adjust_risk_for_regime(
            MAX_RISK_PER_TRADE_PCT, regime
        )
        shares = self._risk_mgr.calculate_position_size(
            capital, entry_price, stop_loss, adj_risk
        )

        if shares <= 0:
            raise ValueError(
                f"Position size is 0 for {ticker} "
                f"(entry={entry_price}, SL={stop_loss}, risk={adj_risk}%)"
            )

        # Run failsafe checks
        self._guard.run_all_checks(
            shares=shares, price=entry_price, broker=self._broker
        )

        logger.info(
            "Submitting bracket order: %s | %d shares @ IDR %,.0f | "
            "SL: IDR %,.0f | TP: IDR %,.0f | Regime: %s",
            ticker, shares, entry_price, stop_loss, take_profit, regime,
        )

        # Submit all three orders
        result = self._broker.submit_bracket_order(
            ticker=ticker,
            shares=shares,
            entry_price=entry_price,
            stop_loss_price=stop_loss,
            take_profit_price=take_profit,
        )

        logger.info("Bracket order result: %s", result)
        return result
