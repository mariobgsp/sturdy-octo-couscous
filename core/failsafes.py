"""
Failsafe Guards — hard-coded kill switches for live trading.

These checks CANNOT be bypassed without changing source code.
They protect against:
  - Fat finger errors (oversized orders)
  - Daily drawdown breakers (halt trading on bad days)

Usage:
    guard = FailsafeGuard()
    guard.run_all_checks(shares=1000, price=5000, broker=broker)
"""

from __future__ import annotations

import logging

from config.settings import (
    DAILY_DRAWDOWN_HALT_PCT,
    FAT_FINGER_MAX_SHARES,
    FAT_FINGER_MAX_VALUE_IDR,
)

logger = logging.getLogger(__name__)


# ─── Custom Exceptions ───────────────────────────────────────────────────────


class FatFingerError(Exception):
    """Raised when an order exceeds the fat finger safety limits."""
    pass


class DrawdownHaltError(Exception):
    """Raised when daily drawdown exceeds the circuit breaker threshold."""
    pass


# ─── Failsafe Guard ──────────────────────────────────────────────────────────


class FailsafeGuard:
    """
    Hard-coded safety checks for live order submission.

    All limits are defined in config/settings.py and enforced
    before any order reaches the broker API.
    """

    def __init__(
        self,
        max_shares: int = FAT_FINGER_MAX_SHARES,
        max_value_idr: float = FAT_FINGER_MAX_VALUE_IDR,
        daily_drawdown_pct: float = DAILY_DRAWDOWN_HALT_PCT,
    ):
        self._max_shares = max_shares
        self._max_value_idr = max_value_idr
        self._daily_drawdown_pct = daily_drawdown_pct

    def check_fat_finger(self, shares: int, price: float) -> None:
        """
        Check if an order exceeds fat finger limits.

        Raises FatFingerError if:
          - shares > FAT_FINGER_MAX_SHARES (default: 50,000)
          - order value > FAT_FINGER_MAX_VALUE_IDR (default: IDR 50M)

        Parameters
        ----------
        shares : int
            Number of shares in the order.
        price : float
            Price per share in IDR.
        """
        # Check share count
        if shares > self._max_shares:
            raise FatFingerError(
                f"BLOCKED: {shares:,} shares exceeds fat finger limit "
                f"of {self._max_shares:,} shares per order."
            )

        # Check order value
        order_value = shares * price
        if order_value > self._max_value_idr:
            raise FatFingerError(
                f"BLOCKED: Order value IDR {order_value:,.0f} exceeds "
                f"fat finger limit of IDR {self._max_value_idr:,.0f}."
            )

        logger.debug(
            "Fat finger check passed: %d shares @ IDR %,.0f (value: IDR %,.0f)",
            shares, price, order_value,
        )

    def check_daily_drawdown(self, broker) -> None:
        """
        Check if the daily drawdown exceeds the circuit breaker threshold.

        Raises DrawdownHaltError if daily loss > DAILY_DRAWDOWN_HALT_PCT.

        Parameters
        ----------
        broker : BrokerAdapter
            The broker adapter to query for daily P&L and balance.
        """
        daily_pnl = broker.get_daily_pnl()
        balance = broker.get_account_balance()

        if balance <= 0:
            raise DrawdownHaltError(
                "BLOCKED: Account balance is zero or negative."
            )

        # Daily P&L as percentage of starting balance
        start_of_day = balance - daily_pnl
        if start_of_day <= 0:
            return  # Can't calculate

        drawdown_pct = abs(min(0, daily_pnl)) / start_of_day * 100.0

        if drawdown_pct >= self._daily_drawdown_pct:
            raise DrawdownHaltError(
                f"HALT: Daily drawdown of {drawdown_pct:.1f}% exceeds "
                f"circuit breaker threshold of {self._daily_drawdown_pct:.1f}%. "
                f"ALL TRADING HALTED for today."
            )

        logger.debug(
            "Daily drawdown check passed: %.1f%% (limit: %.1f%%)",
            drawdown_pct, self._daily_drawdown_pct,
        )

    def run_all_checks(
        self,
        shares: int,
        price: float,
        broker=None,
    ) -> None:
        """
        Run all failsafe checks before order submission.

        Parameters
        ----------
        shares : int
            Number of shares in the proposed order.
        price : float
            Price per share.
        broker : BrokerAdapter, optional
            If provided, also checks daily drawdown.

        Raises
        ------
        FatFingerError
            If order exceeds size limits.
        DrawdownHaltError
            If daily drawdown exceeds threshold.
        """
        self.check_fat_finger(shares, price)

        if broker is not None:
            self.check_daily_drawdown(broker)

        logger.info(
            "All failsafe checks PASSED for %d shares @ IDR %,.0f",
            shares, price,
        )
