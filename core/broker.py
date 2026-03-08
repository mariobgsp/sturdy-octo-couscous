"""
Broker Adapter — abstract interface for IDX broker order routing.

Provides:
  - Abstract BrokerAdapter base class
  - SimulatedBroker for testing without a live broker connection
  - OrderResult dataclass for tracking order status

To connect a real broker, subclass BrokerAdapter and implement
the abstract methods for your specific broker API.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

logger = logging.getLogger(__name__)


# ─── Data Structures ─────────────────────────────────────────────────────────


@dataclass
class OrderResult:
    """Result of an order submission."""

    order_id: str
    ticker: str
    side: str  # "BUY" or "SELL"
    shares: int
    price: float
    status: str  # "FILLED", "PENDING", "REJECTED", "CANCELLED"
    timestamp: str = ""
    message: str = ""
    order_type: str = "LIMIT"  # "LIMIT", "MARKET", "STOP"

    def __str__(self) -> str:
        return (
            f"[{self.status}] {self.side} {self.shares}x {self.ticker} "
            f"@ {self.price:,.0f} (ID: {self.order_id})"
        )


@dataclass
class BracketOrderResult:
    """Result of a bracket order (entry + stop-loss + take-profit)."""

    entry: OrderResult
    stop_loss: OrderResult
    take_profit: OrderResult

    def __str__(self) -> str:
        return (
            f"Bracket: Entry={self.entry.status} "
            f"SL={self.stop_loss.status} "
            f"TP={self.take_profit.status}"
        )


# ─── Abstract Broker ─────────────────────────────────────────────────────────


class BrokerAdapter(ABC):
    """
    Abstract base for broker connections.

    Subclass this and implement the abstract methods for your
    specific IDX broker's API (e.g., Indo Premier, Mandiri, BCA).
    """

    @abstractmethod
    def submit_order(
        self,
        ticker: str,
        side: str,
        shares: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> OrderResult:
        """Submit a single order."""
        ...

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel a pending order. Returns True if successful."""
        ...

    @abstractmethod
    def get_account_balance(self) -> float:
        """Get the current account balance in IDR."""
        ...

    @abstractmethod
    def get_positions(self) -> list[dict]:
        """Get list of open positions."""
        ...

    @abstractmethod
    def get_daily_pnl(self) -> float:
        """Get the current day's P&L in IDR."""
        ...

    def submit_bracket_order(
        self,
        ticker: str,
        shares: int,
        entry_price: float,
        stop_loss_price: float,
        take_profit_price: float,
    ) -> BracketOrderResult:
        """
        Submit a bracket order: Entry + Stop-Loss + Take-Profit.

        Default implementation submits three separate orders.
        Override if your broker supports native bracket orders.
        """
        entry = self.submit_order(
            ticker, "BUY", shares, entry_price, "LIMIT"
        )
        sl = self.submit_order(
            ticker, "SELL", shares, stop_loss_price, "STOP"
        )
        tp = self.submit_order(
            ticker, "SELL", shares, take_profit_price, "LIMIT"
        )
        return BracketOrderResult(entry=entry, stop_loss=sl, take_profit=tp)


# ─── Simulated Broker ────────────────────────────────────────────────────────


class SimulatedBroker(BrokerAdapter):
    """
    In-memory simulated broker for testing.

    All orders are immediately "filled" and tracked in memory.
    Use this for dry-run testing before connecting a real broker.
    """

    def __init__(self, initial_balance: float = 100_000_000) -> None:
        self._balance = initial_balance
        self._start_balance = initial_balance
        self._positions: list[dict] = []
        self._orders: list[OrderResult] = []
        self._order_counter = 0
        self._daily_pnl = 0.0

    def _next_id(self) -> str:
        self._order_counter += 1
        return f"SIM-{self._order_counter:06d}"

    def submit_order(
        self,
        ticker: str,
        side: str,
        shares: int,
        price: float,
        order_type: str = "LIMIT",
    ) -> OrderResult:
        """Simulate an order fill."""
        order_id = self._next_id()
        cost = price * shares

        if side == "BUY":
            if cost > self._balance:
                result = OrderResult(
                    order_id=order_id,
                    ticker=ticker,
                    side=side,
                    shares=shares,
                    price=price,
                    status="REJECTED",
                    timestamp=datetime.now().isoformat(),
                    message="Insufficient funds",
                    order_type=order_type,
                )
                logger.warning("Order REJECTED: %s", result)
                self._orders.append(result)
                return result

            self._balance -= cost
            self._positions.append({
                "ticker": ticker,
                "shares": shares,
                "entry_price": price,
                "order_type": order_type,
            })
        elif side == "SELL":
            self._balance += cost
            self._daily_pnl += cost - (
                price * shares
            )  # Simplified P&L tracking
            # Remove from positions
            self._positions = [
                p for p in self._positions if p["ticker"] != ticker
            ]

        result = OrderResult(
            order_id=order_id,
            ticker=ticker,
            side=side,
            shares=shares,
            price=price,
            status="FILLED",
            timestamp=datetime.now().isoformat(),
            message="Simulated fill",
            order_type=order_type,
        )

        logger.info("Order FILLED: %s", result)
        self._orders.append(result)
        return result

    def cancel_order(self, order_id: str) -> bool:
        """Cancel a simulated order (always succeeds)."""
        for order in self._orders:
            if order.order_id == order_id and order.status == "PENDING":
                order.status = "CANCELLED"
                logger.info("Order CANCELLED: %s", order_id)
                return True
        logger.warning("Order not found or already filled: %s", order_id)
        return False

    def get_account_balance(self) -> float:
        """Return current simulated balance."""
        return self._balance

    def get_positions(self) -> list[dict]:
        """Return current simulated positions."""
        return list(self._positions)

    def get_daily_pnl(self) -> float:
        """Return simulated daily P&L."""
        return self._balance - self._start_balance

    def reset_daily_pnl(self) -> None:
        """Reset daily P&L tracking (call at start of each day)."""
        self._start_balance = self._balance
        self._daily_pnl = 0.0

    @property
    def order_history(self) -> list[OrderResult]:
        """Full order audit trail."""
        return list(self._orders)
