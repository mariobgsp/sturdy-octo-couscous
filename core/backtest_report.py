"""
Backtest Report Card Generator.

Calculates performance metrics from a list of completed backtest trades:
  - Total Trades, Win Rate
  - Expected Value (avg P&L per trade)
  - Max Drawdown (peak-to-trough equity %)
  - Sharpe Ratio (annualized)
  - Profit Factor (gross profit / gross loss)
  - Average Win / Average Loss
  - Longest Win/Loss Streaks
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from config.settings import MAX_DRAWDOWN_THRESHOLD, MIN_PROFIT_FACTOR

logger = logging.getLogger(__name__)


@dataclass
class ReportCard:
    """Performance summary for a backtest period."""

    label: str = ""
    total_trades: int = 0
    winners: int = 0
    losers: int = 0
    win_rate: float = 0.0
    expected_value: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    profit_factor: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_holding_days: float = 0.0
    longest_win_streak: int = 0
    longest_loss_streak: int = 0
    total_return_pct: float = 0.0
    initial_capital: float = 0.0
    final_equity: float = 0.0
    start_date: str = ""
    end_date: str = ""
    total_days: int = 0

    # Flags
    drawdown_flag: bool = False  # True if MDD > threshold
    profit_factor_flag: bool = False  # True if PF < threshold


def generate_report_card(
    trades: list,
    initial_capital: float,
    label: str = "",
    start_date: str = "",
    end_date: str = "",
    total_days: int = 0,
) -> ReportCard:
    """
    Generate a performance report card from backtest trades.

    Parameters
    ----------
    trades : list[BacktestTrade]
        Completed trades from the backtester.
    initial_capital : float
        Starting capital in IDR.
    label : str
        Label for the report (e.g., "Training", "Blind Test").

    Returns
    -------
    ReportCard
        Complete performance metrics.
    """
    card = ReportCard(
        label=label,
        initial_capital=initial_capital,
        start_date=start_date,
        end_date=end_date,
        total_days=total_days,
    )

    if not trades:
        return card

    card.total_trades = len(trades)

    # ── Win/Loss Breakdown ────────────────────────────────────────
    pnls = [t.pnl for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    card.winners = len(wins)
    card.losers = len(losses)
    card.win_rate = (card.winners / card.total_trades) * 100.0

    card.gross_profit = sum(wins) if wins else 0.0
    card.gross_loss = abs(sum(losses)) if losses else 0.0

    card.avg_win = np.mean(wins) if wins else 0.0
    card.avg_loss = abs(np.mean(losses)) if losses else 0.0

    # ── Expected Value ────────────────────────────────────────────
    card.expected_value = np.mean(pnls)

    # ── Profit Factor ─────────────────────────────────────────────
    if card.gross_loss > 0:
        card.profit_factor = card.gross_profit / card.gross_loss
    else:
        card.profit_factor = float("inf") if card.gross_profit > 0 else 0.0

    card.profit_factor_flag = card.profit_factor < MIN_PROFIT_FACTOR

    # ── Max Drawdown ──────────────────────────────────────────────
    equity_curve = _build_equity_curve(trades, initial_capital)
    if len(equity_curve) > 0:
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = ((peak - eq) / peak) * 100.0
            if dd > max_dd:
                max_dd = dd
        card.max_drawdown_pct = round(max_dd, 2)
    card.drawdown_flag = card.max_drawdown_pct > MAX_DRAWDOWN_THRESHOLD

    # ── Sharpe Ratio (annualized) ─────────────────────────────────
    if len(pnls) > 1:
        returns = np.array(pnls) / initial_capital
        mean_return = np.mean(returns)
        std_return = np.std(returns, ddof=1)
        if std_return > 0:
            # Annualize: assume ~250 trading days / year
            trades_per_year = 250 / max(
                1, total_days / max(1, card.total_trades)
            )
            card.sharpe_ratio = round(
                (mean_return / std_return) * np.sqrt(trades_per_year), 2
            )

    # ── Streaks ───────────────────────────────────────────────────
    card.longest_win_streak, card.longest_loss_streak = _calc_streaks(pnls)

    # ── Holding Days ──────────────────────────────────────────────
    holding_days = [t.holding_days for t in trades]
    card.avg_holding_days = round(np.mean(holding_days), 1) if holding_days else 0.0

    # ── Total Return ──────────────────────────────────────────────
    card.final_equity = initial_capital + sum(pnls)
    card.total_return_pct = round(
        ((card.final_equity / initial_capital) - 1.0) * 100.0, 2
    )

    return card


def _build_equity_curve(
    trades: list, initial_capital: float
) -> list[float]:
    """Build an equity curve from sequential trade P&Ls."""
    curve = [initial_capital]
    equity = initial_capital
    for t in trades:
        equity += t.pnl
        curve.append(equity)
    return curve


def _calc_streaks(pnls: list[float]) -> tuple[int, int]:
    """Calculate longest consecutive win and loss streaks."""
    max_win = 0
    max_loss = 0
    cur_win = 0
    cur_loss = 0

    for pnl in pnls:
        if pnl > 0:
            cur_win += 1
            cur_loss = 0
            max_win = max(max_win, cur_win)
        else:
            cur_loss += 1
            cur_win = 0
            max_loss = max(max_loss, cur_loss)

    return max_win, max_loss


def print_report_card(
    train: ReportCard | None = None,
    test: ReportCard | None = None,
) -> str:
    """Generate a formatted console report card."""

    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  BACKTEST REPORT CARD")
    lines.append("=" * 70)

    for card in [train, test]:
        if card is None:
            continue

        lines.append("")
        lines.append(f"  -- {card.label.upper()} --")
        lines.append(f"  Period: {card.start_date} -> {card.end_date}")
        lines.append(f"  Trading Days: {card.total_days}")
        lines.append("")

        lines.append(
            f"  Total Trades:     {card.total_trades}"
        )
        lines.append(
            f"  Win / Loss:       {card.winners} / {card.losers}"
        )
        lines.append(
            f"  Win Rate:         {card.win_rate:.1f}%"
        )
        lines.append("")

        ev_fmt = f"IDR {card.expected_value:,.0f}"
        lines.append(f"  Expected Value:   {ev_fmt} per trade")

        aw_fmt = f"IDR {card.avg_win:,.0f}" if card.avg_win else "N/A"
        al_fmt = f"IDR {card.avg_loss:,.0f}" if card.avg_loss else "N/A"
        lines.append(f"  Avg Win:          {aw_fmt}")
        lines.append(f"  Avg Loss:         {al_fmt}")
        lines.append(f"  Avg Holding:      {card.avg_holding_days:.0f} days")
        lines.append("")

        # Profit Factor
        pf_str = (
            f"{card.profit_factor:.2f}"
            if card.profit_factor != float("inf")
            else "INF"
        )
        pf_flag = " [!] BELOW 1.5" if card.profit_factor_flag else " [OK]"
        lines.append(f"  Profit Factor:    {pf_str}{pf_flag}")

        # Sharpe
        lines.append(f"  Sharpe Ratio:     {card.sharpe_ratio:.2f}")

        # Max Drawdown
        dd_flag = (
            " [!] EXCEEDS 15%" if card.drawdown_flag else " [OK]"
        )
        lines.append(
            f"  Max Drawdown:     {card.max_drawdown_pct:.1f}%{dd_flag}"
        )
        lines.append("")

        # Streaks
        lines.append(
            f"  Win Streak:       {card.longest_win_streak}"
        )
        lines.append(
            f"  Loss Streak:      {card.longest_loss_streak}"
        )
        lines.append("")

        # Total return
        lines.append(
            f"  Initial Capital:  IDR {card.initial_capital:,.0f}"
        )
        lines.append(
            f"  Final Equity:     IDR {card.final_equity:,.0f}"
        )

        sign = "+" if card.total_return_pct >= 0 else ""
        lines.append(
            f"  Total Return:     {sign}{card.total_return_pct:.1f}%"
        )

        lines.append("  " + "-" * 50)

    lines.append("")
    lines.append("=" * 70)
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report
