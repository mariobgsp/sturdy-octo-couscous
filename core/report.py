"""
Daily scan report generator for the IHSG swing trading system.

Produces both console and HTML reports from scan results,
regime data, and portfolio state.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from config.settings import MAX_PORTFOLIO_HEAT_PCT, PROJECT_ROOT

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports"


def generate_console_report(
    scan_result,
    regime_snapshot,
    portfolio,
    elapsed: float = 0,
) -> str:
    """
    Generate a formatted console report string.

    Parameters
    ----------
    scan_result : ScanResult
    regime_snapshot : RegimeSnapshot
    portfolio : Portfolio
    elapsed : float
        Scan duration in seconds.

    Returns
    -------
    str
        Formatted report for terminal display.
    """
    stats = scan_result.stats
    lines: list[str] = []

    # Header
    lines.append("")
    lines.append("=" * 72)
    lines.append("  IHSG DAILY SCAN REPORT")
    lines.append(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 72)

    # Regime
    lines.append("")
    lines.append(f"  MARKET REGIME: {regime_snapshot.regime.value}")
    lines.append(
        f"  IHSG Close: {regime_snapshot.close:,.0f}  "
        f"SMA(50): {regime_snapshot.sma_short:,.0f}  "
        f"SMA(200): {regime_snapshot.sma_long:,.0f}  "
        f"ATR(14): {regime_snapshot.atr_value:,.0f}"
    )
    lines.append(f"  As-of: {regime_snapshot.as_of_date}")

    # Engine permissions
    engines = ["fvg_pullback", "momentum_breakout", "buying_on_weakness"]
    active = [e for e in engines if regime_snapshot.allows_engine(e)]
    lines.append(f"  Active engines: {', '.join(active) if active else 'NONE'}")

    # Scan stats
    lines.append("")
    lines.append("-" * 72)
    lines.append(
        f"  Scanned: {stats.get('total_scanned', 0)}  "
        f"With data: {stats.get('total_with_data', 0)}  "
        f"Duration: {elapsed:.1f}s"
    )

    # Portfolio heat
    lines.append("")
    heat = portfolio.heat
    heat_bar = _heat_bar(heat, MAX_PORTFOLIO_HEAT_PCT)
    lines.append(
        f"  Portfolio Heat: {heat:.2f}% / {MAX_PORTFOLIO_HEAT_PCT}%  "
        f"{heat_bar}"
    )
    lines.append(
        f"  Open Positions: {portfolio.num_positions}  "
        f"Available Cash: IDR {portfolio.capital:,.0f}"
    )

    # === TRADE BUCKET ===
    lines.append("")
    lines.append("=" * 72)
    if scan_result.trade:
        lines.append(f"  TRADE SIGNALS ({len(scan_result.trade)} stocks)")
        lines.append("=" * 72)
        for rank, entry in enumerate(scan_result.trade, start=1):
            lines.append("")
            lines.append(f"  #{rank}  {entry.ticker}")
            lines.append(f"  {'=' * 40}")
            lines.append(f"  Engine:    {entry.signal}")
            lines.append(f"  Entry:     IDR {entry.price:>12,.0f}")

            d = entry.details
            if "stop_loss" in d:
                lines.append(f"  Stop:      IDR {d['stop_loss']:>12,.0f}")
            if "trailing_stop" in d:
                lines.append(f"  Trail:     IDR {d['trailing_stop']:>12,.0f}")
            if "position_size" in d:
                lots = d["position_size"] // 100
                lines.append(
                    f"  Size:      {d['position_size']:>12,} shares "
                    f"({lots} lots)"
                )
            if "risk_amount" in d:
                lines.append(
                    f"  Risk:      IDR {d['risk_amount']:>12,.0f}  "
                    f"({d.get('risk_pct', '?')}%)"
                )
            lines.append(f"  Score:     {entry.score:.2f}")
            if "volume_ratio" in d:
                lines.append(f"  Vol Ratio: {d['volume_ratio']:.2f}x")
            if "rsi" in d:
                lines.append(f"  RSI:       {d['rsi']:.1f}")
    else:
        lines.append("  NO TRADE SIGNALS TODAY")
        lines.append("=" * 72)

    # === WAIT BUCKET ===
    if scan_result.wait:
        lines.append("")
        lines.append(f"  --- WAIT ({len(scan_result.wait)} stocks) ---")
        for entry in scan_result.wait[:10]:
            detail_str = _format_wait_detail(entry)
            lines.append(f"  {entry.ticker:8s} [{entry.condition}] {detail_str}")
        if len(scan_result.wait) > 10:
            lines.append(f"  ... and {len(scan_result.wait) - 10} more")

    # === AVOID SUMMARY ===
    avoid_bk = stats.get("avoid_breakdown", {})
    lines.append("")
    lines.append(f"  --- AVOID ({len(scan_result.avoid)} stocks) ---")
    lines.append(
        f"  Low ADTV: {avoid_bk.get('low_adtv', 0)}  "
        f"Penny: {avoid_bk.get('penny_stock', 0)}  "
        f"Below SMA200: {avoid_bk.get('below_sma200', 0)}  "
        f"Earnings: {avoid_bk.get('earnings_proximity', 0)}  "
        f"No data: {avoid_bk.get('insufficient_data', 0)}"
    )

    if scan_result.skipped:
        lines.append(f"  Skipped: {len(scan_result.skipped)} (no data)")

    lines.append("")
    lines.append("=" * 72)
    lines.append("")

    return "\n".join(lines)


def generate_html_report(
    scan_result,
    regime_snapshot,
    portfolio,
    elapsed: float = 0,
) -> Path:
    """
    Generate an HTML report and save to reports/ directory.

    Returns the path to the saved HTML file.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    path = REPORTS_DIR / f"{date_str}.html"

    stats = scan_result.stats
    heat = portfolio.heat

    # Build trade cards HTML
    trade_cards = ""
    if scan_result.trade:
        for rank, entry in enumerate(scan_result.trade, start=1):
            d = entry.details
            trade_cards += _html_trade_card(rank, entry, d)
    else:
        trade_cards = '<div class="card no-signal">No trade signals today</div>'

    # Wait table rows
    wait_rows = ""
    for entry in scan_result.wait[:20]:
        detail_str = _format_wait_detail(entry)
        wait_rows += (
            f"<tr><td>{entry.ticker}</td>"
            f"<td>{entry.condition}</td>"
            f"<td>{detail_str}</td></tr>\n"
        )

    # Avoid breakdown
    avoid_bk = stats.get("avoid_breakdown", {})

    regime_class = regime_snapshot.regime.value.lower()
    heat_pct = min(heat / MAX_PORTFOLIO_HEAT_PCT * 100, 100)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>IHSG Daily Scan - {date_str}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0f1117; color: #e0e0e0; padding: 24px; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; color: #fff; margin-bottom: 4px; }}
  .date {{ color: #888; font-size: 0.9rem; margin-bottom: 20px; }}

  .regime {{ padding: 16px; border-radius: 8px; margin-bottom: 20px; }}
  .regime.bull {{ background: linear-gradient(135deg, #1a3a1a, #0f1117); border-left: 4px solid #4caf50; }}
  .regime.caution {{ background: linear-gradient(135deg, #3a3a1a, #0f1117); border-left: 4px solid #ff9800; }}
  .regime.bear {{ background: linear-gradient(135deg, #3a1a1a, #0f1117); border-left: 4px solid #f44336; }}
  .regime-label {{ font-size: 1.2rem; font-weight: 700; }}
  .regime-label.bull {{ color: #4caf50; }}
  .regime-label.caution {{ color: #ff9800; }}
  .regime-label.bear {{ color: #f44336; }}
  .regime-detail {{ color: #aaa; font-size: 0.85rem; margin-top: 6px; }}
  .regime-engines {{ color: #888; font-size: 0.8rem; margin-top: 4px; }}

  .stats-row {{ display: flex; gap: 16px; margin-bottom: 20px; flex-wrap: wrap; }}
  .stat-box {{ background: #1a1d27; border-radius: 8px; padding: 14px 18px; flex: 1; min-width: 140px; }}
  .stat-label {{ color: #888; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }}
  .stat-value {{ font-size: 1.3rem; font-weight: 700; color: #fff; margin-top: 4px; }}

  .heat-bar {{ background: #1a1d27; border-radius: 8px; padding: 14px 18px; margin-bottom: 20px; }}
  .heat-track {{ background: #2a2d37; border-radius: 4px; height: 10px; margin-top: 8px; overflow: hidden; }}
  .heat-fill {{ height: 100%; border-radius: 4px; transition: width 0.5s ease; }}
  .heat-fill.ok {{ background: linear-gradient(90deg, #4caf50, #8bc34a); }}
  .heat-fill.warning {{ background: linear-gradient(90deg, #ff9800, #f44336); }}

  .section-title {{ font-size: 1.1rem; font-weight: 600; margin: 24px 0 12px; color: #fff; }}
  .card {{ background: #1a1d27; border-radius: 8px; padding: 18px; margin-bottom: 12px; border-left: 4px solid #2196f3; }}
  .card.no-signal {{ border-left-color: #555; color: #888; text-align: center; padding: 30px; }}
  .card-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; }}
  .card-rank {{ font-size: 1.5rem; font-weight: 800; color: #2196f3; }}
  .card-ticker {{ font-size: 1.3rem; font-weight: 700; color: #fff; }}
  .card-engine {{ font-size: 0.8rem; color: #888; background: #2a2d37; padding: 2px 8px; border-radius: 4px; }}
  .card-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }}
  .card-field {{ }}
  .field-label {{ font-size: 0.7rem; color: #888; text-transform: uppercase; }}
  .field-value {{ font-size: 0.95rem; font-weight: 600; color: #e0e0e0; }}
  .field-value.stop {{ color: #f44336; }}
  .field-value.risk {{ color: #ff9800; }}

  table {{ width: 100%; border-collapse: collapse; background: #1a1d27; border-radius: 8px; overflow: hidden; }}
  th {{ background: #2a2d37; color: #aaa; font-size: 0.75rem; text-transform: uppercase; padding: 10px 12px; text-align: left; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #2a2d37; font-size: 0.85rem; }}

  .avoid-summary {{ display: flex; gap: 12px; flex-wrap: wrap; margin-top: 8px; }}
  .avoid-chip {{ background: #2a2d37; padding: 4px 10px; border-radius: 4px; font-size: 0.8rem; color: #aaa; }}

  .avoid-donut-row {{ display: flex; align-items: center; gap: 24px; margin-top: 12px; flex-wrap: wrap; }}
  .donut {{ width: 120px; height: 120px; border-radius: 50%; position: relative; flex-shrink: 0; }}
  .donut-hole {{ width: 70px; height: 70px; background: #1a1d27; border-radius: 50%; position: absolute; top: 25px; left: 25px; display: flex; align-items: center; justify-content: center; }}
  .donut-hole span {{ color: #fff; font-weight: 700; font-size: 1.1rem; }}
  .donut-legend {{ display: flex; flex-direction: column; gap: 6px; }}
  .legend-item {{ display: flex; align-items: center; gap: 8px; font-size: 0.82rem; color: #ccc; }}
  .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

  .field-value.predicted {{ color: #64b5f6; }}
  .field-value.positive {{ color: #4caf50; }}
  .field-value.negative {{ color: #f44336; }}
</style>
</head>
<body>
<div class="container">

<h1>IHSG Daily Scan Report</h1>
<div class="date">{datetime.now().strftime('%A, %d %B %Y %H:%M')}</div>

<div class="regime {regime_class}">
  <div class="regime-label {regime_class}">REGIME: {regime_snapshot.regime.value}</div>
  <div class="regime-detail">
    IHSG Close: {regime_snapshot.close:,.0f} &nbsp;|&nbsp;
    SMA(50): {regime_snapshot.sma_short:,.0f} &nbsp;|&nbsp;
    SMA(200): {regime_snapshot.sma_long:,.0f} &nbsp;|&nbsp;
    ATR(14): {regime_snapshot.atr_value:,.0f} &nbsp;|&nbsp;
    Hurst: {regime_snapshot.hurst_value:.2f}
  </div>
  <div class="regime-engines">
    Active: {', '.join(e for e in ['FVG Pullback', 'Momentum Breakout', 'B.O.W.', 'Wyckoff Spring', 'Vol Climax Reversal']
                       if regime_snapshot.allows_engine(['fvg_pullback', 'momentum_breakout', 'buying_on_weakness', 'wyckoff_spring', 'volume_climax_reversal'][['FVG Pullback', 'Momentum Breakout', 'B.O.W.', 'Wyckoff Spring', 'Vol Climax Reversal'].index(e)])) or 'None'}
  </div>
</div>

<div class="stats-row">
  <div class="stat-box">
    <div class="stat-label">Scanned</div>
    <div class="stat-value">{stats.get('total_scanned', 0)}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Avoid</div>
    <div class="stat-value">{len(scan_result.avoid)}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Wait</div>
    <div class="stat-value">{len(scan_result.wait)}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Trade</div>
    <div class="stat-value">{len(scan_result.trade)}</div>
  </div>
  <div class="stat-box">
    <div class="stat-label">Duration</div>
    <div class="stat-value">{elapsed:.1f}s</div>
  </div>
</div>

<div class="heat-bar">
  <div class="stat-label">
    Portfolio Heat: {heat:.2f}% / {MAX_PORTFOLIO_HEAT_PCT}%
    &nbsp;&nbsp;|&nbsp;&nbsp;
    Positions: {portfolio.num_positions} &nbsp;|&nbsp;
    Cash: IDR {portfolio.capital:,.0f}
  </div>
  <div class="heat-track">
    <div class="heat-fill {'warning' if heat_pct > 75 else 'ok'}"
         style="width: {heat_pct:.0f}%"></div>
  </div>
</div>

<div class="section-title">Trade Signals</div>
{trade_cards}

<div class="section-title">Wait List ({len(scan_result.wait)})</div>
{'<table><tr><th>Ticker</th><th>Condition</th><th>Details</th></tr>' + wait_rows + '</table>' if wait_rows else '<div class="card no-signal">No stocks in Wait bucket</div>'}

<div class="section-title">Avoid Summary ({len(scan_result.avoid)})</div>
{_html_avoid_donut(avoid_bk, len(scan_result.avoid))}

</div>
</body>
</html>"""

    path.write_text(html, encoding="utf-8")
    logger.info("HTML report saved to %s", path)
    return path


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _heat_bar(heat: float, max_heat: float, width: int = 20) -> str:
    """ASCII heat bar for console output."""
    pct = min(heat / max_heat, 1.0)
    filled = int(pct * width)
    bar = "#" * filled + "-" * (width - filled)
    return f"[{bar}]"


def _format_wait_detail(entry) -> str:
    """Format Wait entry details as a string."""
    d = entry.details
    if entry.condition == "tight_consolidation":
        return (
            f"range={d.get('range_pct', '?')}% "
            f"over {d.get('window', '?')}d, "
            f"price={d.get('price', '?')}"
        )
    elif entry.condition == "fvg_approach":
        return (
            f"FVG [{d.get('gap_low', '?')}-{d.get('gap_high', '?')}] "
            f"on {d.get('fvg_date', '?')}, "
            f"dist={d.get('distance', '?')}"
        )
    elif entry.condition == "trade_overflow":
        return (
            f"signal={d.get('signal', '?')}, "
            f"score={d.get('score', '?')}"
        )
    return str(d)


def _html_trade_card(rank: int, entry, d: dict) -> str:
    """Generate HTML for a single trade card."""
    stop_html = f"<div class='field-value stop'>IDR {d['stop_loss']:,.0f}</div>" if "stop_loss" in d else ""
    trail_html = f"<div class='field-value'>IDR {d['trailing_stop']:,.0f}</div>" if "trailing_stop" in d else ""
    size_html = ""
    if "position_size" in d:
        lots = d["position_size"] // 100
        size_html = f"<div class='field-value'>{d['position_size']:,} ({lots} lots)</div>"
    risk_html = ""
    if "risk_amount" in d:
        risk_html = f"<div class='field-value risk'>IDR {d['risk_amount']:,.0f} ({d.get('risk_pct', '?')}%)</div>"

    # Phase 5 predictive fields
    predicted_html = ""
    if "predicted_return" in d:
        pr = d["predicted_return"]
        pr_class = "positive" if pr >= 0 else "negative"
        predicted_html = f"""
        <div class="card-field">
          <div class="field-label">Predicted 5d Return</div>
          <div class="field-value {pr_class}">{pr * 100:+.2f}%</div>
        </div>"""

    projection_html = ""
    if "projected_upper" in d and "projected_lower" in d:
        projection_html = f"""
        <div class="card-field">
          <div class="field-label">ATR Projection</div>
          <div class="field-value predicted">{d['projected_lower']:,.0f} — {d['projected_upper']:,.0f}</div>
        </div>"""

    cr_html = ""
    if "closing_range" in d:
        cr_val = d["closing_range"]
        cr_label = "Strong" if cr_val > 0.7 else "Weak" if cr_val < 0.3 else "Neutral"
        cr_class = "positive" if cr_val > 0.7 else "negative" if cr_val < 0.3 else "predicted"
        cr_html = f"""
        <div class="card-field">
          <div class="field-label">Closing Range</div>
          <div class="field-value {cr_class}">{cr_val:.2f} ({cr_label})</div>
        </div>"""

    return f"""
    <div class="card">
      <div class="card-header">
        <div><span class="card-rank">#{rank}</span> <span class="card-ticker">{entry.ticker}</span></div>
        <span class="card-engine">{entry.signal}</span>
      </div>
      <div class="card-grid">
        <div class="card-field">
          <div class="field-label">Entry</div>
          <div class="field-value">IDR {entry.price:,.0f}</div>
        </div>
        <div class="card-field">
          <div class="field-label">Stop-Loss</div>
          {stop_html}
        </div>
        <div class="card-field">
          <div class="field-label">Trailing Stop</div>
          {trail_html}
        </div>
        <div class="card-field">
          <div class="field-label">Position Size</div>
          {size_html}
        </div>
        <div class="card-field">
          <div class="field-label">Risk</div>
          {risk_html}
        </div>
        <div class="card-field">
          <div class="field-label">Score</div>
          <div class="field-value">{entry.score:.2f}</div>
        </div>
        {predicted_html}
        {projection_html}
        {cr_html}
      </div>
    </div>"""


def _html_avoid_donut(avoid_bk: dict, total_avoid: int) -> str:
    """Generate a CSS-only donut chart for the Avoid breakdown."""
    categories = [
        ("Low ADTV", avoid_bk.get("low_adtv", 0), "#f44336"),
        ("Penny", avoid_bk.get("penny_stock", 0), "#ff9800"),
        ("Below SMA200", avoid_bk.get("below_sma200", 0), "#2196f3"),
        ("Earnings", avoid_bk.get("earnings_proximity", 0), "#9c27b0"),
        ("No Data", avoid_bk.get("insufficient_data", 0), "#607d8b"),
    ]

    if total_avoid == 0:
        return '<div class="card no-signal">No stocks avoided</div>'

    # Build conic-gradient segments
    segments = []
    cumulative = 0.0
    for _, count, color in categories:
        if count > 0:
            pct = (count / total_avoid) * 100.0
            segments.append(f"{color} {cumulative:.1f}% {cumulative + pct:.1f}%")
            cumulative += pct

    gradient = f"conic-gradient({', '.join(segments)})" if segments else "conic-gradient(#2a2d37 0% 100%)"

    # Legend items
    legend_items = ""
    for label, count, color in categories:
        if count > 0:
            pct = (count / total_avoid) * 100.0
            legend_items += (
                f'<div class="legend-item">'
                f'<div class="legend-dot" style="background:{color}"></div>'
                f'{label}: {count} ({pct:.0f}%)'
                f'</div>\n'
            )

    return f"""
    <div class="avoid-donut-row">
      <div class="donut" style="background: {gradient}">
        <div class="donut-hole"><span>{total_avoid}</span></div>
      </div>
      <div class="donut-legend">
        {legend_items}
      </div>
    </div>"""

