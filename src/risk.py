from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import streamlit as st
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from src.data_connector import get_paper_trading_client
from src.formatting import (
    enum_text,
    field as object_field,
    first_field,
    format_datetime,
    format_money,
    format_percent,
    format_plain_number,
    normalize_records,
    to_float,
)

if TYPE_CHECKING:
    from alpaca.trading.client import TradingClient

    from src.execution import OrderPlan


RISK_CONFIG_FILE = Path("risk_config.json")

DEFAULT_STOP_LOSS_PCT = 0.20
DEFAULT_MAX_ALLOCATION_PCT = 0.25
MIN_STOP_LOSS_PCT = 0.10
MAX_STOP_LOSS_PCT = 0.50
MIN_ALLOCATION_PCT = 0.10
MAX_ALLOCATION_LIMIT_PCT = 0.50
STOP_LOSS_WATCH_RATIO = 0.50
ALLOCATION_WATCH_RATIO = 0.80
RISK_FLAG_CLEAR = "Active"
RISK_FLAG_WATCH = "Watch"
RISK_FLAG_BREACH = "Breach"

RISK_REFRESH_SECONDS = 10.0
CLOSE_COOLDOWN_SECONDS = 90.0
MAX_RISK_EVENTS = 50
RISK_EVENTS_VISIBLE_ROWS = 5

RISK_CONFIG_STATE_KEY = "risk_config"
RISK_EVENTS_STATE_KEY = "risk_events"
RISK_COOLDOWN_STATE_KEY = "risk_close_cooldowns"


@dataclass(frozen=True)
class RiskConfig:
    """Portfolio risk rules. Percentages are stored as fractions (0.20 = 20%)."""

    enabled: bool = True
    stop_loss_pct: float = DEFAULT_STOP_LOSS_PCT
    max_allocation_pct: float = DEFAULT_MAX_ALLOCATION_PCT


@dataclass(frozen=True)
class PositionRiskStatus:
    """Risk evaluation of one open paper position."""

    symbol: str
    qty: float
    market_value: float | None
    allocation: float | None
    unrealized_plpc: float | None
    stop_loss_breached: bool
    allocation_breached: bool
    stop_loss_watch: bool
    allocation_watch: bool


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)


def _percent_label(fraction: float) -> str:
    return f"{fraction * 100:g}%"


def _risk_logger():
    # Imported lazily because src.execution imports this module.
    from src.execution import get_paper_trading_logger

    return get_paper_trading_logger()


def load_risk_config(path: Path = RISK_CONFIG_FILE) -> RiskConfig:
    """Load saved risk settings, falling back to defaults for a missing or bad file."""
    if not path.exists():
        return RiskConfig()

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return RiskConfig()

    if not isinstance(data, dict):
        return RiskConfig()

    stop_loss = to_float(data.get("stop_loss_pct"))
    allocation = to_float(data.get("max_allocation_pct"))

    return RiskConfig(
        enabled=bool(data.get("enabled", True)),
        stop_loss_pct=(
            _clamp(stop_loss, MIN_STOP_LOSS_PCT, MAX_STOP_LOSS_PCT)
            if stop_loss is not None
            else DEFAULT_STOP_LOSS_PCT
        ),
        max_allocation_pct=(
            _clamp(allocation, MIN_ALLOCATION_PCT, MAX_ALLOCATION_LIMIT_PCT)
            if allocation is not None
            else DEFAULT_MAX_ALLOCATION_PCT
        ),
    )


def save_risk_config(config: RiskConfig, path: Path = RISK_CONFIG_FILE) -> None:
    path.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")


def position_unrealized_plpc(position: Any) -> float | None:
    """Unrealized P&L fraction for a position, computed from prices when missing."""
    plpc = to_float(object_field(position, "unrealized_plpc"))
    if plpc is not None:
        return plpc

    entry_price = to_float(object_field(position, "avg_entry_price"))
    current_price = to_float(object_field(position, "current_price"))
    if not entry_price or current_price is None:
        return None

    change = current_price / entry_price - 1.0
    qty = to_float(object_field(position, "qty")) or 0.0
    return -change if qty < 0 else change


def evaluate_position_risk(
    position: Any,
    portfolio_value: float | None,
    config: RiskConfig,
) -> PositionRiskStatus | None:
    """Evaluate one position against the stop-loss and allocation rules."""
    symbol = str(object_field(position, "symbol", "") or "").upper()
    qty = to_float(object_field(position, "qty")) or 0.0
    if not symbol or abs(qty) <= 1e-9:
        return None

    market_value = to_float(object_field(position, "market_value"))
    allocation = (
        abs(market_value) / portfolio_value
        if market_value is not None and portfolio_value is not None and portfolio_value > 0
        else None
    )
    unrealized_plpc = position_unrealized_plpc(position)

    stop_loss_breached = (
        unrealized_plpc is not None and unrealized_plpc <= -config.stop_loss_pct
    )
    allocation_breached = (
        allocation is not None and allocation > config.max_allocation_pct + 1e-9
    )
    stop_loss_watch = (
        unrealized_plpc is not None
        and unrealized_plpc <= -(config.stop_loss_pct * STOP_LOSS_WATCH_RATIO)
        and not stop_loss_breached
    )
    allocation_watch = (
        allocation is not None
        and allocation >= config.max_allocation_pct * ALLOCATION_WATCH_RATIO
        and not allocation_breached
    )

    return PositionRiskStatus(
        symbol=symbol,
        qty=qty,
        market_value=market_value,
        allocation=allocation,
        unrealized_plpc=unrealized_plpc,
        stop_loss_breached=stop_loss_breached,
        allocation_breached=allocation_breached,
        stop_loss_watch=stop_loss_watch,
        allocation_watch=allocation_watch,
    )


def evaluate_portfolio_risk(
    positions: list[Any],
    portfolio_value: float | None,
    config: RiskConfig,
) -> list[PositionRiskStatus]:
    statuses = []
    for position in positions:
        status = evaluate_position_risk(position, portfolio_value, config)
        if status is not None:
            statuses.append(status)

    statuses.sort(key=lambda status: status.allocation or 0.0, reverse=True)
    return statuses


def apply_risk_to_order_plan(
    order_plan: OrderPlan,
    portfolio_value: float | None,
    config: RiskConfig,
    current_symbol_market_value: float = 0.0,
) -> OrderPlan:
    """
    Cap a BUY plan so the resulting position stays within the allocation limit.

    Returns the plan unchanged for non-BUY actions, when enforcement is off, or
    when no portfolio value is available to size the cap against.
    """
    if not config.enabled or order_plan.action != "BUY" or order_plan.quantity is None:
        return order_plan

    if portfolio_value is None or portfolio_value <= 0:
        return order_plan

    limit_label = _percent_label(config.max_allocation_pct)
    allowed_notional = (
        config.max_allocation_pct * portfolio_value
        - max(current_symbol_market_value, 0.0)
    )
    planned_notional = order_plan.quantity * order_plan.close_price
    if planned_notional <= allowed_notional + 1e-9:
        return order_plan

    capped_quantity = math.floor(max(allowed_notional, 0.0) / order_plan.close_price)
    if capped_quantity < 1:
        return replace(
            order_plan,
            action="NONE",
            side=None,
            quantity=None,
            reason=(
                f"{order_plan.reason} BUY blocked by risk rule: the {limit_label} "
                "allocation limit leaves no room for one share."
            ),
        )

    return replace(
        order_plan,
        quantity=float(capped_quantity),
        reason=(
            f"{order_plan.reason} BUY size capped from "
            f"{order_plan.quantity:g} to {capped_quantity:g} shares by the "
            f"{limit_label} allocation limit."
        ),
    )


def fetch_pending_sell_symbols(client: TradingClient) -> set[str]:
    """Symbols that already have an open sell order, to avoid duplicate closes."""
    orders = normalize_records(
        client.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    )
    symbols = set()
    for order in orders:
        side = enum_text(object_field(order, "side")).lower()
        symbol = str(object_field(order, "symbol", "") or "").upper()
        if symbol and "sell" in side:
            symbols.add(symbol)
    return symbols


def enforce_stop_losses(
    client: TradingClient,
    statuses: list[PositionRiskStatus],
    pending_sell_symbols: set[str],
    cooldowns: dict[str, pd.Timestamp],
    config: RiskConfig,
) -> list[dict[str, Any]]:
    """Submit market close orders for stop-loss breaches; returns event records."""
    logger = _risk_logger()
    events: list[dict[str, Any]] = []
    now = pd.Timestamp.now(tz="UTC")

    for status in statuses:
        if not status.stop_loss_breached:
            continue

        symbol = status.symbol
        if symbol in pending_sell_symbols:
            continue

        last_close = cooldowns.get(symbol)
        if (
            last_close is not None
            and (now - last_close).total_seconds() < CLOSE_COOLDOWN_SECONDS
        ):
            continue

        cooldowns[symbol] = now
        loss_text = format_percent(status.unrealized_plpc)
        limit_text = _percent_label(config.stop_loss_pct)

        try:
            client.close_position(symbol)
        except Exception as exc:
            message = (
                f"Stop loss triggered for {symbol} at {loss_text} "
                f"(limit -{limit_text}), but the close order failed: {exc}"
            )
            logger.error("RISK | %s", message)
            events.append(
                {"time": now, "symbol": symbol, "message": message, "error": True}
            )
            continue

        message = (
            f"Stop loss triggered for {symbol} at {loss_text} "
            f"(limit -{limit_text}); submitted market close order."
        )
        logger.info("RISK | %s", message)
        events.append(
            {"time": now, "symbol": symbol, "message": message, "error": False}
        )

    return events


def get_active_risk_config() -> RiskConfig:
    config = st.session_state.get(RISK_CONFIG_STATE_KEY)
    if config is None:
        config = load_risk_config()
        st.session_state[RISK_CONFIG_STATE_KEY] = config
    return config


def position_risk_flag(status: PositionRiskStatus) -> str:
    if status.stop_loss_breached or status.allocation_breached:
        return RISK_FLAG_BREACH
    if status.stop_loss_watch or status.allocation_watch:
        return RISK_FLAG_WATCH
    return RISK_FLAG_CLEAR


def position_risk_tooltip(status: PositionRiskStatus, config: RiskConfig) -> str:
    """Explain why a position is in Watch or Breach state."""
    flag = position_risk_flag(status)
    if flag == RISK_FLAG_CLEAR:
        return ""

    messages = []
    allocation_text = format_percent(status.allocation)
    max_allocation_text = _percent_label(config.max_allocation_pct)
    stop_loss_text = _percent_label(config.stop_loss_pct)

    if status.allocation_breached:
        messages.append(
            f"Current allocation {allocation_text} is greater than the set "
            f"{max_allocation_text} limit."
        )
    elif status.allocation_watch:
        watch_limit = _percent_label(config.max_allocation_pct * ALLOCATION_WATCH_RATIO)
        messages.append(
            f"Allocation {allocation_text} is in Watch because it is at or above "
            f"{watch_limit}, which is 80% of the set {max_allocation_text} limit."
        )

    if status.stop_loss_breached:
        messages.append(
            f"Drawdown {format_percent(status.unrealized_plpc)} is at or below "
            f"the set -{stop_loss_text} stop-loss limit."
        )
    elif status.stop_loss_watch:
        watch_loss = _percent_label(config.stop_loss_pct * STOP_LOSS_WATCH_RATIO)
        messages.append(
            f"Drawdown {format_percent(status.unrealized_plpc)} is in Watch because "
            f"it is at or below -{watch_loss}, half of the set -{stop_loss_text} "
            "stop-loss limit."
        )

    return " ".join(messages)


def _risk_statuses_dataframe(statuses: list[PositionRiskStatus]) -> pd.DataFrame:
    rows = [
        {
            "Symbol": status.symbol,
            "Qty": format_plain_number(status.qty),
            "Market Value": format_money(status.market_value),
            "Allocation": format_percent(status.allocation),
            "Unrealized %": format_percent(status.unrealized_plpc),
            "Status": position_risk_flag(status),
        }
        for status in statuses
    ]
    return pd.DataFrame(rows)


def _risk_events_dataframe(events: list[dict[str, Any]]) -> pd.DataFrame:
    rows = [
        {
            "Time": format_datetime(event.get("time")),
            "Symbol": event.get("symbol", "n/a"),
            "Event": event.get("message", ""),
        }
        for event in reversed(events)
    ]
    return pd.DataFrame(rows)


def _render_risk_config_form(config: RiskConfig) -> None:
    with st.expander("Risk Controls", expanded=False):
        with st.form("risk_config_form"):
            enabled = st.toggle(
                "Enforce risk rules",
                value=config.enabled,
                key="risk_enabled_input",
                help=(
                    "When on, positions breaching the stop loss are closed "
                    "automatically and new buys are capped at the allocation "
                    "limit. When off, breaches are only highlighted."
                ),
            )
            stop_loss_percent = st.slider(
                "Stop loss (% loss per position)",
                min_value=MIN_STOP_LOSS_PCT * 100,
                max_value=MAX_STOP_LOSS_PCT * 100,
                value=config.stop_loss_pct * 100,
                step=1.0,
                format="%.0f%%",
                key="risk_stop_loss_input",
                help="Close a position when its unrealized loss reaches this percentage.",
            )
            allocation_percent = st.slider(
                "Max allocation per position (% of portfolio)",
                min_value=MIN_ALLOCATION_PCT * 100,
                max_value=MAX_ALLOCATION_LIMIT_PCT * 100,
                value=config.max_allocation_pct * 100,
                step=1.0,
                format="%.0f%%",
                key="risk_allocation_input",
                help=(
                    "New buys are sized so a single position never exceeds this "
                    "share of portfolio value. Positions that drift above it are "
                    "flagged but not sold."
                ),
            )
            saved = st.form_submit_button(
                "Save risk settings",
                type="primary",
                width="stretch",
            )

        if saved:
            new_config = RiskConfig(
                enabled=bool(enabled),
                stop_loss_pct=float(stop_loss_percent) / 100,
                max_allocation_pct=float(allocation_percent) / 100,
            )
            save_risk_config(new_config)
            st.session_state[RISK_CONFIG_STATE_KEY] = new_config
            st.success("Risk settings saved.")


@st.fragment(run_every=RISK_REFRESH_SECONDS)
def _render_risk_status_panel(show_position_table: bool = True) -> None:
    config = get_active_risk_config()

    try:
        client = get_paper_trading_client()
        account = client.get_account()
        positions = normalize_records(client.get_all_positions())
    except Exception as exc:
        st.warning(f"Could not load account data for risk checks: {exc}")
        return

    portfolio_value = to_float(first_field(account, "portfolio_value", "equity"))
    statuses = evaluate_portfolio_risk(positions, portfolio_value, config)

    enforcement_text = "Enforcement ON" if config.enabled else "Enforcement OFF"
    st.caption(
        f"Stop loss -{_percent_label(config.stop_loss_pct)} · "
        f"Max allocation {_percent_label(config.max_allocation_pct)} · "
        f"{enforcement_text} · checks every {RISK_REFRESH_SECONDS:.0f}s "
        "while the terminal is open."
    )

    if config.enabled and any(status.stop_loss_breached for status in statuses):
        try:
            pending_sell_symbols = fetch_pending_sell_symbols(client)
        except Exception as exc:
            st.warning(
                "Skipped stop-loss enforcement this cycle: could not check "
                f"open orders for duplicate closes ({exc})."
            )
        else:
            cooldowns = st.session_state.setdefault(RISK_COOLDOWN_STATE_KEY, {})
            events = enforce_stop_losses(
                client,
                statuses,
                pending_sell_symbols,
                cooldowns,
                config,
            )
            if events:
                history = st.session_state.setdefault(RISK_EVENTS_STATE_KEY, [])
                history.extend(events)
                del history[:-MAX_RISK_EVENTS]

    if not config.enabled and any(
        status.stop_loss_breached or status.allocation_breached
        for status in statuses
    ):
        st.warning(
            "Risk rules are breached but enforcement is off; no orders were submitted."
        )

    if show_position_table:
        if not statuses:
            st.info("No open paper positions to monitor.")
        else:
            st.dataframe(
                _risk_statuses_dataframe(statuses),
                hide_index=True,
                width="stretch",
            )

    history = st.session_state.get(RISK_EVENTS_STATE_KEY) or []
    if history:
        st.markdown("**Risk Events**")
        st.dataframe(
            _risk_events_dataframe(history[-RISK_EVENTS_VISIBLE_ROWS:]),
            hide_index=True,
            width="stretch",
        )


def render_risk_management_panel(
    show_heading: bool = True,
    show_position_table: bool = True,
) -> None:
    if show_heading:
        st.subheader("Risk Management")

    config = get_active_risk_config()
    _render_risk_config_form(config)
    _render_risk_status_panel(show_position_table=show_position_table)
