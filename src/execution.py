from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
from alpaca.common.enums import Sort
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import (
    GetOrdersRequest,
    GetPortfolioHistoryRequest,
    MarketOrderRequest,
)

from src.config import AlpacaSettings
from src.data_connector import get_paper_trading_client
from src.formatting import (
    EASTERN_TZ,
    enum_text,
    field as object_field,
    first_field,
    format_datetime,
    format_money,
    format_percent,
    format_plain_number,
    money_class,
    normalize_records,
    sort_timestamp,
    to_float,
)


DEFAULT_ORDER_NOTIONAL = 10_000.0
PAPER_ACCOUNT_ORDER_LIMIT = 50
ML_STRATEGY_DISPLAY_NAME = "ML Logistic Regression"

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "paper_trading.log"


@dataclass(frozen=True)
class OrderPlan:
    symbol: str
    action: str
    side: str | None
    quantity: float | None
    latest_position: int
    current_position: float
    close_price: float
    probability: float
    bar_timestamp: pd.Timestamp | None
    requested_notional: float
    reason: str


@dataclass(frozen=True)
class ExecutionReport:
    """Outcome of one paper-trading decision, for display and submission logs."""

    symbol: str
    bar_timestamp: pd.Timestamp | None
    close_price: float
    probability: float
    signal: str
    action: str
    current_position: float
    order_id: str | None = None
    order_status: str | None = None
    order_qty: float | None = None
    market_open: bool | None = None
    dry_run: bool = False
    message: str = ""
    log_lines: list[str] = field(default_factory=list)


def get_paper_trading_logger() -> logging.Logger:
    """Return a logger that writes signal and order events to logs/paper_trading.log."""
    logger = logging.getLogger("paper_trading")

    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        LOG_DIR.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger


def read_paper_trading_log(max_lines: int = 80) -> str:
    if not LOG_FILE.exists():
        return "No paper-trading log has been written yet."

    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    recent_lines = lines[-max_lines:]
    return "\n".join(recent_lines) if recent_lines else "No paper-trading log entries yet."


def fetch_paper_account_snapshot() -> dict[str, Any]:
    trading_client = get_paper_trading_client()
    order_request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=PAPER_ACCOUNT_ORDER_LIMIT,
        direction=Sort.DESC,
        nested=True,
    )

    return {
        "account": trading_client.get_account(),
        "positions": normalize_records(trading_client.get_all_positions()),
        "orders": normalize_records(trading_client.get_orders(order_request)),
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    }


def portfolio_history_request(timeframe_label: str) -> GetPortfolioHistoryRequest:
    now = pd.Timestamp.now(tz="UTC")

    if timeframe_label == "YTD":
        start = pd.Timestamp(
            year=now.tz_convert(EASTERN_TZ).year,
            month=1,
            day=1,
            tz=EASTERN_TZ,
        )
        return GetPortfolioHistoryRequest(
            start=start.tz_convert("UTC").to_pydatetime(),
            end=now.to_pydatetime(),
            timeframe="1D",
        )

    request_args = {
        "1D": {"period": "1D", "timeframe": "5Min"},
        "1W": {"period": "1W", "timeframe": "15Min"},
        "1M": {"period": "1M", "timeframe": "1D"},
        "3M": {"period": "3M", "timeframe": "1D"},
        "1Y": {"period": "1A", "timeframe": "1D"},
        "ALL": {"period": "all", "timeframe": "1D"},
    }.get(timeframe_label, {"period": "1M", "timeframe": "1D"})

    return GetPortfolioHistoryRequest(**request_args)


def portfolio_history_to_dataframe(history: Any) -> pd.DataFrame:
    timestamps = object_field(history, "timestamp", []) or []
    equities = object_field(history, "equity", []) or []
    profit_losses = object_field(history, "profit_loss", []) or []
    profit_loss_pcts = object_field(history, "profit_loss_pct", []) or []

    row_count = min(len(timestamps), len(equities))
    rows = []
    for index in range(row_count):
        equity = to_float(equities[index])
        timestamp = pd.to_datetime(
            timestamps[index],
            unit="s",
            utc=True,
            errors="coerce",
        )

        if equity is None or pd.isna(timestamp):
            continue

        profit_loss = (
            to_float(profit_losses[index])
            if index < len(profit_losses)
            else None
        )
        profit_loss_pct = (
            to_float(profit_loss_pcts[index])
            if index < len(profit_loss_pcts)
            else None
        )

        rows.append(
            {
                "timestamp": timestamp.tz_convert(EASTERN_TZ),
                "equity": equity,
                "profit_loss": profit_loss,
                "profit_loss_pct": profit_loss_pct,
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "equity", "profit_loss", "profit_loss_pct"]
        )

    return pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)


def fetch_portfolio_history_dataframe(timeframe_label: str) -> pd.DataFrame:
    trading_client = get_paper_trading_client()
    history = trading_client.get_portfolio_history(
        portfolio_history_request(timeframe_label)
    )
    return portfolio_history_to_dataframe(history)


def position_symbol(position: Any) -> str:
    return str(object_field(position, "symbol", "") or "").upper()


def current_strategy_state_from_positions(positions: list[Any]) -> dict[str, Any]:
    open_positions = []
    for position in positions:
        qty = to_float(object_field(position, "qty")) or 0.0
        if abs(qty) <= 1e-9:
            continue

        sort_value = abs(to_float(object_field(position, "market_value")) or qty)
        open_positions.append((sort_value, position))

    if not open_positions:
        return {
            "equity": "Flat",
            "status": "Flat",
            "has_position": False,
        }

    open_positions.sort(key=lambda item: item[0], reverse=True)
    active_symbols = [
        position_symbol(position)
        for _, position in open_positions
        if position_symbol(position)
    ]
    active_symbol = active_symbols[0] if active_symbols else "n/a"

    return {
        "equity": active_symbol,
        "status": "Active",
        "has_position": True,
    }


def fetch_current_strategy_state() -> dict[str, Any]:
    client = get_paper_trading_client()
    positions = normalize_records(client.get_all_positions())
    return current_strategy_state_from_positions(positions)


def resolve_strategy_display_state(
    strategy_state: dict[str, Any],
    active_config: dict[str, Any] | None,
    stop_report: dict[str, Any] | None,
) -> dict[str, Any]:
    stopped_positions = stop_report.get("stopped_positions") if stop_report else []
    has_pending_stop_report = bool(
        stop_report is not None
        and stopped_positions
        and not stop_report.get("error")
    )
    started_without_position = (
        active_config is not None
        and not strategy_state["has_position"]
        and not has_pending_stop_report
    )

    if strategy_state["has_position"]:
        display_strategy = (
            active_config["strategy"]
            if active_config is not None
            and active_config.get("equity") == strategy_state["equity"]
            else ML_STRATEGY_DISPLAY_NAME
        )
        display_equity = strategy_state["equity"]
        display_status = strategy_state["status"]
    elif has_pending_stop_report:
        display_strategy = (
            active_config["strategy"] if active_config is not None else ML_STRATEGY_DISPLAY_NAME
        )
        display_equity = stopped_positions[0].get("symbol", "n/a")
        display_status = "Stopped"
    elif started_without_position:
        display_strategy = active_config["strategy"]
        display_equity = active_config["equity"]
        display_status = "Active (Flat)"
    else:
        display_strategy = "None"
        display_equity = "Flat"
        display_status = "Inactive"

    return {
        "display_strategy": display_strategy,
        "display_equity": display_equity,
        "display_status": display_status,
        "has_pending_stop_report": has_pending_stop_report,
        "started_without_position": started_without_position,
        "stopped_positions": stopped_positions or [],
    }


def build_strategy_stop_error_report(exc: Exception) -> dict[str, Any]:
    return {
        "message": f"Could not stop strategy: {exc}",
        "closed_count": None,
        "orders": [],
        "stopped_at": pd.Timestamp.now(tz="UTC"),
        "error": True,
    }


def build_strategy_cancel_error_report(
    exc: Exception,
    source_stopped_at: str,
) -> dict[str, Any]:
    return {
        "message": f"Could not reactivate strategy: {exc}",
        "orders": [],
        "source_stopped_at": source_stopped_at,
        "reactivated_at": pd.Timestamp.now(tz="UTC"),
        "error": True,
    }


def build_strategy_start_state(
    strategy: str,
    equity: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = pd.Timestamp.now(tz="UTC")
    active_config = {
        "strategy": strategy,
        "equity": equity,
        "started_at": started_at,
    }
    start_report = {
        "message": (
            f"Started {strategy} on {equity}. "
            "The strategy is active in flat mode until a paper position is opened."
        ),
        "started_at": started_at,
    }
    return active_config, start_report


def sum_position_field(positions: list[Any], field_name: str) -> float:
    total = 0.0
    for position in positions:
        total += to_float(object_field(position, field_name)) or 0.0
    return total


def order_status(order: Any) -> str:
    return enum_text(object_field(order, "status")).lower()


def order_side(order: Any) -> str:
    return enum_text(object_field(order, "side")).lower()


def order_result_label(status: str) -> str:
    normalized = status.lower()
    if normalized == "filled":
        return "Success"
    if "cancel" in normalized:
        return "Cancelled"
    if normalized == "partially_filled":
        return "Partial"
    if normalized in {"rejected", "expired", "stopped", "suspended"}:
        return normalized.replace("_", " ").title()
    if normalized in {"new", "accepted", "pending_new", "accepted_for_bidding"}:
        return "Open"
    return normalized.replace("_", " ").title() if normalized else "Unknown"


def calculate_recent_realized_pnl(orders: list[Any]) -> float:
    lots_by_symbol: dict[str, list[dict[str, float]]] = {}
    realized_pnl = 0.0

    sorted_orders = sorted(
        orders,
        key=lambda order: sort_timestamp(
            first_field(order, "filled_at", "submitted_at", "created_at")
        ),
    )

    for order in sorted_orders:
        status = order_status(order)
        filled_qty = to_float(object_field(order, "filled_qty")) or 0.0
        fill_price = to_float(object_field(order, "filled_avg_price"))

        if (
            filled_qty <= 0
            or fill_price is None
            or status not in {"filled", "partially_filled"}
        ):
            continue

        symbol = str(object_field(order, "symbol", "") or "").upper()
        side = order_side(order)
        if not symbol or side not in {"buy", "sell"}:
            continue

        lots = lots_by_symbol.setdefault(symbol, [])
        if side == "buy":
            lots.append({"qty": filled_qty, "price": fill_price})
            continue

        remaining = filled_qty
        while remaining > 0 and lots:
            lot = lots[0]
            matched_qty = min(remaining, lot["qty"])
            realized_pnl += matched_qty * (fill_price - lot["price"])
            lot["qty"] -= matched_qty
            remaining -= matched_qty

            if lot["qty"] <= 1e-9:
                lots.pop(0)

    return realized_pnl


def orders_to_dataframe(orders: list[Any]) -> pd.DataFrame:
    rows = []
    for order in orders:
        status = order_status(order)
        side = order_side(order)
        submitted_at = first_field(order, "submitted_at", "created_at")
        filled_at = object_field(order, "filled_at")

        rows.append(
            {
                "Submitted": format_datetime(submitted_at),
                "Filled": format_datetime(filled_at),
                "Symbol": str(object_field(order, "symbol", "") or "").upper(),
                "Side": side.upper() if side else "n/a",
                "Result": order_result_label(status),
                "Status": status.replace("_", " ").title() if status else "Unknown",
                "Type": enum_text(first_field(order, "type", "order_type"))
                .replace("_", " ")
                .title(),
                "Qty": format_plain_number(object_field(order, "qty")),
                "Filled Qty": format_plain_number(object_field(order, "filled_qty")),
                "Avg Fill": format_money(object_field(order, "filled_avg_price"), currency=""),
                "Limit": format_money(object_field(order, "limit_price"), currency=""),
                "Stop": format_money(object_field(order, "stop_price"), currency=""),
                "Order ID": str(object_field(order, "id", "") or ""),
            }
        )

    return pd.DataFrame(rows)


def orders_to_exchange_log_dataframe(orders: list[Any]) -> pd.DataFrame:
    rows = []
    for order in orders:
        order_id = str(object_field(order, "id", "") or "")
        symbol = str(object_field(order, "symbol", "") or "").upper()
        side = order_side(order).upper() or "ORDER"
        qty = format_plain_number(object_field(order, "qty"))
        status = order_status(order)
        order_type = enum_text(first_field(order, "type", "order_type")).replace(
            "_",
            " ",
        )
        submitted_at = first_field(order, "submitted_at", "created_at")
        filled_at = object_field(order, "filled_at")
        canceled_at = first_field(order, "canceled_at", "cancelled_at")
        expired_at = object_field(order, "expired_at")
        failed_at = object_field(order, "failed_at")
        fill_price = object_field(order, "filled_avg_price")

        if submitted_at is not None:
            rows.append(
                {
                    "Time": format_datetime(submitted_at),
                    "Message": (
                        f"Submitted {side} {order_type} order for {qty} shares of "
                        f"{symbol}; order id {order_id}; status {status or 'unknown'}."
                    ),
                    "_sort": sort_timestamp(submitted_at),
                }
            )

        if filled_at is not None:
            rows.append(
                {
                    "Time": format_datetime(filled_at),
                    "Message": (
                        f"Filled order {order_id} for {symbol}; filled quantity "
                        f"{format_plain_number(object_field(order, 'filled_qty'))} at "
                        f"{format_money(fill_price, currency='')}."
                    ),
                    "_sort": sort_timestamp(filled_at),
                }
            )

        if canceled_at is not None:
            rows.append(
                {
                    "Time": format_datetime(canceled_at),
                    "Message": f"Cancelled order {order_id} for {symbol}.",
                    "_sort": sort_timestamp(canceled_at),
                }
            )

        if expired_at is not None:
            rows.append(
                {
                    "Time": format_datetime(expired_at),
                    "Message": f"Expired order {order_id} for {symbol}.",
                    "_sort": sort_timestamp(expired_at),
                }
            )

        if failed_at is not None:
            rows.append(
                {
                    "Time": format_datetime(failed_at),
                    "Message": (
                        f"Failed order {order_id} for {symbol}; "
                        f"status {status or 'unknown'}."
                    ),
                    "_sort": sort_timestamp(failed_at),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Time", "Message"])

    result = pd.DataFrame(rows).sort_values("_sort", ascending=False)
    return result.drop(columns=["_sort"]).reset_index(drop=True)


def positions_to_dataframe(
    positions: list[Any],
    portfolio_value: float | None,
) -> pd.DataFrame:
    rows = []
    for position in positions:
        market_value = to_float(object_field(position, "market_value"))
        cost_basis = to_float(object_field(position, "cost_basis"))
        unrealized = to_float(object_field(position, "unrealized_pl"))
        allocation = (
            market_value / portfolio_value
            if market_value is not None and portfolio_value not in {None, 0}
            else None
        )

        rows.append(
            {
                "Symbol": position_symbol(position),
                "Side": (enum_text(object_field(position, "side")) or "long").title(),
                "Allocation": format_percent(allocation),
                "Qty": format_plain_number(object_field(position, "qty")),
                "Avg Price": format_money(object_field(position, "avg_entry_price"), currency=""),
                "Current Price": format_money(object_field(position, "current_price"), currency=""),
                "Market Value": format_money(market_value),
                "Cost Basis": format_money(cost_basis),
                "Unrealized P&L": format_money(unrealized),
                "Unrealized %": format_percent(object_field(position, "unrealized_plpc")),
                "Daily P&L": format_money(object_field(position, "unrealized_intraday_pl")),
                "_sort": abs(market_value or 0.0),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "Symbol",
                "Side",
                "Allocation",
                "Qty",
                "Avg Price",
                "Current Price",
                "Market Value",
                "Cost Basis",
                "Unrealized P&L",
                "Unrealized %",
                "Daily P&L",
            ]
        )

    result = pd.DataFrame(rows).sort_values("_sort", ascending=False)
    return result.drop(columns=["_sort"]).reset_index(drop=True)


def account_portfolio_value(account: Any) -> float | None:
    return to_float(first_field(account, "portfolio_value", "equity"))


def account_details_dataframe(
    account: Any,
    positions: list[Any],
    orders: list[Any],
) -> pd.DataFrame:
    details = [
        ("Account status", enum_text(object_field(account, "status")) or "n/a"),
        ("Currency", enum_text(object_field(account, "currency")) or "USD"),
        ("Cash", format_money(object_field(account, "cash"))),
        ("Buying power", format_money(object_field(account, "buying_power"))),
        ("Portfolio value", format_money(first_field(account, "portfolio_value", "equity"))),
        ("Equity", format_money(object_field(account, "equity"))),
        ("Last equity", format_money(object_field(account, "last_equity"))),
        ("Long market value", format_money(object_field(account, "long_market_value"))),
        ("Maintenance margin", format_money(object_field(account, "maintenance_margin"))),
        ("Open positions", str(len(positions))),
        ("Recent orders loaded", str(len(orders))),
    ]
    return pd.DataFrame(details, columns=["Field", "Value"])


def build_account_cards(
    account: Any,
    positions: list[Any],
    orders: list[Any],
) -> list[dict[str, str]]:
    portfolio_value = to_float(first_field(account, "portfolio_value", "equity"))
    buying_power = to_float(object_field(account, "buying_power"))
    cash = to_float(object_field(account, "cash"))
    unrealized_pnl = sum_position_field(positions, "unrealized_pl")
    total_cost = sum_position_field(positions, "cost_basis")
    unrealized_pct = unrealized_pnl / total_cost if total_cost else None
    realized_pnl = calculate_recent_realized_pnl(orders)
    recent_order = orders[0] if orders else None

    if recent_order is None:
        recent_value = "No orders"
        recent_note = "No recent paper order history returned"
        recent_class = "neutral"
    else:
        status = order_status(recent_order)
        recent_value = order_result_label(status)
        recent_note = (
            f"{order_side(recent_order).upper()} "
            f"{format_plain_number(object_field(recent_order, 'qty'))} "
            f"{str(object_field(recent_order, 'symbol', '') or '').upper()}"
        )
        if status == "filled":
            recent_class = "positive"
        elif status in {"rejected", "expired"} or "cancel" in status:
            recent_class = "negative"
        else:
            recent_class = "neutral"

    return [
        {
            "label": "Portfolio Value",
            "value": format_money(portfolio_value),
            "note": f"Cash {format_money(cash)}",
            "class": "neutral",
        },
        {
            "label": "Buying Power",
            "value": format_money(buying_power),
            "note": "Available paper funds",
            "class": "neutral",
        },
        {
            "label": "Unrealized P&L",
            "value": format_money(unrealized_pnl),
            "note": f"{format_percent(unrealized_pct)} on open cost basis",
            "class": money_class(unrealized_pnl),
        },
        {
            "label": "Realized P&L (recent)",
            "value": format_money(realized_pnl),
            "note": "FIFO estimate from loaded fills",
            "class": money_class(realized_pnl),
        },
        {
            "label": "Recent Order",
            "value": recent_value,
            "note": recent_note,
            "class": recent_class,
        },
    ]


def get_current_position(
    symbol: str,
    trading_client: TradingClient | None = None,
) -> float:
    """Return current paper shares for symbol, or 0 when there is no open position."""
    client = trading_client or get_paper_trading_client()

    try:
        position = client.get_open_position(symbol)
    except APIError:
        return 0.0

    return float(position.qty)


def get_latest_signal(signal_df: pd.DataFrame) -> dict[str, Any]:
    """Extract the latest usable ML signal row from a model-generated signal DataFrame."""
    required_columns = {"close", "ml_probability", "ml_position"}
    missing = sorted(required_columns - set(signal_df.columns))
    if missing:
        raise ValueError(f"Signal DataFrame is missing required columns: {missing}")

    ready = signal_df.dropna(subset=["close", "ml_probability", "ml_position"])
    if ready.empty:
        raise ValueError("Signal DataFrame does not contain a usable latest signal row.")

    latest = ready.iloc[-1]
    timestamp = (
        pd.Timestamp(latest["timestamp"])
        if "timestamp" in ready.columns and pd.notna(latest["timestamp"])
        else None
    )
    latest_position = int(latest["ml_position"])
    probability = float(latest["ml_probability"])
    close_price = float(latest["close"])
    trade_signal = (
        int(latest["ml_trade_signal"])
        if "ml_trade_signal" in ready.columns and pd.notna(latest["ml_trade_signal"])
        else None
    )

    return {
        "timestamp": timestamp,
        "close_price": close_price,
        "probability": probability,
        "position": latest_position,
        "trade_signal": trade_signal,
        "label": "LONG" if latest_position == 1 else "FLAT",
    }


def build_order_plan(
    symbol: str,
    latest_signal: dict[str, Any],
    current_position: float,
    notional: float = DEFAULT_ORDER_NOTIONAL,
    available_cash: float | None = None,
) -> OrderPlan:
    """Convert latest long/flat signal and current paper position into an order plan."""
    if notional <= 0:
        raise ValueError("notional must be positive.")

    close_price = float(latest_signal["close_price"])
    if close_price <= 0:
        raise ValueError("latest close price must be positive.")

    latest_position = int(latest_signal["position"])
    probability = float(latest_signal["probability"])
    timestamp = latest_signal.get("timestamp")

    if latest_position == 1 and current_position <= 0:
        effective_notional = min(notional, available_cash) if available_cash is not None else notional
        quantity = math.floor(effective_notional / close_price)

        if quantity < 1:
            return OrderPlan(
                symbol=symbol,
                action="NONE",
                side=None,
                quantity=None,
                latest_position=latest_position,
                current_position=current_position,
                close_price=close_price,
                probability=probability,
                bar_timestamp=timestamp,
                requested_notional=notional,
                reason="LONG signal, but available cash is insufficient to buy one share.",
            )

        return OrderPlan(
            symbol=symbol,
            action="BUY",
            side="buy",
            quantity=float(quantity),
            latest_position=latest_position,
            current_position=current_position,
            close_price=close_price,
            probability=probability,
            bar_timestamp=timestamp,
            requested_notional=notional,
            reason="LONG signal and no current paper position.",
        )

    if latest_position == 1 and current_position > 0:
        return OrderPlan(
            symbol=symbol,
            action="HOLD",
            side=None,
            quantity=None,
            latest_position=latest_position,
            current_position=current_position,
            close_price=close_price,
            probability=probability,
            bar_timestamp=timestamp,
            requested_notional=notional,
            reason="LONG signal and paper account already holds shares.",
        )

    if latest_position == 0 and current_position > 0:
        return OrderPlan(
            symbol=symbol,
            action="SELL",
            side="sell",
            quantity=float(current_position),
            latest_position=latest_position,
            current_position=current_position,
            close_price=close_price,
            probability=probability,
            bar_timestamp=timestamp,
            requested_notional=notional,
            reason="FLAT signal and paper account has an open position.",
        )

    return OrderPlan(
        symbol=symbol,
        action="NONE",
        side=None,
        quantity=None,
        latest_position=latest_position,
        current_position=current_position,
        close_price=close_price,
        probability=probability,
        bar_timestamp=timestamp,
        requested_notional=notional,
        reason="FLAT signal and paper account is already flat.",
    )


def submit_paper_order(
    order_plan: OrderPlan,
    trading_client: TradingClient | None = None,
) -> Any:
    """Submit the market order described by an order plan to Alpaca paper trading."""
    client = trading_client or get_paper_trading_client()

    if order_plan.action == "BUY":
        if order_plan.quantity is None:
            raise ValueError("BUY order plan is missing quantity.")

        return client.submit_order(
            MarketOrderRequest(
                symbol=order_plan.symbol,
                qty=int(order_plan.quantity),
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )

    if order_plan.action == "SELL":
        return client.close_position(order_plan.symbol)

    return None


def execute_latest_signal(
    symbol: str,
    signal_df: pd.DataFrame,
    notional: float = DEFAULT_ORDER_NOTIONAL,
    settings: AlpacaSettings | None = None,
    trading_client: TradingClient | None = None,
    dry_run: bool = False,
) -> ExecutionReport:
    """
    Execute the latest model signal in Alpaca paper trading.

    This function intentionally does not fetch market data, compute features,
    apply PCA, train a model, or generate ML signals. It receives the model
    output and handles only paper-account inspection, order planning, order
    submission, and logging.
    """
    logger = get_paper_trading_logger()
    log_lines: list[str] = []

    def log(message: str) -> None:
        logger.info(message)
        log_lines.append(message)

    client = trading_client or get_paper_trading_client(settings)
    latest_signal = get_latest_signal(signal_df)
    signal_label = str(latest_signal["label"])

    log(f"=== Paper execution run for {symbol} ===")
    log(
        f"Latest signal: {signal_label} | "
        f"P(next-day up)={float(latest_signal['probability']):.4f} | "
        f"close={float(latest_signal['close_price']):.2f}"
    )

    clock = client.get_clock()
    log(f"Market open: {clock.is_open} (next open {clock.next_open}, next close {clock.next_close})")

    current_position = get_current_position(symbol, trading_client=client)
    log(f"Current paper position in {symbol}: {current_position:g} shares")

    available_cash = None
    if int(latest_signal["position"]) == 1 and current_position <= 0:
        account = client.get_account()
        available_cash = float(account.cash)

    order_plan = build_order_plan(
        symbol=symbol,
        latest_signal=latest_signal,
        current_position=current_position,
        notional=notional,
        available_cash=available_cash,
    )
    log(f"Order plan: {order_plan.action} | {order_plan.reason}")

    order = None
    if not dry_run and order_plan.action in {"BUY", "SELL"}:
        order = submit_paper_order(order_plan, trading_client=client)
        log(
            f"Submitted paper {order_plan.action} for {symbol} | "
            f"order id {order.id} | status {order.status}"
        )
    elif dry_run:
        log("Dry run enabled. No paper order submitted.")

    log(f"=== Run complete: signal={signal_label}, action={order_plan.action} ===")

    return ExecutionReport(
        symbol=symbol,
        bar_timestamp=order_plan.bar_timestamp,
        close_price=order_plan.close_price,
        probability=order_plan.probability,
        signal=signal_label,
        action=order_plan.action,
        current_position=current_position,
        order_id=str(order.id) if order is not None else None,
        order_status=str(order.status.value if hasattr(order.status, "value") else order.status)
        if order is not None
        else None,
        order_qty=order_plan.quantity if order is not None else None,
        market_open=bool(clock.is_open),
        dry_run=dry_run,
        message=order_plan.reason,
        log_lines=log_lines,
    )


def summarize_close_position_response(response: Any) -> dict[str, str]:
    body = object_field(response, "body")
    order_id = first_field(response, "order_id", default=object_field(body, "id"))
    status = first_field(response, "status", default=object_field(body, "status"))
    symbol = first_field(response, "symbol", default=object_field(body, "symbol"))

    return {
        "Symbol": str(symbol or "n/a").upper(),
        "Order ID": str(order_id or "n/a"),
        "Status": enum_text(status) or "submitted",
    }


def summarize_stopped_position(position: Any) -> dict[str, Any]:
    symbol = str(object_field(position, "symbol", "") or "").upper()
    qty = to_float(object_field(position, "qty")) or 0.0
    market_value = to_float(object_field(position, "market_value"))

    return {
        "symbol": symbol,
        "qty": qty,
        "display_qty": format_plain_number(qty),
        "market_value": market_value,
    }


def stop_all_paper_positions() -> dict[str, Any]:
    logger = get_paper_trading_logger()
    client = get_paper_trading_client()
    positions = normalize_records(client.get_all_positions())
    stopped_positions = [
        stopped_position
        for stopped_position in (
            summarize_stopped_position(position)
            for position in positions
        )
        if stopped_position["symbol"] and stopped_position["qty"] > 0
    ]

    logger.info("=== Strategy STOP requested: closing all paper positions ===")
    responses = normalize_records(client.close_all_positions(cancel_orders=True))
    orders = [summarize_close_position_response(response) for response in responses]

    if not positions and not orders:
        logger.info(
            "STOP request found no open paper positions; "
            "cancel-orders request was still sent."
        )
        return {
            "message": (
                "No open paper positions to exit. "
                "Open paper orders were cancelled if present."
            ),
            "closed_count": 0,
            "orders": [],
            "stopped_positions": [],
            "stopped_at": pd.Timestamp.now(tz="UTC"),
        }

    logger.info(
        "STOP request submitted close-all for %s open paper position(s); "
        "%s response(s) returned.",
        len(positions),
        len(orders),
    )

    return {
        "message": (
            f"Submitted close-all request for {len(positions)} open "
            "paper position(s)."
        ),
        "closed_count": len(positions),
        "orders": orders,
        "stopped_positions": stopped_positions,
        "stopped_at": pd.Timestamp.now(tz="UTC"),
    }


def _order_quantity(value: float) -> int | float:
    if abs(value - round(value)) < 1e-9:
        return int(round(value))
    return value


def reactivate_stopped_strategy(stop_report: dict[str, Any]) -> dict[str, Any]:
    stopped_positions = stop_report.get("stopped_positions") or []
    if not stopped_positions:
        raise ValueError("No stopped position was recorded for this STOP action.")

    logger = get_paper_trading_logger()
    client = get_paper_trading_client()
    submitted_orders = []

    logger.info("=== Strategy STOP cancellation requested: reactivating position ===")

    for stopped_position in stopped_positions:
        symbol = str(stopped_position.get("symbol", "") or "").upper()
        target_qty = to_float(stopped_position.get("qty")) or 0.0
        if not symbol or target_qty <= 0:
            continue

        current_qty = get_current_position(symbol, trading_client=client)
        buy_qty = max(target_qty - current_qty, 0.0)

        if buy_qty <= 1e-9:
            logger.info(
                "Reactivation skipped for %s; current paper quantity %.8g already "
                "meets target %.8g.",
                symbol,
                current_qty,
                target_qty,
            )
            submitted_orders.append(
                {
                    "Symbol": symbol,
                    "Qty": format_plain_number(0),
                    "Order ID": "n/a",
                    "Status": "Already Active",
                }
            )
            continue

        order_qty = _order_quantity(buy_qty)
        order = client.submit_order(
            MarketOrderRequest(
                symbol=symbol,
                qty=order_qty,
                side=OrderSide.BUY,
                time_in_force=TimeInForce.DAY,
            )
        )
        logger.info(
            "Submitted reactivation BUY for %s | qty %s | order id %s | status %s",
            symbol,
            format_plain_number(buy_qty),
            order.id,
            order.status,
        )
        submitted_orders.append(
            {
                "Symbol": symbol,
                "Qty": format_plain_number(buy_qty),
                "Order ID": str(order.id),
                "Status": enum_text(order.status) or "submitted",
            }
        )

    if not submitted_orders:
        raise ValueError("No valid stopped position was available to reactivate.")

    return {
        "message": "Submitted strategy reactivation order.",
        "orders": submitted_orders,
        "source_stopped_at": str(stop_report.get("stopped_at")),
        "reactivated_at": pd.Timestamp.now(tz="UTC"),
    }
