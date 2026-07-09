from __future__ import annotations

import html
from dataclasses import asdict
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from alpaca.common.enums import Sort
from plotly.subplots import make_subplots
from alpaca.data.timeframe import TimeFrameUnit
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from src.backtester import build_buy_hold_result, build_ml_strategy_spec, run_backtest
from src.company import get_company_name
from src.company_search import CompanyMatch, get_company_choices
from src.data_connector import get_historical_client, get_paper_trading_client
from src.execution import LOG_FILE, execute_latest_signal
from src.features import build_feature_pca_pipeline, transform_latest_features
from src.historical import fetch_daily_ohlcv, get_historical_bars
from src.live_quotes import get_live_quote_manager
from src.metrics import build_metrics_table
from src.models import PROBABILITY_THRESHOLD, run_ml_signal_pipeline, score_pca_features
from src.plots import (
    plot_drawdowns,
    plot_pca_explained_variance,
    plot_portfolio_values,
    plot_signal_chart,
)


st.set_page_config(page_title="Alpaca Market Data Terminal", layout="wide")


LIVE_QUOTE_REFRESH_SECONDS = 1.0
PAPER_ACCOUNT_REFRESH_SECONDS = 10.0
PAPER_ACCOUNT_ORDER_LIMIT = 50
EASTERN_TZ = "America/New_York"
ML_HISTORY_YEARS = 5
ML_PERIODS_PER_YEAR = 252
ML_SIGNAL_INDICATORS = ["MACD", "RSI 14", "Bollinger Bands"]
ML_MODEL_CACHE_STATE_KEY = "ml_model_cache"
ML_EXECUTION_REPORTS_STATE_KEY = "ml_last_execution_reports"
ML_LATEST_SIGNALS_STATE_KEY = "ml_latest_signal_frames"


RANGE_PRESETS = {
    "1D": pd.DateOffset(days=1),
    "5D": pd.DateOffset(days=5),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}


def resolve_date_range(
    selected_range: str,
    custom_days: int | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map a range button to an explicit calendar start/end range."""
    resolved_end = (
        pd.Timestamp(end)
        if end is not None
        else pd.Timestamp.now(tz="UTC").floor("min")
    )

    if resolved_end.tzinfo is None:
        resolved_end = resolved_end.tz_localize("UTC")
    else:
        resolved_end = resolved_end.tz_convert("UTC")

    if selected_range == "Custom":
        offset = pd.DateOffset(days=int(custom_days or 30))
    else:
        offset = RANGE_PRESETS[selected_range]

    return resolved_end - offset, resolved_end


def resolve_tick_spec(
    selected_tick: str,
    custom_tick: int | None = None,
) -> tuple[int, TimeFrameUnit, int]:
    """Map a tick selector value to request timeframe and optional aggregate factor."""
    if selected_tick == "Custom":
        custom_tick_minutes = int(custom_tick or 1)

        if custom_tick_minutes <= 59:
            return custom_tick_minutes, TimeFrameUnit.Minute, 1

        if custom_tick_minutes % 60 == 0:
            return custom_tick_minutes // 60, TimeFrameUnit.Hour, 1

        raise ValueError(
            "Custom tick must be 1-59 minutes or a whole-hour minute value "
            "(60, 120, 180, ...)."
        )

    if selected_tick.endswith("m"):
        return int(selected_tick[:-1]), TimeFrameUnit.Minute, 1

    if selected_tick in {"1D", "5D"}:
        aggregate = 5 if selected_tick == "5D" else 1
        return 1, TimeFrameUnit.Day, aggregate

    if selected_tick in {"1M", "3M"}:
        return int(selected_tick[:-1]), TimeFrameUnit.Month, 1

    if selected_tick == "1h":
        return 1, TimeFrameUnit.Hour, 1

    return 1, TimeFrameUnit.Minute, 1


def aggregate_bars_by_days(df: pd.DataFrame, days: int) -> pd.DataFrame:
    """Aggregate daily bars into multi-day OHLCV bars."""
    if days <= 1 or df.empty:
        return df

    resampled = (
        df.set_index("timestamp")
        .resample(f"{days}D", label="right")
        .agg(
            {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
            }
        )
    )

    return resampled.dropna(subset=["open", "high", "low", "close"]).reset_index()


def prepare_historical_display_df(
    df: pd.DataFrame,
    timeframe_unit: TimeFrameUnit,
) -> pd.DataFrame:
    """Return a display copy with chart timestamps shown in Eastern time."""
    if df.empty or "timestamp" not in df.columns:
        return df

    display_df = df.copy()
    timestamps = pd.to_datetime(display_df["timestamp"], utc=True)

    if timeframe_unit in {TimeFrameUnit.Minute, TimeFrameUnit.Hour}:
        display_df["timestamp"] = (
            timestamps.dt.tz_convert(EASTERN_TZ).dt.tz_localize(None)
        )
    else:
        display_df["timestamp"] = timestamps.dt.date

    return display_df


def render_invalid_symbol_message(target=st) -> None:
    """Show an empty-chart style message for invalid ticker input."""
    target.markdown(
        """
        <div style="
            height: 560px;
            display: flex;
            align-items: center;
            justify-content: center;
            text-align: center;
            border: 1px solid #e5e7eb;
            border-radius: 4px;
        ">
            <div>
                <div style="font-size: 1.2rem; font-weight: 700;">
                    This symbol doesn't exist
                </div>
                <div style="margin-top: 0.5rem; color: #6b7280;">
                    Try picking another one for your analysis, and you'll see the data here.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=LIVE_QUOTE_REFRESH_SECONDS)
def render_live_quote(symbol: str, is_valid_symbol: bool) -> None:
    """
    Refresh only the live quote area.

    Because this is a fragment, only this function reruns on the timer while
    the Alpaca websocket stream keeps receiving quote and trade events.
    """
    manager = get_live_quote_manager(st.session_state)

    if not is_valid_symbol:
        manager.stop()
        st.info("No live quote for invalid symbol.")
        return

    snapshot = manager.get_snapshot(symbol)
    if snapshot is None:
        st.error(f"Could not start live quote stream: {manager.error}")
        return

    st.metric("Bid", snapshot.bid_display)
    st.metric("Ask", snapshot.ask_display)
    st.metric("Last", snapshot.last_trade_display)
    if snapshot.updated_at is None:
        st.caption("Waiting for first streamed update.")
    else:
        st.caption(f"Updated at: {snapshot.updated_at_display}")


def _latest_ml_signal_row(signal_df: pd.DataFrame) -> pd.Series | None:
    if signal_df.empty or "ml_probability" not in signal_df.columns:
        return None

    ready = signal_df.dropna(subset=["ml_probability"])
    if ready.empty:
        return None

    return ready.iloc[-1]


def _get_ml_model_cache() -> dict:
    return st.session_state.setdefault(ML_MODEL_CACHE_STATE_KEY, {})


def _get_ml_execution_reports() -> dict:
    return st.session_state.setdefault(ML_EXECUTION_REPORTS_STATE_KEY, {})


def _get_ml_latest_signal_frames() -> dict:
    return st.session_state.setdefault(ML_LATEST_SIGNALS_STATE_KEY, {})


def _read_paper_trading_log(max_lines: int = 80) -> str:
    if not LOG_FILE.exists():
        return "No paper-trading log has been written yet."

    lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    recent_lines = lines[-max_lines:]
    return "\n".join(recent_lines) if recent_lines else "No paper-trading log entries yet."


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _first_field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = _field(obj, name, default=None)
        if value is not None:
            return value
    return default


def _enum_text(value: Any) -> str:
    if value is None:
        return ""
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def _to_float(value: Any) -> float | None:
    if value is None:
        return None

    raw_value = getattr(value, "value", value)
    if isinstance(raw_value, str):
        raw_value = raw_value.strip().replace("$", "").replace(",", "")
        if not raw_value or raw_value.lower() in {"none", "nan", "null"}:
            return None

    try:
        if pd.isna(raw_value):
            return None
    except (TypeError, ValueError):
        pass

    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return None


def _format_money(value: Any, currency: str = "USD") -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "n/a"

    sign = "-" if numeric < 0 else ""
    suffix = f" {currency}" if currency else ""
    return f"{sign}${abs(numeric):,.2f}{suffix}"


def _format_plain_number(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "n/a"

    if abs(numeric - round(numeric)) < 1e-9:
        return f"{numeric:,.0f}"

    return f"{numeric:,.4f}".rstrip("0").rstrip(".")


def _format_percent(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None:
        return "n/a"

    return f"{numeric:.2%}"


def _format_datetime(value: Any) -> str:
    if value is None:
        return "n/a"

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return str(value)

    if pd.isna(timestamp):
        return "n/a"

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")

    return timestamp.tz_convert(EASTERN_TZ).strftime("%Y-%m-%d %H:%M:%S E.T.")


def _sort_timestamp(value: Any) -> pd.Timestamp:
    if value is None:
        return pd.Timestamp.min.tz_localize("UTC")

    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.Timestamp.min.tz_localize("UTC")

    if pd.isna(timestamp):
        return pd.Timestamp.min.tz_localize("UTC")

    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")

    return timestamp.tz_convert("UTC")


def _money_class(value: Any) -> str:
    numeric = _to_float(value)
    if numeric is None or abs(numeric) < 1e-12:
        return "neutral"
    return "positive" if numeric > 0 else "negative"


def _normalize_records(records: Any) -> list[Any]:
    if records is None:
        return []
    if isinstance(records, dict):
        return list(records.values())
    return list(records)


def _fetch_paper_account_snapshot() -> dict[str, Any]:
    trading_client = get_paper_trading_client()
    order_request = GetOrdersRequest(
        status=QueryOrderStatus.ALL,
        limit=PAPER_ACCOUNT_ORDER_LIMIT,
        direction=Sort.DESC,
        nested=True,
    )

    return {
        "account": trading_client.get_account(),
        "positions": _normalize_records(trading_client.get_all_positions()),
        "orders": _normalize_records(trading_client.get_orders(order_request)),
        "fetched_at": pd.Timestamp.now(tz="UTC"),
    }


def _position_symbol(position: Any) -> str:
    return str(_field(position, "symbol", "") or "").upper()


def _selected_position(positions: list[Any], symbol: str) -> Any | None:
    selected_symbol = symbol.strip().upper()
    if not selected_symbol:
        return None

    for position in positions:
        if _position_symbol(position) == selected_symbol:
            return position

    return None


def _sum_position_field(positions: list[Any], field_name: str) -> float:
    total = 0.0
    for position in positions:
        total += _to_float(_field(position, field_name)) or 0.0
    return total


def _order_status(order: Any) -> str:
    return _enum_text(_field(order, "status")).lower()


def _order_side(order: Any) -> str:
    return _enum_text(_field(order, "side")).lower()


def _order_result_label(status: str) -> str:
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


def _calculate_recent_realized_pnl(orders: list[Any]) -> float:
    lots_by_symbol: dict[str, list[dict[str, float]]] = {}
    realized_pnl = 0.0

    sorted_orders = sorted(
        orders,
        key=lambda order: _sort_timestamp(
            _first_field(order, "filled_at", "submitted_at", "created_at")
        ),
    )

    for order in sorted_orders:
        status = _order_status(order)
        filled_qty = _to_float(_field(order, "filled_qty")) or 0.0
        fill_price = _to_float(_field(order, "filled_avg_price"))

        if filled_qty <= 0 or fill_price is None or status not in {"filled", "partially_filled"}:
            continue

        symbol = str(_field(order, "symbol", "") or "").upper()
        side = _order_side(order)
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


def _orders_to_dataframe(orders: list[Any]) -> pd.DataFrame:
    rows = []
    for order in orders:
        status = _order_status(order)
        side = _order_side(order)
        submitted_at = _first_field(order, "submitted_at", "created_at")
        filled_at = _field(order, "filled_at")

        rows.append(
            {
                "Submitted": _format_datetime(submitted_at),
                "Filled": _format_datetime(filled_at),
                "Symbol": str(_field(order, "symbol", "") or "").upper(),
                "Side": side.upper() if side else "n/a",
                "Result": _order_result_label(status),
                "Status": status.replace("_", " ").title() if status else "Unknown",
                "Type": _enum_text(_first_field(order, "type", "order_type")).replace("_", " ").title(),
                "Qty": _format_plain_number(_field(order, "qty")),
                "Filled Qty": _format_plain_number(_field(order, "filled_qty")),
                "Avg Fill": _format_money(_field(order, "filled_avg_price"), currency=""),
                "Limit": _format_money(_field(order, "limit_price"), currency=""),
                "Stop": _format_money(_field(order, "stop_price"), currency=""),
                "Order ID": str(_field(order, "id", "") or ""),
            }
        )

    return pd.DataFrame(rows)


def _orders_to_exchange_log_dataframe(orders: list[Any]) -> pd.DataFrame:
    rows = []
    for order in orders:
        order_id = str(_field(order, "id", "") or "")
        symbol = str(_field(order, "symbol", "") or "").upper()
        side = _order_side(order).upper() or "ORDER"
        qty = _format_plain_number(_field(order, "qty"))
        status = _order_status(order)
        order_type = _enum_text(_first_field(order, "type", "order_type")).replace("_", " ")
        submitted_at = _first_field(order, "submitted_at", "created_at")
        filled_at = _field(order, "filled_at")
        canceled_at = _first_field(order, "canceled_at", "cancelled_at")
        expired_at = _field(order, "expired_at")
        failed_at = _field(order, "failed_at")
        fill_price = _field(order, "filled_avg_price")

        if submitted_at is not None:
            rows.append(
                {
                    "Time": _format_datetime(submitted_at),
                    "Message": (
                        f"Submitted {side} {order_type} order for {qty} shares of "
                        f"{symbol}; order id {order_id}; status {status or 'unknown'}."
                    ),
                    "_sort": _sort_timestamp(submitted_at),
                }
            )

        if filled_at is not None:
            rows.append(
                {
                    "Time": _format_datetime(filled_at),
                    "Message": (
                        f"Filled order {order_id} for {symbol}; filled quantity "
                        f"{_format_plain_number(_field(order, 'filled_qty'))} at "
                        f"{_format_money(fill_price, currency='')}."
                    ),
                    "_sort": _sort_timestamp(filled_at),
                }
            )

        if canceled_at is not None:
            rows.append(
                {
                    "Time": _format_datetime(canceled_at),
                    "Message": f"Cancelled order {order_id} for {symbol}.",
                    "_sort": _sort_timestamp(canceled_at),
                }
            )

        if expired_at is not None:
            rows.append(
                {
                    "Time": _format_datetime(expired_at),
                    "Message": f"Expired order {order_id} for {symbol}.",
                    "_sort": _sort_timestamp(expired_at),
                }
            )

        if failed_at is not None:
            rows.append(
                {
                    "Time": _format_datetime(failed_at),
                    "Message": f"Failed order {order_id} for {symbol}; status {status or 'unknown'}.",
                    "_sort": _sort_timestamp(failed_at),
                }
            )

    if not rows:
        return pd.DataFrame(columns=["Time", "Message"])

    result = pd.DataFrame(rows).sort_values("_sort", ascending=False)
    return result.drop(columns=["_sort"]).reset_index(drop=True)


def _positions_to_dataframe(positions: list[Any], portfolio_value: float | None) -> pd.DataFrame:
    rows = []
    for position in positions:
        market_value = _to_float(_field(position, "market_value"))
        cost_basis = _to_float(_field(position, "cost_basis"))
        unrealized = _to_float(_field(position, "unrealized_pl"))
        allocation = (
            market_value / portfolio_value
            if market_value is not None and portfolio_value not in {None, 0}
            else None
        )

        rows.append(
            {
                "Symbol": _position_symbol(position),
                "Side": (_enum_text(_field(position, "side")) or "long").title(),
                "Allocation": _format_percent(allocation),
                "Qty": _format_plain_number(_field(position, "qty")),
                "Avg Price": _format_money(_field(position, "avg_entry_price"), currency=""),
                "Current Price": _format_money(_field(position, "current_price"), currency=""),
                "Market Value": _format_money(market_value),
                "Cost Basis": _format_money(cost_basis),
                "Unrealized P&L": _format_money(unrealized),
                "Unrealized %": _format_percent(_field(position, "unrealized_plpc")),
                "Daily P&L": _format_money(_field(position, "unrealized_intraday_pl")),
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


def _account_details_dataframe(account: Any, positions: list[Any], orders: list[Any]) -> pd.DataFrame:
    details = [
        ("Account status", _enum_text(_field(account, "status")) or "n/a"),
        ("Currency", _enum_text(_field(account, "currency")) or "USD"),
        ("Cash", _format_money(_field(account, "cash"))),
        ("Buying power", _format_money(_field(account, "buying_power"))),
        ("Portfolio value", _format_money(_first_field(account, "portfolio_value", "equity"))),
        ("Equity", _format_money(_field(account, "equity"))),
        ("Last equity", _format_money(_field(account, "last_equity"))),
        ("Long market value", _format_money(_field(account, "long_market_value"))),
        ("Maintenance margin", _format_money(_field(account, "maintenance_margin"))),
        ("Open positions", str(len(positions))),
        ("Recent orders loaded", str(len(orders))),
    ]
    return pd.DataFrame(details, columns=["Field", "Value"])


def _selected_position_dataframe(position: Any | None, symbol: str, is_valid_symbol: bool) -> pd.DataFrame:
    if not is_valid_symbol:
        return pd.DataFrame(
            [{"Field": "Selected equity", "Value": symbol or "No valid symbol selected"}]
        )

    if position is None:
        return pd.DataFrame(
            [
                {"Field": "Selected equity", "Value": symbol},
                {"Field": "Position", "Value": "Flat"},
                {"Field": "Qty", "Value": "0"},
                {"Field": "Market value", "Value": _format_money(0)},
            ]
        )

    details = [
        ("Selected equity", _position_symbol(position)),
        ("Position", (_enum_text(_field(position, "side")) or "long").title()),
        ("Qty", _format_plain_number(_field(position, "qty"))),
        ("Available qty", _format_plain_number(_field(position, "qty_available"))),
        ("Average entry", _format_money(_field(position, "avg_entry_price"), currency="")),
        ("Current price", _format_money(_field(position, "current_price"), currency="")),
        ("Market value", _format_money(_field(position, "market_value"))),
        ("Cost basis", _format_money(_field(position, "cost_basis"))),
        ("Unrealized P&L", _format_money(_field(position, "unrealized_pl"))),
        ("Unrealized %", _format_percent(_field(position, "unrealized_plpc"))),
        ("Daily P&L", _format_money(_field(position, "unrealized_intraday_pl"))),
    ]
    return pd.DataFrame(details, columns=["Field", "Value"])


def _paper_account_styles() -> None:
    st.markdown(
        """
        <style>
        .paper-account-heading {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 0.75rem;
            margin: 0.25rem 0 0.6rem;
        }
        .paper-account-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #111827;
        }
        .paper-account-subtitle {
            color: #6b7280;
            font-size: 0.82rem;
            margin-top: 0.15rem;
        }
        .paper-account-shell {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.55rem;
            margin: 0.25rem 0 0.65rem;
        }
        .paper-card-grid {
            display: grid;
            grid-template-columns: repeat(5, minmax(130px, 1fr));
            gap: 0.5rem;
        }
        .paper-card {
            background: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.55rem 0.65rem;
            min-height: 62px;
        }
        .paper-card-label {
            color: #6b7280;
            font-size: 0.72rem;
            font-weight: 650;
            line-height: 1.2;
            margin-bottom: 0.22rem;
        }
        .paper-card-value {
            color: #111827;
            font-size: 1rem;
            font-weight: 750;
            line-height: 1.18;
            overflow-wrap: anywhere;
        }
        .paper-card-value.positive {
            color: #059669;
        }
        .paper-card-value.negative {
            color: #dc2626;
        }
        .paper-card-note {
            color: #6b7280;
            font-size: 0.68rem;
            margin-top: 0.25rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
        }
        @media (max-width: 1200px) {
            .paper-card-grid {
                grid-template-columns: repeat(3, minmax(150px, 1fr));
            }
        }
        @media (max-width: 720px) {
            .paper-card-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_metric_cards(cards: list[dict[str, str]]) -> None:
    card_html = []
    for card in cards:
        value_class = html.escape(card.get("class", "neutral"))
        card_html.append(
            '<div class="paper-card">'
            f'<div class="paper-card-label">{html.escape(card["label"])}</div>'
            f'<div class="paper-card-value {value_class}">{html.escape(card["value"])}</div>'
            f'<div class="paper-card-note">{html.escape(card["note"])}</div>'
            "</div>"
        )

    st.markdown(
        '<div class="paper-account-shell">'
        '<div class="paper-card-grid">'
        f'{"".join(card_html)}'
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )


def _build_account_cards(
    account: Any,
    positions: list[Any],
    orders: list[Any],
    symbol: str,
    is_valid_symbol: bool,
) -> list[dict[str, str]]:
    portfolio_value = _to_float(_first_field(account, "portfolio_value", "equity"))
    buying_power = _to_float(_field(account, "buying_power"))
    cash = _to_float(_field(account, "cash"))
    unrealized_pnl = _sum_position_field(positions, "unrealized_pl")
    total_cost = _sum_position_field(positions, "cost_basis")
    unrealized_pct = unrealized_pnl / total_cost if total_cost else None
    realized_pnl = _calculate_recent_realized_pnl(orders)
    recent_order = orders[0] if orders else None

    if recent_order is None:
        recent_value = "No orders"
        recent_note = "No recent paper order history returned"
        recent_class = "neutral"
    else:
        status = _order_status(recent_order)
        recent_value = _order_result_label(status)
        recent_note = (
            f"{_order_side(recent_order).upper()} "
            f"{_format_plain_number(_field(recent_order, 'qty'))} "
            f"{str(_field(recent_order, 'symbol', '') or '').upper()}"
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
            "value": _format_money(portfolio_value),
            "note": f"Cash {_format_money(cash)}",
            "class": "neutral",
        },
        {
            "label": "Buying Power",
            "value": _format_money(buying_power),
            "note": "Available paper funds",
            "class": "neutral",
        },
        {
            "label": "Unrealized P&L",
            "value": _format_money(unrealized_pnl),
            "note": f"{_format_percent(unrealized_pct)} on open cost basis",
            "class": _money_class(unrealized_pnl),
        },
        {
            "label": "Realized P&L (recent)",
            "value": _format_money(realized_pnl),
            "note": "FIFO estimate from loaded fills",
            "class": _money_class(realized_pnl),
        },
        {
            "label": "Recent Order",
            "value": recent_value,
            "note": recent_note,
            "class": recent_class,
        },
    ]


@st.fragment(run_every=PAPER_ACCOUNT_REFRESH_SECONDS)
def render_paper_account_panel(symbol: str, is_valid_symbol: bool) -> None:
    _paper_account_styles()

    title_col, action_col = st.columns([0.82, 0.18], vertical_alignment="center")
    with title_col:
        st.markdown(
            """
            <div class="paper-account-heading">
                <div>
                    <div class="paper-account-title">Paper Account</div>
                    <div class="paper-account-subtitle">
                        Live Alpaca paper portfolio, holdings, orders, and execution journal.
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with action_col:
        st.button("Refresh Account", key="paper_account_refresh", width="stretch")

    try:
        snapshot = _fetch_paper_account_snapshot()
    except Exception as exc:
        st.warning(f"Could not load Alpaca paper account: {exc}")
        return

    account = snapshot["account"]
    positions = snapshot["positions"]
    orders = snapshot["orders"]
    portfolio_value = _to_float(_first_field(account, "portfolio_value", "equity"))

    _render_metric_cards(
        _build_account_cards(
            account=account,
            positions=positions,
            orders=orders,
            symbol=symbol,
            is_valid_symbol=is_valid_symbol,
        )
    )

    st.caption(
        "Account data refreshes every "
        f"{PAPER_ACCOUNT_REFRESH_SECONDS:.0f}s. Last refresh: "
        f"{_format_datetime(snapshot['fetched_at'])}."
    )

    tab_overview, tab_holdings, tab_transactions = st.tabs(
        ["Overview", "Holdings", "Transactions"]
    )

    with tab_overview:
        st.markdown("**Account Details**")
        st.dataframe(
            _account_details_dataframe(account, positions, orders),
            hide_index=True,
            width="stretch",
        )

    with tab_holdings:
        holdings_df = _positions_to_dataframe(positions, portfolio_value)
        if holdings_df.empty:
            st.info("No open paper positions.")
        else:
            st.dataframe(holdings_df, hide_index=True, width="stretch")

    with tab_transactions:
        order_history_df = _orders_to_dataframe(orders)
        exchange_log_df = _orders_to_exchange_log_dataframe(orders)

        st.markdown("**Order History**")
        if order_history_df.empty:
            st.info("No recent paper orders returned by Alpaca.")
        else:
            st.dataframe(order_history_df, hide_index=True, width="stretch")

        st.markdown("**Alpaca Order Event Log**")
        if exchange_log_df.empty:
            st.info("No exchange order events returned by Alpaca.")
        else:
            st.dataframe(exchange_log_df, hide_index=True, width="stretch")

        st.markdown("**Local Execution Log**")
        st.code(
            _read_paper_trading_log(max_lines=160),
            language="text",
        )


def _build_ml_results(
    symbol: str,
    years: int,
    probability_threshold: float,
    test_size: float,
) -> dict:
    price_df = fetch_daily_ohlcv(symbol, years=years)
    if price_df.empty:
        raise ValueError(f"No daily OHLCV bars returned for {symbol}.")

    pca_result = build_feature_pca_pipeline(
        price_df,
        price_col="close",
        test_size=test_size,
    )
    ml_signal_result = run_ml_signal_pipeline(
        pca_result,
        probability_threshold=probability_threshold,
        trade_on_test_only=True,
    )
    signal_df = ml_signal_result.signal_df

    test_signal_df = signal_df[signal_df["ml_sample_type"].eq("test")].copy()
    if len(test_signal_df) < 2:
        test_signal_df = signal_df.dropna(subset=["ml_probability"]).copy()

    if len(test_signal_df) < 2:
        raise ValueError("Not enough ML signal rows to run the backtest.")

    ml_spec = build_ml_strategy_spec()
    ml_result = run_backtest(test_signal_df, ml_spec)
    buy_hold_result = build_buy_hold_result(test_signal_df)
    results = [buy_hold_result, ml_result]

    return {
        "price_df": price_df,
        "pca_result": pca_result,
        "ml_signal_result": ml_signal_result,
        "signal_df": signal_df,
        "backtest_df": test_signal_df,
        "ml_result": ml_result,
        "buy_hold_result": buy_hold_result,
        "results": results,
        "metrics_table": build_metrics_table(
            results,
            periods_per_year=ML_PERIODS_PER_YEAR,
        ),
        "trained_at": pd.Timestamp.now(tz="UTC"),
    }


def _train_ml_panel_state(
    symbol: str,
    years: int,
    probability_threshold: float,
    test_size: float,
) -> dict:
    return {
        "symbol": symbol,
        "years": years,
        "probability_threshold": probability_threshold,
        "test_size": test_size,
        **_build_ml_results(
            symbol=symbol,
            years=years,
            probability_threshold=probability_threshold,
            test_size=test_size,
        ),
    }


def _build_fresh_latest_signal(
    symbol: str,
    panel_state: dict,
) -> pd.DataFrame:
    fresh_price_df = fetch_daily_ohlcv(symbol, years=int(panel_state["years"]))
    if fresh_price_df.empty:
        raise ValueError(f"No latest daily OHLCV bars returned for {symbol}.")

    latest_pca_df = transform_latest_features(
        fresh_price_df,
        panel_state["pca_result"],
        price_col="close",
    )
    ml_signal_result = panel_state["ml_signal_result"]
    return score_pca_features(
        pca_frame=latest_pca_df,
        component_columns=ml_signal_result.component_columns,
        model=ml_signal_result.model,
        probability_threshold=ml_signal_result.probability_threshold,
    )


def _execution_report_dict(report) -> dict:
    report_dict = asdict(report)
    report_dict["bar_timestamp"] = str(report_dict["bar_timestamp"])
    return report_dict


def render_ml_trading_panel(symbol: str, is_valid_symbol: bool) -> None:
    st.subheader("ML Trading Signal")

    if not is_valid_symbol:
        st.info("Choose a valid ticker before training the ML signal.")
        return

    model_cache = _get_ml_model_cache()
    panel_state = model_cache.get(symbol)

    cached_threshold = (
        float(panel_state.get("probability_threshold", PROBABILITY_THRESHOLD))
        if panel_state is not None
        else float(PROBABILITY_THRESHOLD)
    )
    cached_test_size = (
        float(panel_state.get("test_size", 0.20))
        if panel_state is not None
        else 0.20
    )

    control_cols = st.columns([1, 1, 1])
    with control_cols[0]:
        probability_threshold = st.slider(
            "Long probability threshold",
            min_value=0.50,
            max_value=0.90,
            value=cached_threshold,
            step=0.01,
            key=f"ml_threshold_{symbol}",
        )
    with control_cols[1]:
        test_size = st.slider(
            "Backtest holdout",
            min_value=0.10,
            max_value=0.50,
            value=cached_test_size,
            step=0.05,
            key=f"ml_holdout_{symbol}",
        )
    with control_cols[2]:
        order_notional = st.number_input(
            "Paper order notional",
            min_value=100.0,
            max_value=1_000_000.0,
            value=100_000.0,
            step=10_000.0,
            key=f"ml_order_notional_{symbol}",
        )

    years = ML_HISTORY_YEARS
    train_button_label = (
        "Train Model / Run Backtest"
        if panel_state is None
        else "Retrain Model / Run Backtest"
    )

    if panel_state is None:
        st.info("Train once for this equity. Later paper orders reuse the cached model and refresh only the latest signal.")
    elif (
        abs(float(probability_threshold) - cached_threshold) > 1e-9
        or abs(float(test_size) - cached_test_size) > 1e-9
    ):
        st.info("The changed model controls will apply after retraining.")

    if st.button(train_button_label, key=f"ml_retrain_{symbol}"):
        with st.spinner("Fetching 5 years of daily bars, fitting PCA, and training ML model..."):
            try:
                model_cache[symbol] = _train_ml_panel_state(
                    symbol=symbol,
                    years=years,
                    probability_threshold=probability_threshold,
                    test_size=test_size,
                )
                panel_state = model_cache[symbol]
                _get_ml_latest_signal_frames().pop(symbol, None)
                _get_ml_execution_reports().pop(symbol, None)
                st.success("Model trained for this equity.")
            except Exception as exc:
                st.error(f"Could not train ML signal: {exc}")
                return

    if panel_state is None:
        return

    trained_at = panel_state.get("trained_at")
    trained_at_display = str(trained_at) if trained_at is not None else "current session"
    st.caption(
        "Using cached model for "
        f"{symbol} trained at {trained_at_display}. "
        "Paper orders refresh latest data without retraining."
    )

    signal_df = panel_state["signal_df"]
    latest = _latest_ml_signal_row(signal_df)
    if latest is None:
        st.warning("The ML pipeline did not produce a latest signal row.")
        return

    summary_cols = st.columns(4)
    summary_cols[0].metric("Cached Signal", str(latest["ml_signal"]))
    summary_cols[1].metric("P(next day up)", f"{float(latest['ml_probability']):.2%}")
    summary_cols[2].metric("Training Rows", int(signal_df["ml_sample_type"].eq("train").sum()))
    summary_cols[3].metric("Backtest Rows", int(signal_df["ml_sample_type"].eq("test").sum()))

    tab_metrics, tab_charts, tab_trades, tab_paper = st.tabs(
        ["Metrics", "Charts", "Trades", "Paper Order"]
    )

    with tab_metrics:
        st.dataframe(panel_state["metrics_table"], width="stretch")
        st.dataframe(
            signal_df[
                [
                    "timestamp",
                    "close",
                    "ml_sample_type",
                    "ml_probability",
                    "ml_signal",
                    "ml_position",
                    "ml_trade_signal",
                ]
            ].tail(20),
            width="stretch",
        )

    with tab_charts:
        variance = signal_df.attrs.get("ml_pca_explained_variance_ratio", [])
        if variance:
            st.plotly_chart(
                plot_pca_explained_variance(variance, threshold=0.80),
                width="stretch",
            )
        st.plotly_chart(
            plot_signal_chart(
                panel_state["ml_result"],
                ML_SIGNAL_INDICATORS,
                TimeFrameUnit.Day,
            ),
            width="stretch",
        )
        st.plotly_chart(
            plot_portfolio_values(panel_state["results"], TimeFrameUnit.Day),
            width="stretch",
        )
        st.plotly_chart(
            plot_drawdowns(panel_state["results"], TimeFrameUnit.Day),
            width="stretch",
        )

    with tab_trades:
        trades = panel_state["ml_result"].trades
        if trades.empty:
            st.info("No closed ML trades in the current backtest window.")
        else:
            st.dataframe(trades, width="stretch")

    with tab_paper:
        st.caption("Orders submitted here use Alpaca paper trading credentials only.")
        if st.button("Refresh Signal and Submit Paper Order", key=f"ml_submit_{symbol}"):
            with st.spinner("Fetching latest bars, scoring the cached model, and submitting a paper order..."):
                try:
                    latest_signal_df = _build_fresh_latest_signal(symbol, panel_state)
                    _get_ml_latest_signal_frames()[symbol] = latest_signal_df
                    report = execute_latest_signal(
                        symbol,
                        signal_df=latest_signal_df,
                        notional=float(order_notional),
                    )
                    _get_ml_execution_reports()[symbol] = report
                except Exception as exc:
                    st.error(f"Could not submit paper-trading action: {exc}")

        latest_signal_df = _get_ml_latest_signal_frames().get(symbol)
        if latest_signal_df is not None:
            st.dataframe(
                latest_signal_df[
                    [
                        "timestamp",
                        "close",
                        "ml_probability",
                        "ml_signal",
                        "ml_position",
                        "ml_trade_signal",
                    ]
                ],
                width="stretch",
            )

        report = _get_ml_execution_reports().get(symbol)
        if report is not None:
            st.json(_execution_report_dict(report))
            if report.log_lines:
                st.markdown("Latest execution log")
                st.code("\n".join(report.log_lines), language="text")

        st.markdown("Recent paper-trading log")
        st.code(_read_paper_trading_log(), language="text")

st.title("Mini Market Terminal")


if "ticker_input" not in st.session_state:
    st.session_state.ticker_input = "HOOD"

if "selected_symbol" not in st.session_state:
    st.session_state.selected_symbol = st.session_state.ticker_input


equity_choices: list[CompanyMatch] = get_company_choices()
equity_by_label = {match.display: match for match in equity_choices}
equity_by_symbol = {match.symbol: match for match in equity_choices}

equity_placeholder = "Select or search an equity"
equity_options = [equity_placeholder, *equity_by_label.keys()]


def sync_from_equity() -> None:
    match = equity_by_label.get(st.session_state.equity_selection)

    if match is None:
        return

    st.session_state.selected_symbol = match.symbol
    st.session_state.ticker_input = match.symbol


def sync_from_ticker() -> None:
    symbol = st.session_state.ticker_input.strip().upper()

    if symbol:
        st.session_state.selected_symbol = symbol


current_symbol = st.session_state.selected_symbol.strip().upper()
current_match = equity_by_symbol.get(current_symbol)

st.session_state.ticker_input = current_symbol

if current_match is not None:
    st.session_state.equity_selection = current_match.display
else:
    st.session_state.equity_selection = equity_placeholder


st.sidebar.selectbox(
    "Stocks & ETFs",
    options=equity_options,
    key="equity_selection",
    on_change=sync_from_equity,
)


symbol_input = st.sidebar.text_input(
    "Ticker",
    key="ticker_input",
    on_change=sync_from_ticker,
)

symbol_input = symbol_input.strip().upper()

if symbol_input:
    st.session_state.selected_symbol = symbol_input

symbol = symbol_input
selected_match = equity_by_symbol.get(symbol)
is_valid_symbol = bool(symbol) and (not equity_by_symbol or symbol in equity_by_symbol)


time_range = st.sidebar.radio(
    "Time range",
    options=[*RANGE_PRESETS.keys(), "Custom"],
    index=0,
    horizontal=True,
)

if time_range == "Custom":
    custom_days = st.sidebar.slider(
        "Custom range (calendar days)",
        min_value=1,
        max_value=1827,
        value=30,
    )
else:
    custom_days = None


tick_choice = st.sidebar.radio(
    "Tick size",
    options=["1m", "5m", "15m", "30m", "1h", "1D", "5D", "1M", "3M", "Custom"],
    index=1,
    horizontal=True,
)

if tick_choice == "Custom":
    custom_tick = st.sidebar.slider(
        "Custom tick size (minutes)",
        min_value=1,
        max_value=240,
        value=5,
    )
else:
    custom_tick = None


range_start, range_end = resolve_date_range(time_range, custom_days)

try:
    timeframe_value, timeframe_unit, aggregate_factor = resolve_tick_spec(
        tick_choice,
        custom_tick,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()


try:
    client = get_historical_client()
except ValueError as exc:
    st.error(str(exc))
    st.stop()


render_paper_account_panel(symbol, is_valid_symbol)

left, right = st.columns([2, 1])


with left:
    if not is_valid_symbol:
        company_name = symbol or "Invalid symbol"
    elif selected_match is not None and selected_match.symbol == symbol:
        company_name = selected_match.name
    else:
        company_name = get_company_name(symbol)

    st.subheader(f"{company_name} ({symbol})")

    chart_area = st.empty()
    table_area = st.empty()

    if not is_valid_symbol:
        render_invalid_symbol_message(chart_area)
        table_area.markdown("")
    else:
        requested_key = (
            f"{symbol}|{range_start.isoformat()}|{range_end.isoformat()}|"
            f"{timeframe_value}|"
            f"{timeframe_unit.value}|{aggregate_factor}"
        )

        has_data = (
            "historical_df" in st.session_state
            and st.session_state.get("historical_key") == requested_key
        )

        if not has_data:
            with st.spinner("Loading historical bars..."):
                request_value = timeframe_value
                request_unit = timeframe_unit

                if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
                    request_value = 1

                bars = get_historical_bars(
                    client=client,
                    symbol=symbol,
                    timeframe_value=request_value,
                    timeframe_unit=request_unit,
                    start=range_start.to_pydatetime(),
                    end=range_end.to_pydatetime(),
                )

                if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
                    bars = aggregate_bars_by_days(bars, aggregate_factor)

                st.session_state.historical_df = bars
                st.session_state.historical_key = requested_key

        df = st.session_state.historical_df

        if df.empty:
            chart_area.warning("No historical bars returned for this symbol.")
            table_area.markdown("")
        else:
            display_df = prepare_historical_display_df(df, timeframe_unit)

            fig = make_subplots(
                rows=2,
                cols=1,
                shared_xaxes=True,
                row_heights=[0.72, 0.28],
                vertical_spacing=0.05,
            )

            fig.add_trace(
                go.Candlestick(
                    x=display_df["timestamp"],
                    open=display_df["open"],
                    high=display_df["high"],
                    low=display_df["low"],
                    close=display_df["close"],
                    name="Price",
                ),
                row=1,
                col=1,
            )

            fig.add_trace(
                go.Bar(
                    x=display_df["timestamp"],
                    y=display_df["volume"],
                    name="Volume",
                ),
                row=2,
                col=1,
            )

            fig.update_layout(
                height=640,
                xaxis_rangeslider_visible=False,
            )

            fig.update_xaxes(
                title_text="Time (E.T.)",
                row=2,
                col=1,
            )

            # Fixed deprecation warning:
            # use_container_width=True -> width="stretch"
            chart_area.plotly_chart(fig, width="stretch")

            # Fixed deprecation warning:
            # use_container_width=True -> width="stretch"
            with table_area.expander("OHLCV Data", expanded=False):
                st.dataframe(display_df.tail(50), width="stretch")


with right:
    st.subheader("Live Quote")

    # Only this quote area refreshes automatically.
    # The chart/table/sidebar will not refresh every second anymore.
    render_live_quote(symbol, is_valid_symbol)


st.divider()
with st.expander("ML Trading Signal", expanded=False):
    render_ml_trading_panel(symbol, is_valid_symbol)
