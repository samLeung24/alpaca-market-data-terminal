from __future__ import annotations

import html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.company import get_company_name
from src.company_search import CompanyMatch, get_company_choices
from src.execution import (
    account_details_dataframe,
    account_portfolio_value,
    build_account_cards,
    build_strategy_cancel_error_report,
    build_strategy_start_state,
    build_strategy_stop_error_report,
    fetch_current_strategy_state,
    fetch_paper_account_snapshot,
    fetch_portfolio_history_dataframe,
    ML_STRATEGY_DISPLAY_NAME,
    orders_to_dataframe,
    orders_to_exchange_log_dataframe,
    positions_to_dataframe,
    reactivate_stopped_strategy,
    read_paper_trading_log,
    resolve_strategy_display_state,
    stop_all_paper_positions,
)
from src.formatting import format_datetime
from src.historical import (
    RANGE_PRESETS,
    fetch_historical_chart_bars,
    resolve_date_range,
    resolve_tick_spec,
)
from src.indicators import add_selected_indicators
from src.live_quotes import get_live_quote_manager
from src.plots import (
    add_lower_indicator_window,
    add_selected_indicator_traces,
    prepare_historical_display_df,
    selected_lower_indicator_windows,
)


st.set_page_config(page_title="Alpaca Market Data Terminal", layout="wide")


LIVE_QUOTE_REFRESH_SECONDS = 1.0
PAPER_ACCOUNT_REFRESH_SECONDS = 10.0
PAPER_PORTFOLIO_TIMEFRAMES = ("1D", "1W", "1M", "3M", "YTD", "1Y", "ALL")
PAPER_PORTFOLIO_TIMEFRAME_STATE_KEY = "paper_portfolio_timeframe"
PAPER_PORTFOLIO_DEFAULT_TIMEFRAME = "1D"
TRANSACTION_LOG_VISIBLE_ROWS = 5
TRANSACTION_TABLE_ROW_HEIGHT = 32
TRANSACTION_TABLE_HEIGHT = (TRANSACTION_LOG_VISIBLE_ROWS + 1) * TRANSACTION_TABLE_ROW_HEIGHT + 6
LOCAL_LOG_HEIGHT = TRANSACTION_LOG_VISIBLE_ROWS * 22 + 24
BULLISH_COLOR = "#1abc9c"
BEARISH_COLOR = "#e74c3c"
BULLISH_BAR_COLOR = "rgba(26, 188, 156, 0.45)"
BEARISH_BAR_COLOR = "rgba(231, 76, 60, 0.45)"
INDICATOR_OPTIONS = [
    "SMA 50",
    "SMA 200",
    "EMA 12",
    "EMA 26",
    "EMA 20",
    "MACD",
    "RSI 14",
    "Bollinger Bands",
    "Momentum 10",
    "Stochastic Oscillator",
]
STRATEGY_STOP_REPORT_STATE_KEY = "strategy_stop_report"
STRATEGY_CANCEL_REPORT_STATE_KEY = "strategy_cancel_report"
STRATEGY_START_FORM_STATE_KEY = "strategy_start_form_open"
STRATEGY_ACTIVE_CONFIG_STATE_KEY = "strategy_active_config"
STRATEGY_START_REPORT_STATE_KEY = "strategy_start_report"
STRATEGY_START_OPTIONS = (
    "Trend-following",
    "Mean-reversion",
    "Multi-factor",
    ML_STRATEGY_DISPLAY_NAME,
)


def render_font_tokens() -> None:
    st.markdown(
        """
        <style>
        :root {
            --terminal-font-xs: 0.75rem;
            --terminal-font-sm: 0.85rem;
            --terminal-font-md: 1rem;
            --terminal-font-lg: 1.2rem;
            --terminal-font-xl: 1.5rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_search_header() -> None:
    st.sidebar.markdown(
        """
        <div style="
            display: flex;
            align-items: center;
            gap: 0.55rem;
            margin: 0.35rem 0 1rem;
            color: #31333f;
        ">
            <svg width="25" height="25" viewBox="0 0 24 24" fill="none"
                 xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                <circle cx="11" cy="11" r="7" stroke="currentColor"
                        stroke-width="2.25"/>
                <path d="M16.2 16.2L21 21" stroke="currentColor"
                      stroke-width="2.25" stroke-linecap="round"/>
            </svg>
            <span style="
                font-size: var(--terminal-font-xl);
                font-weight: 750;
                line-height: 1.15;
            ">Search</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


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
                <div style="font-size: var(--terminal-font-lg); font-weight: 700;">
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

    quote_items = (
        ("Bid", snapshot.bid_display),
        ("Ask", snapshot.ask_display),
        ("Last", snapshot.last_trade_display),
    )
    quote_html = "".join(
        '<div class="equity-quote-item">'
        f'<span class="equity-quote-label">{html.escape(str(label))}</span>'
        f'<span class="equity-quote-value">{html.escape(str(value))}</span>'
        "</div>"
        for label, value in quote_items
    )

    if snapshot.updated_at is None:
        updated_text = "Waiting for first streamed update."
    else:
        updated_text = f"Updated at: {snapshot.updated_at_display}"

    st.markdown(
        """
        <style>
        .equity-quote-row {
            display: flex;
            align-items: baseline;
            gap: 1.2rem;
            flex-wrap: wrap;
            margin: -0.2rem 0 0.8rem;
        }
        .equity-quote-item {
            display: inline-flex;
            align-items: baseline;
            gap: 0.32rem;
            min-width: 5.2rem;
        }
        .equity-quote-label {
            color: #6b7280;
            font-size: var(--terminal-font-sm);
            font-weight: 600;
        }
        .equity-quote-value {
            color: #31333f;
            font-size: var(--terminal-font-lg);
            font-weight: 650;
            line-height: 1.2;
        }
        .equity-quote-updated {
            color: #6b7280;
            font-size: var(--terminal-font-xs);
            margin: -0.35rem 0 0.9rem;
        }
        </style>
        """
        f'<div class="equity-quote-row">{quote_html}</div>'
        f'<div class="equity-quote-updated">{html.escape(str(updated_text))}</div>',
        unsafe_allow_html=True,
    )


def _paper_account_styles() -> None:
    st.markdown(
        """
        <style>
        .paper-account-shell {
            background: #ffffff;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 0.55rem;
            margin: 0.25rem 0 0.65rem;
        }
        .paper-card-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
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
            font-size: var(--terminal-font-sm);
            font-weight: 650;
            line-height: 1.2;
            margin-bottom: 0.22rem;
        }
        .paper-card-value {
            color: #111827;
            font-size: var(--terminal-font-md);
            font-weight: 750;
            line-height: 1.18;
            overflow-wrap: anywhere;
        }
        .paper-card-value.positive {
            color: #1abc9c;
        }
        .paper-card-value.negative {
            color: #e74c3c;
        }
        .paper-card-note {
            color: #6b7280;
            font-size: var(--terminal-font-xs);
            margin-top: 0.25rem;
            line-height: 1.25;
            overflow-wrap: anywhere;
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
    display_cards = [card for card in cards if card.get("label") != "Recent Order"]
    for card in display_cards:
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


def _render_strategy_status(
    strategy: str,
    equity: str,
    status: str,
) -> None:
    items = [
        ("Strategy", strategy, "strategy"),
        ("Equity", equity, "equity"),
        ("Status", status, "status"),
    ]
    item_html = "".join(
        '<div class="strategy-status-item">'
        f'<div class="strategy-status-label">{html.escape(label)}</div>'
        f'<div class="strategy-status-value {css_class}">{html.escape(value)}</div>'
        "</div>"
        for label, value, css_class in items
    )
    st.markdown(
        """
        <style>
        .strategy-status-grid {
            display: grid;
            grid-template-columns: minmax(0, 1.35fr) minmax(0, 0.75fr) minmax(0, 0.9fr);
            gap: 0.85rem;
            margin: 0.4rem 0 1rem;
        }
        .strategy-status-label {
            color: #31333f;
            font-size: var(--terminal-font-sm);
            font-weight: 600;
            line-height: 1.2;
            margin-bottom: 0.35rem;
        }
        .strategy-status-value {
            color: #31333f;
            font-size: var(--terminal-font-lg);
            font-weight: 500;
            line-height: 1.15;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .strategy-status-value.equity,
        .strategy-status-value.status {
            font-size: var(--terminal-font-lg);
        }
        </style>
        """
        f'<div class="strategy-status-grid">{item_html}</div>',
        unsafe_allow_html=True,
    )


def _strategy_start_styles() -> None:
    st.markdown(
        """
        <style>
        div[data-testid="stForm"] button[kind="primary"] {
            background: #1abc9c;
            border-color: #1abc9c;
            color: #ffffff;
        }
        div[data-testid="stForm"] button[kind="primary"]:hover {
            background: #159a80;
            border-color: #159a80;
            color: #ffffff;
        }
        div[data-testid="stForm"] button[kind="primary"]:focus {
            box-shadow: 0 0 0 0.2rem rgba(26, 188, 156, 0.25);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_start_trading_controls(
    start_equity_options: list[str],
    equity_by_label: dict[str, CompanyMatch],
) -> None:
    _strategy_start_styles()
    has_equity_options = len(start_equity_options) > 1

    with st.form("strategy_start_launcher_form"):
        start_form_requested = st.form_submit_button(
            "Start Trading",
            type="primary",
            width="stretch",
            disabled=not has_equity_options,
        )

    if not has_equity_options:
        st.caption("No equity universe is available for strategy startup.")

    if start_form_requested:
        st.session_state[STRATEGY_START_FORM_STATE_KEY] = True

    if not st.session_state.get(STRATEGY_START_FORM_STATE_KEY, False):
        return

    with st.form("strategy_start_form"):
        selected_equity_label = st.selectbox(
            "Equity",
            options=start_equity_options,
            key="strategy_start_equity_selection",
        )
        selected_strategy = st.selectbox(
            "Strategy",
            options=STRATEGY_START_OPTIONS,
            key="strategy_start_selection",
        )
        start_submitted = st.form_submit_button(
            "Start",
            type="primary",
            width="stretch",
        )
        st.warning(
            "Before proceeding, test this strategy in the Backtesting module "
            "for the selected equity and review its risk and performance."
        )

    if not start_submitted:
        return

    selected_equity = equity_by_label.get(selected_equity_label)
    if selected_equity is None:
        st.warning("Select an equity before starting a trading strategy.")
        return

    symbol = selected_equity.symbol
    active_config, start_report = build_strategy_start_state(selected_strategy, symbol)
    st.session_state[STRATEGY_ACTIVE_CONFIG_STATE_KEY] = active_config
    st.session_state[STRATEGY_START_REPORT_STATE_KEY] = start_report
    st.session_state[STRATEGY_START_FORM_STATE_KEY] = False
    st.session_state.pop(STRATEGY_STOP_REPORT_STATE_KEY, None)
    st.session_state.pop(STRATEGY_CANCEL_REPORT_STATE_KEY, None)
    st.rerun()


def _render_portfolio_history_chart(history_df: pd.DataFrame) -> None:
    if history_df.empty or len(history_df) < 2:
        st.info("No portfolio history returned by Alpaca for this time frame.")
        return

    first_equity = float(history_df["equity"].iloc[0])
    last_equity = float(history_df["equity"].iloc[-1])
    portfolio_change = last_equity - first_equity
    line_color = BULLISH_COLOR if portfolio_change >= 0 else BEARISH_COLOR
    min_equity = float(history_df["equity"].min())
    max_equity = float(history_df["equity"].max())
    padding = max((max_equity - min_equity) * 0.12, abs(last_equity) * 0.005, 1.0)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history_df["timestamp"],
            y=history_df["equity"],
            mode="lines",
            line={"color": line_color, "width": 2.4},
            hovertemplate="%{x|%b %d, %Y %H:%M}<br>%{y:$,.2f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=first_equity,
        line_dash="dot",
        line_color="#9ca3af",
        line_width=1.4,
        opacity=0.85,
    )
    fig.update_layout(
        height=290,
        margin={"l": 56, "r": 12, "t": 8, "b": 48},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        hovermode="x unified",
        font={"color": "#374151", "size": 11},
    )
    fig.update_xaxes(
        title_text="Time / Date",
        showgrid=True,
        gridcolor="#eef2f7",
        zeroline=False,
        showline=True,
        linecolor="#d1d5db",
        showticklabels=True,
        ticks="outside",
        tickfont={"size": 10},
        title_font={"size": 11},
    )
    fig.update_yaxes(
        title_text="Value",
        showgrid=True,
        gridcolor="#eef2f7",
        zeroline=False,
        showline=True,
        linecolor="#d1d5db",
        showticklabels=True,
        tickprefix="$",
        tickformat=",.0f",
        tickfont={"size": 10},
        title_font={"size": 11},
        range=[min_equity - padding, max_equity + padding],
    )

    st.plotly_chart(
        fig,
        width="stretch",
        config={"displayModeBar": False},
    )


def _render_portfolio_history_panel() -> None:
    if PAPER_PORTFOLIO_TIMEFRAME_STATE_KEY not in st.session_state:
        st.session_state[PAPER_PORTFOLIO_TIMEFRAME_STATE_KEY] = PAPER_PORTFOLIO_DEFAULT_TIMEFRAME

    selected_timeframe = st.session_state[PAPER_PORTFOLIO_TIMEFRAME_STATE_KEY]
    try:
        history_df = fetch_portfolio_history_dataframe(selected_timeframe)
    except Exception as exc:
        st.warning(f"Could not load Alpaca portfolio history: {exc}")
        history_df = pd.DataFrame()

    _render_portfolio_history_chart(history_df)
    st.segmented_control(
        "Portfolio chart time frame",
        options=PAPER_PORTFOLIO_TIMEFRAMES,
        key=PAPER_PORTFOLIO_TIMEFRAME_STATE_KEY,
        label_visibility="collapsed",
        width="stretch",
    )


@st.fragment(run_every=PAPER_ACCOUNT_REFRESH_SECONDS)
def render_paper_account_panel() -> None:
    _paper_account_styles()

    title_col, action_col = st.columns([0.9, 0.1], vertical_alignment="center")
    with title_col:
        st.subheader("Paper Account")
    with action_col:
        st.button(
            "⟳",
            key="paper_account_refresh",
            help="Refresh account",
            width="stretch",
        )

    try:
        snapshot = fetch_paper_account_snapshot()
    except Exception as exc:
        st.warning(f"Could not load Alpaca paper account: {exc}")
        return

    account = snapshot["account"]
    positions = snapshot["positions"]
    orders = snapshot["orders"]
    portfolio_value = account_portfolio_value(account)

    _render_metric_cards(
        build_account_cards(
            account=account,
            positions=positions,
            orders=orders,
        )
    )

    st.caption(
        "Account data refreshes every "
        f"{PAPER_ACCOUNT_REFRESH_SECONDS:.0f}s. Last refresh: "
        f"{format_datetime(snapshot['fetched_at'])}."
    )
    _render_portfolio_history_panel()

    tab_overview, tab_holdings, tab_transactions = st.tabs(
        ["Overview", "Holdings", "Transactions"]
    )

    with tab_overview:
        st.markdown("**Account Details**")
        st.dataframe(
            account_details_dataframe(account, positions, orders),
            hide_index=True,
            width="stretch",
        )

    with tab_holdings:
        holdings_df = positions_to_dataframe(positions, portfolio_value)
        if holdings_df.empty:
            st.info("No open paper positions.")
        else:
            st.dataframe(holdings_df, hide_index=True, width="stretch")

    with tab_transactions:
        order_history_df = orders_to_dataframe(orders)
        exchange_log_df = orders_to_exchange_log_dataframe(orders)

        st.markdown("**Order History**")
        if order_history_df.empty:
            st.info("No recent paper orders returned by Alpaca.")
        else:
            st.dataframe(
                order_history_df,
                hide_index=True,
                width="stretch",
                height=TRANSACTION_TABLE_HEIGHT,
                row_height=TRANSACTION_TABLE_ROW_HEIGHT,
            )

        st.markdown("**Alpaca Order Event Log**")
        if exchange_log_df.empty:
            st.info("No exchange order events returned by Alpaca.")
        else:
            st.dataframe(
                exchange_log_df,
                hide_index=True,
                width="stretch",
                height=TRANSACTION_TABLE_HEIGHT,
                row_height=TRANSACTION_TABLE_ROW_HEIGHT,
            )

        st.markdown("**Local Execution Log**")
        st.code(
            read_paper_trading_log(max_lines=160),
            language="text",
            height=LOCAL_LOG_HEIGHT,
        )


def render_strategy_management_panel(
    start_equity_options: list[str],
    equity_by_label: dict[str, CompanyMatch],
) -> None:
    st.subheader("Trading Strategy Management")

    try:
        strategy_state = fetch_current_strategy_state()
    except Exception as exc:
        strategy_state = {
            "equity": "n/a",
            "status": "Unavailable",
            "has_position": False,
            "error": str(exc),
        }

    active_config = st.session_state.get(STRATEGY_ACTIVE_CONFIG_STATE_KEY)
    report = st.session_state.get(STRATEGY_STOP_REPORT_STATE_KEY)
    display_state = resolve_strategy_display_state(
        strategy_state,
        active_config,
        report,
    )

    _render_strategy_status(
        display_state["display_strategy"],
        display_state["display_equity"],
        display_state["display_status"],
    )

    if strategy_state["has_position"]:
        stop_col, info_col = st.columns([0.22, 0.78], vertical_alignment="center")
        with stop_col:
            stop_clicked = st.button(
                "STOP",
                key=f"strategy_stop_all_{strategy_state['equity']}",
                type="primary",
                width="stretch",
            )
        with info_col:
            st.caption("STOP cancels open paper orders and submits closing orders for all open paper positions.")
    else:
        stop_clicked = False

    if strategy_state.get("error"):
        st.warning(f"Could not load current paper holding: {strategy_state['error']}")

    if (
        not strategy_state["has_position"]
        and not display_state["has_pending_stop_report"]
        and active_config is None
    ):
        _render_start_trading_controls(start_equity_options, equity_by_label)

    start_report = st.session_state.get(STRATEGY_START_REPORT_STATE_KEY)
    if display_state["started_without_position"] and start_report is not None:
        started_at = start_report.get("started_at")
        started_text = format_datetime(started_at) if started_at is not None else "n/a"
        st.success(f"{start_report.get('message', '')} Started at: {started_text}.")

    if stop_clicked:
        with st.spinner("Submitting close-all request to Alpaca paper trading..."):
            try:
                st.session_state[STRATEGY_STOP_REPORT_STATE_KEY] = stop_all_paper_positions()
                st.session_state.pop(STRATEGY_CANCEL_REPORT_STATE_KEY, None)
                st.session_state.pop(STRATEGY_START_REPORT_STATE_KEY, None)
            except Exception as exc:
                st.session_state[STRATEGY_STOP_REPORT_STATE_KEY] = (
                    build_strategy_stop_error_report(exc)
                )

    report = st.session_state.get(STRATEGY_STOP_REPORT_STATE_KEY)
    if report is None:
        return

    stopped_at = report.get("stopped_at")
    timestamp_text = format_datetime(stopped_at) if stopped_at is not None else "n/a"
    message = f"{report.get('message', '')} Last STOP action: {timestamp_text}."

    if report.get("error"):
        st.error(message)
    else:
        st.success(message)

    orders = report.get("orders") or []
    if orders:
        st.dataframe(pd.DataFrame(orders), hide_index=True, width="stretch")

    stopped_positions = report.get("stopped_positions") or []
    source_stopped_at = str(report.get("stopped_at"))
    cancel_report = st.session_state.get(STRATEGY_CANCEL_REPORT_STATE_KEY)
    cancel_matches_report = (
        cancel_report is not None
        and cancel_report.get("source_stopped_at") == source_stopped_at
    )

    if stopped_positions and not report.get("error") and not cancel_matches_report:
        cancel_col, cancel_info_col = st.columns([0.22, 0.78], vertical_alignment="center")
        with cancel_col:
            cancel_clicked = st.button(
                "Cancel STOP",
                key=f"strategy_cancel_stop_{source_stopped_at}",
                width="stretch",
            )
        with cancel_info_col:
            st.caption("Cancel reactivates the strategy by buying back the stopped position.")

        if cancel_clicked:
            with st.spinner("Reactivating strategy in Alpaca paper trading..."):
                try:
                    st.session_state[STRATEGY_CANCEL_REPORT_STATE_KEY] = reactivate_stopped_strategy(report)
                except Exception as exc:
                    st.session_state[STRATEGY_CANCEL_REPORT_STATE_KEY] = (
                        build_strategy_cancel_error_report(exc, source_stopped_at)
                    )
            cancel_report = st.session_state.get(STRATEGY_CANCEL_REPORT_STATE_KEY)
            cancel_matches_report = True

    if cancel_matches_report:
        reactivated_at = cancel_report.get("reactivated_at")
        reactivated_text = (
            format_datetime(reactivated_at)
            if reactivated_at is not None
            else "n/a"
        )
        cancel_message = (
            f"{cancel_report.get('message', '')} "
            f"Last reactivation action: {reactivated_text}."
        )
        if cancel_report.get("error"):
            st.error(cancel_message)
        else:
            st.success(cancel_message)

        cancel_orders = cancel_report.get("orders") or []
        if cancel_orders:
            st.dataframe(pd.DataFrame(cancel_orders), hide_index=True, width="stretch")


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

render_font_tokens()
render_sidebar_search_header()
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


selected_indicators = st.sidebar.multiselect(
    "Indicators",
    options=INDICATOR_OPTIONS,
    default=[],
)


range_start, range_end = resolve_date_range(time_range, custom_days)

try:
    timeframe_value, timeframe_unit, aggregate_factor = resolve_tick_spec(
        tick_choice,
        custom_tick,
    )
except ValueError as exc:
    st.error(str(exc))
    st.stop()


equity_panel, trading_panel = st.columns([1.35, 1], gap="large")


with equity_panel:
    if not is_valid_symbol:
        company_name = symbol or "Invalid symbol"
    elif selected_match is not None and selected_match.symbol == symbol:
        company_name = selected_match.name
    else:
        company_name = get_company_name(symbol)

    st.subheader(f"{company_name} ({symbol})")
    render_live_quote(symbol, is_valid_symbol)

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
                try:
                    bars = fetch_historical_chart_bars(
                        symbol=symbol,
                        start=range_start,
                        end=range_end,
                        timeframe_value=timeframe_value,
                        timeframe_unit=timeframe_unit,
                        aggregate_factor=aggregate_factor,
                    )
                except ValueError as exc:
                    chart_area.error(str(exc))
                    table_area.markdown("")
                    st.stop()

                st.session_state.historical_df = bars
                st.session_state.historical_key = requested_key

        df = st.session_state.historical_df

        if df.empty:
            chart_area.warning("No historical bars returned for this symbol.")
            table_area.markdown("")
        else:
            analysis_df = add_selected_indicators(df, selected_indicators)
            display_df = prepare_historical_display_df(analysis_df, timeframe_unit)
            lower_windows = selected_lower_indicator_windows(
                display_df,
                selected_indicators,
            )
            rows = 2 + len(lower_windows)
            if lower_windows:
                row_heights = [
                    0.58,
                    0.16,
                    *([0.26 / len(lower_windows)] * len(lower_windows)),
                ]
            else:
                row_heights = [0.74, 0.26]

            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                row_heights=row_heights,
                vertical_spacing=0.025,
            )

            fig.add_trace(
                go.Candlestick(
                    x=display_df["timestamp"],
                    open=display_df["open"],
                    high=display_df["high"],
                    low=display_df["low"],
                    close=display_df["close"],
                    name="Price",
                    increasing_line_color=BULLISH_COLOR,
                    increasing_fillcolor=BULLISH_COLOR,
                    decreasing_line_color=BEARISH_COLOR,
                    decreasing_fillcolor=BEARISH_COLOR,
                ),
                row=1,
                col=1,
            )

            add_selected_indicator_traces(
                fig,
                display_df,
                selected_indicators,
                row=1,
                col=1,
            )

            volume_colors = [
                BULLISH_BAR_COLOR
                if close >= open_
                else BEARISH_BAR_COLOR
                for open_, close in zip(display_df["open"], display_df["close"])
            ]
            fig.add_trace(
                go.Bar(
                    x=display_df["timestamp"],
                    y=display_df["volume"],
                    name="Volume",
                    marker_color=volume_colors,
                    opacity=0.85,
                    showlegend=False,
                ),
                row=2,
                col=1,
            )

            for offset, indicator_name in enumerate(lower_windows, start=3):
                add_lower_indicator_window(fig, display_df, indicator_name, offset)

            fig.update_layout(
                height=640 + 115 * len(lower_windows),
                margin={"l": 15, "r": 15, "t": 20, "b": 60},
                xaxis_rangeslider_visible=False,
                bargap=0,
                legend={
                    "orientation": "h",
                    "yanchor": "top",
                    "y": -0.12,
                    "xanchor": "center",
                    "x": 0.5,
                    "font": {"size": 10},
                    "itemsizing": "constant",
                },
            )

            fig.update_yaxes(title_text="Price", row=1, col=1)
            fig.update_yaxes(title_text="Volume", row=2, col=1)
            fig.update_xaxes(
                title_text="Time (E.T.)",
                row=rows,
                col=1,
            )

            # Fixed deprecation warning:
            # use_container_width=True -> width="stretch"
            chart_area.plotly_chart(fig, width="stretch")

            # Fixed deprecation warning:
            # use_container_width=True -> width="stretch"
            with table_area.expander("OHLCV Table", expanded=False):
                st.dataframe(display_df.tail(50), width="stretch")


with trading_panel:
    render_strategy_management_panel(equity_options, equity_by_label)
    st.divider()
    render_paper_account_panel()
