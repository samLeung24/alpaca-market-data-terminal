from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from alpaca.data.timeframe import TimeFrameUnit

from src.company import get_company_name
from src.company_search import CompanyMatch, get_company_choices
from src.data_connector import get_historical_client
from src.historical import get_historical_bars
from src.live_quotes import get_live_quote_manager


st.set_page_config(page_title="Alpaca Market Data Terminal", layout="wide")


LIVE_QUOTE_REFRESH_SECONDS = 1.0
EASTERN_TZ = "America/New_York"


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


st.title("Mini Market Data Terminal v1.0")


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
            table_area.dataframe(display_df.tail(50), width="stretch")


with right:
    st.subheader("Live Quote")

    # Only this quote area refreshes automatically.
    # The chart/table/sidebar will not refresh every second anymore.
    render_live_quote(symbol, is_valid_symbol)
