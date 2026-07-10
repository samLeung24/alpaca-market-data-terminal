from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go
from alpaca.data.timeframe import TimeFrameUnit
from plotly.subplots import make_subplots


EASTERN_TZ = "America/New_York"
BULLISH_COLOR = "#1abc9c"
BEARISH_COLOR = "#e74c3c"
BULLISH_BAR_COLOR = "rgba(26, 188, 156, 0.45)"
BEARISH_BAR_COLOR = "rgba(231, 76, 60, 0.45)"
BENCHMARK_NAME = "Buy & Hold"
BENCHMARK_LINE = dict(color="#111111", width=2.4)
STRATEGY_LINE_COLORS = {
    "Trend Following": "#4F7CAC",
    "Mean Reversion": "#C9823B",
    "Custom Multi-Factor": "#7A6FA8",
    "ML Signal": "#1B7F5C",
    "ML Logistic Regression": "#1B7F5C",
}


def get_result_line_style(result_name: str) -> dict[str, Any] | None:
    if result_name.startswith(BENCHMARK_NAME):
        return BENCHMARK_LINE

    color = STRATEGY_LINE_COLORS.get(result_name)
    if color is None:
        return None

    return {"color": color, "width": 2}


def prepare_historical_display_df(
    df: pd.DataFrame,
    timeframe_unit: TimeFrameUnit,
) -> pd.DataFrame:
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


def plot_portfolio_values(
    results: list[Any],
    timeframe_unit: TimeFrameUnit,
) -> go.Figure:
    fig = go.Figure()

    for result in results:
        history = prepare_historical_display_df(result.history, timeframe_unit)
        line_style = get_result_line_style(result.name)
        fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["portfolio_value"],
                mode="lines",
                name=result.name,
                line=line_style,
            )
        )

    fig.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=45, b=20),
        title=dict(text="Portfolio Value", font=dict(size=24)),
        yaxis_title="Portfolio Value",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_drawdowns(
    results: list[Any],
    timeframe_unit: TimeFrameUnit,
) -> go.Figure:
    fig = go.Figure()

    for result in results:
        history = prepare_historical_display_df(result.history, timeframe_unit)
        line_style = get_result_line_style(result.name)
        fig.add_trace(
            go.Scatter(
                x=history["timestamp"],
                y=history["drawdown"],
                mode="lines",
                name=result.name,
                line=line_style,
            )
        )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=45, b=20),
        title=dict(text="Drawdown", font=dict(size=24)),
        yaxis_title="Drawdown",
        yaxis_tickformat=".0%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_pca_explained_variance(
    explained_variance_ratio: Any,
    threshold: float = 0.80,
) -> go.Figure:

    ratios = pd.Series(explained_variance_ratio, dtype=float).reset_index(drop=True)
    components = [f"PC{i + 1}" for i in range(len(ratios))]
    cumulative = ratios.cumsum()

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=components,
            y=ratios,
            name="Per component",
            marker_color="rgba(27, 127, 92, 0.55)",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=components,
            y=cumulative,
            mode="lines+markers",
            name="Cumulative",
            line=dict(color="#111111", width=2),
        )
    )

    fig.add_hline(
        y=threshold,
        line_dash="dash",
        line_color="rgba(220, 38, 38, 0.8)",
        annotation_text=f"{threshold:.0%} threshold",
        annotation_position="bottom right",
    )

    fig.update_layout(
        height=360,
        margin=dict(l=20, r=20, t=45, b=20),
        title=dict(text="PCA Explained Variance", font=dict(size=24)),
        yaxis_title="Explained Variance",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1.02],
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def _has_column(df: pd.DataFrame, column: str) -> bool:
    return column in df.columns and df[column].notna().any()


def _constant_line(
    display_df: pd.DataFrame,
    value: float,
    name: str,
    color: str = "rgba(107, 114, 128, 0.65)",
) -> go.Scatter:
    return go.Scatter(
        x=display_df["timestamp"],
        y=[value] * len(display_df),
        mode="lines",
        name=name,
        line=dict(color=color, width=1, dash="dash"),
        showlegend=False,
    )


def add_selected_indicator_traces(
    fig: go.Figure,
    display_df: pd.DataFrame,
    selected_indicators: list[str],
    row: int = 1,
    col: int = 1,
) -> None:
    if "Bollinger Bands" in selected_indicators and all(
        _has_column(display_df, column)
        for column in ["bb_upper_20", "bb_middle_20", "bb_lower_20"]
    ):
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["bb_upper_20"],
                mode="lines",
                name="BB Upper",
                line=dict(color="#64748b", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["bb_lower_20"],
                mode="lines",
                name="BB Lower",
                fill="tonexty",
                fillcolor="rgba(148, 163, 184, 0.18)",
                line=dict(color="#64748b", width=1),
                showlegend=False,
            ),
            row=row,
            col=col,
        )

        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["bb_middle_20"],
                mode="lines",
                name="BB Middle",
                line=dict(color="#475569", width=1, dash="dot"),
                legendgroup="bollinger",
            ),
            row=row,
            col=col,
        )

    indicator_columns = []
    if "SMA 50" in selected_indicators:
        indicator_columns.append(("sma_50", "SMA 50", "#2563eb"))
    if "SMA 200" in selected_indicators:
        indicator_columns.append(("sma_200", "SMA 200", "#7c3aed"))
    if "EMA 12" in selected_indicators:
        indicator_columns.append(("ema_12", "EMA 12", "#f59e0b"))
    if "EMA 26" in selected_indicators:
        indicator_columns.append(("ema_26", "EMA 26", "#0f766e"))
    if "EMA 20" in selected_indicators:
        indicator_columns.append(("ema_20", "EMA 20", "#dc2626"))

    for column, label, color in indicator_columns:
        if _has_column(display_df, column):
            fig.add_trace(
                go.Scatter(
                    x=display_df["timestamp"],
                    y=display_df[column],
                    mode="lines",
                    name=label,
                    line=dict(color=color, width=1.3),
                ),
                row=row,
                col=col,
            )


def _selected_lower_windows(
    display_df: pd.DataFrame,
    selected_indicators: list[str],
) -> list[str]:
    windows = []

    if "MACD" in selected_indicators and all(
        _has_column(display_df, column)
        for column in ["macd", "macd_signal", "macd_histogram"]
    ):
        windows.append("MACD")

    if "RSI 14" in selected_indicators and _has_column(display_df, "rsi_14"):
        windows.append("RSI 14")

    if "Momentum 10" in selected_indicators and _has_column(
        display_df,
        "momentum_10",
    ):
        windows.append("Momentum 10")

    if "Stochastic Oscillator" in selected_indicators and all(
        _has_column(display_df, column)
        for column in ["stochastic_k_14", "stochastic_d_3"]
    ):
        windows.append("Stochastic Oscillator")

    return windows


def _add_lower_indicator_window(
    fig: go.Figure,
    display_df: pd.DataFrame,
    indicator_name: str,
    row: int,
    bullish_bar_color: str = BULLISH_BAR_COLOR,
    bearish_bar_color: str = BEARISH_BAR_COLOR,
) -> None:
    if indicator_name == "MACD":
        histogram_colors = [
            bullish_bar_color if value >= 0 else bearish_bar_color
            for value in display_df["macd_histogram"].fillna(0)
        ]
        fig.add_trace(
            go.Bar(
                x=display_df["timestamp"],
                y=display_df["macd_histogram"],
                name="MACD Histogram",
                marker_color=histogram_colors,
                opacity=0.75,
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["macd"],
                mode="lines",
                name="MACD",
                line=dict(color="#7c3aed", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["macd_signal"],
                mode="lines",
                name="MACD Signal",
                line=dict(color="#f59e0b", width=1.2),
            ),
            row=row,
            col=1,
        )
        fig.update_yaxes(title_text="MACD", row=row, col=1)
        return

    if indicator_name == "RSI 14":
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["rsi_14"],
                mode="lines",
                name="RSI 14",
                line=dict(color="#7c3aed", width=1.2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(_constant_line(display_df, 70, "RSI 70"), row=row, col=1)
        fig.add_trace(_constant_line(display_df, 30, "RSI 30"), row=row, col=1)
        fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=1)
        return

    if indicator_name == "Momentum 10":
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["momentum_10"],
                mode="lines",
                name="Momentum 10",
                line=dict(color="#0891b2", width=1.2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(_constant_line(display_df, 0, "Momentum 0"), row=row, col=1)
        fig.update_yaxes(title_text="Momentum", row=row, col=1)
        return

    if indicator_name == "Stochastic Oscillator":
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["stochastic_k_14"],
                mode="lines",
                name="Stoch %K",
                line=dict(color="#7c3aed", width=1.2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["stochastic_d_3"],
                mode="lines",
                name="Stoch %D",
                line=dict(color="#f59e0b", width=1.2),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(_constant_line(display_df, 80, "Stoch 80"), row=row, col=1)
        fig.add_trace(_constant_line(display_df, 20, "Stoch 20"), row=row, col=1)
        fig.update_yaxes(title_text="Stoch", range=[0, 100], row=row, col=1)


def selected_lower_indicator_windows(
    display_df: pd.DataFrame,
    selected_indicators: list[str],
) -> list[str]:
    return _selected_lower_windows(display_df, selected_indicators)


def add_lower_indicator_window(
    fig: go.Figure,
    display_df: pd.DataFrame,
    indicator_name: str,
    row: int,
    bullish_bar_color: str = BULLISH_BAR_COLOR,
    bearish_bar_color: str = BEARISH_BAR_COLOR,
) -> None:
    _add_lower_indicator_window(
        fig,
        display_df,
        indicator_name,
        row,
        bullish_bar_color=bullish_bar_color,
        bearish_bar_color=bearish_bar_color,
    )


def plot_signal_chart(
    result: Any,
    selected_indicators: list[str],
    timeframe_unit: TimeFrameUnit,
) -> go.Figure:
    display_df = prepare_historical_display_df(result.signals, timeframe_unit)
    buy_points = display_df[display_df["buy_signal"].fillna(False).astype(bool)]
    sell_points = display_df[display_df["sell_signal"].fillna(False).astype(bool)]
    lower_windows = _selected_lower_windows(display_df, selected_indicators)

    rows = 2 + len(lower_windows)
    if lower_windows:
        row_heights = [0.54, 0.16, *([0.30 / len(lower_windows)] * len(lower_windows))]
    else:
        row_heights = [0.74, 0.26]

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.025,
    )

    has_ohlc = all(
        _has_column(display_df, column)
        for column in ["open", "high", "low", "close"]
    )

    if has_ohlc:
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
    else:
        fig.add_trace(
            go.Scatter(
                x=display_df["timestamp"],
                y=display_df["close"],
                mode="lines",
                name="Close",
                line=dict(color="#2563eb", width=2),
            ),
            row=1,
            col=1,
        )

    add_selected_indicator_traces(fig, display_df, selected_indicators, row=1, col=1)

    buy_y = buy_points["low"] if "low" in buy_points.columns else buy_points["close"]
    sell_y = sell_points["high"] if "high" in sell_points.columns else sell_points["close"]

    fig.add_trace(
        go.Scatter(
            x=buy_points["timestamp"],
            y=buy_y,
            mode="markers",
            name="Buy",
            marker=dict(symbol="triangle-up", size=11, color="#111111"),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=sell_points["timestamp"],
            y=sell_y,
            mode="markers",
            name="Sell",
            marker=dict(symbol="triangle-down", size=11, color="#111111"),
        ),
        row=1,
        col=1,
    )

    if _has_column(display_df, "volume"):
        if has_ohlc:
            volume_colors = [
                BULLISH_BAR_COLOR if close >= open_ else BEARISH_BAR_COLOR
                for open_, close in zip(display_df["open"], display_df["close"])
            ]
        else:
            volume_colors = "rgba(100, 116, 139, 0.45)"

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
        _add_lower_indicator_window(fig, display_df, indicator_name, offset)

    chart_height = 600 + 115 * len(lower_windows)
    fig.update_layout(
        height=chart_height,
        margin=dict(l=15, r=15, t=55, b=95),
        title=dict(text=result.name, font=dict(size=18), x=0.5, xanchor="center"),
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        bargap=0,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.13,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            itemsizing="constant",
        ),
    )

    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    fig.update_xaxes(title_text="Time (E.T.)", row=rows, col=1)

    return fig
