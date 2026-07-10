from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import get_settings
from src.data_connector import get_historical_client, resolve_data_feed


RANGE_PRESETS = {
    "1D": pd.DateOffset(days=1),
    "5D": pd.DateOffset(days=5),
    "1M": pd.DateOffset(months=1),
    "3M": pd.DateOffset(months=3),
    "6M": pd.DateOffset(months=6),
    "1Y": pd.DateOffset(years=1),
    "5Y": pd.DateOffset(years=5),
}

INDICATOR_WARMUP_BARS = {
    "SMA 50": 50,
    "SMA 200": 200,
    "EMA 12": 36,
    "EMA 20": 60,
    "EMA 26": 78,
    "MACD": 78,
    "RSI 14": 14,
    "Bollinger Bands": 20,
    "Momentum 10": 10,
    "Stochastic Oscillator": 14,
}


def resolve_date_range(
    selected_range: str,
    custom_days: int | None = None,
    end: pd.Timestamp | None = None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Map a range selector value to an explicit calendar start/end range."""
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


def get_indicator_warmup_bars(selected_indicators: list[str]) -> int:
    """Return the largest hidden-bar lookback needed by selected indicators."""
    return max(
        (INDICATOR_WARMUP_BARS.get(indicator, 0) for indicator in selected_indicators),
        default=0,
    )


def resolve_indicator_fetch_start(
    display_start: pd.Timestamp,
    timeframe_value: int,
    timeframe_unit: TimeFrameUnit,
    aggregate_factor: int,
    selected_indicators: list[str],
) -> pd.Timestamp:
    """Expand a chart fetch start so indicators have hidden warm-up bars."""
    warmup_bars = get_indicator_warmup_bars(selected_indicators)
    if warmup_bars <= 0:
        return display_start

    resolved_start = pd.Timestamp(display_start)
    if resolved_start.tzinfo is None:
        resolved_start = resolved_start.tz_localize("UTC")
    else:
        resolved_start = resolved_start.tz_convert("UTC")

    if timeframe_unit == TimeFrameUnit.Minute:
        minutes_per_bar = max(1, int(timeframe_value))
        bars_per_session = max(1, math.floor(390 / minutes_per_bar))
        sessions = math.ceil(warmup_bars / bars_per_session)
        calendar_days = math.ceil(sessions * 7 / 5) + 3
        return resolved_start - pd.DateOffset(days=calendar_days)

    if timeframe_unit == TimeFrameUnit.Hour:
        hours_per_bar = max(1, int(timeframe_value))
        bars_per_session = max(1, math.floor(6.5 / hours_per_bar))
        sessions = math.ceil(warmup_bars / bars_per_session)
        calendar_days = math.ceil(sessions * 7 / 5) + 3
        return resolved_start - pd.DateOffset(days=calendar_days)

    if timeframe_unit == TimeFrameUnit.Day:
        days_per_bar = max(1, int(aggregate_factor))
        trading_days = warmup_bars * days_per_bar
        calendar_days = math.ceil(trading_days * 7 / 5) + 10
        return resolved_start - pd.DateOffset(days=calendar_days)

    if timeframe_unit == TimeFrameUnit.Month:
        months_per_bar = max(1, int(timeframe_value))
        return resolved_start - pd.DateOffset(months=warmup_bars * months_per_bar)

    return resolved_start


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


def fetch_historical_chart_bars(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    timeframe_value: int,
    timeframe_unit: TimeFrameUnit,
    aggregate_factor: int = 1,
    client: StockHistoricalDataClient | None = None,
) -> pd.DataFrame:
    """Fetch bars for the terminal chart, applying multi-day aggregation if needed."""
    request_value = timeframe_value
    request_unit = timeframe_unit

    if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
        request_value = 1

    historical_client = client or get_historical_client()
    bars = get_historical_bars(
        client=historical_client,
        symbol=symbol,
        timeframe_value=request_value,
        timeframe_unit=request_unit,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
    )

    if timeframe_unit == TimeFrameUnit.Day and aggregate_factor > 1:
        return aggregate_bars_by_days(bars, aggregate_factor)

    return bars


def get_historical_bars(
    client: StockHistoricalDataClient,
    symbol: str,
    days: int = 30,
    timeframe_value: int = 5,
    timeframe_unit: TimeFrameUnit = TimeFrameUnit.Minute,
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    end = end or datetime.now(UTC)
    start = start or end - timedelta(days=days)
    settings = get_settings()

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(timeframe_value, timeframe_unit),
        start=start,
        end=end,
        feed=resolve_data_feed(settings.data_feed),
    )

    bars = client.get_stock_bars(request)
    df = bars.df
    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    return df.reset_index()


def fetch_daily_ohlcv(
    symbol: str,
    years: int = 5,
    client: StockHistoricalDataClient | None = None,
    feed_name: str | None = None,
) -> pd.DataFrame:
    """Fetch daily OHLCV bars for one ticker over the requested year window."""
    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25) + 14)

    settings = get_settings()
    historical_client = client or get_historical_client(settings)
    data_feed = resolve_data_feed(feed_name or settings.data_feed)

    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=start,
        end=end,
        feed=data_feed,
        limit=10_000,
    )

    bars = historical_client.get_stock_bars(request)
    df = bars.df
    if df.empty:
        return df

    if isinstance(df.index, pd.MultiIndex):
        df = df.xs(symbol, level="symbol")

    result = df.reset_index()
    if "timestamp" in result.columns:
        result = result.sort_values("timestamp")
    return result.reset_index(drop=True)
