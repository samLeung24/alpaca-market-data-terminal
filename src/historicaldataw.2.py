from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from src.config import get_settings
from src.data_connector import get_historical_client, resolve_data_feed

REQUIRED_DAILY_COLUMNS = ["symbol", "timestamp", "open", "high", "low", "close", "volume"]
OPTIONAL_DAILY_COLUMNS = ["trade_count", "vwap"]
DEFAULT_TICKERS = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]


def normalize_symbol(symbol: str) -> str:
    """Return a clean uppercase ticker symbol from user input."""
    normalized_symbol = str(symbol).strip().upper()

    if not normalized_symbol:
        raise ValueError("Ticker symbol cannot be blank.")

    return normalized_symbol


def _to_utc_datetime(value: str | date | datetime | None) -> datetime | None:
    """Convert date-like user input to a timezone-aware UTC datetime."""
    if value is None:
        return None

    timestamp = pd.Timestamp(value)

    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(UTC)
    else:
        timestamp = timestamp.tz_convert(UTC)

    return timestamp.to_pydatetime()


def get_default_date_range(years: int = 5, extra_days: int = 14) -> tuple[datetime, datetime]:
    """
    Return the default assignment date range.

    The assignment asks for at least five years of daily data. A small buffer is
    added so leap years, weekends, and market holidays do not accidentally make
    the request slightly shorter than five full calendar years.
    """
    if years < 1:
        raise ValueError("years must be at least 1.")

    end = datetime.now(UTC)
    start = end - timedelta(days=int(years * 365.25) + extra_days)

    return start, end


def _empty_daily_frame() -> pd.DataFrame:
    """Return an empty DataFrame with the columns expected by the assignment."""
    return pd.DataFrame(columns=REQUIRED_DAILY_COLUMNS + OPTIONAL_DAILY_COLUMNS)


def _require_columns(df: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _flatten_alpaca_bars(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Convert Alpaca's returned bar DataFrame into a normal single-ticker DataFrame.

    alpaca-py usually returns bars with a MultiIndex of (symbol, timestamp).
    This function keeps the user's selected ticker and makes timestamp a normal
    column so the indicator and backtesting modules can use it easily.
    """
    if df is None or df.empty:
        return _empty_daily_frame()

    result = df.copy()

    if isinstance(result.index, pd.MultiIndex):
        symbol_level = "symbol" if "symbol" in result.index.names else 0

        try:
            result = result.xs(symbol, level=symbol_level, drop_level=True)
        except KeyError:
            return _empty_daily_frame()

    result = result.reset_index()
    result.columns = [str(column).lower() for column in result.columns]

    if "timestamp" not in result.columns and "index" in result.columns:
        result = result.rename(columns={"index": "timestamp"})

    result["symbol"] = symbol

    return result


def clean_daily_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Alpaca daily bars for this assignment.

    Output format:
        symbol, timestamp, open, high, low, close, volume, trade_count, vwap

    The returned DataFrame is sorted oldest to newest and is ready for
    src.indicators.add_required_indicators().
    """
    if df is None or df.empty:
        return _empty_daily_frame()

    result = df.copy()
    result.columns = [str(column).lower() for column in result.columns]

    _require_columns(result, REQUIRED_DAILY_COLUMNS)

    result["symbol"] = result["symbol"].astype(str).str.upper().str.strip()
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True, errors="coerce")

    for column in ["open", "high", "low", "close", "volume"]:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    for column in OPTIONAL_DAILY_COLUMNS:
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")

    result = result.dropna(subset=["symbol", "timestamp", "open", "high", "low", "close", "volume"])
    result = result.sort_values("timestamp")
    result = result.drop_duplicates(subset=["symbol", "timestamp"], keep="last")
    result = result.reset_index(drop=True)

    ordered_columns = REQUIRED_DAILY_COLUMNS + [
        column for column in OPTIONAL_DAILY_COLUMNS if column in result.columns
    ]

    return result[ordered_columns]


def validate_daily_history(
    df: pd.DataFrame,
    years: int = 5,
    min_trading_days_per_year: int = 200,
) -> None:
    """
    Raise a helpful error if the returned data is too short for the assignment.

    A normal year has roughly 252 U.S. trading days. The default threshold of 200
    days per year is intentionally conservative so holidays, recent IPOs, or
    partial current-year data do not create false failures.
    """
    if df is None or df.empty:
        raise ValueError("No historical data was returned by Alpaca.")

    minimum_rows = years * min_trading_days_per_year

    if len(df) < minimum_rows:
        symbol = df["symbol"].iloc[0] if "symbol" in df.columns and not df.empty else "the selected ticker"
        raise ValueError(
            f"Only {len(df):,} daily bars were returned for {symbol}. "
            f"The assignment needs about {minimum_rows:,}+ daily bars for {years} years. "
            "Try a large, liquid ticker such as AAPL, MSFT, SPY, QQQ, or NVDA, "
            "or check your Alpaca data feed setting."
        )


def fetch_daily_ohlcv(
    symbol: str,
    years: int = 5,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    client: StockHistoricalDataClient | None = None,
    feed_name: str | None = None,
    validate_history: bool = True,
) -> pd.DataFrame:
    """
    Fetch daily OHLCV bars from Alpaca for one user-selected ticker.

    This is the main data function for the assignment.

    Parameters
    ----------
    symbol:
        User-selected ticker, such as AAPL, MSFT, SPY, QQQ, or NVDA.
    years:
        Number of calendar years to request when start is not provided.
    start, end:
        Optional custom date range. If start is omitted, the function requests
        at least `years` years of daily data ending today.
    client:
        Optional Alpaca historical client. Leave as None to use the existing
        project credentials from src.config and src.data_connector.
    feed_name:
        Optional Alpaca data feed override, for example "iex" or "sip".
    validate_history:
        If True, checks that the returned dataset is long enough for the assignment.

    Returns
    -------
    pd.DataFrame
        Clean daily bars with symbol, timestamp, open, high, low, close, volume,
        and optional Alpaca fields such as trade_count and vwap.
    """
    normalized_symbol = normalize_symbol(symbol)

    default_start, default_end = get_default_date_range(years=years)
    request_start = _to_utc_datetime(start) or default_start
    request_end = _to_utc_datetime(end) or default_end

    if request_start >= request_end:
        raise ValueError("start must be earlier than end.")

    settings = get_settings() if client is None or feed_name is None else None
    historical_client = client or get_historical_client(settings)
    data_feed = resolve_data_feed(feed_name or settings.data_feed if settings else feed_name or "iex")

    request = StockBarsRequest(
        symbol_or_symbols=normalized_symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Day),
        start=request_start,
        end=request_end,
        feed=data_feed,
        limit=10_000,
    )

    bars = historical_client.get_stock_bars(request)
    raw_df = _flatten_alpaca_bars(bars.df, normalized_symbol)
    clean_df = clean_daily_ohlcv(raw_df)

    if validate_history:
        validate_daily_history(clean_df, years=years)

    return clean_df


def fetch_multiple_daily_ohlcv(
    symbols: Iterable[str],
    years: int = 5,
    start: str | date | datetime | None = None,
    end: str | date | datetime | None = None,
    feed_name: str | None = None,
    validate_history: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Fetch daily OHLCV data for several tickers.

    This keeps each ticker in its own DataFrame so each teammate can pass one
    selected ticker into the indicator, strategy, and backtesting modules.
    """
    settings = get_settings()
    client = get_historical_client(settings)

    results: dict[str, pd.DataFrame] = {}

    for symbol in symbols:
        normalized_symbol = normalize_symbol(symbol)
        results[normalized_symbol] = fetch_daily_ohlcv(
            symbol=normalized_symbol,
            years=years,
            start=start,
            end=end,
            client=client,
            feed_name=feed_name or settings.data_feed,
            validate_history=validate_history,
        )

    return results


def save_daily_ohlcv_to_csv(
    df: pd.DataFrame,
    output_dir: str | Path = "data",
) -> Path:
    """Save one ticker's cleaned assignment data to a CSV file."""
    if df is None or df.empty:
        raise ValueError("Cannot save an empty DataFrame.")

    _require_columns(df, ["symbol", "timestamp"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    symbol = str(df["symbol"].iloc[0]).upper()
    start_date = pd.to_datetime(df["timestamp"].min()).date().isoformat()
    end_date = pd.to_datetime(df["timestamp"].max()).date().isoformat()

    file_path = output_path / f"{symbol}_daily_ohlcv_{start_date}_to_{end_date}.csv"
    df.to_csv(file_path, index=False)

    return file_path


def load_assignment_data(
    symbol: str,
    years: int = 5,
    save_csv: bool = False,
    output_dir: str | Path = "data",
) -> pd.DataFrame:
    """
    Convenience wrapper for the assignment workflow.

    It fetches the required five-year daily OHLCV DataFrame and optionally saves
    a CSV copy for reproducibility.
    """
    df = fetch_daily_ohlcv(symbol=symbol, years=years)

    if save_csv:
        save_daily_ohlcv_to_csv(df, output_dir=output_dir)

    return df