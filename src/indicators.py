from __future__ import annotations

import numpy as np
import pandas as pd


def _require_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def _get_numeric_price(df: pd.DataFrame, price_col: str) -> pd.Series:
    _require_columns(df, [price_col])
    return pd.to_numeric(df[price_col], errors="coerce")


def add_simple_moving_average(
    df: pd.DataFrame,
    window: int = 50,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    result[f"sma_{window}"] = price.rolling(window=window).mean()

    return result


def add_simple_moving_averages(
    df: pd.DataFrame,
    short_window: int = 50,
    long_window: int = 200,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()

    result = add_simple_moving_average(
        result,
        window=short_window,
        price_col=price_col,
    )

    result = add_simple_moving_average(
        result,
        window=long_window,
        price_col=price_col,
    )

    return result


def add_exponential_moving_average(
    df: pd.DataFrame,
    span: int = 12,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    result[f"ema_{span}"] = price.ewm(span=span, adjust=False).mean()

    return result


def add_exponential_moving_averages(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()

    result = add_exponential_moving_average(
        result,
        span=short_window,
        price_col=price_col,
    )

    result = add_exponential_moving_average(
        result,
        span=long_window,
        price_col=price_col,
    )

    return result


def add_macd(
    df: pd.DataFrame,
    short_window: int = 12,
    long_window: int = 26,
    signal_window: int = 9,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    result[f"ema_{short_window}"] = price.ewm(
        span=short_window,
        adjust=False,
    ).mean()

    result[f"ema_{long_window}"] = price.ewm(
        span=long_window,
        adjust=False,
    ).mean()

    result["macd"] = result[f"ema_{short_window}"] - result[f"ema_{long_window}"]

    result["macd_signal"] = result["macd"].ewm(
        span=signal_window,
        adjust=False,
    ).mean()

    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    return result


def add_relative_strength_index(
    df: pd.DataFrame,
    period: int = 14,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    delta = price.diff()

    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain, index=result.index).rolling(window=period).mean()
    avg_loss = pd.Series(loss, index=result.index).rolling(window=period).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    result[f"rsi_{period}"] = 100 - (100 / (1 + rs))

    result[f"rsi_{period}"] = result[f"rsi_{period}"].mask(
        (avg_loss == 0) & (avg_gain > 0),
        100,
    )

    result[f"rsi_{period}"] = result[f"rsi_{period}"].mask(
        (avg_gain == 0) & (avg_loss > 0),
        0,
    )

    result[f"rsi_{period}"] = result[f"rsi_{period}"].mask(
        (avg_gain == 0) & (avg_loss == 0),
        50,
    )

    return result


def add_bollinger_bands(
    df: pd.DataFrame,
    window: int = 20,
    number_of_std: float = 2.0,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    result[f"bb_middle_{window}"] = price.rolling(window=window).mean()
    result[f"bb_std_{window}"] = price.rolling(window=window).std()

    result[f"bb_upper_{window}"] = (
        result[f"bb_middle_{window}"] + result[f"bb_std_{window}"] * number_of_std
    )

    result[f"bb_lower_{window}"] = (
        result[f"bb_middle_{window}"] - result[f"bb_std_{window}"] * number_of_std
    )

    return result


def add_momentum(
    df: pd.DataFrame,
    period: int = 10,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    result[f"momentum_{period}"] = price.diff(period)

    return result


def add_stochastic_oscillator(
    df: pd.DataFrame,
    period: int = 14,
    signal_window: int = 3,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()
    price = _get_numeric_price(result, price_col)

    result[f"stochastic_low_{period}"] = price.rolling(period).min()
    result[f"stochastic_high_{period}"] = price.rolling(period).max()

    result[f"stochastic_k_{period}"] = (
        (price - result[f"stochastic_low_{period}"])
        / (result[f"stochastic_high_{period}"] - result[f"stochastic_low_{period}"])
        * 100
    )

    result[f"stochastic_d_{signal_window}"] = (
        result[f"stochastic_k_{period}"].rolling(signal_window).mean()
    )

    return result


def add_required_indicators(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    result = df.copy()

    result = add_simple_moving_averages(
        result,
        short_window=50,
        long_window=200,
        price_col=price_col,
    )

    result = add_exponential_moving_averages(
        result,
        short_window=12,
        long_window=26,
        price_col=price_col,
    )

    result = add_macd(
        result,
        short_window=12,
        long_window=26,
        signal_window=9,
        price_col=price_col,
    )

    result = add_relative_strength_index(
        result,
        period=14,
        price_col=price_col,
    )

    result = add_bollinger_bands(
        result,
        window=20,
        number_of_std=2,
        price_col=price_col,
    )

    result = add_momentum(
        result,
        period=10,
        price_col=price_col,
    )

    result = add_stochastic_oscillator(
        result,
        period=14,
        signal_window=3,
        price_col=price_col,
    )

    return result


def add_all_indicators(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    return add_required_indicators(df, price_col=price_col)


def add_selected_indicators(
    df: pd.DataFrame,
    selected_indicators: list[str],
    price_col: str = "close",
) -> pd.DataFrame:
    """Add the indicator columns needed by the selected chart overlays/windows."""
    result = add_required_indicators(df, price_col=price_col)

    if "EMA 20" in selected_indicators and "ema_20" not in result.columns:
        result = add_exponential_moving_average(
            result,
            span=20,
            price_col=price_col,
        )

    return result
