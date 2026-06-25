from __future__ import annotations

import pandas as pd

from src.indicators import add_required_indicators


def generate_macd_sma_trend_signals(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    """
    Strategy 1: MACD + SMA200 Trend Filter.

    Buy when:
        MACD > MACD Signal
        Close > SMA200

    Sell when:
        MACD < MACD Signal
        Close < SMA200

    This is a long-only strategy:
        1 = long
        0 = cash
    """
    result = df.copy()

    required_columns = {
        price_col,
        "sma_200",
        "macd",
        "macd_signal",
    }

    if not required_columns.issubset(result.columns):
        result = add_required_indicators(result, price_col=price_col)

    entry_condition = (
        (result["macd"] > result["macd_signal"])
        & (result[price_col] > result["sma_200"])
    )

    exit_condition = (
        (result["macd"] < result["macd_signal"])
        | (result[price_col] < result["sma_200"])
    )

    current_position = 0
    positions = []
    trade_signals = []

    for should_enter, should_exit in zip(
        entry_condition.fillna(False),
        exit_condition.fillna(False),
    ):
        trade_signal = 0

        if current_position == 0 and should_enter:
            current_position = 1
            trade_signal = 1

        elif current_position == 1 and should_exit:
            current_position = 0
            trade_signal = -1

        positions.append(current_position)
        trade_signals.append(trade_signal)

    result["trend_entry_condition"] = entry_condition.fillna(False)
    result["trend_exit_condition"] = exit_condition.fillna(False)
    result["trend_position"] = positions
    result["trend_trade_signal"] = trade_signals
    result["trend_buy_signal"] = result["trend_trade_signal"] == 1
    result["trend_sell_signal"] = result["trend_trade_signal"] == -1

    return result


def generate_trend_following_signals(
    df: pd.DataFrame,
    price_col: str = "close",
) -> pd.DataFrame:
    return generate_macd_sma_trend_signals(df, price_col=price_col)