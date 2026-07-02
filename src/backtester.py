from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from src.metrics import INITIAL_CAPITAL, calculate_drawdown
from src.strategies import (
    generate_custom_multifactor_signals,
    generate_macd_sma_trend_signals,
    generate_rsi_bollinger_mean_reversion_signals,
)


@dataclass(frozen=True)
class StrategySpec:
    name: str
    signal_function: Callable[..., pd.DataFrame]
    position_col: str
    trade_signal_col: str
    buy_signal_col: str
    sell_signal_col: str


@dataclass(frozen=True)
class StrategyResult:
    name: str
    signals: pd.DataFrame
    history: pd.DataFrame
    trades: pd.DataFrame


STRATEGY_SPECS = {
    "Trend Following": StrategySpec(
        name="Trend Following",
        signal_function=generate_macd_sma_trend_signals,
        position_col="trend_position",
        trade_signal_col="trend_trade_signal",
        buy_signal_col="trend_buy_signal",
        sell_signal_col="trend_sell_signal",
    ),
    "Mean Reversion": StrategySpec(
        name="Mean Reversion",
        signal_function=generate_rsi_bollinger_mean_reversion_signals,
        position_col="mean_reversion_position",
        trade_signal_col="mean_reversion_trade_signal",
        buy_signal_col="mean_reversion_buy_signal",
        sell_signal_col="mean_reversion_sell_signal",
    ),
    "Custom Multi-Factor": StrategySpec(
        name="Custom Multi-Factor",
        signal_function=generate_custom_multifactor_signals,
        position_col="custom_position",
        trade_signal_col="custom_trade_signal",
        buy_signal_col="custom_buy_signal",
        sell_signal_col="custom_sell_signal",
    ),
}


def build_ml_strategy_spec() -> StrategySpec:
    # src/models.py is a placeholder; import lazily so module import keeps working.
    try:
        from src.models import generate_ml_signals
    except ImportError as exc:
        raise ImportError(
            "src/models.py does not define generate_ml_signals yet."
        ) from exc

    return StrategySpec(
        name="ML Signal",
        signal_function=generate_ml_signals,
        position_col="ml_position",
        trade_signal_col="ml_trade_signal",
        buy_signal_col="ml_buy_signal",
        sell_signal_col="ml_sell_signal",
    )


def build_trade_log(
    history: pd.DataFrame,
    trade_signal_col: str = "trade_signal",
) -> pd.DataFrame:
    trades = []
    entry_time = None
    entry_price = None
    shares = None

    for row in history.itertuples(index=False):
        trade_signal = getattr(row, trade_signal_col)
        timestamp = getattr(row, "timestamp")
        close = float(getattr(row, "close"))
        portfolio_value = float(getattr(row, "portfolio_value"))

        if trade_signal == 1 and entry_price is None:
            entry_time = timestamp
            entry_price = close
            shares = portfolio_value / close if close else 0.0

        elif trade_signal == -1 and entry_price is not None:
            exit_time = timestamp
            exit_price = close
            pnl = (exit_price - entry_price) * float(shares or 0.0)
            trade_return = exit_price / entry_price - 1 if entry_price else 0.0
            holding_delta = pd.Timestamp(exit_time) - pd.Timestamp(entry_time)

            trades.append(
                {
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "shares": shares,
                    "pnl": pnl,
                    "return": trade_return,
                    "holding_days": holding_delta.days,
                }
            )

            entry_time = None
            entry_price = None
            shares = None

    return pd.DataFrame(
        trades,
        columns=[
            "entry_time",
            "entry_price",
            "exit_time",
            "exit_price",
            "shares",
            "pnl",
            "return",
            "holding_days",
        ],
    )


def run_backtest(
    signal_df: pd.DataFrame,
    strategy: StrategySpec,
    initial_capital: float = INITIAL_CAPITAL,
) -> StrategyResult:
    result = signal_df.copy()
    result["close"] = pd.to_numeric(result["close"], errors="coerce")
    result[strategy.position_col] = result[strategy.position_col].fillna(0).astype(int)
    result[strategy.trade_signal_col] = (
        result[strategy.trade_signal_col].fillna(0).astype(int)
    )

    result["asset_return"] = result["close"].pct_change().fillna(0)
    result["strategy_return"] = (
        result[strategy.position_col].shift(1).fillna(0) * result["asset_return"]
    )
    result["buy_hold_return"] = result["asset_return"]
    result["portfolio_value"] = initial_capital * (1 + result["strategy_return"]).cumprod()
    result["buy_hold_value"] = initial_capital * (1 + result["buy_hold_return"]).cumprod()
    result["drawdown"] = calculate_drawdown(result["portfolio_value"])
    result["position"] = result[strategy.position_col]
    result["trade_signal"] = result[strategy.trade_signal_col]
    result["buy_signal"] = result[strategy.buy_signal_col]
    result["sell_signal"] = result[strategy.sell_signal_col]

    history = result[
        [
            "timestamp",
            "close",
            "position",
            "trade_signal",
            "buy_signal",
            "sell_signal",
            "asset_return",
            "strategy_return",
            "buy_hold_return",
            "portfolio_value",
            "buy_hold_value",
            "drawdown",
        ]
    ]

    trades = build_trade_log(history)
    return StrategyResult(
        name=strategy.name,
        signals=result,
        history=history,
        trades=trades,
    )


def build_buy_hold_result(
    price_df: pd.DataFrame,
    initial_capital: float = INITIAL_CAPITAL,
) -> StrategyResult:
    history = price_df[["timestamp", "close"]].copy()
    history["asset_return"] = history["close"].pct_change().fillna(0)
    history["strategy_return"] = history["asset_return"]
    history["portfolio_value"] = initial_capital * (
        1 + history["strategy_return"]
    ).cumprod()
    history["drawdown"] = calculate_drawdown(history["portfolio_value"])
    history["position"] = 1
    history["trade_signal"] = 0
    history["buy_signal"] = False
    history["sell_signal"] = False

    return StrategyResult(
        name="Buy & Hold",
        signals=history.copy(),
        history=history,
        trades=pd.DataFrame(
            columns=[
                "entry_time",
                "entry_price",
                "exit_time",
                "exit_price",
                "shares",
                "pnl",
                "return",
                "holding_days",
            ]
        ),
    )
