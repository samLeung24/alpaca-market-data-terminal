from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from src.config import AlpacaSettings, get_settings
from src.historical import fetch_daily_ohlcv


# No switch for live env. only paper trading
PAPER_ONLY = True

DEFAULT_ORDER_NOTIONAL = 10_000.0
DEFAULT_HISTORY_YEARS = 5
DEFAULT_PROBABILITY_THRESHOLD = 0.6

LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "paper_trading.log"

_ML_CONTRACT_HINT = (
    "src/features.py and src/models.py are placeholders and do not implement "
    "add_ml_features / generate_ml_signals yet."
)


@dataclass(frozen=True)
class ExecutionReport:
    """Outcome of one execute_latest_signal() run, for display and submission logs."""

    symbol: str
    bar_timestamp: pd.Timestamp
    close_price: float
    probability: float
    signal: str  # "LONG" or "FLAT"
    action: str  # "BUY", "SELL", "HOLD", or "NONE"
    order_id: str | None = None
    order_status: str | None = None
    order_qty: float | None = None
    market_open: bool | None = None
    log_lines: list[str] = field(default_factory=list)


def get_paper_trading_logger() -> logging.Logger:
    """Return a logger that writes signal/order events to console and logs/paper_trading.log."""
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


def get_trading_client(settings: AlpacaSettings | None = None) -> TradingClient:
    """Build an Alpaca trading client locked to the paper environment."""
    settings = settings or get_settings()
    return TradingClient(settings.api_key, settings.secret_key, paper=PAPER_ONLY)


def _import_ml_pipeline():
    # features/models are placeholder modules; import lazily so the app still loads.
    try:
        from src import features, models

        add_ml_features = features.add_ml_features
        generate_ml_signals = models.generate_ml_signals
    except (ImportError, AttributeError) as exc:
        raise ImportError(_ML_CONTRACT_HINT) from exc

    threshold = getattr(models, "PROBABILITY_THRESHOLD", DEFAULT_PROBABILITY_THRESHOLD)
    return add_ml_features, generate_ml_signals, threshold


def _get_open_position_qty(client: TradingClient, symbol: str) -> float:
    try:
        position = client.get_open_position(symbol)
    except APIError:
        return 0.0
    return float(position.qty)


def execute_latest_signal(
    symbol: str,
    notional: float = DEFAULT_ORDER_NOTIONAL,
    years: int = DEFAULT_HISTORY_YEARS,
    settings: AlpacaSettings | None = None,
    trading_client: TradingClient | None = None,
) -> ExecutionReport:
    """
    Run the full ML signal pipeline once and act on the latest signal.

    Fetch daily bars, compute features, apply PCA, generate the ML signal,
    then reconcile with the current paper position:
        LONG signal + no position  -> submit market BUY (paper only)
        LONG signal + position     -> HOLD, no order
        FLAT signal + position     -> close the position (market SELL)
        FLAT signal + no position  -> no order

    Long-only and unleveraged: the buy notional is capped at available cash.
    """
    logger = get_paper_trading_logger()
    log_lines: list[str] = []

    def log(message: str) -> None:
        logger.info(message)
        log_lines.append(message)

    add_ml_features, generate_ml_signals, threshold = _import_ml_pipeline()

    settings = settings or get_settings()
    client = trading_client or get_trading_client(settings)

    log(f"=== Paper trading run for {symbol} (paper={PAPER_ONLY}) ===")

    bars = fetch_daily_ohlcv(symbol, years=years)
    if bars.empty:
        raise ValueError(f"No daily bars returned for {symbol}.")

    feature_df = add_ml_features(bars)
    signals = generate_ml_signals(feature_df, price_col="close")

    latest = signals.iloc[-1]
    bar_timestamp = pd.Timestamp(latest["timestamp"])
    close_price = float(latest["close"])
    probability = float(latest["ml_probability"])
    is_long = int(latest["ml_position"]) == 1
    signal_label = "LONG" if is_long else "FLAT"

    log(
        f"Latest bar {bar_timestamp.date()} close={close_price:.2f} | "
        f"P(next-day up)={probability:.4f} vs threshold {threshold:.2f} "
        f"-> signal {signal_label}"
    )

    clock = client.get_clock()
    log(f"Market open: {clock.is_open} (next open {clock.next_open}, next close {clock.next_close})")

    held_qty = _get_open_position_qty(client, symbol)
    log(f"Current paper position in {symbol}: {held_qty:g} shares")

    action = "NONE"
    order = None
    order_qty: float | None = None

    if is_long and held_qty == 0:
        account = client.get_account()
        available_cash = float(account.cash)
        effective_notional = min(notional, available_cash)
        qty = math.floor(effective_notional / close_price)

        if qty < 1:
            log(
                f"LONG signal but cannot afford one share "
                f"(cash ${available_cash:,.2f}, close ${close_price:.2f}). No order."
            )
        else:
            action = "BUY"
            order_qty = float(qty)
            order = client.submit_order(
                MarketOrderRequest(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY,
                    time_in_force=TimeInForce.DAY,
                )
            )
            log(
                f"Submitted paper BUY {qty} {symbol} @ market "
                f"(~${qty * close_price:,.2f}) | order id {order.id} | status {order.status}"
            )

    elif is_long and held_qty > 0:
        action = "HOLD"
        log(f"LONG signal and already holding {held_qty:g} shares. No order.")

    elif not is_long and held_qty > 0:
        action = "SELL"
        order_qty = held_qty
        order = client.close_position(symbol)
        log(
            f"FLAT signal: closing {held_qty:g} {symbol} @ market | "
            f"order id {order.id} | status {order.status}"
        )

    else:
        log("FLAT signal and no open position. No order.")

    log(f"=== Run complete: signal={signal_label}, action={action} ===")

    return ExecutionReport(
        symbol=symbol,
        bar_timestamp=bar_timestamp,
        close_price=close_price,
        probability=probability,
        signal=signal_label,
        action=action,
        order_id=str(order.id) if order is not None else None,
        order_status=str(order.status.value if hasattr(order.status, "value") else order.status)
        if order is not None
        else None,
        order_qty=order_qty,
        market_open=bool(clock.is_open),
        log_lines=log_lines,
    )
