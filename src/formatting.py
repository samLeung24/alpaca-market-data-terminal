from __future__ import annotations

from typing import Any

import pandas as pd


EASTERN_TZ = "America/New_York"


def field(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def first_field(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        value = field(obj, name, default=None)
        if value is not None:
            return value
    return default


def enum_text(value: Any) -> str:
    if value is None:
        return ""
    raw_value = getattr(value, "value", value)
    return str(raw_value)


def to_float(value: Any) -> float | None:
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


def format_money(value: Any, currency: str = "USD") -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"

    sign = "-" if numeric < 0 else ""
    suffix = f" {currency}" if currency else ""
    return f"{sign}${abs(numeric):,.2f}{suffix}"


def format_plain_number(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"

    if abs(numeric - round(numeric)) < 1e-9:
        return f"{numeric:,.0f}"

    return f"{numeric:,.4f}".rstrip("0").rstrip(".")


def format_percent(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None:
        return "n/a"

    return f"{numeric:.2%}"


def format_datetime(value: Any) -> str:
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


def sort_timestamp(value: Any) -> pd.Timestamp:
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


def money_class(value: Any) -> str:
    numeric = to_float(value)
    if numeric is None or abs(numeric) < 1e-12:
        return "neutral"
    return "positive" if numeric > 0 else "negative"


def normalize_records(records: Any) -> list[Any]:
    if records is None:
        return []
    if isinstance(records, dict):
        return list(records.values())
    return list(records)
