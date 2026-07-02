from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


DEFAULT_FEATURE_COLUMNS: list[str] = [
    # Returns and rolling statistics required by the assignment
    "log_return",
    "rolling_mean_10",
    "rolling_std_10",
    "rolling_mean_20",
    "rolling_std_20",
    # Trend indicators
    "sma_10",
    "sma_20",
    "sma_50",
    "ema_12",
    "ema_26",
    "ema_20",
    "macd",
    "macd_signal",
    "macd_histogram",
    "adx_14",
    "plus_di_14",
    "minus_di_14",
    # Momentum indicators
    "rsi_14",
    "stoch_k_14",
    "stoch_d_3",
    "williams_r_14",
    # Volatility indicators
    "bb_middle_20",
    "bb_upper_20",
    "bb_lower_20",
    "bb_width_20",
    "bb_percent_b_20",
    "atr_14",
    # Volume indicators
    "obv",
    "cmf_20",
    "volume_sma_20",
    "volume_zscore_20",
]


@dataclass(frozen=True)
class PCAFeatureResult:
    """Container for the fitted scaler/PCA objects and transformed feature data."""

    data: pd.DataFrame
    feature_columns: list[str]
    pca_columns: list[str]
    scaler: StandardScaler
    pca: PCA
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    X_train_pca: pd.DataFrame
    X_test_pca: pd.DataFrame
    train_index: pd.Index
    test_index: pd.Index
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: np.ndarray


_REQUIRED_OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]


def _add_lowercase_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Allow the same code to work with Alpaca lowercase columns or notebook-style Close/Open columns."""

    result = df.copy()

    if "timestamp" not in result.columns and isinstance(result.index, pd.DatetimeIndex):
        index_name = result.index.name or "timestamp"
        result = result.reset_index().rename(columns={index_name: "timestamp"})

    lower_map = {str(column).lower(): column for column in result.columns}
    for standard_name in ["timestamp", *_REQUIRED_OHLCV_COLUMNS]:
        if standard_name not in result.columns and standard_name in lower_map:
            result[standard_name] = result[lower_map[standard_name]]

    return result


def _prepare_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    result = _add_lowercase_aliases(df)

    missing = [column for column in _REQUIRED_OHLCV_COLUMNS if column not in result.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")

    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    for column in _REQUIRED_OHLCV_COLUMNS:
        result[column] = pd.to_numeric(result[column], errors="coerce")

    return result


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    previous_close = close.shift(1)
    ranges = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def _wilder_smoothing(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def build_ml_features(
    df: pd.DataFrame,
    price_col: str = "close",
    target_col: str = "target",
) -> pd.DataFrame:
    """
    Build the technical-indicator feature set used by the PCA/ML pipeline.

    The target is binary and uses information from the next row only:
    1 when next-period return is positive, 0 when it is non-positive.
    The final row has an unknown target, which is intentionally kept as NaN so
    the same function can be used for a latest-data paper-trading signal.
    """

    result = _prepare_price_frame(df)
    if price_col not in result.columns:
        raise ValueError(f"Missing price column: {price_col}")

    open_ = result["open"]
    high = result["high"]
    low = result["low"]
    close = pd.to_numeric(result[price_col], errors="coerce")
    volume = pd.to_numeric(result["volume"], errors="coerce").fillna(0)

    # Required returns and rolling stats
    result["log_return"] = np.log(close / close.shift(1))
    result["rolling_mean_10"] = close.rolling(window=10, min_periods=10).mean()
    result["rolling_std_10"] = close.rolling(window=10, min_periods=10).std()
    result["rolling_mean_20"] = close.rolling(window=20, min_periods=20).mean()
    result["rolling_std_20"] = close.rolling(window=20, min_periods=20).std()

    # Trend: SMA, EMA, MACD, ADX
    result["sma_10"] = close.rolling(window=10, min_periods=10).mean()
    result["sma_20"] = close.rolling(window=20, min_periods=20).mean()
    result["sma_50"] = close.rolling(window=50, min_periods=50).mean()
    result["ema_12"] = close.ewm(span=12, adjust=False, min_periods=12).mean()
    result["ema_26"] = close.ewm(span=26, adjust=False, min_periods=26).mean()
    result["ema_20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    result["macd"] = result["ema_12"] - result["ema_26"]
    result["macd_signal"] = result["macd"].ewm(span=9, adjust=False, min_periods=9).mean()
    result["macd_histogram"] = result["macd"] - result["macd_signal"]

    true_range = _true_range(high, low, close)
    result["atr_14"] = _wilder_smoothing(true_range, period=14)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=result.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=result.index,
    )
    plus_di = 100 * _wilder_smoothing(plus_dm, 14) / result["atr_14"].replace(0, np.nan)
    minus_di = 100 * _wilder_smoothing(minus_dm, 14) / result["atr_14"].replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    result["plus_di_14"] = plus_di
    result["minus_di_14"] = minus_di
    result["adx_14"] = _wilder_smoothing(dx, period=14)

    # Momentum: RSI, stochastic oscillator, Williams %R
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = _wilder_smoothing(gains, period=14)
    avg_loss = _wilder_smoothing(losses, period=14)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    result["rsi_14"] = 100 - (100 / (1 + rs))
    result["rsi_14"] = result["rsi_14"].mask((avg_loss == 0) & (avg_gain > 0), 100)
    result["rsi_14"] = result["rsi_14"].mask((avg_gain == 0) & (avg_loss > 0), 0)
    result["rsi_14"] = result["rsi_14"].mask((avg_gain == 0) & (avg_loss == 0), 50)

    low_14 = low.rolling(window=14, min_periods=14).min()
    high_14 = high.rolling(window=14, min_periods=14).max()
    stochastic_range = (high_14 - low_14).replace(0, np.nan)
    result["stoch_k_14"] = 100 * (close - low_14) / stochastic_range
    result["stoch_d_3"] = result["stoch_k_14"].rolling(window=3, min_periods=3).mean()
    result["williams_r_14"] = -100 * (high_14 - close) / stochastic_range

    # Volatility: Bollinger Bands and ATR
    bb_middle = close.rolling(window=20, min_periods=20).mean()
    bb_std = close.rolling(window=20, min_periods=20).std()
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std
    result["bb_middle_20"] = bb_middle
    result["bb_upper_20"] = bb_upper
    result["bb_lower_20"] = bb_lower
    result["bb_width_20"] = (bb_upper - bb_lower) / bb_middle.replace(0, np.nan)
    result["bb_percent_b_20"] = (close - bb_lower) / (bb_upper - bb_lower).replace(0, np.nan)

    # Volume: OBV and CMF
    direction = np.sign(close.diff()).fillna(0)
    result["obv"] = (direction * volume).cumsum()

    money_flow_multiplier = ((close - low) - (high - close)) / (high - low).replace(0, np.nan)
    money_flow_volume = money_flow_multiplier * volume
    result["cmf_20"] = (
        money_flow_volume.rolling(window=20, min_periods=20).sum()
        / volume.rolling(window=20, min_periods=20).sum().replace(0, np.nan)
    )
    result["volume_sma_20"] = volume.rolling(window=20, min_periods=20).mean()
    volume_std_20 = volume.rolling(window=20, min_periods=20).std()
    result["volume_zscore_20"] = (volume - result["volume_sma_20"]) / volume_std_20.replace(0, np.nan)

    # Binary next-period target for classifier training.
    result["next_return"] = close.pct_change().shift(-1)
    result[target_col] = np.nan
    known_target = result["next_return"].notna()
    result.loc[known_target, target_col] = (result.loc[known_target, "next_return"] > 0).astype(int)

    result = result.replace([np.inf, -np.inf], np.nan)
    return result


def get_feature_columns(
    df: pd.DataFrame | None = None,
    include_missing: bool = False,
) -> list[str]:
    """Return the ML feature columns, optionally filtered to columns present in df."""

    if df is None or include_missing:
        return DEFAULT_FEATURE_COLUMNS.copy()
    return [column for column in DEFAULT_FEATURE_COLUMNS if column in df.columns]


def build_feature_pca_pipeline(
    df: pd.DataFrame,
    feature_columns: Sequence[str] | None = None,
    target_col: str = "target",
    price_col: str = "close",
    test_size: float = 0.20,
    variance_threshold: float = 0.80,
) -> PCAFeatureResult:
    """
    Standardize ML features, fit PCA on the chronological training set, and
    transform all feature-complete rows.

    PCA is fit on training rows only to avoid leaking test/latest information.
    Rows with complete features but unknown targets are transformed too, which
    lets the same fitted model produce a latest paper-trading signal.
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0 < variance_threshold <= 1:
        raise ValueError("variance_threshold must be in the range (0, 1].")

    result = df.copy()
    available_features = get_feature_columns(result)
    if feature_columns is None and len(available_features) < len(DEFAULT_FEATURE_COLUMNS):
        result = build_ml_features(result, price_col=price_col, target_col=target_col)

    selected_features = list(feature_columns) if feature_columns is not None else get_feature_columns(result)
    missing_features = [column for column in selected_features if column not in result.columns]
    if missing_features:
        raise ValueError(f"Missing selected feature columns: {missing_features}")
    if target_col not in result.columns:
        raise ValueError(f"Missing target column: {target_col}")

    modeling_columns = [*selected_features, target_col]
    result[modeling_columns] = result[modeling_columns].apply(pd.to_numeric, errors="coerce")
    result = result.replace([np.inf, -np.inf], np.nan)

    feature_ready = result.dropna(subset=selected_features).copy()
    supervised = feature_ready.dropna(subset=[target_col]).copy()
    if len(supervised) < 30:
        raise ValueError(
            "Not enough rows after feature engineering. Use more history or shorter indicator windows."
        )

    split_index = int(len(supervised) * (1 - test_size))
    if split_index <= 0 or split_index >= len(supervised):
        raise ValueError("The train/test split produced an empty train or test set.")

    train_index = supervised.index[:split_index]
    test_index = supervised.index[split_index:]

    X_train = supervised.loc[train_index, selected_features].copy()
    X_test = supervised.loc[test_index, selected_features].copy()
    y_train = supervised.loc[train_index, target_col].astype(int).copy()
    y_test = supervised.loc[test_index, target_col].astype(int).copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    pca = PCA(n_components=variance_threshold, svd_solver="full")
    X_train_pca_array = pca.fit_transform(X_train_scaled)

    all_scaled = scaler.transform(feature_ready[selected_features])
    all_pca_array = pca.transform(all_scaled)

    pca_columns = [f"PC{i + 1}" for i in range(all_pca_array.shape[1])]
    for component_number, column in enumerate(pca_columns):
        feature_ready[column] = all_pca_array[:, component_number]

    X_train_pca = pd.DataFrame(
        X_train_pca_array,
        index=train_index,
        columns=pca_columns,
    )
    X_test_pca = feature_ready.loc[test_index, pca_columns].copy()

    return PCAFeatureResult(
        data=feature_ready,
        feature_columns=selected_features,
        pca_columns=pca_columns,
        scaler=scaler,
        pca=pca,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        X_train_pca=X_train_pca,
        X_test_pca=X_test_pca,
        train_index=train_index,
        test_index=test_index,
        explained_variance_ratio=pca.explained_variance_ratio_,
        cumulative_explained_variance=np.cumsum(pca.explained_variance_ratio_),
    )
