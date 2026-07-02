from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

from src.features import (
    PCAFeatureResult,
    build_feature_pca_pipeline,
    build_ml_features,
)


ML_POSITION_COL = "ml_position"
ML_TRADE_SIGNAL_COL = "ml_trade_signal"
ML_BUY_SIGNAL_COL = "ml_buy_signal"
ML_SELL_SIGNAL_COL = "ml_sell_signal"


def train_classifier(
    pca_result: PCAFeatureResult,
    classifier: BaseEstimator | None = None,
) -> BaseEstimator:
    """Train the assignment classifier on PCA components."""

    if pca_result.y_train.nunique() < 2:
        raise ValueError(
            "Training target has only one class. Use more data or a different date range/ticker."
        )

    model = classifier or RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=10,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(pca_result.X_train_pca, pca_result.y_train.astype(int))
    return model


def _positive_class_probability(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The classifier must support predict_proba because the assignment uses a 0.60 probability threshold."
        )

    probabilities = model.predict_proba(X)
    classes = list(getattr(model, "classes_", []))
    if 1 not in classes:
        raise ValueError("The trained classifier does not contain class 1 in model.classes_.")

    positive_class_position = classes.index(1)
    return probabilities[:, positive_class_position]


def run_ml_signal_pipeline(
    df: pd.DataFrame,
    price_col: str = "close",
    probability_threshold: float = 0.60,
    test_size: float = 0.20,
    variance_threshold: float = 0.80,
    classifier: BaseEstimator | None = None,
    trade_on_test_only: bool = True,
) -> pd.DataFrame:
    """
    End-to-end Feature -> PCA -> classifier -> long/flat signal pipeline.

    Output columns are compatible with src.backtester.run_backtest when wired
    through a StrategySpec using the ML_* column constants below.
    """

    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0 and 1.")

    result = build_ml_features(df, price_col=price_col)
    pca_result = build_feature_pca_pipeline(
        result,
        price_col=price_col,
        test_size=test_size,
        variance_threshold=variance_threshold,
    )
    model = train_classifier(pca_result, classifier=classifier)

    pca_input = pca_result.data[pca_result.pca_columns]
    positive_probability = _positive_class_probability(model, pca_input)
    raw_signal = (positive_probability > probability_threshold).astype(int)
    predicted_target = (positive_probability >= 0.50).astype(int)

    result["ml_probability"] = np.nan
    result["ml_predicted_target"] = np.nan
    result["ml_raw_signal"] = 0
    result["ml_signal"] = "Flat"
    result["ml_sample_type"] = "not_ready"

    for column in pca_result.pca_columns:
        result[column] = np.nan
        result.loc[pca_result.data.index, column] = pca_result.data[column]

    prediction_index = pca_result.data.index
    result.loc[prediction_index, "ml_probability"] = positive_probability
    result.loc[prediction_index, "ml_predicted_target"] = predicted_target
    result.loc[prediction_index, "ml_raw_signal"] = raw_signal
    result.loc[prediction_index, "ml_signal"] = np.where(raw_signal == 1, "Long", "Flat")

    result.loc[pca_result.train_index, "ml_sample_type"] = "train"
    result.loc[pca_result.test_index, "ml_sample_type"] = "test"
    latest_unlabeled_index = pca_result.data.index[pca_result.data["target"].isna()]
    result.loc[latest_unlabeled_index, "ml_sample_type"] = "latest_unlabeled"

    if trade_on_test_only:
        eligible_index = pca_result.test_index.union(latest_unlabeled_index)
    else:
        eligible_index = prediction_index

    result[ML_POSITION_COL] = 0
    result.loc[eligible_index, ML_POSITION_COL] = result.loc[eligible_index, "ml_raw_signal"].astype(int)
    result[ML_POSITION_COL] = result[ML_POSITION_COL].fillna(0).astype(int)

    result[ML_TRADE_SIGNAL_COL] = (
        result[ML_POSITION_COL].diff().fillna(result[ML_POSITION_COL]).astype(int)
    )
    result[ML_BUY_SIGNAL_COL] = result[ML_TRADE_SIGNAL_COL] == 1
    result[ML_SELL_SIGNAL_COL] = result[ML_TRADE_SIGNAL_COL] == -1

    # Helpful generic aliases for logs or paper-trading scripts.
    result["position"] = result[ML_POSITION_COL]
    result["trade_signal"] = result[ML_TRADE_SIGNAL_COL]
    result["buy_signal"] = result[ML_BUY_SIGNAL_COL]
    result["sell_signal"] = result[ML_SELL_SIGNAL_COL]

    # Store compact metadata for display/debugging without changing backtest logic.
    result.attrs["ml_feature_columns"] = pca_result.feature_columns
    result.attrs["ml_pca_columns"] = pca_result.pca_columns
    result.attrs["ml_pca_explained_variance_ratio"] = pca_result.explained_variance_ratio.tolist()
    result.attrs["ml_pca_cumulative_explained_variance"] = (
        pca_result.cumulative_explained_variance.tolist()
    )
    result.attrs["ml_model"] = model.__class__.__name__
    result.attrs["ml_probability_threshold"] = probability_threshold

    return result
