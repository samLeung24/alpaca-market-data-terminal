from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.features import (
    PCAFeatureResult,
    build_feature_pca_pipeline,
    build_ml_features,
)


# Assignment probability threshold:
# Long if model probability > 0.60
# Flat if model probability <= 0.60
PROBABILITY_THRESHOLD = 0.60

ML_POSITION_COL = "ml_position"
ML_TRADE_SIGNAL_COL = "ml_trade_signal"
ML_BUY_SIGNAL_COL = "ml_buy_signal"
ML_SELL_SIGNAL_COL = "ml_sell_signal"


def train_classifier(
    pca_result: PCAFeatureResult,
    classifier: BaseEstimator | None = None,
) -> BaseEstimator:
    """
    Train the assignment classifier on PCA components.

    This version uses Logistic Regression by default instead of Random Forest.
    The model is trained on PCA-transformed features.
    """

    if pca_result.y_train.nunique() < 2:
        raise ValueError(
            "Training target has only one class. Use more data or a different date range/ticker."
        )

    model = classifier or LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )

    model.fit(pca_result.X_train_pca, pca_result.y_train.astype(int))
    return model


def _positive_class_probability(model: BaseEstimator, X: pd.DataFrame) -> np.ndarray:
    """
    Return the predicted probability of class 1.

    Class 1 means:
        next-day return > 0
    """

    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The classifier must support predict_proba because the assignment uses "
            "a 0.60 probability threshold."
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
    probability_threshold: float = PROBABILITY_THRESHOLD,
    test_size: float = 0.20,
    variance_threshold: float = 0.80,
    classifier: BaseEstimator | None = None,
    trade_on_test_only: bool = True,
) -> pd.DataFrame:
    """
    End-to-end ML signal pipeline.

    Steps:
        1. Build technical indicator features.
        2. Define binary target:
               target = 1 if next-day return > 0 else 0
        3. Standardize features.
        4. Apply PCA.
        5. Keep PCA components explaining at least 80% of variance.
        6. Train Logistic Regression on PCA components.
        7. Predict probability of next-day positive return.
        8. Generate signal:
               Long if probability > 0.60
               Flat if probability <= 0.60

    Output columns are compatible with the backtester through:

        ml_position
        ml_trade_signal
        ml_buy_signal
        ml_sell_signal
    """

    if not 0 < probability_threshold < 1:
        raise ValueError("probability_threshold must be between 0 and 1.")

    # ------------------------------------------------------------------
    # 1. Feature engineering and target creation
    # ------------------------------------------------------------------
    result = build_ml_features(df, price_col=price_col)

    # ------------------------------------------------------------------
    # 2. Standardization + PCA pipeline
    # ------------------------------------------------------------------
    pca_result = build_feature_pca_pipeline(
        result,
        price_col=price_col,
        test_size=test_size,
        variance_threshold=variance_threshold,
    )

    # ------------------------------------------------------------------
    # 3. Train Logistic Regression model
    # ------------------------------------------------------------------
    model = train_classifier(pca_result, classifier=classifier)

    # ------------------------------------------------------------------
    # 4. Predict probabilities using PCA components
    # ------------------------------------------------------------------
    pca_input = pca_result.data[pca_result.pca_columns]

    positive_probability = _positive_class_probability(model, pca_input)

    raw_signal = (positive_probability > probability_threshold).astype(int)
    predicted_target = (positive_probability >= 0.50).astype(int)

    # ------------------------------------------------------------------
    # 5. Initialize ML output columns
    # ------------------------------------------------------------------
    result["ml_probability"] = np.nan
    result["ml_predicted_target"] = np.nan
    result["ml_raw_signal"] = 0
    result["ml_signal"] = "Flat"
    result["ml_sample_type"] = "not_ready"

    # Add PCA component columns back into the main result DataFrame
    for column in pca_result.pca_columns:
        result[column] = np.nan
        result.loc[pca_result.data.index, column] = pca_result.data[column]

    # ------------------------------------------------------------------
    # 6. Store model predictions/signals
    # ------------------------------------------------------------------
    prediction_index = pca_result.data.index

    result.loc[prediction_index, "ml_probability"] = positive_probability
    result.loc[prediction_index, "ml_predicted_target"] = predicted_target
    result.loc[prediction_index, "ml_raw_signal"] = raw_signal
    result.loc[prediction_index, "ml_signal"] = np.where(raw_signal == 1, "Long", "Flat")

    # Mark rows as train/test/latest
    result.loc[pca_result.train_index, "ml_sample_type"] = "train"
    result.loc[pca_result.test_index, "ml_sample_type"] = "test"

    latest_unlabeled_index = pca_result.data.index[pca_result.data["target"].isna()]
    result.loc[latest_unlabeled_index, "ml_sample_type"] = "latest_unlabeled"

    # ------------------------------------------------------------------
    # 7. Decide which rows are eligible for trading
    # ------------------------------------------------------------------
    # For a realistic backtest, only trade on the test set.
    # The latest unlabeled row is included so paper-trading can use the
    # most recent signal.
    if trade_on_test_only:
        eligible_index = pca_result.test_index.union(latest_unlabeled_index)
    else:
        eligible_index = prediction_index

    result[ML_POSITION_COL] = 0
    result.loc[eligible_index, ML_POSITION_COL] = (
        result.loc[eligible_index, "ml_raw_signal"].astype(int)
    )
    result[ML_POSITION_COL] = result[ML_POSITION_COL].fillna(0).astype(int)

    # ------------------------------------------------------------------
    # 8. Convert position into trade signals
    # ------------------------------------------------------------------
    # ml_position:
    #     1 = Long
    #     0 = Flat
    #
    # ml_trade_signal:
    #     1  = Buy
    #    -1  = Sell
    #     0  = Hold current state
    result[ML_TRADE_SIGNAL_COL] = (
        result[ML_POSITION_COL]
        .diff()
        .fillna(result[ML_POSITION_COL])
        .astype(int)
    )

    result[ML_BUY_SIGNAL_COL] = result[ML_TRADE_SIGNAL_COL] == 1
    result[ML_SELL_SIGNAL_COL] = result[ML_TRADE_SIGNAL_COL] == -1

    # Helpful generic aliases for logs or paper-trading scripts.
    result["position"] = result[ML_POSITION_COL]
    result["trade_signal"] = result[ML_TRADE_SIGNAL_COL]
    result["buy_signal"] = result[ML_BUY_SIGNAL_COL]
    result["sell_signal"] = result[ML_SELL_SIGNAL_COL]

    # ------------------------------------------------------------------
    # 9. Store metadata for display/debugging
    # ------------------------------------------------------------------
    result.attrs["ml_feature_columns"] = pca_result.feature_columns
    result.attrs["ml_pca_columns"] = pca_result.pca_columns
    result.attrs["ml_pca_explained_variance_ratio"] = (
        pca_result.explained_variance_ratio.tolist()
    )
    result.attrs["ml_pca_cumulative_explained_variance"] = (
        pca_result.cumulative_explained_variance.tolist()
    )
    result.attrs["ml_model"] = model.__class__.__name__
    result.attrs["ml_probability_threshold"] = probability_threshold
    result.attrs["ml_trade_on_test_only"] = trade_on_test_only

    return result


def generate_ml_signals(
    df: pd.DataFrame,
    price_col: str = "close",
    probability_threshold: float = PROBABILITY_THRESHOLD,
    test_size: float = 0.20,
    variance_threshold: float = 0.80,
    classifier: BaseEstimator | None = None,
    trade_on_test_only: bool = True,
) -> pd.DataFrame:
    """
    Convenience wrapper used by the app, backtester, or paper-trading script.

    This simply calls run_ml_signal_pipeline().
    """

    return run_ml_signal_pipeline(
        df=df,
        price_col=price_col,
        probability_threshold=probability_threshold,
        test_size=test_size,
        variance_threshold=variance_threshold,
        classifier=classifier,
        trade_on_test_only=trade_on_test_only,
    )