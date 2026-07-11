# Feature Engineering and ML Workflow

This document describes the current feature engineering, PCA, logistic
regression, backtesting, and paper-trading workflow in the Alpaca market data
terminal.

The system has two user-facing modes:

- `backtesting.py`: evaluates rule-based strategies and the ML strategy on
  historical data.
- `trading.py`: trains a selected-equity ML model, refreshes the latest signal,
  and submits Alpaca paper-trading orders from the latest signal.

## High-Level Data Flow

```text
Selected ticker
  -> Alpaca daily OHLCV data
  -> src/features.py
     -> build_ml_features()
     -> build_feature_pca_pipeline()
  -> src/models.py
     -> train_classifier()
     -> run_ml_signal_pipeline()
  -> src/backtester.py
     -> build_ml_strategy_spec()
     -> run_backtest()
     -> build_buy_hold_result()
  -> src/metrics.py and src/plots.py
     -> metrics table and charts
  -> src/execution.py
     -> execute_latest_signal() for Alpaca paper trading
```

## Module Boundaries

### `src/features.py`

Owns feature engineering, target construction, chronological train/test split,
standardization, PCA fitting, and PCA transformation.

Important objects and functions:

- `PCAFeatureResult`: container for feature data, fitted scaler, fitted PCA,
  train/test PCA matrices, labels, indices, and explained variance.
- `build_ml_features(df, price_col="close", target_col="target")`: creates the
  technical-indicator feature matrix and binary next-period target.
- `get_feature_columns(df=None, include_missing=False)`: returns the feature
  columns used by the ML pipeline.
- `build_feature_pca_pipeline(...)`: builds features if needed, splits rows
  chronologically, fits `StandardScaler` and `PCA` on training rows only, then
  transforms all feature-ready rows.
- `transform_features_with_pipeline(...)`: transforms new data with an already
  fitted scaler and PCA object.
- `transform_latest_features(...)`: returns only the latest PCA-transformed row
  for fast trading-mode inference.

`src/features.py` should not train the classifier or submit trades.

### `src/models.py`

Owns classifier training, probability scoring, and ML signal generation. It
starts after PCA-ready data exists.

Important objects and functions:

- `PROBABILITY_THRESHOLD = 0.60`: default cutoff for a long signal.
- `MLSignalResult`: container for the signal DataFrame, fitted model, PCA
  result, PCA component columns, threshold, and signal policy.
- `train_classifier(pca_result, classifier=None)`: trains the classifier on
  `pca_result.X_train_pca` and `pca_result.y_train`.
- `run_ml_signal_pipeline(...)`: trains logistic regression and returns an
  `MLSignalResult`.
- `score_pca_features(...)`: scores any PCA-ready frame with an already fitted
  model and appends standard ML signal columns.
- `generate_ml_signals(...)`: compatibility wrapper that returns only the
  signal DataFrame.
- `predict_latest_signal(...)`: returns the latest long/flat signal summary.

The selected model is logistic regression:

```python
LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    solver="lbfgs",
    random_state=42,
)
```

Signal rule:

```text
P(next-day return > 0) > 0.60 -> Long
P(next-day return > 0) <= 0.60 -> Flat
```

`src/models.py` should not build raw features, fit PCA, run backtests, or submit
Alpaca orders.

### `src/backtester.py`

Owns portfolio simulation and trade logs for both rule-based strategies and the
ML signal.

Important functions:

- `build_ml_strategy_spec()`: maps ML signal columns to the generic backtester
  contract.
- `run_backtest(signal_df, strategy, initial_capital=...)`: simulates strategy
  returns, portfolio value, drawdown, and trade events.
- `build_buy_hold_result(df, initial_capital=...)`: builds the buy-and-hold
  benchmark.
- `build_trade_log(...)`: creates closed-trade records from trade signals.

### `src/execution.py`

Owns Alpaca paper-trading execution only. It receives model-generated signal
rows and does not know how features, PCA, or model training work.

Important functions:

- `get_latest_signal(signal_df)`: extracts the latest usable ML signal row.
- `get_current_position(symbol, trading_client=None)`: reads the current paper
  position for the ticker.
- `build_order_plan(...)`: converts the latest signal and current paper
  position into a `BUY`, `SELL`, `HOLD`, or `NONE` action.
- `submit_paper_order(order_plan, trading_client=None)`: submits the planned
  paper order to Alpaca.
- `execute_latest_signal(...)`: coordinates signal extraction, paper-position
  lookup, order planning, order submission, and logging.

Order behavior is long-only:

- Long signal and no current position: paper buy.
- Long signal and existing position: hold.
- Flat signal and open position: close paper position.
- Flat signal and no position: no order.

### `src/data_connector.py`

Owns Alpaca client creation.

Important functions:

- `get_historical_client(...)`: creates the Alpaca historical data client.
- `get_stream_client(...)`: creates the Alpaca streaming data client.
- `get_paper_trading_client(...)`: creates `TradingClient(..., paper=True)`.

## Feature Set

`build_ml_features()` uses daily OHLCV data and creates features from several
groups.

Returns and rolling statistics:

- `log_return`
- `rolling_mean_10`
- `rolling_std_10`
- `rolling_mean_20`
- `rolling_std_20`

Trend:

- `sma_10`
- `sma_20`
- `sma_50`
- `ema_12`
- `ema_26`
- `ema_20`
- `macd`
- `macd_signal`
- `macd_histogram`
- `adx_14`
- `plus_di_14`
- `minus_di_14`

Momentum:

- `rsi_14`
- `stoch_k_14`
- `stoch_d_3`
- `williams_r_14`

Volatility:

- `bb_middle_20`
- `bb_upper_20`
- `bb_lower_20`
- `bb_width_20`
- `bb_percent_b_20`
- `atr_14`

Volume:

- `obv`
- `cmf_20`
- `volume_sma_20`
- `volume_zscore_20`

Target:

- `target = 1` when the next-period return is positive.
- `target = 0` when the next-period return is non-positive.
- The final row has an unknown target and is kept as `NaN` so it can still be
  transformed for latest-signal inference.

## PCA and Leakage Control

`build_feature_pca_pipeline()` avoids future-data leakage by using a
chronological split:

1. Build all feature columns and the next-period target.
2. Drop rows that are not feature-complete.
3. Use only rows with known targets for supervised train/test splitting.
4. Split chronologically with the configured `test_size`.
5. Fit `StandardScaler` on training features only.
6. Fit `PCA(n_components=variance_threshold, svd_solver="full")` on scaled
   training features only.
7. Transform training, test, and latest feature-ready rows with the fitted
   scaler/PCA objects.

The default PCA variance threshold is `0.80`, so the pipeline keeps enough
principal components to explain at least 80% of training-set feature variance.

## ML Strategy Rationale

The machine-learning strategy uses logistic regression to estimate whether the next daily return is more likely to be positive or non-positive. The model is still systematic and long-only: it does not make discretionary decisions, and it does not short the asset.

The target variable is:

- `1` if the next-period return is positive.
- `0` if the next-period return is non-positive.

The model uses features based on recent price, trend, momentum, volatility, and volume behavior. The rationale is that these variables may contain weak information about short-term direction. For example, recent momentum, volatility expansion, moving-average relationships, or volume pressure may slightly change the probability that the next return is positive.

The strategy does not go long every time the model predicts class `1`. Instead, it uses a stricter probability threshold:

- If `P(next-day return > 0) > 0.60`, the strategy goes Long.
- If `P(next-day return > 0) <= 0.60`, the strategy stays Flat.

The reason for using a 0.60 threshold is to make the strategy more selective. A 0.50 threshold would simply mean that the model thinks an up day is slightly more likely than a down day. By requiring 0.60, the system only enters when the model’s estimated probability is stronger.

PCA is used because many technical indicators are correlated with each other. For example, moving averages, MACD, and rolling returns all contain related information about recent price behavior. PCA reduces dimensionality and helps control multicollinearity before logistic regression is trained.

The pipeline also controls for look-ahead bias. The chronological split ensures that earlier observations are used for training and later observations are reserved for testing. The scaler and PCA are fit only on the training rows, then applied to the holdout rows. This prevents information from the test period from leaking into the model training process.

The ML strategy should be interpreted carefully. Logistic regression is useful because it is simple, transparent, and less prone to overfitting than more complex models, but financial prediction is still difficult. The model may find weak relationships in historical data that do not remain stable in the future. Therefore, the holdout backtest and comparison against buy-and-hold are more important than training accuracy alone.

## Backtesting Mode

The ML backtesting flow lives in `backtesting.py`.

Important functions:

- `get_daily_price_data_for_ml(...)`: ensures the ML workflow uses daily bars
  even when the chart uses another timeframe.
- `build_ml_logistic_regression_backtest(...)`: builds PCA features, trains
  logistic regression, filters to the holdout rows, and runs the ML and
  buy-and-hold backtests.
- `retrain_ml_logistic_regression_backtest(...)`: refetches/prepares daily
  data and rebuilds the ML holdout backtest.
- `get_ml_backtest_cache()` and `build_ml_backtest_cache_key(...)`: cache
  backtest results by symbol, date range, threshold, split, and initial capital.

Default controls:

- Initial capital: `$100,000`
- Probability threshold: `0.60`
- Holdout test split: `0.20`
- PCA variance threshold: `0.80`

The ML strategy is tested only on rows labeled `ml_sample_type == "test"`.
This means the model trains on earlier data and evaluates on later unseen daily
bars.

## Trading Mode

The ML trading panel lives in `trading.py`.

Important functions:

- `render_ml_trading_panel(symbol, is_valid_symbol)`: renders controls, cached
  model state, metrics, charts, trades, paper-order controls, and logs.
- `_build_ml_results(...)`: fetches 5 years of daily bars, builds PCA features,
  trains the model, generates ML signals, runs a holdout backtest, and builds
  metrics.
- `_train_ml_panel_state(...)`: creates the cached per-symbol model state.
- `_build_fresh_latest_signal(...)`: fetches latest daily bars and transforms
  the newest row through the cached scaler/PCA/model path.
- `_read_paper_trading_log(...)`: reads recent log lines for display in the UI.

Training and inference are intentionally separated:

1. User selects an equity.
2. User clicks `Train Model / Run Backtest` or `Retrain Model / Run Backtest`.
3. The fitted feature/PCA/model state is cached in `st.session_state`.
4. User clicks `Refresh Signal and Submit Paper Order`.
5. The app fetches latest daily data.
6. The app uses `transform_latest_features()` to apply the cached scaler/PCA
   pipeline.
7. The app uses `score_pca_features()` to score the latest row with the cached
   logistic regression model.
8. The resulting latest signal is passed to `execute_latest_signal()`.

This keeps the trading panel responsive because paper-order clicks do not
retrain the model.

## Standard ML Signal Columns

After `score_pca_features()` or `run_ml_signal_pipeline()`, downstream modules
expect these columns:

- `ml_probability`: probability that the next-period return is positive.
- `ml_predicted_target`: class prediction using a 0.50 threshold.
- `ml_raw_signal`: raw long/flat decision from the configured threshold.
- `ml_signal`: display label, either `Long` or `Flat`.
- `ml_position`: numeric long-only position, `1` for long and `0` for flat.
- `ml_trade_signal`: position change, `1` buy, `-1` sell, `0` no change.
- `ml_buy_signal`: boolean buy marker.
- `ml_sell_signal`: boolean sell marker.
- `ml_sample_type`: `train`, `test`, `latest_unlabeled`, or `not_ready`.

## Collaboration Notes

When modifying this workflow, keep these boundaries intact:

- Put raw feature construction, target alignment, scaling, and PCA in
  `src/features.py`.
- Put classifier training and signal generation in `src/models.py`.
- Put portfolio simulation in `src/backtester.py`.
- Put paper-account inspection, order planning, order submission, and logging
  in `src/execution.py`.
- Put Alpaca client construction in `src/data_connector.py`.
- Keep Streamlit display and session-state orchestration in `trading.py` and
  `backtesting.py`.

This separation makes it easier to test each layer independently and prevents
paper-trading execution code from depending on feature-engineering details.
