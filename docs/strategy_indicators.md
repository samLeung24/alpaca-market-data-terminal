# Strategy and Indicator Documentation

This document summarizes the strategies and indicators used by the strategy backtester. The goal of this page is to explain not only the trading rules, but also the rationale behind each strategy, the market behavior each one is trying to exploit, and the main limitations of the approach.

The strategies are designed for exploratory analysis, classroom demonstration, and Alpaca paper-trading tests. They are not presented as production-ready trading systems or guaranteed sources of alpha.

---

## Strategy Overview

All implemented rule-based strategies are long-only:

- `1` means the strategy is invested.
- `0` means the strategy is in cash.

The strategies are also systematic. Once the user selects a strategy, the buy and sell decisions come from predefined rules rather than discretionary clicking.

The backtester also includes a buy-and-hold benchmark for comparison. This benchmark is important because a trading strategy should not only be evaluated in isolation. It should also be compared against a simple passive alternative.

If a strategy performs worse than buy-and-hold, then its extra complexity may not be justified.

---

## Strategy 1: Trend Following

The trend-following strategy uses MACD and a 200-period simple moving average.

### Signal Rules

Buy when:

- MACD is above the MACD signal line.
- Closing price is above SMA200.

Sell when:

- MACD falls below the MACD signal line.
- Closing price falls below SMA200.

### Purpose

This strategy tries to participate when price momentum is positive and the stock is trading above its longer-term trend.

### Strategy Intuition

The intuition behind this strategy is trend persistence. In financial markets, prices sometimes continue moving in the same direction because investors react to information at different speeds, buying pressure can reinforce itself, and strong trends can attract additional market participants.

The 200-period simple moving average acts as a long-term trend filter. If the price is above SMA200, the asset is treated as being in a healthier long-term environment. MACD then adds a shorter-term momentum confirmation. If MACD is above its signal line, recent momentum is positive.

The strategy only enters when both conditions are true: the long-term trend is positive and short-term momentum is improving.

### Market Behavior Being Exploited

This strategy tries to exploit sustained upward trends. The market behavior we are trying to capture is that when an asset is already trading above its long-term trend and short-term momentum is improving, the move may continue.

### Why It Could Generate Returns

This strategy could generate returns by staying invested during large upward trends while moving to cash during weaker or declining regimes.

Instead of holding through every market condition, it tries to be exposed mainly when trend and momentum are aligned.

### Main Weakness

The main weakness is that trend-following indicators are lagging. The strategy may enter after a move has already started and exit only after weakness has already appeared.

It can also get whipsawed in sideways markets. In a sideways market, MACD and price can repeatedly cross their thresholds without a strong trend developing, causing the strategy to buy and sell too often.

---

## Strategy 2: Mean Reversion

The mean-reversion strategy uses RSI and Bollinger Bands.

### Signal Rules

Buy when:

- RSI14 is below 30.
- Closing price is below the lower Bollinger Band.

Sell when:

- RSI14 is above 70.
- Closing price is above the upper Bollinger Band.

### Purpose

This strategy looks for short-term oversold conditions and exits after the price recovers into an overbought area.

### Strategy Intuition

The intuition behind mean reversion is that prices sometimes move too far in one direction in the short term and then return toward a more normal range.

Instead of trying to follow a strong trend, this strategy looks for moments when the asset appears temporarily overextended. When the price appears unusually low relative to its recent behavior, the strategy treats it as a possible buying opportunity. When the price moves unusually high, the strategy exits the position to lock in gains or avoid a reversal.

RSI below 30 suggests that the asset may be oversold. A close below the lower Bollinger Band means the price is unusually low relative to its recent volatility-adjusted range. When both conditions occur together, the strategy treats the move as a possible short-term overreaction.

The exit rule works in the opposite direction. RSI above 70 and a close above the upper Bollinger Band suggest that the asset has rebounded strongly and may now be overextended to the upside.

### Market Behavior Being Exploited

This strategy tries to exploit short-term overreaction. Prices can temporarily fall too far because of liquidity pressure, stop-loss selling, emotional reactions to news, or short-term market noise.

### Why It Could Generate Returns

The strategy could generate returns if oversold price moves normalize and the asset bounces back toward its recent range.

The goal is to buy after unusually sharp weakness and exit after a strong recovery.

### Main Weakness

The main weakness is that not every oversold move is temporary. Sometimes a sharp decline is the beginning of a real downtrend.

In that case, a mean-reversion strategy can buy too early and hold through further losses. This is the risk of “catching a falling knife.”

---

## Strategy 3: Custom Multi-Factor

The custom strategy combines trend, momentum, and volatility breakout signals.

### Signal Rules

Buy when all of these are true:

- Closing price is above SMA200.
- EMA20 is above SMA50.
- MACD is above the MACD signal line.
- RSI14 is between 50 and 70.
- Closing price is above the upper Bollinger Band.

Sell when any of these are true:

- Closing price falls below EMA20.
- MACD falls below the MACD signal line.
- RSI14 falls below 45.
- Closing price falls below the middle Bollinger Band.

### Purpose

This strategy is stricter on entry than exit. It waits for trend, momentum, and breakout conditions to align before buying, then exits when price or momentum starts to weaken.

### Strategy Intuition

The intuition behind this strategy is trend acceleration. The strategy does not buy just because one indicator looks positive. Instead, it requires several independent signals to align before entering.

The long-term trend filter checks that the asset is above SMA200. The EMA20 above SMA50 condition checks that the shorter-term trend is stronger than the medium-term trend. MACD confirms positive momentum. RSI between 50 and 70 attempts to capture strength without buying an extremely overbought condition. Finally, the Bollinger Band breakout requires the price to move above its recent volatility-adjusted range.

Together, these filters are meant to identify assets that are already in a healthy uptrend and are now breaking out with stronger momentum.

### Market Behavior Being Exploited

This strategy tries to exploit breakout continuation. The market behavior behind the strategy is that strong breakouts can attract additional buyers, confirm demand, and lead to further upside movement.

### Why It Could Generate Returns

This strategy could generate returns by entering only when multiple signals suggest strong upside pressure.

Because the entry conditions are stricter, the strategy tries to avoid weaker signals and reduce false entries. It is designed to participate in stronger trend environments rather than trade every small signal.

### Main Weakness

The main weakness is that the entry rule may be too strict. The strategy can miss early parts of a move and enter late, after much of the price increase has already happened.

It can also lose money during false breakouts, where price briefly moves above the upper Bollinger Band but then reverses.

---

## Buy-and-Hold Benchmark

The backtester includes a buy-and-hold benchmark.

The benchmark invests at the beginning of the sample and remains invested until the end. This creates a simple comparison point for the active strategies.

This benchmark matters because active trading rules should be compared against a passive alternative. A strategy may look profitable on its own but still perform worse than simply buying and holding the asset.

The comparison also helps show whether the strategy is actually adding value or just benefiting from the asset’s general upward movement over the sample period.

---

## Machine-Learning Strategy Note

The project also includes a machine-learning strategy based on logistic regression, PCA, and technical-indicator features.

That workflow is documented separately in:

- `docs/feature_model.md`

The ML strategy estimates the probability that the next-period return will be positive. If the predicted probability is greater than the configured threshold, the strategy goes Long. Otherwise, it stays Flat.

The ML strategy is also long-only and systematic. It does not manually choose trades, and it is tested against a buy-and-hold benchmark over the same holdout period.

---

## Indicators

The backtester can calculate and display these indicators:

- SMA50
- SMA200
- EMA12
- EMA20
- EMA26
- MACD
- MACD signal
- MACD histogram
- RSI14
- Bollinger Bands
- Momentum 10
- Stochastic oscillator

---

## Indicator Purpose

### Moving Averages

Moving averages smooth price data and help identify the direction of the trend.

SMA200 is used as a long-term trend filter. EMA20 and SMA50 help identify shorter-term and medium-term trend conditions.

### MACD

MACD is used as a momentum and trend-change indicator. When MACD is above the MACD signal line, the strategy treats short-term momentum as positive. When MACD falls below the signal line, momentum is treated as weakening.

### RSI

RSI is used to identify overbought and oversold conditions.

In the mean-reversion strategy, RSI below 30 is treated as oversold, while RSI above 70 is treated as overbought.

In the custom multi-factor strategy, RSI between 50 and 70 is used as a momentum filter. This range suggests positive momentum without requiring an extremely overbought reading.

### Bollinger Bands

Bollinger Bands are volatility-adjusted price bands.

In the mean-reversion strategy, a close below the lower Bollinger Band suggests the price may be unusually low relative to recent volatility.

In the custom strategy, a close above the upper Bollinger Band is used as a breakout signal.

### Stochastic Oscillator and Momentum

The stochastic oscillator and momentum indicators are available for analysis and visualization. They help measure the speed and direction of recent price movement.

---

## Limitations

These strategies are for exploratory analysis and classroom demonstration. They do not include commissions, slippage, taxes, funding costs, borrow costs, or dividend reinvestment.

This matters because real-world trading is affected by both explicit and implicit costs. Even a strategy that looks profitable in a simplified backtest may perform worse after transaction costs and implementation frictions are included.

The strategies also rely on historical price behavior. There is no guarantee that patterns observed in the backtest will continue in the future.

Other limitations include:

- The strategies are long-only and cannot profit directly from short opportunities.
- The rule-based strategies use fixed parameters rather than optimized or adaptive parameters.
- The backtests do not fully model order execution quality.
- The strategies may be sensitive to the selected ticker, date range, and market regime.
- The indicators are lagging or reactive, meaning they respond to price behavior after it has already happened.
- The ML strategy may find weak historical relationships that do not remain stable out of sample.

---

## Potential Improvements

Future improvements could include:

- Adding explicit transaction-cost and slippage modeling.
- Adding dividend adjustments for longer backtests.
- Testing the strategies across a broader universe of tickers.
- Adding walk-forward validation for strategy parameters.
- Adding take-profit rules and more detailed stop-loss logic.
- Adding portfolio-level risk constraints across multiple tickers.
- Adding volatility-based position sizing.
- Adding limit-order execution logic instead of relying only on market-style paper execution.
- Improving trade logs to explain which exact rule triggered each entry and exit.
- Adding more robust out-of-sample testing for the ML strategy.

---

## Lessons Learned

The biggest lesson from the strategy backtester is that building a trading system is not just about finding a signal.

A complete system needs:

- Clean data.
- Clear signal rules.
- Backtesting infrastructure.
- Risk controls.
- Secure credential handling.
- Logging.
- A user interface that makes the system state easy to understand.
- A benchmark for comparison.

Another important lesson is that generating alpha is extremely hard. When strategies are compared against a simple buy-and-hold benchmark, the extra return is not always strong or consistent.

This suggests that markets are relatively efficient. Even when a strategy has a reasonable intuition, it can still struggle to beat a simple benchmark, especially after accounting for transaction costs, taxes, slippage, and other real-world frictions.

Overall, the project shows that a strategy can make sense in theory and still be difficult to implement profitably in practice.

---

## Final Notes

These strategies are useful because they demonstrate different systematic trading ideas:

- Trend following tries to capture persistent upward movement.
- Mean reversion tries to capture short-term overreaction.
- The custom multi-factor strategy tries to capture strong trend acceleration and breakout behavior.
- Buy-and-hold provides a passive benchmark.
- The ML strategy tests whether technical features can provide weak predictive information about next-period direction.

The main purpose of this documentation is to explain the logic, intuition, trade-offs, and limitations behind the strategy layer of the Alpaca Market Data and Algorithmic Trading Terminal.