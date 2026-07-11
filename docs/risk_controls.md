# Risk Controls Documentation

This document explains the risk management layer used by the Alpaca Market Data and Algorithmic Trading Terminal. The goal of the risk system is to make sure that strategy signals are not sent directly to Alpaca paper trading without additional portfolio-level checks.

The trading strategies generate long/flat signals. The execution layer converts those signals into paper orders. The risk layer sits between the strategy signal and the submitted order so that the system can monitor exposure, cap oversized positions, and close positions that breach the configured stop-loss rule.

The risk controls are used in paper trading mode only. No real-money orders are submitted.

---

## High-Level Purpose

The risk module is designed to answer three questions before or during paper trading:

1. Is a new buy order too large relative to the paper portfolio?
2. Has an open position lost more than the allowed amount?
3. Should the system only warn the user, or should it automatically enforce the rule?

This separation is important because a trading signal and a risk decision are not the same thing. A strategy may say “go long,” but the risk module determines whether the resulting trade is acceptable under the current portfolio limits.

---

## Module Boundary

The risk management logic lives mainly in:

```text
src/risk.py
```

The execution logic that applies the risk checks lives in:

```text
src/execution.py
```

The Streamlit user interface displays the risk panel inside:

```text
trading.py
```

The local risk settings are stored in:

```text
risk_config.json
```

The repository includes:

```text
risk_config.example.json
```

This example file shows the expected format without committing personal local settings.

---

## Risk Configuration

The risk configuration contains three main fields:

```json
{
  "enabled": true,
  "stop_loss_pct": 0.20,
  "max_allocation_pct": 0.25
}
```

The values are stored as decimals:

- `0.20` means 20%.
- `0.25` means 25%.

The default configuration is:

- Risk enforcement: enabled.
- Stop-loss limit: 20% unrealized loss per position.
- Maximum allocation limit: 25% of the paper portfolio per position.

When the user changes the settings in the UI and clicks save, the system writes the updated settings to `risk_config.json`. This allows the same risk limits to be restored when the app is restarted.

If the local config file is missing or invalid, the system falls back to the default risk settings.

---

## Risk Rule 1: Stop-Loss Limit

The stop-loss rule checks each open paper position’s unrealized percentage return.

If a position’s unrealized loss is equal to or worse than the configured stop-loss limit, the position is considered a risk breach.

With the default setting:

```text
Stop-loss limit = 20%
```

A position breaches the rule when:

```text
Unrealized return <= -20%
```

### Example

If the paper account holds AAPL and the unrealized return is `-21%`, the stop-loss rule is breached.

If risk enforcement is on, the system submits a paper close-position request through Alpaca.

If risk enforcement is off, the system does not submit an order, but it still shows the breach in the UI.

### Rationale

The purpose of the stop-loss rule is to prevent a strategy from holding a losing paper position indefinitely. This is especially important because the rule-based and ML strategies are not guaranteed to exit quickly in every market environment.

The strategy proposes the trade, but the stop-loss rule acts as a safety layer when the position moves too far against the paper account.

---

## Risk Rule 2: Maximum Allocation Per Position

The maximum allocation rule limits how much of the portfolio can be allocated to a single position.

With the default setting:

```text
Maximum allocation = 25% of portfolio value
```

The system checks the planned buy order before it is submitted. If the planned position would exceed the maximum allocation, the system reduces the order size. If the remaining allowed notional is too small to buy at least one share, the buy order is blocked.

### Example

Suppose the paper portfolio value is `$100,000`.

With a 25% allocation cap, the maximum allowed position size is:

```text
$100,000 × 25% = $25,000
```

If the strategy attempts to buy `$30,000` of a stock, the risk module caps the order so that the resulting position stays within the `$25,000` limit.

If the account already holds a position close to the limit, and there is not enough remaining room to buy one more share, the system blocks the buy order.

### Rationale

The purpose of the allocation rule is to avoid excessive concentration in one ticker. Without this rule, a strategy could allocate too much of the paper portfolio to a single asset, increasing idiosyncratic risk.

This is especially useful because the system allows multiple strategies and tickers to be started from the trading terminal. A maximum allocation rule helps keep the paper account from becoming too concentrated.

---

## Strategy Capital Allocation vs. Risk Allocation Cap

The system has two related but different allocation concepts.

### Strategy Capital Allocation

When a strategy is started in the UI, the user chooses how much available cash that strategy may request for a trade.

The default strategy capital allocation is 10% of available cash.

This controls the initial notional amount requested by the strategy.

### Risk Maximum Allocation

The risk maximum allocation is a portfolio-level cap.

The default risk cap is 25% of the total paper portfolio value per position.

This acts as a final check after the strategy has already proposed a trade. Even if the strategy requests a larger order, the risk module can reduce or block the buy if it would exceed the configured cap.

### Why Both Are Needed

The strategy capital allocation controls how aggressive the strategy is when it opens a trade.

The risk allocation cap controls the maximum portfolio exposure allowed in a single position.

Together, they make the system safer than using strategy signals alone.

---

## Watch vs. Breach Status

The risk panel separates risk status into three states:

```text
Active
Watch
Breach
```

### Active

A position is marked `Active` when it is within the configured risk limits.

### Watch

A position is marked `Watch` when it is approaching a risk limit but has not breached it yet.

For allocation risk, the system enters Watch when the position reaches 80% of the configured allocation limit.

For stop-loss risk, the system enters Watch when the loss reaches half of the configured stop-loss limit.

With the default settings:

```text
Max allocation = 25%
Allocation Watch threshold = 20%

Stop-loss = 20%
Stop-loss Watch threshold = -10%
```

This gives the user an early warning before a position becomes a full breach.

### Breach

A position is marked `Breach` when it has crossed one of the configured limits.

A breach can occur because:

- The position allocation is above the maximum allowed allocation.
- The unrealized loss is worse than the stop-loss limit.

---

## Enforcement Modes

The risk system has two modes.

### Enforcement Off

When enforcement is off, the system only monitors the paper portfolio.

In this mode:

- Risk limits are still calculated.
- Watch and Breach statuses are still displayed.
- The UI warns the user when a rule is breached.
- No automatic corrective orders are submitted.

This mode is useful for observing how often a strategy would violate risk limits without letting the system act automatically.

### Enforcement On

When enforcement is on, the system actively applies the risk rules.

In this mode:

- New buy orders are capped or blocked if they would exceed the maximum allocation limit.
- Positions that breach the stop-loss rule are closed through Alpaca paper trading.
- Risk events are written to the local execution log.

This mode more closely represents how a real trading system would separate signal generation from risk approval.

---

## How Risk Controls Interact With Execution

The execution process follows this order:

```text
Strategy signal generated
        ↓
Execution layer reads current paper position
        ↓
Execution layer builds order plan
        ↓
Risk module checks the order plan
        ↓
Order is allowed, reduced, blocked, or position is closed
        ↓
Alpaca paper order is submitted when appropriate
```

The execution layer is long-only:

- Long signal and no current position: build a buy plan.
- Long signal and existing position: hold.
- Flat signal and existing position: close the position.
- Flat signal and no position: do nothing.

For buy orders, the risk module checks the planned order against the maximum allocation rule before the order reaches Alpaca.

For existing positions, the risk panel checks whether a stop-loss breach has occurred. If enforcement is enabled, a stop-loss breach triggers a paper close-position request.

---

## Order Type and Position Sizing

The system submits paper orders through Alpaca paper trading.

The current implementation uses market-style paper execution logic:

- Buy orders are submitted when the latest signal is Long and the account does not already hold the position.
- Sell or close-position requests are submitted when the signal is Flat or when a stop-loss is breached.
- Buy quantity is rounded down to whole shares.
- A buy order is refused if the available cash cannot purchase at least one share.
- The order planner uses available paper cash, so the system does not intentionally rely on leverage.

The risk module does not create the trading signal. It only modifies or blocks the order plan after the signal has already been generated.

---

## Risk Monitoring in the UI

The risk controls are displayed in the Streamlit trading terminal.

The panel allows the user to:

- Turn risk enforcement on or off.
- Adjust the stop-loss percentage.
- Adjust the maximum allocation percentage.
- Save the risk settings locally.
- View current position risk status.
- View recent risk events.

The risk status panel refreshes while the terminal is open, allowing the user to monitor positions, unrealized P&L, allocation, and risk status without manually inspecting the account.

The UI also displays paper account information, including:

- Portfolio value.
- Cash and buying power.
- Open positions.
- Unrealized P&L.
- Recent orders.
- Alpaca order events.
- Local execution logs.

This helps make the system state easier to understand while paper trading.

---

## Logging and Audit Trail

The system logs important paper-trading and risk events.

Examples of logged events include:

- Latest strategy signal.
- Current paper position.
- Generated order plan.
- Risk-adjusted order plan.
- Submitted paper order.
- Stop-loss close-position event.
- Failed risk enforcement attempt.

This is important because trading systems need an audit trail. If the system buys, sells, blocks an order, or closes a position, the user should be able to understand why the action happened.

---

## What the Current Risk Controls Do Not Do

The current risk system is intentionally simple. It provides basic but meaningful controls, but it is not a full production risk engine.

The current implementation does not include:

- Portfolio-level maximum drawdown liquidation.
- Take-profit rules.
- Volatility-based position sizing.
- Correlation limits across multiple assets.
- Sector or industry exposure limits.
- Limit-order execution logic.
- Full transaction-cost or slippage modeling.
- Real-time intraday risk based on every tick.

The current allocation rule can cap or block new buy orders, but existing positions that drift above the allocation cap are flagged rather than automatically sold because of allocation alone. Automatic closing is currently tied to the stop-loss rule.

---

## Why These Controls Matter

The risk controls matter because even a systematic strategy can behave poorly in live or paper trading.

A strategy may:

- Enter a position before a large reversal.
- Hold a losing trade too long.
- Allocate too much capital to one ticker.
- Generate a valid signal during an unfavorable account state.
- Produce a trade that is too large relative to the portfolio.

The risk module reduces these problems by adding a second decision layer. The strategy decides what it wants to do. The risk system decides whether the trade is allowed under the portfolio rules.

---

## Project Requirement Mapping

The project asks for a complete Alpaca-based trading system with strategy logic, execution, UI controls, risk limits, logging, and documentation.

This risk module addresses those requirements by providing:

- A separate risk module for limits and checks.
- Configurable risk settings through `risk_config.json`.
- UI controls for stop-loss and maximum allocation.
- Position monitoring through the paper trading terminal.
- Automatic enforcement when risk enforcement is enabled.
- Warning-only monitoring when risk enforcement is disabled.
- Logging of risk events and paper-trading actions.
- Integration with the execution layer before orders reach Alpaca paper trading.

Together, these features make the trading system more realistic because the strategies are not allowed to operate without risk checks.