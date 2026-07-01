# alpaca-market-data-terminal

Mini market data and strategy backtesting terminal using Alpaca REST APIs

## Executive Summary

This project connects to Alpaca market data to retrieve historical OHLCV,
display interactive price and volume charts, stream live bid/ask/last-trade
updates, and test simple long-only trading strategies against a buy-and-hold
benchmark.

The repository currently contains two Streamlit apps:

- `app.py`: market data terminal with historical OHLCV charts and live quote
  updates.
- `backtesting.py`: strategy backtester with buy/sell markers, portfolio value,
  drawdown, and performance metrics.

## Demo Video

Market data terminal: https://youtu.be/Zx6PTew7rmc

Strategy backtester: https://youtu.be/NmMOAOslkIA

## Screenshots

### Market Data Terminal

![Historical chart and live quote panel](screenshots/UI_1.png)

![Historical data table](screenshots/UI_2.png)

### Strategy Backtester

![Trading signals](screenshots/tradingSignals.png)

![Portfolio value comparison](screenshots/portfolioValue.png)

![Maximum drawdown comparison](screenshots/maxDrawdown.png)

## Setup

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate alpaca-terminal
```

Create a local `.env` file from the example:

```bash
cp .env.example .env
```

Then add your Alpaca API key and secret to `.env`:

```text
ALPACA_API_KEY=your_paper_api_key_here
ALPACA_SECRET_KEY=your_paper_secret_key_here
ALPACA_DATA_FEED=iex
```

## Run

Run the market data terminal:

```bash
streamlit run app.py
```

Run the strategy backtester:

```bash
streamlit run backtesting.py
```

## Documentation

Strategy and indicator details are documented separately in
[`docs/strategy_indicators.md`](docs/strategy_indicators.md).

## Repository Structure

```text
alpaca-market-data-terminal/
├── app.py                  # Streamlit market data terminal
├── backtesting.py          # Streamlit strategy backtester
├── docs/
│   ├── Week2_Report.pdf        # Homework 2 report
│   └── strategy_indicators.md  # Strategy and indicator documentation
├── src/
│   ├── __init__.py
│   ├── config.py           # Loads Alpaca credentials and feed settings
│   ├── data_connector.py   # Builds Alpaca historical and streaming clients
│   ├── historical.py       # Fetches historical OHLCV bar data
│   ├── live_quotes.py      # Streams latest quotes/trades with Alpaca websocket
│   ├── company.py          # Resolves ticker symbols to company names
│   ├── company_search.py   # Provides ticker/company search choices
│   ├── indicators.py       # Adds technical indicator columns
│   ├── strategies.py       # Generates long-only strategy signals
│   ├── backtester.py       # Simulates strategy and buy-and-hold performance
│   ├── metrics.py          # Calculates and formats performance metrics
│   └── plots.py            # Builds Plotly charts for backtest results
├── screenshots/
│   ├── UI_1.png            # Historical chart and live quote screenshot
│   ├── UI_2.png            # Historical data table screenshot
│   ├── maxDrawdown.png     # Backtester drawdown chart screenshot
│   ├── portfolioValue.png  # Backtester portfolio value screenshot
│   └── tradingSignals.png  # Backtester buy/sell signal screenshot
├── .env.example            # Template for required Alpaca API credentials
├── .gitignore              # Excludes local secrets, caches, and system files
├── environment.yml         # Conda environment specification
├── requirements.txt        # Python package requirements
├── LICENSE                 # Project license
├── SKILL.md                # Project-specific workflow notes
└── README.md               # Project overview, setup, and usage instructions
```

## Behavioral Notes

During after-hours periods, live quote updates may be sparse in the market data terminal, but the panel should still show the last available quote.

The strategy backtester is intended for exploratory analysis. It is not a
production trading or portfolio accounting system.

## Security Notes

Do not commit `.env` or real API credentials. Commit `.env.example` only.
