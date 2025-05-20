# Coinbase Checker

A cryptocurrency trading signal generator that identifies potential trading opportunities on Coinbase using market data analysis and technical indicators.

## Features

- Automatically fetches real-time market data from Coinbase API
- Analyzes cryptocurrencies for trading signals based on multiple factors:
  - Momentum analysis
  - Order book pressure
  - Support/resistance levels
  - Volume surge detection
  - Peak detection (avoids coins near their recent high)
  - Reversal pattern detection
- Ranks trading opportunities with detailed metrics
- Tracks signal performance over time
- Adapts parameters based on historical performance

## Setup

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Unix/MacOS: `source venv/bin/activate`
4. Install dependencies: `pip install pandas numpy requests`
5. Create a logs directory: `mkdir logs`

## Usage

Run the main script to fetch data and generate signals:

```
python coinbase_checker_signals.py
```

The program will:
1. Fetch market data from Coinbase
2. Process and analyze the data
3. Generate trading signals
4. Output the top trading opportunities in the console
5. Save results to CSV files

## Output Files

- `crypto_signals.csv`: Complete signal data for all analyzed cryptocurrencies
- `ranked_opportunities.csv`: Top-ranked trading opportunities
- `all_cryptos_merged.csv`: Merged market data for analysis
- `signal_history.json`: History of tracked signals
- `signal_performance_history.json`: Performance metrics of past signals

## How It Works

The system uses a multi-factor analysis approach to identify potential trading opportunities:

1. Data collection from Coinbase API for USD trading pairs
2. Technical analysis including momentum, volatility, order book, and support/resistance
3. Peak avoidance to prevent buying at local highs
4. Reversal detection to identify potential bottoming patterns
5. Opportunity ranking with volume-weighted scoring
6. Performance tracking to adapt parameters over time

## Disclaimer

This tool is for educational and research purposes only. It is not financial advice, and trading cryptocurrencies involves significant risk of loss. 