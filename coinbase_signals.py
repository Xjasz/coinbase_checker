import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json

# Set up logging
logger = logging.getLogger("coinbase_signals")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/coinbase_signals.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Constants for price targets
DEFAULT_STOP_LOSS_PCT = 0.02  # 2% stop loss by default
MIN_TAKE_PROFIT_PCT = 0.03    # 3% minimum take profit
DEFAULT_TAKE_PROFIT_PCT = 0.05  # 5% take profit by default
PERFORMANCE_HISTORY_FILE = "sig_data/signal_performance_history.json"
SIGNAL_HISTORY_FILE = "sig_data/signal_history.json"
TOP_RANK_COUNT = 5  # Number of top opportunities to track for adaptation (Tier 1)
TIER_2_RANK_MAX = 10  # Maximum rank for Tier 2
TIER_3_RANK_MAX = 20  # Maximum rank for Tier 3

# Start date for performance tracking - can be adjusted manually
PERFORMANCE_TRACKING_START_DATE = datetime.utcnow().isoformat()

def preprocess_data(data):
    """
    Preprocess the merged cryptocurrency data for signal generation.
    
    Args:
        data (DataFrame): The merged cryptocurrency data
        
    Returns:
        DataFrame: Preprocessed data with additional columns
    """
    df = data.copy()
    
    # Ensure datetime types
    for col in ['ticker_time', 'candle_start', 'candle_end', 'request_time']:
        df[col] = pd.to_datetime(df[col])
    
    # Handle missing data - ensure points are in chronological order
    df.sort_values(['symbol', 'ticker_time'], inplace=True)
    
    # Calculate time deltas between consecutive records (for detecting gaps)
    df['time_delta'] = df.groupby('symbol')['ticker_time'].diff().dt.total_seconds() / 60
    
    # Calculate basic price derivatives
    df['pct_change_abs'] = df['pct_change'].abs()
    df['price_volatility'] = df.groupby('symbol')['pct_change'].rolling(12).std().reset_index(level=0, drop=True)
    
    # Calculate volume-based metrics
    df['volume_ratio'] = df['vol5m'] / df.groupby('symbol')['vol5m'].rolling(24).mean().reset_index(level=0, drop=True)
    df['usd_volume_ratio'] = df['usd_vol5m'] / df.groupby('symbol')['usd_vol5m'].rolling(24).mean().reset_index(level=0, drop=True)
    
    # Order book imbalance metrics
    df['bid_ask_ratio'] = df['total_bid_volume'] / df['total_ask_volume'].replace(0, np.nan)
    df['depth_ratio'] = df['bid_depth'] / df['ask_depth'].replace(0, np.nan)
    
    # Handle NaNs
    for col in ['price_volatility', 'volume_ratio', 'usd_volume_ratio', 'bid_ask_ratio', 'depth_ratio']:
        df[col] = df[col].fillna(0)
    
    return df

def calculate_momentum_score(data, window=24):
    """
    Calculate momentum-based signals
    
    Args:
        data (DataFrame): Preprocessed cryptocurrency data
        window (int): Lookback window for momentum calculation
        
    Returns:
        DataFrame: Data with momentum signals
    """
    df = data.copy()
    
    # Calculate short-term momentum (1-hour)
    df['momentum_1h'] = df.groupby('symbol')['ticker_price'].pct_change(12)
    
    # Calculate medium-term momentum (6-hour) 
    df['momentum_6h'] = df.groupby('symbol')['ticker_price'].pct_change(window)
    
    # Calculate momentum strength ratio (short-term vs medium-term)
    df['momentum_ratio'] = df['momentum_1h'] / df['momentum_6h'].replace(0, np.nan)
    df['momentum_ratio'] = df['momentum_ratio'].fillna(0)
    
    # Calculate momentum score (ranges from -100 to 100)
    df['momentum_score'] = (100 * np.tanh(df['momentum_1h'] * 10) * 
                           (0.7 + 0.3 * df['volume_ratio'].clip(0, 3)))
    
    return df

def calculate_volatility_breakout(data, lookback=72, threshold_mult=2.0):
    """
    Calculate volatility breakout signals
    
    Args:
        data (DataFrame): Preprocessed cryptocurrency data
        lookback (int): Number of periods to establish volatility baseline
        threshold_mult (float): Multiplier for volatility threshold
        
    Returns:
        DataFrame: Data with volatility breakout signals
    """
    df = data.copy()
    
    # Calculate historical volatility (standard deviation of returns)
    df['hist_volatility'] = df.groupby('symbol')['pct_change'].rolling(lookback).std().reset_index(level=0, drop=True)
    
    # Calculate average trading range
    df['avg_range'] = df.groupby('symbol').apply(
        lambda x: (x['high'] - x['low']) / x['low']
    ).reset_index(level=0, drop=True)
    df['avg_range_baseline'] = df.groupby('symbol')['avg_range'].rolling(lookback).mean().reset_index(level=0, drop=True)
    
    # Calculate breakout threshold
    df['breakout_threshold'] = df['avg_range_baseline'] * threshold_mult
    
    # Current candle range relative to historical average
    df['current_range_ratio'] = df['avg_range'] / df['avg_range_baseline'].replace(0, np.nan)
    df['current_range_ratio'] = df['current_range_ratio'].fillna(1)
    
    # Calculate breakout score (ranges from 0 to 100)
    df['is_range_expanding'] = df['current_range_ratio'] > 1.0
    df['volatility_breakout_score'] = 100 * np.tanh(
        (df['current_range_ratio'] - 1) * 
        df['volume_ratio'].clip(0, 3) * 
        df['is_range_expanding']
    )
    
    return df

def calculate_order_book_pressure(data, ma_period=24):
    """
    Calculate order book pressure signals
    
    Args:
        data (DataFrame): Preprocessed cryptocurrency data
        ma_period (int): Moving average period for baseline
        
    Returns:
        DataFrame: Data with order book pressure signals
    """
    df = data.copy()
    
    # Order book imbalance
    df['bid_ask_imbalance'] = (df['total_bid_volume'] - df['total_ask_volume']) / (df['total_bid_volume'] + df['total_ask_volume'])
    
    # Normalize and smooth the imbalance
    df['bid_ask_imbalance_ma'] = df.groupby('symbol')['bid_ask_imbalance'].rolling(ma_period).mean().reset_index(level=0, drop=True)
    df['bid_ask_imbalance_std'] = df.groupby('symbol')['bid_ask_imbalance'].rolling(ma_period).std().reset_index(level=0, drop=True)
    
    # Z-score of current imbalance
    df['bid_ask_pressure_z'] = (df['bid_ask_imbalance'] - df['bid_ask_imbalance_ma']) / df['bid_ask_imbalance_std'].replace(0, 1)
    
    # Calculate order book pressure score (ranges from -100 to 100)
    df['order_book_score'] = 100 * np.tanh(df['bid_ask_pressure_z'] * 0.5)
    
    # Add depth factor (more depth = more reliable signal)
    total_depth = df['bid_depth'] + df['ask_depth']
    df['depth_factor'] = np.tanh(total_depth / 5000)
    df['order_book_score'] = df['order_book_score'] * df['depth_factor']
    
    return df

def calculate_volume_surge(data, baseline_window=72, threshold=2.0):
    """
    Calculate volume surge signals
    
    Args:
        data (DataFrame): Preprocessed cryptocurrency data
        baseline_window (int): Lookback window for volume baseline
        threshold (float): Volume surge threshold multiplier
        
    Returns:
        DataFrame: Data with volume surge signals
    """
    df = data.copy()
    
    # Calculate volume baseline
    df['volume_baseline'] = df.groupby('symbol')['vol5m'].rolling(baseline_window).mean().reset_index(level=0, drop=True)
    df['usd_volume_baseline'] = df.groupby('symbol')['usd_vol5m'].rolling(baseline_window).mean().reset_index(level=0, drop=True)
    
    # Calculate volume surge ratio
    df['volume_surge_ratio'] = df['vol5m'] / df['volume_baseline'].replace(0, np.nan)
    df['usd_volume_surge_ratio'] = df['usd_vol5m'] / df['usd_volume_baseline'].replace(0, np.nan)
    
    # Fill NaNs
    df['volume_surge_ratio'] = df['volume_surge_ratio'].fillna(1)
    df['usd_volume_surge_ratio'] = df['usd_volume_surge_ratio'].fillna(1)
    
    # Calculate volume surge score (ranges from 0 to 100)
    df['volume_surge_score'] = 100 * np.tanh((df['volume_surge_ratio'] - threshold) * 0.5) * (df['volume_surge_ratio'] > threshold)
    df['usd_volume_surge_score'] = 100 * np.tanh((df['usd_volume_surge_ratio'] - threshold) * 0.5) * (df['usd_volume_surge_ratio'] > threshold)
    
    # Combine scores with price direction
    df['price_direction'] = np.sign(df['pct_change'])
    df['volume_surge_buy_signal'] = df['volume_surge_score'] * (df['price_direction'] > 0)
    
    return df

def calculate_support_resistance(data, lookback=144, n_levels=3, threshold=0.01):
    """
    Calculate support and resistance levels
    
    Args:
        data (DataFrame): Preprocessed cryptocurrency data
        lookback (int): Lookback window for identifying levels
        n_levels (int): Number of support/resistance levels to identify
        threshold (float): Price proximity threshold to consider a level hit
        
    Returns:
        DataFrame: Data with support/resistance signals
    """
    df = data.copy()
    
    # Function to find support/resistance levels for a symbol
    def find_levels(symbol_data):
        # Create a copy to avoid SettingWithCopyWarning
        df_symbol = symbol_data.copy()
        
        # Get price high/low history
        highs = df_symbol['high'].values
        lows = df_symbol['low'].values
        
        # Find peaks and troughs using simple algorithm
        # A peak is a point higher than its neighbors
        peak_indices = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
                peak_indices.append(i)
        
        # A trough is a point lower than its neighbors
        trough_indices = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
                trough_indices.append(i)
        
        # Extract the price values at peaks and troughs
        if len(peak_indices) > 0:
            resistance_levels = highs[peak_indices]
            resistance_levels = np.sort(resistance_levels)[-n_levels:] if len(resistance_levels) >= n_levels else resistance_levels
        else:
            resistance_levels = []
            
        if len(trough_indices) > 0:
            support_levels = lows[trough_indices]
            support_levels = np.sort(support_levels)[:n_levels] if len(support_levels) >= n_levels else support_levels
        else:
            support_levels = []
        
        # Calculate distance to nearest support/resistance
        current_price = df_symbol['ticker_price'].iloc[-1]
        
        # Distance to support (positive if price is above support)
        if len(support_levels) > 0:
            support_distances = [(current_price - level) / current_price for level in support_levels]
            nearest_support_dist = min([d for d in support_distances if d > 0], default=float('inf'))
            nearest_support = current_price * (1 - nearest_support_dist) if nearest_support_dist != float('inf') else None
        else:
            nearest_support_dist = float('inf')
            nearest_support = None
            
        # Distance to resistance (positive if price is below resistance)
        if len(resistance_levels) > 0:
            resistance_distances = [(level - current_price) / current_price for level in resistance_levels]
            nearest_resistance_dist = min([d for d in resistance_distances if d > 0], default=float('inf'))
            nearest_resistance = current_price * (1 + nearest_resistance_dist) if nearest_resistance_dist != float('inf') else None
        else:
            nearest_resistance_dist = float('inf')
            nearest_resistance = None
            
        # Create result columns
        df_symbol['nearest_support'] = nearest_support
        df_symbol['nearest_resistance'] = nearest_resistance
        df_symbol['distance_to_support'] = nearest_support_dist if nearest_support_dist != float('inf') else np.nan
        df_symbol['distance_to_resistance'] = nearest_resistance_dist if nearest_resistance_dist != float('inf') else np.nan
        
        # Calculate support/resistance score
        if nearest_support is not None and nearest_resistance is not None:
            # Price position between support and resistance (0 = at support, 1 = at resistance)
            price_position = (current_price - nearest_support) / (nearest_resistance - nearest_support) if nearest_resistance != nearest_support else 0.5
            # Score from -100 (at resistance) to 100 (at support)
            df_symbol['sr_score'] = 100 * (1 - 2 * price_position)
        else:
            df_symbol['sr_score'] = 0
            
        return df_symbol
    
    # Apply to each symbol and combine results
    result_dfs = []
    for symbol, group in df.groupby('symbol'):
        # Only process if we have enough data points
        if len(group) >= lookback:
            result_dfs.append(find_levels(group.tail(lookback)))
        else:
            # For symbols with insufficient data, just add NaN columns
            group['nearest_support'] = np.nan
            group['nearest_resistance'] = np.nan
            group['distance_to_support'] = np.nan
            group['distance_to_resistance'] = np.nan
            group['sr_score'] = 0
            result_dfs.append(group)
    
    return pd.concat(result_dfs)

def calculate_combined_signal_score(data):
    """
    Calculate a combined signal score from all individual signals
    
    Args:
        data (DataFrame): Cryptocurrency data with all calculated signals
        
    Returns:
        DataFrame: Data with combined signal score
    """
    df = data.copy()
    
    # Normalize the NaN values in score columns
    score_columns = [
        'momentum_score', 'volatility_breakout_score', 
        'order_book_score', 'volume_surge_buy_signal', 'sr_score'
    ]
    
    for col in score_columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in dataframe, skipping")
            continue
        df[col] = df[col].fillna(0)
    
    # Create a peak detection filter - penalize coins that have recently hit new highs
    df['recent_high'] = df.groupby('symbol')['high'].rolling(24).max().reset_index(level=0, drop=True)
    df['high_proximity'] = (df['ticker_price'] / df['recent_high'])
    
    # If price is within 2% of recent high, apply penalty to avoid buying at the peak
    df['peak_penalty'] = np.where(df['high_proximity'] > 0.98, -50 * (df['high_proximity'] - 0.98) * 50, 0)
    
    # Add a reversal detection factor
    # Calculate short-term momentum (30 min - 1 hour)
    df['very_short_momentum'] = df.groupby('symbol')['ticker_price'].pct_change(6)
    df['short_momentum'] = df.groupby('symbol')['ticker_price'].pct_change(12)
    df['medium_momentum'] = df.groupby('symbol')['ticker_price'].pct_change(36)
    
    # Reversal bonus: Positive when price was down in medium term but up in short term
    # This identifies potential bottoming patterns rather than tops
    df['reversal_bonus'] = np.where(
        (df['medium_momentum'] < -0.02) & (df['short_momentum'] > 0) & (df['very_short_momentum'] > 0),
        50,  # Strong bonus for potential reversals
        0
    )
    
    # Calculate the combined score with weights - REDUCED momentum weight to avoid chasing pumps
    weights = {
        'momentum_score': 0.15,      # REDUCED from 0.25
        'volatility_breakout_score': 0.20,
        'order_book_score': 0.25,    # INCREASED from 0.20
        'volume_surge_buy_signal': 0.15,
        'sr_score': 0.25             # INCREASED from 0.20
    }
    
    # Ensure all required columns exist
    for col in weights.keys():
        if col not in df.columns:
            logger.warning(f"Missing column {col} for combined score, setting to zero")
            df[col] = 0
    
    # Calculate combined score
    df['raw_combined_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)
    
    # Add peak penalty and reversal bonus to the combined score
    df['combined_score'] = df['raw_combined_score'] + df['peak_penalty'] + df['reversal_bonus']
    
    # Normalize to -100 to 100 range
    df['combined_score'] = 100 * np.tanh(df['combined_score'] / 100)
    
    # Strategy categorization
    df['signal_strength'] = pd.cut(
        df['combined_score'], 
        bins=[-100, -60, -20, 20, 60, 100], 
        labels=['Strong Sell', 'Weak Sell', 'Neutral', 'Weak Buy', 'Strong Buy']
    )
    
    return df

def calculate_price_targets(data, atr_periods=14):
    """
    Calculate entry, stop loss and take profit levels
    
    Args:
        data (DataFrame): Preprocessed cryptocurrency data
        atr_periods (int): Periods for ATR calculation
        
    Returns:
        DataFrame: Data with price targets
    """
    df = data.copy()
    
    # Calculate ATR (Average True Range) for dynamic stop loss
    df['tr1'] = df.groupby('symbol')['high'].shift(1) - df.groupby('symbol')['low'].shift(1)
    df['tr2'] = abs(df.groupby('symbol')['high'].shift(1) - df.groupby('symbol')['close'].shift(1))
    df['tr3'] = abs(df.groupby('symbol')['low'].shift(1) - df.groupby('symbol')['close'].shift(1))
    df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
    df['atr'] = df.groupby('symbol')['true_range'].rolling(window=atr_periods).mean().reset_index(level=0, drop=True)
    
    # Calculate volatility-adjusted stop loss (higher for more volatile coins)
    df['volatility_factor'] = df['price_volatility'] / df.groupby('symbol')['price_volatility'].transform('mean')
    df['volatility_factor'] = df['volatility_factor'].fillna(1.0).clip(0.5, 2.0)
    
    # Calculate support/resistance adjusted stop loss
    df['sl_distance'] = DEFAULT_STOP_LOSS_PCT * df['volatility_factor']
    
    # For coins near support, tighten stop loss
    support_factor = 1.0 - (df['sr_score'].clip(-100, 100) / 200)  # 0.5 (at support) to 1.0 (at resistance)
    df['sl_distance'] = df['sl_distance'] * support_factor
    
    # Ensure minimum stop loss
    df['sl_distance'] = df['sl_distance'].clip(0.015, 0.05)  # Between 1.5% and 5%
    
    # Calculate entry price as current price for simplicity
    df['entry_price'] = df['ticker_price']
    
    # Calculate stop loss price
    df['stop_loss_price'] = df['entry_price'] * (1 - df['sl_distance'])
    
    # Calculate take profit based on risk-reward ratio and signal strength
    # Higher combined score = higher profit target
    profit_factor = 1.0 + ((df['combined_score'].clip(0, 100) / 100) * 0.5)  # 1.0 to 1.5
    df['take_profit_distance'] = df['sl_distance'] * 2.0 * profit_factor  # 2:1 risk-reward minimum
    
    # Ensure minimum take profit is met
    df['take_profit_distance'] = df['take_profit_distance'].clip(MIN_TAKE_PROFIT_PCT, 0.15)  # Between 3% and 15%
    
    # Calculate take profit price
    df['take_profit_price'] = df['entry_price'] * (1 + df['take_profit_distance'])
    
    # Calculate risk-reward ratio
    df['risk_reward_ratio'] = df['take_profit_distance'] / df['sl_distance']
    
    return df

def load_signal_history():
    """
    Load signal history from file
    
    Returns:
        dict: Signal history data
    """
    if os.path.exists(SIGNAL_HISTORY_FILE):
        try:
            with open(SIGNAL_HISTORY_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading signal history: {e}")
    return {}

def save_signal_history(history):
    """
    Save signal history to file
    
    Args:
        history (dict): Signal history data
    """
    try:
        with open(SIGNAL_HISTORY_FILE, 'w') as f:
            json.dump(history, f, indent=2, default=str)
    except Exception as e:
        logger.error(f"Error saving signal history: {e}")

def track_signal(symbol, entry_price, stop_loss, take_profit, signal_score, rank, timestamp=None):
    """
    Track a new signal
    
    Args:
        symbol (str): Cryptocurrency symbol
        entry_price (float): Entry price
        stop_loss (float): Stop loss price
        take_profit (float): Take profit price
        signal_score (float): Signal score
        rank (int): Rank of the opportunity
        timestamp (str, optional): Timestamp for the signal. Defaults to current time.
        
    Returns:
        dict: Signal data
    """
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()
        
    signal_data = {
        "symbol": symbol,
        "entry_price": float(entry_price),
        "stop_loss": float(stop_loss),
        "take_profit": float(take_profit),
        "signal_score": float(signal_score),
        "rank": int(rank),
        "timestamp": timestamp,
        "status": "active",
        "result": None,
        "exit_price": None,
        "exit_time": None,
        "profit_pct": None
    }
    
    # Load existing signals
    history = load_signal_history()
    
    # Initialize if empty
    if "signals" not in history:
        history["signals"] = []
        
    # Add new signal
    history["signals"].append(signal_data)
    
    # Save updated history
    save_signal_history(history)
    
    return signal_data

def update_signal_results(current_data):
    """
    Update results for active signals
    
    Args:
        current_data (DataFrame): Current market data
        
    Returns:
        tuple: (updated_signals, all_metrics, tier1_metrics, tier2_metrics, tier3_metrics)
    """
    # Load existing signals
    history = load_signal_history()
    
    if "signals" not in history or not history["signals"]:
        return [], {}, {}, {}, {}
    
    # Create latest price lookup
    latest_prices = {}
    for symbol, group in current_data.groupby('symbol'):
        if not group.empty:
            latest_row = group.iloc[-1]
            latest_prices[symbol] = {
                'price': latest_row['ticker_price'],
                'timestamp': latest_row['ticker_time']
            }
    
    # Load existing performance metrics file if it exists
    existing_metrics = {}
    if os.path.exists(PERFORMANCE_HISTORY_FILE):
        try:
            with open(PERFORMANCE_HISTORY_FILE, 'r') as f:
                existing_metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading existing performance metrics: {e}")
    
    # Initialize metrics with start dates from existing metrics if available
    metrics = {
        "start_date": existing_metrics.get("all_signals", {}).get("start_date", PERFORMANCE_TRACKING_START_DATE),
        "total_signals": 0,
        "active_signals": 0,
        "successful_signals": 0,
        "failed_signals": 0,
        "total_profit_pct": 0,
        "avg_profit_pct": 0,
        "avg_win_pct": 0,
        "avg_loss_pct": 0,
        "win_rate": 0
    }
    
    # Separate metrics for tier 1 (top 5) signals
    tier1_metrics = {
        "start_date": existing_metrics.get("tier1_signals", {}).get("start_date", PERFORMANCE_TRACKING_START_DATE),
        "total_signals": 0,
        "active_signals": 0,
        "successful_signals": 0,
        "failed_signals": 0,
        "total_profit_pct": 0,
        "avg_profit_pct": 0,
        "avg_win_pct": 0,
        "avg_loss_pct": 0,
        "win_rate": 0
    }
    
    # Separate metrics for tier 2 (ranks 6-10) signals
    tier2_metrics = {
        "start_date": existing_metrics.get("tier2_signals", {}).get("start_date", PERFORMANCE_TRACKING_START_DATE),
        "total_signals": 0,
        "active_signals": 0,
        "successful_signals": 0,
        "failed_signals": 0,
        "total_profit_pct": 0,
        "avg_profit_pct": 0,
        "avg_win_pct": 0,
        "avg_loss_pct": 0,
        "win_rate": 0
    }
    
    # Separate metrics for tier 3 (ranks 11-20) signals
    tier3_metrics = {
        "start_date": existing_metrics.get("tier3_signals", {}).get("start_date", PERFORMANCE_TRACKING_START_DATE),
        "total_signals": 0,
        "active_signals": 0,
        "successful_signals": 0,
        "failed_signals": 0,
        "total_profit_pct": 0,
        "avg_profit_pct": 0,
        "avg_win_pct": 0,
        "avg_loss_pct": 0,
        "win_rate": 0
    }
    
    # First update any active signals
    updated_signals = []
    for i, signal in enumerate(history["signals"]):
        if signal["status"] == "active" and signal["symbol"] in latest_prices:
            current_price = latest_prices[signal["symbol"]]['price']
            current_time = latest_prices[signal["symbol"]]['timestamp']
            
            # Check if take profit hit
            if current_price >= signal["take_profit"]:
                signal["status"] = "closed"
                signal["result"] = "take_profit"
                signal["exit_price"] = float(signal["take_profit"])
                signal["exit_time"] = current_time
                signal["profit_pct"] = ((signal["take_profit"] / signal["entry_price"]) - 1) * 100
                
            # Check if stop loss hit
            elif current_price <= signal["stop_loss"]:
                signal["status"] = "closed"
                signal["result"] = "stop_loss"
                signal["exit_price"] = float(signal["stop_loss"])
                signal["exit_time"] = current_time
                signal["profit_pct"] = ((signal["stop_loss"] / signal["entry_price"]) - 1) * 100
            
            history["signals"][i] = signal
    
    # Save updated signal statuses
    save_signal_history(history)
    
    # Now tally up the metrics from the entire history
    all_wins = []
    all_losses = []
    tier1_wins = []
    tier1_losses = []
    tier2_wins = []
    tier2_losses = []
    tier3_wins = []
    tier3_losses = []
    
    # Process all signals to build metrics
    for signal in history["signals"]:
        rank_num = signal.get("rank", 0)
        is_tier1 = rank_num <= TOP_RANK_COUNT
        is_tier2 = TOP_RANK_COUNT < rank_num <= TIER_2_RANK_MAX
        is_tier3 = TIER_2_RANK_MAX < rank_num <= TIER_3_RANK_MAX
        
        # Always count in all_signals
        metrics["total_signals"] += 1
        
        if signal["status"] == "active":
            metrics["active_signals"] += 1
        elif signal["status"] == "closed":
            profit_pct = signal.get("profit_pct", 0)
            
            if signal["result"] == "take_profit":
                metrics["successful_signals"] += 1
                metrics["total_profit_pct"] += profit_pct
                all_wins.append(profit_pct)
            else:  # stop loss
                metrics["failed_signals"] += 1
                metrics["total_profit_pct"] += profit_pct
                all_losses.append(profit_pct)
        
        # Count in tier1 if applicable
        if is_tier1:
            tier1_metrics["total_signals"] += 1
            
            if signal["status"] == "active":
                tier1_metrics["active_signals"] += 1
            elif signal["status"] == "closed":
                profit_pct = signal.get("profit_pct", 0)
                
                if signal["result"] == "take_profit":
                    tier1_metrics["successful_signals"] += 1
                    tier1_metrics["total_profit_pct"] += profit_pct
                    tier1_wins.append(profit_pct)
                else:  # stop loss
                    tier1_metrics["failed_signals"] += 1
                    tier1_metrics["total_profit_pct"] += profit_pct
                    tier1_losses.append(profit_pct)
        
        # Count in tier2 if applicable
        elif is_tier2:  # IMPORTANT: changed from 'if' to 'elif' to avoid double counting
            tier2_metrics["total_signals"] += 1
            
            if signal["status"] == "active":
                tier2_metrics["active_signals"] += 1
            elif signal["status"] == "closed":
                profit_pct = signal.get("profit_pct", 0)
                
                if signal["result"] == "take_profit":
                    tier2_metrics["successful_signals"] += 1
                    tier2_metrics["total_profit_pct"] += profit_pct
                    tier2_wins.append(profit_pct)
                else:  # stop loss
                    tier2_metrics["failed_signals"] += 1
                    tier2_metrics["total_profit_pct"] += profit_pct
                    tier2_losses.append(profit_pct)
                    
        # Count in tier3 if applicable
        elif is_tier3:  # Using elif to avoid double counting
            tier3_metrics["total_signals"] += 1
            
            if signal["status"] == "active":
                tier3_metrics["active_signals"] += 1
            elif signal["status"] == "closed":
                profit_pct = signal.get("profit_pct", 0)
                
                if signal["result"] == "take_profit":
                    tier3_metrics["successful_signals"] += 1
                    tier3_metrics["total_profit_pct"] += profit_pct
                    tier3_wins.append(profit_pct)
                else:  # stop loss
                    tier3_metrics["failed_signals"] += 1
                    tier3_metrics["total_profit_pct"] += profit_pct
                    tier3_losses.append(profit_pct)
        
        # Add recently closed signals to the returned list
        if signal["status"] == "closed" and "exit_time" in signal and signal["exit_time"]:
            try:
                exit_time = pd.to_datetime(signal["exit_time"])
                if (datetime.utcnow() - exit_time).days <= 7:
                    updated_signals.append(signal)
            except:
                pass
    
    # Calculate aggregate metrics for all signals
    closed_signals = metrics["successful_signals"] + metrics["failed_signals"]
    if closed_signals > 0:
        metrics["avg_profit_pct"] = metrics["total_profit_pct"] / closed_signals
        metrics["win_rate"] = (metrics["successful_signals"] / closed_signals) * 100
        metrics["avg_win_pct"] = sum(all_wins) / len(all_wins) if all_wins else 0
        metrics["avg_loss_pct"] = sum(all_losses) / len(all_losses) if all_losses else 0
    
    # Calculate aggregate metrics for tier 1 signals
    closed_tier1_signals = tier1_metrics["successful_signals"] + tier1_metrics["failed_signals"]
    if closed_tier1_signals > 0:
        tier1_metrics["avg_profit_pct"] = tier1_metrics["total_profit_pct"] / closed_tier1_signals
        tier1_metrics["win_rate"] = (tier1_metrics["successful_signals"] / closed_tier1_signals) * 100
        tier1_metrics["avg_win_pct"] = sum(tier1_wins) / len(tier1_wins) if tier1_wins else 0
        tier1_metrics["avg_loss_pct"] = sum(tier1_losses) / len(tier1_losses) if tier1_losses else 0
    
    # Calculate aggregate metrics for tier 2 signals
    closed_tier2_signals = tier2_metrics["successful_signals"] + tier2_metrics["failed_signals"]
    if closed_tier2_signals > 0:
        tier2_metrics["avg_profit_pct"] = tier2_metrics["total_profit_pct"] / closed_tier2_signals
        tier2_metrics["win_rate"] = (tier2_metrics["successful_signals"] / closed_tier2_signals) * 100
        tier2_metrics["avg_win_pct"] = sum(tier2_wins) / len(tier2_wins) if tier2_wins else 0
        tier2_metrics["avg_loss_pct"] = sum(tier2_losses) / len(tier2_losses) if tier2_losses else 0
        
    # Calculate aggregate metrics for tier 3 signals
    closed_tier3_signals = tier3_metrics["successful_signals"] + tier3_metrics["failed_signals"]
    if closed_tier3_signals > 0:
        tier3_metrics["avg_profit_pct"] = tier3_metrics["total_profit_pct"] / closed_tier3_signals
        tier3_metrics["win_rate"] = (tier3_metrics["successful_signals"] / closed_tier3_signals) * 100
        tier3_metrics["avg_win_pct"] = sum(tier3_wins) / len(tier3_wins) if tier3_wins else 0
        tier3_metrics["avg_loss_pct"] = sum(tier3_losses) / len(tier3_losses) if tier3_losses else 0
    
    # Save performance metrics with tiers
    metrics_data = {
        "all_signals": metrics,
        "tier1_signals": tier1_metrics,
        "tier2_signals": tier2_metrics,
        "tier3_signals": tier3_metrics
    }
    
    try:
        with open(PERFORMANCE_HISTORY_FILE, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    except Exception as e:
        logger.error(f"Error saving performance metrics: {e}")
    
    return updated_signals, metrics, tier1_metrics, tier2_metrics, tier3_metrics

def adaptive_parameters(performance_metrics):
    """
    Adapt parameters based on tier 1 signal performance metrics
    
    Args:
        performance_metrics (dict): Performance metrics dictionary containing signals data
        
    Returns:
        dict: Updated parameters
    """
    # Default parameters
    params = {
        "min_score": 60,
        "stop_loss_pct": DEFAULT_STOP_LOSS_PCT,
        "take_profit_pct": DEFAULT_TAKE_PROFIT_PCT
    }
    
    # Check if we have tier 1 signal metrics
    tier1_metrics = performance_metrics.get("tier1_signals", {})
    
    # If we have enough closed tier 1 signals to make decisions
    if tier1_metrics.get("total_signals", 0) >= 5 and tier1_metrics.get("successful_signals", 0) + tier1_metrics.get("failed_signals", 0) >= 3:
        logger.info("Adapting parameters based on tier 1 signal performance")
        win_rate = tier1_metrics.get("win_rate", 50)
        avg_profit = tier1_metrics.get("avg_profit_pct", 0)
        
        # Adjust signal threshold based on win rate
        if win_rate < 40:
            # If win rate is poor, be more selective
            params["min_score"] = 75
            logger.info(f"Low win rate ({win_rate:.2f}%), increasing min_score to 75")
        elif win_rate > 70:
            # If win rate is excellent, can be less selective
            params["min_score"] = 50
            logger.info(f"High win rate ({win_rate:.2f}%), decreasing min_score to 50")
            
        # Adjust take profit based on average profit
        if avg_profit < 1.5:
            # If profit is poor, increase take profit target
            new_tp = max(MIN_TAKE_PROFIT_PCT, DEFAULT_TAKE_PROFIT_PCT * 1.3)
            params["take_profit_pct"] = new_tp
            logger.info(f"Low average profit ({avg_profit:.2f}%), increasing take_profit_pct to {new_tp:.2f}%")
        elif avg_profit > 6.0:
            # If profit is good, can reduce take profit to increase win rate
            new_tp = max(MIN_TAKE_PROFIT_PCT, DEFAULT_TAKE_PROFIT_PCT * 0.9)
            params["take_profit_pct"] = new_tp
            logger.info(f"High average profit ({avg_profit:.2f}%), decreasing take_profit_pct to {new_tp:.2f}%")
    else:
        logger.info("Insufficient tier 1 signal history for parameter adaptation")
    
    return params

def rank_opportunities(signals_df, min_score=60, min_volume_usd=50000):
    """
    Rank trading opportunities based on signal scores
    
    Args:
        signals_df (DataFrame): DataFrame with calculated signals
        min_score (float): Minimum score to consider an opportunity
        min_volume_usd (float): Minimum 5-minute USD volume
        
    Returns:
        DataFrame: Ranked opportunities
    """
    logger.info(f"Starting to rank opportunities with min_score={min_score}, min_volume_usd={min_volume_usd}")
    
    # Get the latest data point for each symbol
    latest_data = signals_df.sort_values('ticker_time').groupby('symbol').tail(1).copy()
    logger.info(f"Found {len(latest_data)} unique symbols in latest data")
    
    # Ensure datetime format for analysis and logging
    if 'ticker_time' in latest_data.columns:
        latest_data['ticker_time'] = pd.to_datetime(latest_data['ticker_time'], format='mixed', errors='coerce')
        
        # Report on freshness of data
        now = datetime.utcnow()
        latest_data['minutes_old'] = (now - latest_data['ticker_time']).dt.total_seconds() / 60
        
        # Log data freshness statistics
        very_fresh = len(latest_data[latest_data['minutes_old'] < 5])
        fresh = len(latest_data[latest_data['minutes_old'] < 10])
        stale = len(latest_data[latest_data['minutes_old'] >= 10])
        
        logger.info(f"Data freshness: {very_fresh} symbols <5min old, {fresh-very_fresh} symbols 5-10min old, {stale} symbols >10min old")
        
        # Report on the stalest symbols
        if stale > 0:
            stalest = latest_data.nlargest(min(5, stale), 'minutes_old')
            for _, row in stalest.iterrows():
                logger.info(f"Stale data: {row['symbol']} is {row['minutes_old']:.1f} minutes old from {row['ticker_time']}")
    
    # Show distribution of scores
    score_ranges = {
        "< 0": len(latest_data[latest_data['combined_score'] < 0]),
        "0-20": len(latest_data[(latest_data['combined_score'] >= 0) & (latest_data['combined_score'] < 20)]),
        "20-40": len(latest_data[(latest_data['combined_score'] >= 20) & (latest_data['combined_score'] < 40)]),
        "40-60": len(latest_data[(latest_data['combined_score'] >= 40) & (latest_data['combined_score'] < 60)]),
        "60-80": len(latest_data[(latest_data['combined_score'] >= 60) & (latest_data['combined_score'] < 80)]),
        "80-100": len(latest_data[(latest_data['combined_score'] >= 80) & (latest_data['combined_score'] <= 100)])
    }
    logger.info(f"Score distribution: {score_ranges}")
    
    # Show volume distribution
    volume_counts = {
        "< 10k": len(latest_data[latest_data['usd_vol5m'] < 10000]),
        "10k-50k": len(latest_data[(latest_data['usd_vol5m'] >= 10000) & (latest_data['usd_vol5m'] < 50000)]),
        "50k-100k": len(latest_data[(latest_data['usd_vol5m'] >= 50000) & (latest_data['usd_vol5m'] < 100000)]),
        "100k-500k": len(latest_data[(latest_data['usd_vol5m'] >= 100000) & (latest_data['usd_vol5m'] < 500000)]),
        "500k+": len(latest_data[latest_data['usd_vol5m'] >= 500000])
    }
    logger.info(f"Volume distribution: {volume_counts}")
    
    # Add a "near peak" indicator for filtering
    if 'high_proximity' in latest_data.columns:
        near_peak = len(latest_data[latest_data['high_proximity'] > 0.98])
        logger.info(f"Coins near their recent peak (>98%): {near_peak}")
        
    # For debugging, show max score and highest volume coins
    if not latest_data.empty:
        max_score_row = latest_data.loc[latest_data['combined_score'].idxmax()]
        max_volume_row = latest_data.loc[latest_data['usd_vol5m'].idxmax()]
        logger.info(f"Max score coin: {max_score_row['symbol']} with score {max_score_row['combined_score']:.2f}")
        logger.info(f"Max volume coin: {max_volume_row['symbol']} with volume ${max_volume_row['usd_vol5m']:.2f}")
    
    # EXPLICIT FILTER for coins near their peak
    # Avoid coins that are within 2% of their recent high
    peak_filter = latest_data['high_proximity'] < 0.98 if 'high_proximity' in latest_data.columns else pd.Series(True, index=latest_data.index)
    logger.info(f"Coins passing peak filter (<98% of recent high): {peak_filter.sum()}")
    
    # Filter for data freshness - only use data from last 10 minutes
    freshness_filter = latest_data['minutes_old'] < 10 if 'minutes_old' in latest_data.columns else pd.Series(True, index=latest_data.index) 
    logger.info(f"Coins passing freshness filter (<10min old): {freshness_filter.sum()}")
    
    # Filter by minimum score and volume
    score_filter = latest_data['combined_score'] >= min_score
    volume_filter = latest_data['usd_vol5m'] >= min_volume_usd
    
    logger.info(f"Coins passing score filter ({min_score}+): {score_filter.sum()}")
    logger.info(f"Coins passing volume filter (${min_volume_usd}+): {volume_filter.sum()}")
    logger.info(f"Coins passing all filters: {(score_filter & volume_filter & peak_filter & freshness_filter).sum()}")
    
    # IMPORTANT: First try our best filtering
    filtered_data = latest_data[score_filter & volume_filter & peak_filter & freshness_filter].copy()
    
    # If we have at least 20 opportunities with optimal filtering, use those
    if len(filtered_data) >= TIER_3_RANK_MAX:
        logger.info("Found sufficient opportunities with optimal filtering")
        opportunities = filtered_data
    else:
        # Try with relaxed score threshold
        logger.info(f"Optimal filtering produced fewer than {TIER_3_RANK_MAX} results, relaxing score filter")
        relaxed_score_filter = latest_data['combined_score'] >= 40  # Relaxed score threshold
        relaxed_filtered_data = latest_data[relaxed_score_filter & volume_filter & peak_filter & freshness_filter].copy()
        
        if len(relaxed_filtered_data) >= TIER_3_RANK_MAX:
            logger.info("Found sufficient opportunities with relaxed score filtering")
            opportunities = relaxed_filtered_data
        else:
            # Try with relaxed volume threshold
            logger.info(f"Relaxed score filtering produced fewer than {TIER_3_RANK_MAX} results, relaxing volume filter")
            relaxed_volume_filter = latest_data['usd_vol5m'] >= 10000  # Relaxed volume threshold
            more_relaxed_data = latest_data[relaxed_score_filter & relaxed_volume_filter & peak_filter & freshness_filter].copy()
            
            if len(more_relaxed_data) >= TIER_3_RANK_MAX:
                logger.info("Found sufficient opportunities with relaxed score and volume filtering")
                opportunities = more_relaxed_data
            else:
                # Try with just the peak filter
                logger.info(f"Relaxed filtering still produced fewer than {TIER_3_RANK_MAX} results, using only peak filter")
                peak_filtered_data = latest_data[peak_filter & freshness_filter].copy()
                
                if len(peak_filtered_data) >= TIER_3_RANK_MAX:
                    logger.info("Found sufficient opportunities with peak filter only")
                    opportunities = peak_filtered_data
                else:
                    # As a last resort, just take all data
                    logger.info(f"Using all available data to get top {TIER_3_RANK_MAX} opportunities")
                    opportunities = latest_data[freshness_filter].copy()
    
    # Create a total opportunity score that considers both signal strength and liquidity
    opportunities['liquidity_factor'] = np.tanh(opportunities['usd_vol5m'] / 1000000)
    
    # Integrate peak avoidance into opportunity score
    if 'high_proximity' in opportunities.columns:
        # Peak avoidance factor: 1.0 for coins <90% of peak, decreasing to 0.5 at peak
        opportunities['peak_avoidance'] = 1.0 - (0.5 * opportunities['high_proximity'].clip(0.9, 1.0) - 0.9) * 5
        opportunities['opportunity_score'] = opportunities['combined_score'] * (0.6 + 0.2 * opportunities['liquidity_factor'] + 0.2 * opportunities['peak_avoidance'])
    else:
        opportunities['opportunity_score'] = opportunities['combined_score'] * (0.7 + 0.3 * opportunities['liquidity_factor'])
    
    # Rank opportunities and ensure we get at least top 20 (Tier 1, 2, and 3)
    n_opportunities = min(len(opportunities), TIER_3_RANK_MAX)
    ranked = opportunities.nlargest(n_opportunities, 'opportunity_score').reset_index(drop=True)
    
    # Add rank column
    ranked['rank'] = ranked.index + 1
    
    logger.info(f"Final ranked opportunities: {len(ranked)}")
    
    # Extract price targets
    result_columns = [
        'rank', 'symbol', 'ticker_price', 'combined_score', 'opportunity_score', 
        'momentum_score', 'volatility_breakout_score', 'order_book_score',
        'volume_surge_buy_signal', 'sr_score', 'usd_vol5m', 'ticker_time',
        'entry_price', 'stop_loss_price', 'take_profit_price', 'risk_reward_ratio'
    ]
    
    # Add peak proximity to results if available
    if 'high_proximity' in ranked.columns:
        result_columns.append('high_proximity')
    if 'reversal_bonus' in ranked.columns:
        result_columns.append('reversal_bonus')
    
    # Return only columns that exist
    available_columns = [col for col in result_columns if col in ranked.columns]
    return ranked[available_columns]

def generate_signals(data):
    """
    Generate trading signals from cryptocurrency data
    
    Args:
        data (DataFrame): Raw cryptocurrency data
        
    Returns:
        DataFrame: Data with all trading signals
    """
    # Preprocess data
    df = preprocess_data(data)
    
    # Calculate all signal components
    df = calculate_momentum_score(df)
    df = calculate_volatility_breakout(df)
    df = calculate_order_book_pressure(df)
    df = calculate_volume_surge(df)
    df = calculate_support_resistance(df)
    
    # Calculate final combined score
    df = calculate_combined_signal_score(df)
    
    # Calculate price targets
    df = calculate_price_targets(df)
    
    return df

def run_signal_analysis(merged_data_path="sig_data/all_cryptos_merged.csv"):
    """
    Run complete signal analysis on merged cryptocurrency data
    
    Args:
        merged_data_path (str): Path to the merged data file
        
    Returns:
        tuple: (signals_df, ranked_opportunities, performance_metrics)
    """
    logger.info(f"Loading data from {merged_data_path}")
    try:
        data = pd.read_csv(merged_data_path)
        logger.info(f"Loaded {len(data)} rows of data")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return None, None, None
    
    # Generate signals
    logger.info("Generating trading signals")
    signals_df = generate_signals(data)
    
    # Update results for existing signals
    logger.info("Updating signal performance tracking")
    updated_signals, all_metrics, tier1_metrics, tier2_metrics, tier3_metrics = update_signal_results(signals_df)
    
    # Get adaptive parameters based on past performance of tier 1 signals
    metrics_for_adaptation = {"tier1_signals": tier1_metrics}
    params = adaptive_parameters(metrics_for_adaptation)
    logger.info(f"Using adaptive parameters: {params}")
    
    # Rank opportunities using adaptive parameters
    logger.info("Ranking trading opportunities")
    ranked = rank_opportunities(
        signals_df, 
        min_score=params["min_score"], 
        min_volume_usd=50000
    )
    
    # Track new signals
    current_time = datetime.utcnow().isoformat()
    
    # Track tier 1 signals (top 5)
    for i, row in ranked.head(TOP_RANK_COUNT).iterrows():
        track_signal(
            row['symbol'], 
            row['entry_price'], 
            row['stop_loss_price'],
            row['take_profit_price'],
            row['combined_score'],
            row['rank'],  # Ranks 1-5 = Tier 1
            current_time
        )
    
    # Track tier 2 signals (ranks 6-10)
    if len(ranked) > TOP_RANK_COUNT:
        tier2_end = min(TIER_2_RANK_MAX, len(ranked))
        for i, row in ranked.iloc[TOP_RANK_COUNT:tier2_end].iterrows():
            track_signal(
                row['symbol'], 
                row['entry_price'], 
                row['stop_loss_price'],
                row['take_profit_price'],
                row['combined_score'],
                row['rank'],  # Ranks 6-10 = Tier 2
                current_time
            )
    
    # Track tier 3 signals (ranks 11-20)
    if len(ranked) > TIER_2_RANK_MAX:
        tier3_end = min(TIER_3_RANK_MAX, len(ranked))
        for i, row in ranked.iloc[TIER_2_RANK_MAX:tier3_end].iterrows():
            track_signal(
                row['symbol'], 
                row['entry_price'], 
                row['stop_loss_price'],
                row['take_profit_price'],
                row['combined_score'],
                row['rank'],  # Ranks 11-20 = Tier 3
                current_time
            )
    
    # Save signals to file
    signals_file = "sig_data/crypto_signals.csv"
    signals_df.to_csv(signals_file, index=False)
    logger.info(f"Saved signals to {signals_file}")
    
    ranked_file = "sig_data/ranked_opportunities.csv"
    ranked.to_csv(ranked_file, index=False)
    logger.info(f"Saved ranked opportunities to {ranked_file}")
    
    # Combine metrics for return - include all tiers
    performance_metrics = {
        "all_signals": all_metrics,
        "tier1_signals": tier1_metrics,
        "tier2_signals": tier2_metrics,
        "tier3_signals": tier3_metrics
    }
    
    return signals_df, ranked, performance_metrics

if __name__ == "__main__":
    run_signal_analysis() 