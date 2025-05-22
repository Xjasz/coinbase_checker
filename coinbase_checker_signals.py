import io
import logging
import os
import random
import sys
from datetime import datetime, timedelta
from glob import glob
from dateutil.parser import isoparse
import numpy as np
import pandas as pd
import requests
import coinbase_signals 

logger = logging.getLogger("coinbase_checker")
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler("logs/coinbase_checker.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
console_handler.setLevel(logging.DEBUG)
logger.addHandler(console_handler)

# Constants
API_URL = "https://api.exchange.coinbase.com"
DATA_DIR = "cb_data"
VOLUME_LEADERBOARD = "sig_data/volume_leaderboard.csv"
os.makedirs(DATA_DIR, exist_ok=True)

VOLUME_24HOUR_THRESHOLD = 500000

EXCLUDED_PRODUCTS = {"LIT-USD","DAR-USD","PRQ-USD","GAL-USD","ORN-USD","MOVE-USD","00-USD"}

def remove_csv_for(product_id):
    fn = os.path.join(DATA_DIR, f"{product_id.replace('-', '_')}.csv")
    if os.path.exists(fn):
        os.remove(fn)
        logger.info(f"Removed old data file for excluded product {product_id}: {fn}")

def get_products():
    r = requests.get(f"{API_URL}/products")
    r.raise_for_status()
    prods = r.json()
    return [p for p in prods if p["quote_currency"] == "USD"and not p["trading_disabled"]and p["id"] not in EXCLUDED_PRODUCTS]

def get_ticker(product_id):
    r = requests.get(f"{API_URL}/products/{product_id}/ticker")
    r.raise_for_status()
    return r.json()

def get_order_book(product_id, level=2):
    r = requests.get(f"{API_URL}/products/{product_id}/book?level={level}")
    r.raise_for_status()
    return r.json()

def get_5min_window(now=None):
    now = now or datetime.utcnow()
    floored_minute = (now.minute // 5) * 5
    end = now.replace(minute=floored_minute, second=0, microsecond=0)
    start = end - timedelta(minutes=5)
    return start, end

def get_candle(product_id, granularity=300):
    start_dt, end_dt = get_5min_window()
    start = start_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    end = end_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    params = {
        "start": start,
        "end": end,
        "granularity": granularity
    }
    url = f"{API_URL}/products/{product_id}/candles"
    try:
        r = requests.get(url, params=params)
        r.raise_for_status()
        results = r.json()
        if not results:
            logger.info(f"No candle data for {product_id} at {end}, falling back to ticker")
            return None
    except Exception as e:
        logger.warning(f"Candle fetch error {product_id} [{params}]: {e}")
        return None
    ts, low, high, open_, close_, vol = results[-1]
    return {"candle_start": datetime.utcfromtimestamp(ts), "candle_end": datetime.utcfromtimestamp(ts+granularity),"open":open_,"high":high,"low":low,"close":close_,"volume":vol}

def load_volume_leaderboard():
    if os.path.exists(VOLUME_LEADERBOARD):
        return pd.read_csv(VOLUME_LEADERBOARD, parse_dates=["last_pull"])
    return pd.DataFrame(columns=["product_id", "usd_vol5m", "last_pull"])

def save_volume_leaderboard(df):
    df.to_csv(VOLUME_LEADERBOARD, index=False)

def should_pull(index, last_pull):
    now = datetime.utcnow()
    current_rank = index + 1
    if pd.isnull(last_pull):
        info_text = f"Missing last_pull -> FORCING PULL"
        return True, info_text
    time_since = now - last_pull
    if time_since <= timedelta(minutes=5):
        return False, None
    if time_since > timedelta(hours=12):
        info_text = f"Last pull {time_since} ago -> FORCING PULL"
        return True, info_text
    if current_rank < 25:
        info_text = "Top 25 -> ALWAYS PULL"
        return True, info_text
    decrement = ((current_rank - 25) // 5) * 5
    probability = max(100 - decrement, 0)
    if probability < 1:
        return False, None
    roll = random.randint(1, 100)
    decision = roll <= probability
    info_text = f"Last pull: {last_pull} | Chance: {probability}% | Rolled: {roll} -> {'PULL' if decision else 'SKIP'}"
    return decision, info_text

def track_coin(product_id, is_new=False, rank=None, extra_info=None):
    now = datetime.utcnow()
    ticker = get_ticker(product_id)
    book = get_order_book(product_id)
    candle = get_candle(product_id)
    ticker_dt = isoparse(ticker.get('time')).replace(tzinfo=None)
    isoticker_time = ticker_dt.isoformat(timespec="milliseconds")
    full_candle = False
    ticker_price = float(ticker.get('price', 0))
    volume24h = float(ticker.get('volume', 0))
    open_c = ticker_price if not candle else float(candle.get('open', 0))
    high_c = ticker_price if not candle else float(candle.get('high', 0))
    low_c = ticker_price if not candle else float(candle.get('low', 0))
    close_c = ticker_price if not candle else float(candle.get('close', 0))
    vol5m = 0.0 if not candle else float(candle.get('volume', 0.0))
    candle_start = isoticker_time if not candle else candle.get('candle_start').isoformat(timespec="milliseconds")
    candle_end = isoticker_time if not candle else candle.get('candle_end').isoformat(timespec="milliseconds")
    if candle:
        dt_candle_end = datetime.fromisoformat(candle_end)
        dt_ticker_plus = ticker_dt + timedelta(minutes=1)
        full_candle = True if dt_candle_end <= dt_ticker_plus else False
    usd_vol24h = ticker_price * volume24h
    usd_vol5m = ticker_price * vol5m
    bids = [(float(b[0]), float(b[1])) for b in book['bids'] if abs(float(b[0]) - ticker_price) / ticker_price <= 0.1]
    asks = [(float(a[0]), float(a[1])) for a in book['asks'] if abs(float(a[0]) - ticker_price) / ticker_price <= 0.1]
    best_bid = float(book['bids'][0][0]) if book['bids'] else 0
    best_ask = float(book['asks'][0][0]) if book['asks'] else 0
    spread = best_ask - best_bid if best_ask and best_bid else 0
    mid_price = (best_ask + best_bid) / 2 if best_ask and best_bid else 0
    total_bid_volume = sum([b[1] for b in bids])
    total_ask_volume = sum([a[1] for a in asks])
    record = {
        "request_time": now.isoformat(timespec="milliseconds"),
        "ticker_time": isoticker_time,
        "candle_start": candle_start,
        "candle_end": candle_end,
        "full_candle": full_candle,
        "ticker_price": ticker_price,
        "open": open_c,
        "high": high_c,
        "low": low_c,
        "close": close_c,
        "volume24h": volume24h,
        "vol5m": vol5m,
        "usd_vol24h": usd_vol24h,
        "usd_vol5m": usd_vol5m,
        "bid_depth": len(bids),
        "ask_depth": len(asks),
        "best_bid": best_bid,
        "best_ask": best_ask,
        "spread": spread,
        "mid_price": mid_price,
        "total_bid_volume": total_bid_volume,
        "total_ask_volume": total_ask_volume
    }
    df = pd.DataFrame([record])
    file_path = os.path.join(DATA_DIR, f"{product_id.replace('-', '_')}.csv")
    df.to_csv(file_path, mode='a', index=False, header=not os.path.exists(file_path))
    return record

def clean_dupes():
    logger.info("clean_dupes...")
    for path in glob(os.path.join(DATA_DIR, "*.csv")):
        df = pd.read_csv(path)
        df['ticker_time'] = pd.to_datetime(df['ticker_time'], format='mixed', errors='coerce')
        df_cleaned = df.sort_values('ticker_time').groupby('candle_start', as_index=False).last()
        df_cleaned = df_cleaned[df.columns]
        if len(df) != len(df_cleaned):
            logger.info(f"{os.path.basename(path)}: original = {len(df)}, cleaned = {len(df_cleaned)}")
            df_cleaned.to_csv(path, index=False)

def merge_active_data():
    logger.info("merge_active_data...")
    files = glob(os.path.join(DATA_DIR, "*.csv"))
    dfs = []
    for f in files:
        df = pd.read_csv(f)
        symbol = os.path.splitext(os.path.basename(f))[0].split("_")[0]
        df["symbol"] = symbol
        dfs.append(df)
    all_data = pd.concat(dfs, ignore_index=True)
    all_data.sort_values(["symbol", "ticker_time"], inplace=True)
    all_data["pct_change"] = all_data.groupby("symbol")["ticker_price"].pct_change()
    all_data = all_data.copy()
    all_data["pct_change"] = all_data["pct_change"].fillna(0.0)
    
    # Ensure datetime format for ticker_time
    all_data['ticker_time'] = pd.to_datetime(all_data['ticker_time'], format='mixed', errors='coerce')
    
    # Filter out symbols without recent data (last 10 minutes)
    recent_cutoff = datetime.utcnow() - timedelta(minutes=10)
    logger.info(f"Filtering out symbols without data since {recent_cutoff.isoformat()}")
    
    # Get most recent data point for each symbol
    latest_data = all_data.sort_values('ticker_time').groupby('symbol').tail(1)
    
    # Find symbols with recent data
    fresh_symbols = latest_data[latest_data['ticker_time'] >= recent_cutoff]['symbol'].unique()
    logger.info(f"Found {len(fresh_symbols)} symbols with data in the last 10 minutes")
    
    # Filter all_data to only include fresh symbols
    fresh_data = all_data[all_data['symbol'].isin(fresh_symbols)].copy()
    logger.info(f"Filtered from {len(all_data)} to {len(fresh_data)} rows by recency")
    
    # Now apply the volume threshold filter
    active_data = fresh_data[fresh_data['usd_vol24h'] >= VOLUME_24HOUR_THRESHOLD].copy()
    logger.info(f"Further filtered to {len(active_data)} rows by volume threshold")
    
    # Apply the 48-hour cutoff and minimum data point requirements
    cutoff = datetime.utcnow() - timedelta(hours=48)
    recent = active_data[active_data['ticker_time'] >= cutoff]
    good_symbols = recent['symbol'].value_counts()[lambda s: s >= 20].index
    active_data = recent[recent['symbol'].isin(good_symbols)].copy()
    logger.info(f"Final dataset contains {len(active_data)} rows across {len(good_symbols)} symbols")
    
    active_data.sort_values(['symbol', 'ticker_time'], inplace=True)
    active_data.to_csv(os.path.join("sig_data/all_cryptos_merged.csv"), index=False)
    return active_data

def run_checker():
    logger.info("run_checker...")
    all_products = get_products()
    for bad in EXCLUDED_PRODUCTS:
        remove_csv_for(bad)

    volume_leaders_df = load_volume_leaderboard()
    volume_leaders_df = volume_leaders_df[~volume_leaders_df["product_id"].isin(EXCLUDED_PRODUCTS)].reset_index(drop=True)

    product_ids = [p["id"] for p in all_products]
    existing_ids = set(volume_leaders_df["product_id"])
    new_ids = [pid for pid in product_ids if pid not in existing_ids]
    new_records = []
    for pid in new_ids:
        c_record = track_coin(pid, is_new=True)
        usd_vol5m = c_record['usd_vol5m']
        candle_start = c_record['candle_start']
        new_records.append({"product_id": pid, "usd_vol5m": usd_vol5m, "last_pull": candle_start})

    if new_records:
        new_df = pd.DataFrame(new_records)
        volume_leaders_df = pd.concat([volume_leaders_df, new_df], ignore_index=True)

    volume_leaders_df.sort_values(by="usd_vol5m", ascending=False, inplace=True)
    save_volume_leaderboard(volume_leaders_df)

    updated_rows = []
    for idx, row in enumerate(volume_leaders_df.itertuples()):
        pid = row.product_id
        last_pull = row.last_pull
        last_pull_dt = pd.to_datetime(last_pull) if pd.notnull(last_pull) else None
        try:
            result, result_text = should_pull(idx, last_pull_dt)
            if result:
                coin_result = track_coin(pid, rank=idx, extra_info=result_text)
                usd_vol5m = coin_result['usd_vol5m']
                candle_start = coin_result['candle_start']
                updated_rows.append((idx, usd_vol5m, candle_start))
        except Exception as e:
            logger.error(f"Error tracking {pid}: {str(e)}")

    for idx, usd_vol5m, candle_start in updated_rows:
        volume_leaders_df.at[idx, "usd_vol5m"] = usd_vol5m
        volume_leaders_df.at[idx, "last_pull"] = candle_start

    volume_leaders_df.sort_values(by="usd_vol5m", ascending=False, inplace=True)
    clean_dupes()
    save_volume_leaderboard(volume_leaders_df)
    merge_active_data()

    logger.info("Running signal analysis after data collection...")
    signals_df, ranked, performance_metrics = coinbase_signals.run_signal_analysis()
    
    if ranked is None or ranked.empty:
        logger.info("\nNO TRADING OPPORTUNITIES FOUND")
    else:
        top_n = min(5, len(ranked))
        logger.info(f"TOP {top_n} TRADING OPPORTUNITIES:")
        header = f"{'Rank':<4} {'Symbol':<8} {'Current Price':<16} {'Score':<8} {'Entry':<16} {'Stop Loss':<16} {'Take Profit':<16} {'Risk/Reward':<10}"
        
        # Add peak proximity and reversal columns if available
        if 'high_proximity' in ranked.columns:
            header += f" {'Peak %':<8}"
        if 'reversal_bonus' in ranked.columns:
            header += f" {'Reversal':<8}"
            
        logger.info(header)
        logger.info("-" * (100 + ('high_proximity' in ranked.columns) * 9 + ('reversal_bonus' in ranked.columns) * 9))
        
        for i, row in ranked.head(top_n).iterrows():
            current_price = f"${row['ticker_price']:.4f}" if row['ticker_price'] < 1000 else f"${row['ticker_price']:.2f}"
            entry = f"${row['entry_price']:.4f}" if 'entry_price' in row and row['entry_price'] < 1000 else "N/A"
            stop_loss = f"${row['stop_loss_price']:.4f}" if 'stop_loss_price' in row and row['stop_loss_price'] < 1000 else "N/A"
            take_profit = f"${row['take_profit_price']:.4f}" if 'take_profit_price' in row and row['take_profit_price'] < 1000 else "N/A"
            risk_reward = f"{row['risk_reward_ratio']:.2f}" if 'risk_reward_ratio' in row else "N/A"
            
            output = f"{int(row['rank']):<4} {row['symbol']:<8} {current_price:<16} {row['opportunity_score']:.2f} {entry:<16} {stop_loss:<16} {take_profit:<16} {risk_reward:<10}"
            
            # Add peak proximity percentage if available
            if 'high_proximity' in row and pd.notnull(row['high_proximity']):
                peak_pct = f"{row['high_proximity']*100:.1f}%"
                output += f" {peak_pct:<8}"
                
            # Add reversal indicator if available
            if 'reversal_bonus' in row and pd.notnull(row['reversal_bonus']):
                reversal = "YES" if row['reversal_bonus'] > 0 else "NO"
                output += f" {reversal:<8}"
                
            logger.info(output)

run_start = datetime.utcnow().isoformat()
logger.info("=" * 80)
logger.info(f"STARTING CB CHECKER  @ {run_start}")
logger.info("=" * 80)
run_checker()
run_end = datetime.utcnow().isoformat()
logger.info("=" * 80)
duration = datetime.utcnow() - datetime.fromisoformat(run_start)
logger.info(f"FINISHED CB CHECKER @ {run_end}  DURATION: {duration}")
logger.info("=" * 80 + "\n")
