import logging
import os
from datetime import datetime
import coinbase_signals as signals
import pandas as pd
import json

logger = logging.getLogger("coinbase_signals_runner")
logger.setLevel(logging.INFO)
file_handler = logging.FileHandler("logs/coinbase_signals_runner.log")
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

def format_price(price):
    """Format price with appropriate decimal places"""
    if price >= 1000:
        return f"${price:.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.8f}"

def display_performance_metrics(metrics):
    """Display performance metrics in a readable format"""
    if not metrics:
        logger.info("No performance metrics available yet")
        return
        
    # Display metrics for top-ranked signals
    top_metrics = metrics.get('top_signals', {})
    if top_metrics:
        logger.info("\nTOP 5 SIGNAL PERFORMANCE METRICS:")
        logger.info(f"Total Top Signals: {top_metrics.get('total_signals', 0)}")
        logger.info(f"Active Top Signals: {top_metrics.get('active_signals', 0)}")
        
        closed_signals = top_metrics.get('successful_signals', 0) + top_metrics.get('failed_signals', 0)
        logger.info(f"Closed Top Signals: {closed_signals}")
        
        if closed_signals > 0:
            win_rate = top_metrics.get('win_rate', 0)
            logger.info(f"Win Rate: {win_rate:.2f}%")
            
            avg_profit = top_metrics.get('avg_profit_pct', 0)
            logger.info(f"Average Profit: {avg_profit:.2f}%")
            
            if top_metrics.get('avg_win_pct') is not None:
                logger.info(f"Average Win: {top_metrics.get('avg_win_pct', 0):.2f}%")
            
            if top_metrics.get('avg_loss_pct') is not None:
                logger.info(f"Average Loss: {top_metrics.get('avg_loss_pct', 0):.2f}%")
        else:
            logger.info("No closed signals yet - metrics will be available once signals complete")
    else:
        logger.info("No top signal performance metrics available yet")

def display_recent_signals(filepath="signal_history.json"):
    """Display recently closed signals, highlighting top-ranked ones"""
    if not os.path.exists(filepath):
        logger.info("No signal history available yet")
        return
        
    try:
        with open(filepath, 'r') as f:
            history = json.load(f)
            
        if "signals" not in history or not history["signals"]:
            logger.info("No signals in history")
            return
            
        # Find recently closed signals (last 10)
        closed_signals = [s for s in history["signals"] if s["status"] == "closed"]
        closed_signals = sorted(closed_signals, key=lambda x: x.get("exit_time", ""), reverse=True)[:10]
        
        if closed_signals:
            logger.info("\nRECENT SIGNAL RESULTS:")
            logger.info(f"{'Symbol':<8} {'Rank':<6} {'Result':<8} {'Profit %':<10} {'Entry':<12} {'Exit':<12} {'Date':<12}")
            logger.info("-" * 75)
            
            for signal in closed_signals:
                rank = f"#{signal['rank']}" if 'rank' in signal else "N/A"
                result_icon = "WIN" if signal["result"] == "take_profit" else "LOSS"
                
                # Highlight top-ranked signals
                is_top = 'rank' in signal and signal['rank'] <= 5
                prefix = "TOP " if is_top else "    "
                
                exit_time = ""
                if "exit_time" in signal and signal["exit_time"]:
                    try:
                        exit_dt = pd.to_datetime(signal["exit_time"])
                        exit_time = exit_dt.strftime("%Y-%m-%d")
                    except:
                        pass
                
                logger.info(f"{prefix}{signal['symbol']:<6} {rank:<6} {result_icon:<8} {signal['profit_pct']:.2f}%      {signal['entry_price']:<12.4f} {signal['exit_price']:<12.4f} {exit_time:<12}")
        else:
            logger.info("\nNo recently closed signals")
            
    except Exception as e:
        logger.error(f"Error displaying recent signals: {e}")

def main():
    start_time = datetime.utcnow()
    logger.info("=" * 80)
    logger.info(f"STARTING COINBASE SIGNAL ANALYSIS @ {start_time.isoformat()}")
    logger.info("=" * 80)
    
    try:
        # Check if merged data file exists
        merged_data_file = "all_cryptos_merged.csv"
        if not os.path.exists(merged_data_file):
            logger.error(f"Merged data file '{merged_data_file}' not found")
            return
            
        # Run signal analysis
        signals_df, ranked, performance_metrics = signals.run_signal_analysis(merged_data_file)
        
        # Display performance metrics
        display_performance_metrics(performance_metrics)
        
        # Display recent signal results
        display_recent_signals()
        
        if ranked is not None and not ranked.empty:
            # Display top opportunities
            top_n = min(5, len(ranked))
            logger.info(f"\nTOP {top_n} TRADING OPPORTUNITIES:")
            logger.info(f"{'Rank':<4} {'Symbol':<8} {'Current Price':<16} {'Score':<8} {'Entry':<16} {'Stop Loss':<16} {'Take Profit':<16} {'Risk/Reward':<10}")
            logger.info("-" * 100)
            
            for i, row in ranked.head(top_n).iterrows():
                current_price = format_price(row['ticker_price'])
                entry = format_price(row['entry_price']) if 'entry_price' in row else "N/A"
                stop_loss = format_price(row['stop_loss_price']) if 'stop_loss_price' in row else "N/A"
                take_profit = format_price(row['take_profit_price']) if 'take_profit_price' in row else "N/A"
                risk_reward = f"{row['risk_reward_ratio']:.2f}" if 'risk_reward_ratio' in row else "N/A"
                
                logger.info(f"{int(row['rank']):<4} {row['symbol']:<8} {current_price:<16} {row['opportunity_score']:.2f} {entry:<16} {stop_loss:<16} {take_profit:<16} {risk_reward:<10}")
    
    except Exception as e:
        logger.error(f"Error during signal analysis: {str(e)}", exc_info=True)
    
    end_time = datetime.utcnow()
    duration = end_time - start_time
    logger.info("=" * 80)
    logger.info(f"FINISHED COINBASE SIGNAL ANALYSIS @ {end_time.isoformat()}")
    logger.info(f"DURATION: {duration}")
    logger.info("=" * 80)

if __name__ == "__main__":
    main() 