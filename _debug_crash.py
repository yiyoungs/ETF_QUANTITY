"""详细追踪关键熊市时期的调仓行为"""
import pandas as pd
import os
import logging

logging.basicConfig(level=logging.WARNING)

from config import CACHE_DIR, StrategyConfig, BacktestConfig, DEFAULT_BACKTEST_CONFIG
from backtester import Backtester

all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})

bt_config = BacktestConfig(start_date='2019-01-01', end_date='2025-12-31')
bt = Backtester(all_data, etf_pool, backtest_config=bt_config)

# 手动回测
print(f"{'日期':<12} {'模式':<6} {'eq_ratio':>8} {'总资产':>12} {'现金':>10} {'持仓':<40}")
print("=" * 120)

for date_str in bt.trading_dates:
    if bt.is_rebalance_day(date_str):
        bt._check_market_mode(date_str)
    
    # 止损
    stop_loss_list = bt.portfolio_manager.check_stop_loss(all_data, date_str, bt.trailing_stop_pct)
    for etf_code in stop_loss_list:
        bt.portfolio_manager.execute_stop_loss(all_data, etf_code, date_str)
    bt.portfolio_manager.update_highest_price(all_data, date_str)
    
    if bt.is_rebalance_day(date_str):
        bt.portfolio_manager.rebalance(
            all_data, etf_pool, date_str,
            periods_weights=bt.periods_weights,
            market_mode=bt.market_mode,
            equity_ratio=bt.equity_ratio
        )
    
    # 每3周打印一次或关键时期
    date_dt = pd.to_datetime(date_str)
    month = date_dt.month
    is_key_period = (date_dt.year in [2021, 2022, 2023, 2024]) and (date_dt.day <= 7)
    
    if bt.is_rebalance_day(date_str) and (is_key_period or date_dt.quarter == 1):
        total_val = bt.portfolio_manager.get_total_value(all_data, date_str)
        pos_str = ", ".join([f"{c}:{s}" for c, s in sorted(bt.portfolio_manager.positions.items())])
        print(f"{date_str} {bt.market_mode:<6} {bt.equity_ratio:>8.2f} {total_val:>12.0f} {bt.portfolio_manager.cash:>10.0f} {pos_str}")

final_val = bt.portfolio_manager.get_total_value(all_data, bt.trading_dates[-1])
print(f"\n最终价值: {final_val:.2f}")
print(f"最终净值: {final_val / StrategyConfig.INITIAL_CAPITAL:.6f}")
