import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO)

from config import CACHE_DIR, StrategyConfig, BacktestConfig, DEFAULT_BACKTEST_CONFIG
from backtester import Backtester
from signal_engine import get_top_momentum_stocks

# 加载缓存数据
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})

bt_config = BacktestConfig(start_date='2019-01-01', end_date='2025-12-31')
bt = Backtester(all_data, etf_pool, backtest_config=bt_config)

print("=" * 100)
print("真实回测：追踪每日市场模式与实际持仓")
print("=" * 100)

# 手动模拟回测流程，但增加输出
bt.portfolio_manager.positions = {}
bt.portfolio_manager.cash = StrategyConfig.INITIAL_CAPITAL
bt.market_mode = 'bear'
bt.equity_ratio = 0.0
bt.bull_signal_count = 0
bt.bear_signal_count = 0
bt.initial_rebalance_done = False

print(f"{'日期':<12} {'周调仓':<5} {'模式':<6} {'eq_ratio':>8} {'现金':>10} {'持仓':<60}")
print("-" * 100)

for date_str in bt.trading_dates:
    if bt.is_rebalance_day(date_str):
        bt._check_market_mode(date_str)
        
    # 止损
    stop_loss_list = bt.portfolio_manager.check_stop_loss(all_data, date_str)
    for etf_code in stop_loss_list:
        bt.portfolio_manager.execute_stop_loss(all_data, etf_code, date_str)
    bt.portfolio_manager.update_highest_price(all_data, date_str)
    
    is_rebal = bt.is_rebalance_day(date_str)
    
    if is_rebal:
        if bt.initial_rebalance_done:
            bt._weekly_rebalance(date_str, equity_ratio=bt.equity_ratio)
        else:
            bt.portfolio_manager.rebalance(
                all_data, etf_pool, date_str,
                periods_weights=bt.periods_weights,
                market_mode=bt.market_mode,
                equity_ratio=bt.equity_ratio
            )
            bt.initial_rebalance_done = True
    
    # 每周五打印一次
    if is_rebal:
        pos_str = ", ".join([f"{c}:{s}" for c, s in sorted(bt.portfolio_manager.positions.items())])
        total_val = bt.portfolio_manager.get_total_value(all_data, date_str)
        print(f"{date_str} {'  OK':<5} {bt.market_mode:<6} {bt.equity_ratio:>8.2f} {bt.portfolio_manager.cash:>10.0f} {pos_str}")

# 最终结果
print()
print("=" * 100)
final_val = bt.portfolio_manager.get_total_value(all_data, bt.trading_dates[-1])
final_nav = final_val / StrategyConfig.INITIAL_CAPITAL
years = (pd.to_datetime(bt.trading_dates[-1]) - pd.to_datetime(bt.trading_dates[0])).days / 365.25
print(f"最终净值: {final_nav:.4f}")
print(f"年化收益: {(final_nav ** (1/years) - 1) * 100:.2f}%")
print(f"交易日: {len(bt.trading_dates)}")
