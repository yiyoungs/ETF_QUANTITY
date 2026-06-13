"""深入调试 _execute_buy：为什么风险资产仓位为 0%？"""
import pandas as pd
import numpy as np
import os

from config import CACHE_DIR, StrategyConfig, BacktestConfig
from backtester import Backtester
from signal_engine import get_top_momentum_stocks, calc_multi_period_momentum, filter_liquidity

# 加载缓存数据
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
codes = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        codes.append(code)
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})

# 用几个典型调仓日做 DEBUG
sample_dates = [
    '2019-06-21',  # 牛市中段
    '2020-07-03',  # 疫情后
    '2021-01-22',  # 牛市切换
    '2024-03-01',  # 牛市
    '2025-06-06',  # 最近
]

print("=" * 80)
print("调试 1：get_top_momentum_stocks 实际返回哪些标的？")
print("=" * 80)

for d in sample_dates:
    print(f"\n[ {d} ]")
    # 流动性过滤
    qualified = filter_liquidity(all_data, etf_pool, d)
    print(f"  - 流动性合格标的数: {len(qualified)}")

    # 动量计算
    mom_df = calc_multi_period_momentum(all_data, qualified, d)
    if mom_df is not None and len(mom_df) > 0:
        print(f"  - 动量计算成功, 候选: {len(mom_df)} 只")
        print(f"  - Top10动量评分:")
        top10 = mom_df.head(10)
        for _, r in top10.iterrows():
            print(f"      {r['code']}: momentum_score={r.get('momentum_score', 'N/A'):.4f}")

    # get_top_momentum_stocks
    top = get_top_momentum_stocks(all_data, etf_pool, d, top_n=5)
    print(f"  - get_top_momentum_stocks 返回: {top}")

# 调试 2：看 MA200 的市场模式判断
print()
print("=" * 80)
print("调试 2：_check_market_mode 返回")
print("=" * 80)

bt_config = BacktestConfig(start_date='2018-01-01', end_date='2025-12-31')
bt = Backtester(all_data, etf_pool, backtest_config=bt_config)

# 手动跑几个调仓日
for date_str in bt.trading_dates:
    if bt.is_rebalance_day(date_str):
        mode = bt._check_market_mode(date_str)
        # 打印沪深300 MA200状态
        idx = all_data['qfq']['510300']
        idx = idx[idx.date <= pd.to_datetime(date_str)]
        if len(idx) >= 202:
            latest = idx.iloc[-2]  # 昨日（避免 lookahead）
            ma200 = idx['close'].rolling(200).mean().iloc[-2]
            close = float(latest['close'])
            print(f"  {date_str}: mode={mode}, equity_ratio={bt.equity_ratio:.2f}, "
                  f"510300 close={close:.2f}, MA200={ma200:.2f}, 偏离={(close-ma200)/ma200*100:.1f}%")
