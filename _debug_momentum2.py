import pandas as pd
import numpy as np
import os
import logging

from config import CACHE_DIR, StrategyConfig, BacktestConfig, DEFAULT_BACKTEST_CONFIG
from signal_engine import get_top_momentum_stocks, filter_liquidity, calc_multi_period_momentum

logging.basicConfig(level=logging.INFO)

# 加载缓存数据
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

etf_pool = pd.DataFrame({'code': [c for c in all_data['hfq'].keys()]})
qfq_data = all_data['qfq']

print("MIN_AMOUNT:", StrategyConfig.MIN_AMOUNT)
print()

# 测试 15 个关键调仓日
test_dates = [
    '2019-01-11', '2019-03-08', '2019-06-21', '2019-09-13', '2019-12-20',
    '2020-01-10', '2020-04-10', '2020-07-03', '2020-10-09', '2020-12-25',
    '2021-01-22', '2021-06-25', '2024-06-28', '2025-01-10', '2025-06-06'
]

print("=" * 100)
print(f"{'日期':<12} {'流动性合格':>8} {'动量合格':>8} {'Top5':<50}")
print("=" * 100)

equity_pool_all = etf_pool[etf_pool['code'] != '511010']

for d in test_dates:
    # 调用实际的函数（flat dict）
    qualified = filter_liquidity(qfq_data, equity_pool_all, d)
    qcodes = qualified
    
    mom_df = calc_multi_period_momentum(qfq_data, qualified, d)
    mom_count = len(mom_df) if mom_df is not None and not mom_df.empty else 0
    
    top5 = get_top_momentum_stocks(qfq_data, equity_pool_all, d, top_n=5)
    
    # Top5 详细
    top5_detail = ""
    if mom_df is not None and len(mom_df) > 0:
        top = mom_df.head(5)
        parts = []
        for _, r in top.iterrows():
            score = r.get('momentum_score', float('nan'))
            parts.append(f"{r['code']}({score:.3f})")
        top5_detail = ", ".join(parts)
    
    print(f"{d:<12} {len(qcodes):>8} {mom_count:>8} {top5_detail}")
    if not top5:
        print(f"  ⚠️ WARNING: get_top_momentum_stocks 返回空列表！")
    # 检查是否因为滚动量不够
    for code in ['510050', '510300', '159915', '512880', '512000']:
        if code not in qcodes and code in qfq_data:
            df = qfq_data[code]
            mask = df['date'] <= pd.to_datetime(d)
            if mask.sum() >= 20:
                df_temp = df[mask].copy()
                rolling = df_temp['amount'].rolling(window=20).mean().shift(1).iloc[-1]
                print(f"  - {code}: 数据天数={mask.sum()}, rolling_amount={rolling:.2e}, MIN_AMOUNT={StrategyConfig.MIN_AMOUNT:.2e}, 合格={rolling >= StrategyConfig.MIN_AMOUNT}")

# 检查 StrategyConfig 关键参数
print()
print("=" * 60)
print("策略参数:")
print(f"  MIN_AMOUNT = {StrategyConfig.MIN_AMOUNT}")
print(f"  TARGET_HOLDINGS = {StrategyConfig.TARGET_HOLDINGS}")
print(f"  MAX_CANDIDATES = {StrategyConfig.MAX_CANDIDATES}")
print(f"  CASH_ETF_CODE = {StrategyConfig.CASH_ETF_CODE}")
