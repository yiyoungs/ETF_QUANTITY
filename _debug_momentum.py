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

# 测试 10 个关键调仓日
test_dates = [
    '2019-01-11', '2019-06-21', '2019-12-20',
    '2020-01-10', '2020-07-03', '2020-12-25',
    '2021-01-22', '2021-06-25',
    '2024-06-28', '2025-01-10', '2025-06-06'
]

print("=" * 90)
print(f"{'日期':<12} {'流动性合格':>8} {'动量合格':>8} {'Top5 结果':<50}")
print("=" * 90)

for d in test_dates:
    equity_pool = etf_pool[etf_pool['code'] != '511010']
    qualified = filter_liquidity(all_data, equity_pool, d)
    qcodes = qualified['code'].tolist() if len(qualified) > 0 else []
    
    # 手动计算动量
    mom_df = calc_multi_period_momentum(all_data, qualified, d)
    mom_count = len(mom_df) if mom_df is not None else 0
    
    # 获取Top5
    top5 = get_top_momentum_stocks(all_data, equity_pool, d, top_n=5)
    
    # 显示前5的分数
    top5_detail = ""
    if mom_df is not None and len(mom_df) > 0:
        top = mom_df.head(5)
        parts = []
        for _, r in top.iterrows():
            parts.append(f"{r['code']}({r.get('momentum_score', '?'):.3f})")
        top5_detail = ", ".join(parts)
    
    print(f"{d:<12} {len(qcodes):>8} {mom_count:>8} {top5_detail}")
    if not top5:
        print(f"  ⚠️ WARNING: get_top_momentum_stocks 返回空列表！")

# 看一下 filter_liquidity 对所有ETF的要求
print()
print("=" * 90)
print("检查 2019-06-21 各ETF的流动性过滤")
print("=" * 90)

d = '2019-06-21'
from signal_engine import filter_liquidity
df = pd.DataFrame({'code': [c for c in all_data['qfq'].keys() if c != '511010']})

# 手动看每只ETF的情况
for code in df['code']:
    data = all_data.get('qfq', all_data)
    if isinstance(data, dict) and code in data:
        etf_data = data[code]
    else:
        continue
    etf_data = etf_data[etf_data['date'] <= pd.to_datetime(d)]
    if len(etf_data) < 60:
        print(f"{code}: 只有 {len(etf_data)} 天数据（需要60），被过滤")
    else:
        recent = etf_data.tail(60)
        avg_vol = recent['amount'].mean() if 'amount' in recent.columns else 'N/A'
        print(f"{code}: {len(etf_data)}天, 最近60天日均成交额={avg_vol:.0e}" if avg_vol != 'N/A' else f"{code}: {len(etf_data)}天")
