"""2018-2025 分年度回测 —— 输出每年收益、回撤、夏普"""
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(os.getcwd(), 'logs', 'backtest_yearly.log'), encoding='utf-8'),
    ]
)

from config import CACHE_DIR, StrategyConfig, BacktestConfig
from backtester import Backtester

# ========= 1. 加载缓存数据 =========
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
codes = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        codes.append(code)
        hfq = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        qfq = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])
        all_data['hfq'][code] = hfq
        all_data['qfq'][code] = qfq

if StrategyConfig.CASH_ETF_CODE not in all_data['hfq']:
    all_data['hfq'][StrategyConfig.CASH_ETF_CODE] = all_data['hfq'][codes[0]]
    all_data['qfq'][StrategyConfig.CASH_ETF_CODE] = all_data['qfq'][codes[0]]

if '510300' not in all_data['hfq']:
    all_data['hfq']['510300'] = all_data['hfq'][codes[0]]
    all_data['qfq']['510300'] = all_data['qfq'][codes[0]]

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})
print(f'加载 ETF 数量: {len(etf_pool)}')
print(f'沪深300 ETF 数据范围: {all_data["hfq"]["510300"].date.min().date()} ~ {all_data["hfq"]["510300"].date.max().date()}')

# ========= 2. 执行 2018-01-01 ~ 2025-12-31 回测 =========
bt_config = BacktestConfig(start_date='2018-01-01', end_date='2025-12-31')
bt = Backtester(all_data, etf_pool, backtest_config=bt_config)
results_df = bt.run()

print(f'\n回测完成, 共 {len(results_df)} 个交易日')
print(f'净值范围: {results_df.nav.iloc[0]:.4f} → {results_df.nav.iloc[-1]:.4f}')
print(f'日期范围: {results_df.date.dt.date.iloc[0]} ~ {results_df.date.dt.date.iloc[-1]}')

# ========= 3. 分年度计算指标 =========
results_df = results_df.copy()
results_df['year'] = results_df['date'].dt.year
results_df['daily_ret'] = results_df['nav'].pct_change().fillna(0)

rows = []
for year, group in results_df.groupby('year'):
    if len(group) < 30:
        continue

    start_nav = group['nav'].iloc[0]
    end_nav = group['nav'].iloc[-1]
    annual_ret = end_nav / start_nav - 1

    group_cummax = group['nav'].cummax()
    group_dd = (group['nav'] - group_cummax) / group_cummax
    max_dd = group_dd.min()

    daily_std = group['daily_ret'].std()
    ann_vol = daily_std * np.sqrt(252)

    n_days = len(group)
    if n_days > 0 and ann_vol > 0:
        year_ret_ann = (1 + annual_ret) ** (252 / n_days) - 1
        sharpe = year_ret_ann / ann_vol
    else:
        sharpe = 0.0

    win_rate = (group['daily_ret'] > 0).sum() / len(group)
    win_days = (group['daily_ret'] > 0).sum()

    rows.append({
        '年份': int(year),
        '交易日': n_days,
        '年度收益(%)': round(annual_ret * 100, 2),
        '年化波动(%)': round(ann_vol * 100, 2),
        '夏普': round(sharpe, 2),
        '最大回撤(%)': round(max_dd * 100, 2),
        '胜率(%)': round(win_rate * 100, 1),
        '上涨天数': int(win_days),
    })

yearly_df = pd.DataFrame(rows)

# 全期指标
all_start = results_df['nav'].iloc[0]
all_end = results_df['nav'].iloc[-1]
all_ret = all_end / all_start - 1
all_cummax = results_df['nav'].cummax()
all_dd = (results_df['nav'] - all_cummax) / all_cummax
all_max_dd = all_dd.min()
all_daily_std = results_df['daily_ret'].std()
all_ann_vol = all_daily_std * np.sqrt(252)
years_val = (results_df['date'].iloc[-1] - results_df['date'].iloc[0]).days / 365.25
all_ann_ret = (1 + all_ret) ** (1 / years_val) - 1
all_sharpe = all_ann_ret / all_ann_vol if all_ann_vol > 0 else 0.0
all_win_rate = (results_df['daily_ret'] > 0).mean()

# ========= 4. 输出结果 =========
print('\n' + '=' * 110)
header = '| 年份 | 交易日 | 年度收益(%) | 年化波动(%) |  夏普 | 最大回撤(%) | 胜率(%) | 上涨天数 |'
print(header)
sep = '|------|--------|-------------|-------------|-------|-------------|---------|----------|'
print(sep)
for _, r in yearly_df.iterrows():
    print(
        f'| {int(r["年份"]):>4d} | {int(r["交易日"]):>6d} | '
        f'{r["年度收益(%)"]:>11.2f} | {r["年化波动(%)"]:>11.2f} | '
        f'{r["夏普"]:>5.2f} | {r["最大回撤(%)"]:>11.2f} | '
        f'{r["胜率(%)"]:>7.1f} | {int(r["上涨天数"]):>8d} |'
    )
print(sep)
print(
    f'| 2018-25 | {len(results_df):>6d} | '
    f'{all_ret*100:>11.2f} | {all_ann_vol*100:>11.2f} | '
    f'{all_sharpe:>5.2f} | {all_max_dd*100:>11.2f} | '
    f'{all_win_rate*100:>7.1f} | {int((results_df["daily_ret"]>0).sum()):>8d} |'
)
print('=' * 110)

print(f'\n全期统计:')
print(f'  总收益: {all_ret*100:.2f}%')
print(f'  年化收益: {all_ann_ret*100:.2f}%')
print(f'  年化波动率: {all_ann_vol*100:.2f}%')
print(f'  夏普比率: {all_sharpe:.2f}')
print(f'  最大回撤: {all_max_dd*100:.2f}%')
print(f'  胜率: {all_win_rate*100:.1f}%')
print(f'  回测年数: {years_val:.2f} 年')
print(f'  最终净值: {all_end:.4f}')

# 保存
os.makedirs('reports', exist_ok=True)
csv_path = 'reports/backtest_yearly_2018_2025.csv'
yearly_df.to_csv(csv_path, index=False, encoding='utf-8-sig')
print(f'\n分年度结果已保存至: {csv_path}')
