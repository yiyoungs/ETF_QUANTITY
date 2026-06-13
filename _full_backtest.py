"""2018-2025 完整回测：净值曲线 + 夏普比率 + 最大回撤"""
import pandas as pd
import numpy as np
import os

from config import StrategyConfig, CACHE_DIR
from backtester import Backtester

# 加载所有ETF数据
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})

# 运行回测
bt = Backtester(all_data, etf_pool)
results = bt.run()

print(f"回测完成: {len(results)} 交易日")
print(f"起始日期: {results['date'].iloc[0]}")
print(f"结束日期: {results['date'].iloc[-1]}")

# 策略指标
nav = results['total_value'].values
initial = nav[0]
final = nav[-1]
total_return = final / initial - 1

daily_returns = pd.Series(nav).pct_change().dropna().values
annualized_return = (1 + total_return) ** (252 / len(nav)) - 1
annualized_vol = daily_returns.std() * np.sqrt(252)
sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

# 最大回撤
cummax = pd.Series(nav).cummax()
drawdown = (nav - cummax) / cummax
max_dd = drawdown.min()
max_dd_start = results['date'].iloc[drawdown.idxmin()]
dd_end_idx = 0
for i in range(drawdown.idxmin(), -1, -1):
    if drawdown.iloc[i] >= 0 or i == 0:
        dd_end_idx = i
        break
max_dd_end = results['date'].iloc[dd_end_idx]

# 胜率
win_rate = (daily_returns > 0).mean()

# 沪深300基准
hs300 = all_data['hfq'].get('510300')
if hs300 is not None:
    hs300 = hs300[(hs300['date'] >= results['date'].iloc[0]) & (hs300['date'] <= results['date'].iloc[-1])]
    hs300_nav = hs300['close'].values
    hs300_total = hs300_nav[-1] / hs300_nav[0] - 1
    hs300_daily = pd.Series(hs300_nav).pct_change().dropna().values
    hs300_ann = (1 + hs300_total) ** (252 / len(hs300_nav)) - 1
    hs300_vol = hs300_daily.std() * np.sqrt(252)
    hs300_sharpe = hs300_ann / hs300_vol if hs300_vol > 0 else 0
    hs300_cummax = pd.Series(hs300_nav).cummax()
    hs300_dd = (hs300_nav - hs300_cummax) / hs300_cummax
    hs300_maxdd = hs300_dd.min()
    hs300_win = (hs300_daily > 0).mean()

print("\n" + "=" * 70)
print(f"{'指标':<25} {'本策略':>12} {'沪深300':>12}")
print("=" * 70)
print(f"{'总资产 (最终)':<25} {final:>12,.0f} {hs300_nav[-1]*initial/hs300_nav[0]:>12,.0f}")
print(f"{'总收益':<25} {total_return*100:>11.2f}% {hs300_total*100:>11.2f}%")
print(f"{'年化收益':<25} {annualized_return*100:>11.2f}% {hs300_ann*100:>11.2f}%")
print(f"{'年化波动':<25} {annualized_vol*100:>11.2f}% {hs300_vol*100:>11.2f}%")
print(f"{'夏普比率':<25} {sharpe:>12.3f} {hs300_sharpe:>12.3f}")
print(f"{'最大回撤':<25} {max_dd*100:>11.2f}% {hs300_maxdd*100:>11.2f}%")
print(f"{'日胜率':<25} {win_rate*100:>11.2f}% {hs300_win*100:>11.2f}%")
print("=" * 70)

# 分年度表现
print("\n" + "=" * 80)
print("策略分年度表现")
print("=" * 80)
print(f"{'年份':<8} {'交易日':>7} {'年度收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'日胜率':>8}")
print("-" * 80)

results_df = results.copy()
results_df['year'] = pd.to_datetime(results_df['date']).dt.year
results_df['daily_ret'] = results_df['total_value'].pct_change()

for year in sorted(results_df['year'].unique()):
    sub = results_df[results_df['year'] == year].reset_index(drop=True)
    if len(sub) < 5:
        continue
    ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
    vol = sub['daily_ret'].std() * np.sqrt(252)
    sharpe_y = ret / vol if vol > 0 else 0
    cummax = sub['total_value'].cummax()
    dd = ((sub['total_value'] - cummax) / cummax).min()
    wr = (sub['daily_ret'] > 0).mean()
    print(f"{year:<8} {len(sub):>7} {ret*100:>9.2f}% {vol*100:>9.2f}% {sharpe_y:>8.3f} {dd*100:>9.2f}% {wr*100:>7.2f}%")

print("=" * 80)

# 保存CSV供进一步分析
results_df.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_2018_2025.csv'), index=False)
print(f"\n结果已保存至 backtest_2018_2025.csv")
