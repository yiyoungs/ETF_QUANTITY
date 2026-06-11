import pandas as pd
import os

# 检查2018-10-12附近的持仓情况
results = pd.read_csv('reports/backtest_results_fixed.csv', parse_dates=['date'])

# 找到2018-10-08和2018-10-12的记录
mask = (results['date'] >= '2018-10-08') & (results['date'] <= '2018-10-15')
subset = results[mask]

print('2018-10-08 ~ 2018-10-15 期间的每日状态:')
print('='*100)
for _, row in subset.iterrows():
    print(f"日期: {row['date'].strftime('%Y-%m-%d')}")
    print(f"  总资产: {row['total_value']:,.2f}")
    print(f"  现金: {row['cash']:,.2f}")
    print(f"  持仓: {row['positions']}")
    print(f"  日收益率: {row['daily_return']:.2%}")
    print()

# 检查持仓是否有过大的市值
print("\n检查持仓明细:")
for _, row in subset.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    positions = eval(row['positions'])
    print(f"{date_str}: {positions}")
