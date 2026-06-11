import pandas as pd

# 读取回测结果
results = pd.read_csv('reports/backtest_results_fixed.csv', parse_dates=['date'])

# 检查2018-10-08到2018-10-15的数据
mask = (results['date'] >= '2018-10-08') & (results['date'] <= '2018-10-15')
subset = results[mask]

print("="*80)
print("2018-10-08 ~ 2018-10-15 每日状态详情")
print("="*80)

for _, row in subset.iterrows():
    date_str = row['date'].strftime('%Y-%m-%d')
    positions = eval(row['positions'])
    
    print(f"\n【{date_str}】")
    print(f"  总资产: {row['total_value']:>15,.2f}")
    print(f"  现金:   {row['cash']:>15,.2f}")
    print(f"  日收益率: {row['daily_return']:>10.2%}")
    print(f"  持仓明细:")
    for code, shares in positions.items():
        print(f"    - {code}: {shares:>8} 份")

# 计算现金变化
print("\n" + "="*80)
print("现金变化分析")
print("="*80)
for i in range(1, len(subset)):
    prev = subset.iloc[i-1]
    curr = subset.iloc[i]
    cash_change = curr['cash'] - prev['cash']
    value_change = curr['total_value'] - prev['total_value']
    print(f"{prev['date'].strftime('%Y-%m-%d')} -> {curr['date'].strftime('%Y-%m-%d')}: 现金变化 {cash_change:>+15,.2f}, 总资产变化 {value_change:>+15,.2f}")

# 检查510050, 510300, 512880的持仓变化
print("\n" + "="*80)
print("持仓变化追踪")
print("="*80)
for i in range(len(subset)):
    row = subset.iloc[i]
    positions = eval(row['positions'])
    for code in ['510050', '510300', '512880', '511010']:
        if code in positions:
            print(f"{row['date'].strftime('%Y-%m-%d')}: {code} = {positions[code]} 份")
