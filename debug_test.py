import numpy as np
import pandas as pd
from backtester import Backtester

dates = pd.date_range('2021-01-01', periods=200, freq='B')

test_data = {
    '510050': pd.DataFrame({
        'date': dates,
        'open': np.linspace(2.0, 2.5, 200),
        'high': np.linspace(2.02, 2.52, 200),
        'low': np.linspace(1.98, 2.48, 200),
        'close': np.linspace(2.0, 2.5, 200),
        'volume': np.ones(200) * 10000000,
        'amount': np.ones(200) * 200000000
    }),
    '511010': pd.DataFrame({
        'date': dates,
        'open': np.linspace(100.0, 100.5, 200),
        'close': np.linspace(100.0, 100.5, 200),
        'volume': np.ones(200) * 5000000,
        'amount': np.ones(200) * 500000000
    })
}

test_pool = pd.DataFrame({
    'code': ['510050', '511010'],
    'name': ['上证50ETF', '国债ETF'],
    'type': ['ETF', 'ETF'],
    'list_date': '2015-01-01',
    'scale': 5000000000
})

backtester = Backtester(test_data, test_pool)
results_df = backtester.run()

print(f'初始净值: {results_df["nav"].iloc[0]}')
print(f'最终净值: {results_df["nav"].iloc[-1]}')
print(f'每日收益率统计:')
print(f'  均值: {results_df["daily_return"].mean():.6f}')
print(f'  标准差: {results_df["daily_return"].std():.6f}')
print(f'  最小值: {results_df["daily_return"].min():.6f}')
print(f'  最大值: {results_df["daily_return"].max():.6f}')

print("\n检查净值变化异常的日期:")
prev_nav = results_df["nav"].iloc[0]
for i in range(1, len(results_df)):
    row = results_df.iloc[i]
    nav_change = (row["nav"] - prev_nav) / prev_nav
    if abs(nav_change) > 0.01:
        print(f'{row["date"].date()}: nav={row["nav"]:.4f}, change={nav_change*100:.2f}%, positions={row["positions"]}')
    prev_nav = row["nav"]

print("\n前10天的持仓变化:")
for i in range(min(10, len(results_df))):
    row = results_df.iloc[i]
    print(f'{row["date"].date()}: total={row["total_value"]:.2f}, cash={row["cash"]:.2f}, positions={row["positions"]}')