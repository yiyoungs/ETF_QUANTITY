import numpy as np
import pandas as pd
from portfolio_manager import PortfolioManager

dates = pd.date_range('2018-01-01', periods=200, freq='B')
test_data = {
    '510050': pd.DataFrame({
        'date': dates,
        'open': np.linspace(2.0, 3.0, 200),
        'high': np.linspace(2.02, 3.02, 200),
        'low': np.linspace(1.98, 2.98, 200),
        'close': np.linspace(2.0, 3.0, 200),
        'volume': np.ones(200) * 10000000,
        'amount': np.ones(200) * 200000000
    }),
    '511010': pd.DataFrame({
        'date': dates,
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

pm = PortfolioManager()
print(f'初始现金: {pm.cash}')

# 第一次调仓
pm.rebalance(test_data, test_pool, '2018-01-01')
print(f'第一次调仓后现金: {pm.cash}')
print(f'持仓: {pm.positions}')
total = pm.get_total_value(test_data, '2018-01-01')
print(f'第一次调仓后总价值: {total}')

# 第二天价值
total = pm.get_total_value(test_data, '2018-01-02')
print(f'第二天总价值: {total}')

# 第二次调仓（假设是周五）
pm.rebalance(test_data, test_pool, '2018-01-05')
print(f'第二次调仓后现金: {pm.cash}')
print(f'持仓: {pm.positions}')
total = pm.get_total_value(test_data, '2018-01-05')
print(f'第二次调仓后总价值: {total}')

# 查看持仓价值计算
for code, shares in pm.positions.items():
    df = test_data[code]
    price = df[df['date'] <= pd.to_datetime('2018-01-05')]['close'].iloc[-1]
    value = shares * price
    print(f'{code}: {shares} 股 @ {price:.2f} = {value:.2f}')
