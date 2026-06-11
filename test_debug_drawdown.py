import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG
import logging

logging.basicConfig(level=logging.INFO)

def generate_crash_test_data():
    """生成包含2018年熊市特征的测试数据"""
    dates = pd.date_range('2016-01-01', periods=1200, freq='B')
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    n_days = len(dates)
    
    base_price = 3.0
    prices = np.ones(n_days) * base_price
    
    phase1_end = int(n_days * 0.2)
    prices[:phase1_end] = np.linspace(base_price, base_price * 1.3, phase1_end)
    
    crash_start = phase1_end
    crash_end = int(n_days * 0.5)
    prices[crash_start:crash_end] = np.linspace(base_price * 1.3, base_price * 0.6, crash_end - crash_start)
    
    recovery_start = crash_end
    recovery_end = int(n_days * 0.8)
    prices[recovery_start:recovery_end] = np.linspace(base_price * 0.6, base_price * 1.2, recovery_end - recovery_start)
    
    prices[recovery_end:] = np.linspace(base_price * 1.2, base_price * 1.4, n_days - recovery_end)
    
    prices += np.random.normal(0, 0.02, n_days)
    
    for etf_code in ['510050', '510300', '510500']:
        close = prices.copy()
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'high': close * (1 + np.random.uniform(0, 0.015, n_days)),
            'low': close * (1 + np.random.uniform(-0.015, 0, n_days)),
            'close': close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        test_data['hfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'close': close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        test_data['dividend'][etf_code] = pd.DataFrame({
            'ex_date': [],
            'dividend_per_share': []
        })
    
    test_data['qfq']['511010'] = pd.DataFrame({
        'date': dates,
        'open': 100.01 + np.random.uniform(-0.01, 0.01, n_days),
        'high': 100.02 + np.random.uniform(0, 0.01, n_days),
        'low': 100.00 + np.random.uniform(-0.01, 0, n_days),
        'close': 100.01 + np.random.uniform(-0.005, 0.005, n_days),
        'volume': np.ones(n_days) * 1000000,
        'amount': np.ones(n_days) * 100000000
    })
    
    test_data['hfq']['511010'] = test_data['qfq']['511010'][['date', 'close', 'volume', 'amount']].copy()
    test_data['dividend']['511010'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    
    etf_pool = pd.DataFrame({
        'code': ['510050', '510300', '510500'],
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF'],
        'type': ['ETF'] * 3,
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    return test_data, etf_pool

def test_drawdown_analysis():
    """测试并分析回撤问题"""
    print("=" * 80)
    print("           回撤分析测试 - 2018年熊市模拟")
    print("=" * 80)
    
    all_data, etf_pool = generate_crash_test_data()
    
    bt_config = BacktestConfig(start_date='2018-01-01', end_date='2019-06-30')
    
    params = {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params, backtest_config=bt_config)
    results_df = backtester.run()
    
    analyzer = Analyzer(results_df)
    metrics = analyzer.calculate_metrics()
    
    print(f"\n回测结果 (默认参数 60/0.08):")
    print(f"  年化收益: {metrics.get('annualized_return', 0) * 100:.2f}%")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"  盈亏比: {metrics.get('profit_factor', 0):.2f}")
    
    print("\n每日持仓变化分析:")
    print("-" * 60)
    
    prev_positions = None
    for idx, row in results_df.iterrows():
        positions = row['positions']
        date = row['date']
        
        if prev_positions is None or positions != prev_positions:
            print(f"{date}: 持仓变化 -> {dict(positions)}")
        
        prev_positions = positions
    
    print("\n资产配置时间分布:")
    equity_days = 0
    cash_days = 0
    
    for idx, row in results_df.iterrows():
        has_equity = any(code != '511010' and shares > 0 for code, shares in row['positions'].items())
        if has_equity:
            equity_days += 1
        else:
            cash_days += 1
    
    total_days = len(results_df)
    print(f"持有权益资产天数: {equity_days} ({equity_days/total_days*100:.1f}%)")
    print(f"持有现金天数: {cash_days} ({cash_days/total_days*100:.1f}%)")
    
    return results_df

if __name__ == "__main__":
    test_drawdown_analysis()