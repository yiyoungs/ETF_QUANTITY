import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
import logging

logging.basicConfig(level=logging.DEBUG)

def generate_extreme_test_data():
    """生成极端下跌的测试数据，确保止损触发"""
    n_days = 600
    dates = pd.date_range('2016-01-01', periods=n_days, freq='B')
    
    etf_list = ['510050', '511010']
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    base_price = 3.0
    
    phase1_end = 60
    phase2_end = 90
    phase3_end = n_days
    
    phase1 = np.linspace(base_price, base_price * 1.3, phase1_end)
    phase2 = np.linspace(base_price * 1.3, base_price * 0.9, phase2_end - phase1_end)
    phase3 = np.linspace(base_price * 0.9, base_price * 1.1, phase3_end - phase2_end)
    
    trend = np.concatenate([phase1, phase2, phase3])
    noise = np.random.normal(0, 0.005, n_days)
    close = trend * (1 + noise)
    
    test_data['qfq']['510050'] = pd.DataFrame({
        'date': dates,
        'open': close * (1 + np.random.uniform(-0.005, 0.005, n_days)),
        'high': close * (1 + np.random.uniform(0, 0.01, n_days)),
        'low': close * (1 + np.random.uniform(-0.01, 0, n_days)),
        'close': close,
        'volume': np.ones(n_days) * 50000000,
        'amount': np.ones(n_days) * 150000000
    })
    
    test_data['hfq']['510050'] = pd.DataFrame({
        'date': dates,
        'close': close * 1.01,
        'volume': np.ones(n_days) * 50000000,
        'amount': np.ones(n_days) * 150000000
    })
    
    test_data['dividend']['510050'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    
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
        'code': ['510050'],
        'name': ['上证50ETF'],
        'type': ['ETF'],
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    print(f"测试数据概览:")
    print(f"  日期范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"  Phase1 (上涨): 0-{phase1_end}天，价格 {base_price} -> {base_price*1.3:.2f}")
    print(f"  Phase2 (下跌): {phase1_end}-{phase2_end}天，价格 {base_price*1.3:.2f} -> {base_price*0.9:.2f} (跌幅 {((1.3-0.9)/1.3*100):.0f}%)")
    print(f"  预期止损触发: 当价格从 {base_price*1.3:.2f} 下跌 {8}% 时，即 {base_price*1.3*0.92:.2f}")
    
    return test_data, etf_pool, dates

def verify_stop_loss():
    """验证止损机制是否工作"""
    print("=" * 80)
    print("           止损机制验证测试")
    print("=" * 80)
    
    all_data, etf_pool, dates = generate_extreme_test_data()
    
    params = {'lookback_days': 30, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params)
    
    print(f"\n参数: lookback_days={backtester.lookback_days}, trailing_stop_pct={backtester.trailing_stop_pct*100:.0f}%")
    
    results_df = backtester.run()
    
    print("\n回测结果:")
    analyzer = Analyzer(results_df)
    metrics = analyzer.calculate_metrics()
    print(f"  年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    print(f"\n每日持仓分析:")
    positions_data = []
    for idx, row in results_df.iterrows():
        date = row['date']
        positions = row['positions']
        
        if isinstance(positions, dict):
            has_510050 = '510050' in positions
            has_511010 = '511010' in positions
        else:
            has_510050 = False
            has_511010 = False
        
        positions_data.append({
            'date': date,
            'has_510050': has_510050,
            'has_511010': has_511010
        })
    
    positions_df = pd.DataFrame(positions_data)
    
    crash_start = dates[60]
    crash_end = dates[90]
    
    print(f"\n下跌期间 ({crash_start.date()} 到 {crash_end.date()}):")
    crash_period = positions_df[(positions_df['date'] >= crash_start) & (positions_df['date'] <= crash_end)]
    print(f"  持有510050天数: {len(crash_period[crash_period['has_510050']])}")
    print(f"  持有511010天数: {len(crash_period[crash_period['has_511010']])}")

if __name__ == "__main__":
    verify_stop_loss()