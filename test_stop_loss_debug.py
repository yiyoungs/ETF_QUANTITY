import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
import logging

logging.basicConfig(level=logging.INFO)

def generate_crash_test_data():
    """生成包含大幅回撤的测试数据"""
    n_days = 500
    dates = pd.date_range('2016-01-01', periods=n_days, freq='B')
    
    etf_list = ['510050', '510300', '511010']
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    for etf_code in etf_list[:-1]:
        base_price = 3.0
        trend = np.linspace(base_price, base_price * 1.2, n_days)
        
        crash_start = int(n_days * 0.3)
        crash_end = int(n_days * 0.45)
        crash_factor = np.ones(n_days)
        crash_factor[crash_start:crash_end] = np.linspace(1, 0.5, crash_end - crash_start)
        crash_factor[crash_end:] = crash_factor[crash_end-1] * np.linspace(1, 1.4, n_days - crash_end)
        
        noise = np.random.normal(0, 0.005, n_days)
        close = trend * crash_factor + noise
        
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
        'code': etf_list[:-1],
        'name': ['上证50ETF', '沪深300ETF'],
        'type': ['ETF'] * 2,
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    print(f"生成的数据范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"崩盘期: 第 {crash_start} 天到第 {crash_end} 天")
    print(f"最大跌幅: {(1 - 0.5) * 100:.0f}%")
    
    return test_data, etf_pool

def test_stop_loss_trigger():
    """测试止损机制是否正常触发"""
    print("=" * 80)
    print("           止损机制测试")
    print("=" * 80)
    
    all_data, etf_pool = generate_crash_test_data()
    
    test_cases = [
        {'lookback_days': 60, 'trailing_stop_pct': 0.05, 'name': '5% 止损'},
        {'lookback_days': 60, 'trailing_stop_pct': 0.08, 'name': '8% 止损'},
        {'lookback_days': 60, 'trailing_stop_pct': 0.15, 'name': '15% 止损'},
    ]
    
    for case in test_cases:
        print(f"\n测试: {case['name']}")
        print(f"参数: lookback_days={case['lookback_days']}, trailing_stop_pct={case['trailing_stop_pct']*100:.0f}%")
        
        params = {
            'lookback_days': case['lookback_days'],
            'trailing_stop_pct': case['trailing_stop_pct']
        }
        
        backtester = Backtester(all_data, etf_pool, params=params)
        results_df = backtester.run()
        
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        print(f"结果: 年化收益={metrics.get('annualized_return', 0)*100:.2f}%, "
              f"夏普={metrics.get('sharpe_ratio', 0):.2f}, "
              f"最大回撤={metrics.get('max_drawdown', 0)*100:.2f}%")
        
        print(f"结果DataFrame列: {results_df.columns.tolist()}")
        print(f"前5行持仓:")
        print(results_df[['date', 'total_assets', 'positions']].head(5))

if __name__ == "__main__":
    test_stop_loss_trigger()