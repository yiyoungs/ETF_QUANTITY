import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from config import BacktestConfig
import logging

logging.basicConfig(level=logging.WARNING)

def generate_stress_test_data():
    """生成极端压力测试数据，确保止损能被触发"""
    dates = pd.date_range('2016-01-01', periods=1000, freq='B')
    n_days = len(dates)
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    base_price = 3.0
    prices = np.ones(n_days) * base_price
    
    prices[:300] = np.linspace(base_price, base_price * 1.5, 300)
    
    crash_start = 300
    crash_end = 400
    prices[crash_start:crash_end] = np.linspace(base_price * 1.5, base_price * 0.7, crash_end - crash_start)
    
    prices[400:] = np.linspace(base_price * 0.7, base_price * 1.3, n_days - 400)
    
    prices += np.random.normal(0, 0.01, n_days)
    
    for etf_code in ['510050', '510300', '510500']:
        close = prices.copy()
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.uniform(-0.005, 0.005, n_days)),
            'high': close * (1 + np.random.uniform(0, 0.008, n_days)),
            'low': close * (1 + np.random.uniform(-0.008, 0, n_days)),
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

def validate_parameter_sensitivity():
    """验证参数敏感性测试是否有效"""
    print("=" * 80)
    print("           参数敏感性验证测试")
    print("=" * 80)
    
    all_data, etf_pool = generate_stress_test_data()
    
    original_start = BacktestConfig.START_DATE
    original_end = BacktestConfig.END_DATE
    
    BacktestConfig.START_DATE = '2017-06-01'
    BacktestConfig.END_DATE = '2018-12-31'
    
    param_combinations = [
        {'lookback_days': 60, 'trailing_stop_pct': 0.05},
        {'lookback_days': 60, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.15},
        {'lookback_days': 60, 'trailing_stop_pct': 0.30},
    ]
    
    results = []
    
    print("\n测试不同止损阈值的效果:")
    print("-" * 60)
    
    for params in param_combinations:
        ts = params['trailing_stop_pct']
        lb = params['lookback_days']
        
        print(f"\n测试: lookback={lb}, stop={ts*100:.0f}%")
        
        backtester = Backtester(all_data, etf_pool, params=params)
        results_df = backtester.run()
        
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        print(f"  年化收益: {metrics.get('annualized_return', 0) * 100:.2f}%")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        print(f"  盈亏比: {metrics.get('profit_factor', 0):.2f}")
        
        stop_loss_count = 0
        
        print(f"  止损触发次数: 需要从日志统计")
        
        results.append({
            'trailing_stop': ts,
            'return': metrics.get('annualized_return', 0),
            'drawdown': metrics.get('max_drawdown', 0),
            'sharpe': metrics.get('sharpe_ratio', 0)
        })
    
    print("\n" + "=" * 80)
    print("           参数敏感性测试结果对比")
    print("=" * 80)
    print(f"{'止损阈值':<12} {'年化收益':<12} {'夏普比率':<12} {'最大回撤':<12}")
    print("-" * 60)
    
    all_same = True
    first_drawdown = results[0]['drawdown']
    
    for r in results:
        print(f"{r['trailing_stop']*100:<10.0f}%   {r['return']*100:>10.2f}%  {r['sharpe']:>12.2f}  {r['drawdown']*100:>12.2f}%")
        if abs(r['drawdown'] - first_drawdown) > 0.01:
            all_same = False
    
    print("\n" + "=" * 80)
    if all_same:
        print("  ⚠️ 警告：所有参数组合结果相同！参数传递可能有问题")
    else:
        print("  ✅ 参数传递正常，不同止损阈值产生不同结果")
    print("=" * 80)
    
    BacktestConfig.START_DATE = original_start
    BacktestConfig.END_DATE = original_end

if __name__ == "__main__":
    validate_parameter_sensitivity()