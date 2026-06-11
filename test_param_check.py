import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
import logging
logger = logging.getLogger(__name__)

def generate_simple_test_data(n_days=1200):
    """生成简单测试数据，包含明显的趋势变化"""
    dates = pd.date_range('2016-01-01', periods=n_days, freq='B')
    
    etf_list = ['510050', '510300', '510500', '511010']
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    for i, etf_code in enumerate(etf_list[:-1]):
        base_price = 3.0 + i
        trend = np.linspace(base_price, base_price * 1.5, n_days)
        
        crash_start = int(n_days * 0.3)
        crash_end = int(n_days * 0.4)
        crash_factor = np.ones(n_days)
        crash_factor[crash_start:crash_end] = np.linspace(1, 0.7, crash_end - crash_start)
        crash_factor[crash_end:] = crash_factor[crash_end-1] * np.linspace(1, 1.3, n_days - crash_end)
        
        noise = np.random.normal(0, 0.01, n_days)
        close = trend * crash_factor + noise
        
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.uniform(-0.005, 0.005, n_days)),
            'high': close * (1 + np.random.uniform(0, 0.008, n_days)),
            'low': close * (1 + np.random.uniform(-0.008, 0, n_days)),
            'close': close,
            'volume': np.ones(n_days) * 10000000,
            'amount': np.ones(n_days) * 200000000
        })
        
        test_data['hfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'close': close * 1.02,
            'volume': np.ones(n_days) * 10000000,
            'amount': np.ones(n_days) * 200000000
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
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF'],
        'type': ['ETF'] * 3,
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    return test_data, etf_pool

def test_parameter_sensitivity():
    """测试参数敏感性"""
    print("=" * 80)
    print("           参数敏感性测试 - 验证参数传递是否正确")
    print("=" * 80)
    
    all_data, etf_pool = generate_simple_test_data()
    
    param_combinations = [
        {'lookback_days': 40, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.08},
        {'lookback_days': 80, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.15},
    ]
    
    results = []
    
    for params in param_combinations:
        lb = params['lookback_days']
        ts = params['trailing_stop_pct']
        
        print(f"\n测试参数: lookback_days={lb}, trailing_stop_pct={ts*100:.0f}%")
        
        backtester = Backtester(all_data, etf_pool, params=params)
        print(f"  Backtester.lookback_days: {backtester.lookback_days}")
        print(f"  Backtester.trailing_stop_pct: {backtester.trailing_stop_pct}")
        print(f"  PortfolioManager.lookback_days: {backtester.portfolio_manager.lookback_days}")
        print(f"  PortfolioManager.trailing_stop_pct: {backtester.portfolio_manager.trailing_stop_pct}")
        
        results_df = backtester.run()
        
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        print(f"  结果: 年化收益={metrics.get('annualized_return', 0)*100:.2f}%, "
              f"夏普={metrics.get('sharpe_ratio', 0):.2f}, "
              f"最大回撤={metrics.get('max_drawdown', 0)*100:.2f}%")
        
        results.append({
            'lookback': lb,
            'stop': ts,
            'return': metrics.get('annualized_return', 0),
            'sharpe': metrics.get('sharpe_ratio', 0),
            'drawdown': metrics.get('max_drawdown', 0)
        })
    
    print("\n" + "=" * 80)
    print("           参数敏感性测试结果汇总")
    print("=" * 80)
    print(f"{'参数组合':<25} {'年化收益':<12} {'夏普比率':<12} {'最大回撤':<12}")
    print("-" * 80)
    
    all_same = True
    first_return = results[0]['return']
    
    for r in results:
        params_str = f"lookback={r['lookback']}, stop={r['stop']*100:.0f}%"
        print(f"{params_str:<25} {r['return']*100:>10.2f}% {r['sharpe']:>12.2f} {r['drawdown']*100:>12.2f}%")
        if abs(r['return'] - first_return) > 0.001:
            all_same = False
    
    print("\n" + "=" * 80)
    if all_same:
        print("  ⚠️ 警告：所有参数组合结果相同！参数传递可能有问题")
    else:
        print("  ✅ 参数传递正常，不同参数产生不同结果")
    print("=" * 80)

if __name__ == "__main__":
    test_parameter_sensitivity()