import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from config import BacktestConfig, StrategyConfig
import logging

logging.basicConfig(level=logging.INFO)

def generate_simple_crash_data():
    """生成简单的崩盘数据"""
    dates = pd.date_range('2024-01-01', periods=100, freq='B')
    n_days = len(dates)
    
    np.random.seed(42)
    
    prices = np.linspace(4.0, 4.5, 30)
    prices = np.concatenate([prices, np.linspace(4.5, 3.6, 40)])
    prices = np.concatenate([prices, np.linspace(3.6, 4.2, n_days - 70)])
    
    prices += np.random.normal(0, 0.005, n_days)
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
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
    
    test_data['dividend']['510050'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    test_data['dividend']['510300'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    test_data['dividend']['510500'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    
    test_data['qfq']['511010'] = pd.DataFrame({
        'date': dates,
        'close': np.linspace(100.0, 100.5, n_days),
        'volume': np.ones(n_days) * 1000000,
        'amount': np.ones(n_days) * 100000000
    })
    
    test_data['hfq']['511010'] = test_data['qfq']['511010'][['date', 'close', 'volume', 'amount']].copy()
    test_data['dividend']['511010'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    
    etf_pool = pd.DataFrame({
        'code': ['510050', '510300', '510500'],
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF'],
        'type': ['ETF'] * 3,
        'list_date': '2023-01-01',
        'scale': 5000000000
    })
    
    return test_data, etf_pool, dates

def test_force_stop_loss():
    """强制测试止损触发"""
    print("=" * 80)
    print("           强制止损测试")
    print("=" * 80)
    
    all_data, etf_pool, dates = generate_simple_crash_data()
    
    original_start = BacktestConfig.START_DATE
    original_end = BacktestConfig.END_DATE
    
    BacktestConfig.START_DATE = '2024-01-15'
    BacktestConfig.END_DATE = '2024-05-31'
    
    print("\n测试数据概览:")
    print(f"  日期范围: {dates[0]} 到 {dates[-1]}")
    print(f"  初始价格: {all_data['hfq']['510050']['close'].iloc[0]:.2f}")
    print(f"  最高价格: {all_data['hfq']['510050']['close'].max():.2f}")
    print(f"  最低价格: {all_data['hfq']['510050']['close'].min():.2f}")
    print(f"  最大跌幅: {(1 - all_data['hfq']['510050']['close'].min() / all_data['hfq']['510050']['close'].max()) * 100:.2f}%")
    
    param_combinations = [
        {'lookback_days': 30, 'trailing_stop_pct': 0.05},
        {'lookback_days': 30, 'trailing_stop_pct': 0.08},
        {'lookback_days': 30, 'trailing_stop_pct': 0.15},
    ]
    
    results = []
    
    print("\n测试不同止损阈值:")
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
        
        results.append({
            'trailing_stop': ts,
            'drawdown': metrics.get('max_drawdown', 0)
        })
    
    print("\n" + "=" * 80)
    print("           参数敏感性测试结果对比")
    print("=" * 80)
    print(f"{'止损阈值':<12} {'最大回撤':<12}")
    print("-" * 60)
    
    all_same = True
    first_drawdown = results[0]['drawdown']
    
    for r in results:
        print(f"{r['trailing_stop']*100:<10.0f}%   {r['drawdown']*100:>12.2f}%")
        if abs(r['drawdown'] - first_drawdown) > 0.01:
            all_same = False
    
    print("\n" + "=" * 80)
    if all_same:
        print("  ⚠️ 警告：所有参数组合结果相同！")
    else:
        print("  ✅ 参数敏感性测试正常！")
    print("=" * 80)
    
    BacktestConfig.START_DATE = original_start
    BacktestConfig.END_DATE = original_end

if __name__ == "__main__":
    test_force_stop_loss()