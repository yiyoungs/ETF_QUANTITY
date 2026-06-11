import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
import logging

logging.basicConfig(level=logging.WARNING)

def generate_vshape_test_data():
    """生成包含V型反转的测试数据，确保止损能够触发"""
    n_days = 800
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
        
        phase1_end = int(n_days * 0.3)
        phase2_end = int(n_days * 0.45)
        phase3_end = int(n_days * 0.6)
        phase4_end = n_days
        
        phase1 = np.linspace(base_price, base_price * 1.4, phase1_end)
        phase2 = np.linspace(base_price * 1.4, base_price * 1.05, phase2_end - phase1_end)
        phase3 = np.linspace(base_price * 1.05, base_price * 1.5, phase3_end - phase2_end)
        phase4 = np.linspace(base_price * 1.5, base_price * 1.6, phase4_end - phase3_end)
        
        trend = np.concatenate([phase1, phase2, phase3, phase4])
        
        noise = np.random.normal(0, 0.01, n_days)
        close = trend * (1 + noise)
        
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.uniform(-0.015, 0.015, n_days)),
            'high': close * (1 + np.random.uniform(0, 0.02, n_days)),
            'low': close * (1 + np.random.uniform(-0.02, 0, n_days)),
            'close': close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        test_data['hfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'close': close * 1.01,
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
    print(f"Phase1 (上涨期): 0-{phase1_end}天，价格 {base_price} -> {base_price*1.4:.2f}")
    print(f"Phase2 (下跌期): {phase1_end}-{phase2_end}天，价格 {base_price*1.4:.2f} -> {base_price*1.05:.2f} (跌幅 {((1.4-1.05)/1.4*100):.0f}%)")
    print(f"Phase3 (恢复上涨): {phase2_end}-{phase3_end}天，价格 {base_price*1.05:.2f} -> {base_price*1.5:.2f}")
    print(f"Phase4 (继续上涨): {phase3_end}-{phase4_end}天，价格 {base_price*1.5:.2f} -> {base_price*1.6:.2f}")
    
    return test_data, etf_pool

def test_sensitivity_fixed():
    """测试参数敏感性，确保不同参数产生不同结果"""
    print("=" * 80)
    print("           参数敏感性测试（修复版）")
    print("=" * 80)
    
    all_data, etf_pool = generate_vshape_test_data()
    
    test_cases = [
        {'lookback_days': 40, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.08},
        {'lookback_days': 80, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.15},
        {'lookback_days': 60, 'trailing_stop_pct': 0.05},
    ]
    
    results = []
    
    for params in test_cases:
        lb = params['lookback_days']
        ts = params['trailing_stop_pct']
        
        print(f"\n测试参数: lookback_days={lb}, trailing_stop_pct={ts*100:.0f}%")
        
        backtester = Backtester(all_data, etf_pool, params=params)
        
        assert backtester.lookback_days == lb, f"lookback_days 不匹配: {backtester.lookback_days} != {lb}"
        assert backtester.trailing_stop_pct == ts, f"trailing_stop_pct 不匹配: {backtester.trailing_stop_pct} != {ts}"
        
        print(f"  参数验证通过 ✓")
        
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
        print("  ⚠️ 警告：所有参数组合结果相同！")
    else:
        print("  ✅ 参数敏感性测试通过！不同参数产生不同结果")
    print("=" * 80)

if __name__ == "__main__":
    test_sensitivity_fixed()