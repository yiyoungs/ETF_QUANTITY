import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
import logging

logging.basicConfig(level=logging.WARNING)

def generate_crash_test_data():
    """生成包含急剧下跌的测试数据"""
    n_days = 500
    dates = pd.date_range('2017-01-01', periods=n_days, freq='B')
    
    etf_list = ['510050', '510300', '511010']
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    for etf_code in etf_list[:-1]:
        base_price = 3.0
        
        phase1_end = int(n_days * 0.4)
        phase2_end = int(n_days * 0.55)
        phase3_end = n_days
        
        phase1 = np.linspace(base_price, base_price * 1.3, phase1_end)
        phase2 = np.linspace(base_price * 1.3, base_price * 0.95, phase2_end - phase1_end)
        phase3 = np.linspace(base_price * 0.95, base_price * 1.2, phase3_end - phase2_end)
        
        trend = np.concatenate([phase1, phase2, phase3])
        noise = np.random.normal(0, 0.008, n_days)
        close = trend * (1 + noise)
        
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
    
    return test_data, etf_pool, dates

def debug_holdings():
    """调试持仓情况"""
    print("=" * 80)
    print("           持仓调试")
    print("=" * 80)
    
    all_data, etf_pool, dates = generate_crash_test_data()
    
    print(f"\n测试数据概览:")
    print(f"  日期范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"  下跌期: 第 {int(len(dates)*0.4)} 天到第 {int(len(dates)*0.55)} 天")
    print(f"  最大跌幅: {((1.3-0.95)/1.3*100):.1f}%")
    
    params = {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params)
    results_df = backtester.run()
    
    print(f"\n回测结果:")
    analyzer = Analyzer(results_df)
    metrics = analyzer.calculate_metrics()
    print(f"  年化收益: {metrics.get('annualized_return', 0)*100:.2f}%")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0)*100:.2f}%")
    
    print(f"\n每日持仓分析:")
    positions_data = []
    for idx, row in results_df.iterrows():
        date = row['date']
        positions = row['positions']
        cash = row['cash']
        total_value = row['total_value']
        
        if isinstance(positions, dict):
            risk_assets = {k: v for k, v in positions.items() if k != '511010'}
            cash_holdings = positions.get('511010', 0)
            has_risk = len(risk_assets) > 0
        else:
            risk_assets = {}
            cash_holdings = 0
            has_risk = False
        
        positions_data.append({
            'date': date,
            'has_risk': has_risk,
            'risk_assets': list(risk_assets.keys()),
            'cash': cash,
            'total_value': total_value
        })
    
    positions_df = pd.DataFrame(positions_data)
    positions_df['date'] = pd.to_datetime(positions_df['date'])
    
    crash_start = dates[int(len(dates)*0.4)]
    crash_end = dates[int(len(dates)*0.55)]
    
    crash_period = positions_df[(positions_df['date'] >= crash_start) & (positions_df['date'] <= crash_end)]
    print(f"\n下跌期间 ({crash_start.date()} 到 {crash_end.date()}):")
    print(f"  总天数: {len(crash_period)}")
    print(f"  持有风险资产天数: {len(crash_period[crash_period['has_risk']])}")
    print(f"  持有现金天数: {len(crash_period[~crash_period['has_risk']])}")
    
    if len(crash_period[crash_period['has_risk']]) > 0:
        print(f"\n下跌期间持有风险资产的日子:")
        for idx, row in crash_period[crash_period['has_risk']].iterrows():
            print(f"  {row['date'].date()}: {row['risk_assets']}")
    
    print(f"\n全回测期:")
    print(f"  持有风险资产天数: {len(positions_df[positions_df['has_risk']])}/{len(positions_df)}")
    print(f"  纯现金天数: {len(positions_df[~positions_df['has_risk']])}/{len(positions_df)}")

if __name__ == "__main__":
    debug_holdings()