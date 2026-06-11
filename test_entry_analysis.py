import pandas as pd
import numpy as np
from signal_engine import filter_liquidity, generate_target_portfolio

def generate_test_data():
    dates = pd.date_range('2017-01-01', periods=500, freq='B')
    n_days = len(dates)
    
    np.random.seed(42)
    
    prices = np.linspace(3.0, 4.0, 150)
    prices = np.concatenate([prices, np.linspace(4.0, 3.68, 50)])
    prices = np.concatenate([prices, np.linspace(3.68, 4.2, n_days - 200)])
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
    
    etf_pool = pd.DataFrame({
        'code': ['510050', '510300', '510500'],
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF'],
        'type': ['ETF'] * 3,
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    return test_data, etf_pool

def analyze_entry_conditions():
    """分析入场条件"""
    print("=" * 80)
    print("           入场条件分析")
    print("=" * 80)
    
    all_data, etf_pool = generate_test_data()
    
    test_dates = ['2017-07-01', '2017-10-01', '2017-12-01', '2018-01-01']
    
    for date in test_dates:
        print(f"\n【{date}】")
        print("-" * 60)
        
        qualified = filter_liquidity(all_data['qfq'], etf_pool, date)
        print(f"符合流动性要求的ETF: {qualified}")
        
        target = generate_target_portfolio(all_data['qfq'], etf_pool, date)
        print(f"目标持仓: {target}")
        
        has_equity = any(code != '511010' for code in target)
        print(f"是否包含权益资产: {'是' if has_equity else '否'}")

if __name__ == "__main__":
    analyze_entry_conditions()