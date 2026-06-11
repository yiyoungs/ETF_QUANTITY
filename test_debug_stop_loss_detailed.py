import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from config import BacktestConfig, StrategyConfig
import logging

logging.basicConfig(level=logging.DEBUG)

def generate_test_data():
    """生成测试数据"""
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

def debug_stop_loss():
    """调试止损机制"""
    print("=" * 80)
    print("           止损机制调试")
    print("=" * 80)
    
    all_data, etf_pool = generate_test_data()
    
    original_start = BacktestConfig.START_DATE
    original_end = BacktestConfig.END_DATE
    
    BacktestConfig.START_DATE = '2017-07-01'
    BacktestConfig.END_DATE = '2018-01-31'
    
    params = {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params)
    
    pm = backtester.portfolio_manager
    
    qfq_data = all_data['qfq']
    hfq_data = all_data['hfq']
    
    test_dates = ['2017-07-01', '2017-12-01', '2017-12-15', '2018-01-01']
    
    print("\n【初始状态检查】")
    print(f"初始现金: {pm.cash}")
    print(f"初始持仓: {pm.positions}")
    print(f"初始最高价记录: {pm.highest_prices}")
    
    for date_str in test_dates:
        print(f"\n【{date_str}】")
        print("-" * 60)
        
        if date_str == '2017-07-01':
            print("执行初始调仓...")
            pm.rebalance(all_data, etf_pool, date_str)
            print(f"调仓后持仓: {pm.positions}")
            print(f"调仓后最高价记录: {pm.highest_prices}")
            print(f"调仓后买入日期: {pm.bought_dates}")
        else:
            print(f"当前持仓: {pm.positions}")
            print(f"当前最高价记录: {pm.highest_prices}")
            
            print("\n止损检查:")
            for etf_code in pm.positions:
                if etf_code == StrategyConfig.CASH_ETF_CODE:
                    continue
                
                df = hfq_data.get(etf_code)
                if df is not None:
                    price = df[df['date'] <= pd.to_datetime(date_str)]['close'].iloc[-1]
                    highest_price = pm.highest_prices.get(etf_code, price)
                    drawdown = (highest_price - price) / highest_price if highest_price > 0 else 0
                    can_sell = pm.can_sell(etf_code, date_str)
                    
                    print(f"  {etf_code}: 现价={price:.4f}, 最高价={highest_price:.4f}, 回撤={drawdown:.2%}, 可卖出={can_sell}")
                    if drawdown >= 0.08:
                        print(f"  ⚠️ 应该触发止损！")
            
            pm.update_highest_price(all_data, date_str)
            print(f"更新最高价后记录: {pm.highest_prices}")
    
    BacktestConfig.START_DATE = original_start
    BacktestConfig.END_DATE = original_end

if __name__ == "__main__":
    debug_stop_loss()