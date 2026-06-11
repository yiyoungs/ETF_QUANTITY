import pandas as pd
import numpy as np
from portfolio_manager import PortfolioManager
from config import StrategyConfig

def debug_check_stop_loss():
    """调试check_stop_loss方法"""
    print("=" * 80)
    print("           调试 check_stop_loss 方法")
    print("=" * 80)
    
    pm = PortfolioManager({'trailing_stop_pct': 0.08})
    
    pm.positions = {'510050': 1000}
    pm.bought_dates = {'510050': '2024-01-01'}
    pm.highest_prices = {'510050': 4.0}
    
    dates = pd.date_range('2024-01-01', periods=30, freq='B')
    test_data = {
        'hfq': {
            '510050': pd.DataFrame({
                'date': dates,
                'close': [4.0] * 10 + [3.9, 3.8, 3.7, 3.68, 3.65] + [3.6] * 15
            })
        },
        'dividend': {}
    }
    
    print(f"\nCash ETF Code: {StrategyConfig.CASH_ETF_CODE}")
    print(f"持仓中的ETF: {list(pm.positions.keys())}")
    
    test_date = '2024-01-18'
    print(f"\n测试日期: {test_date}")
    
    hfq_data = test_data.get('hfq', test_data)
    print(f"hfQ数据键: {list(hfq_data.keys())}")
    
    for etf_code in list(pm.positions.keys()):
        print(f"\n处理 ETF: {etf_code}")
        
        if etf_code == StrategyConfig.CASH_ETF_CODE:
            print(f"  跳过现金ETF")
            continue
        
        can_sell = pm.can_sell(etf_code, test_date)
        print(f"  can_sell: {can_sell}")
        
        df = hfq_data.get(etf_code)
        print(f"  df is None: {df is None}")
        print(f"  df is empty: {df.empty if df is not None else 'N/A'}")
        
        if df is not None and not df.empty:
            price = df[df['date'] <= pd.to_datetime(test_date)]['close'].iloc[-1]
            print(f"  价格: {price}")
            
            highest_price = pm.highest_prices.get(etf_code, price)
            print(f"  最高价: {highest_price}")
            
            if highest_price > 0:
                drawdown = (highest_price - price) / highest_price
                threshold = 0.08
                print(f"  回撤: {drawdown}")
                print(f"  阈值: {threshold}")
                print(f"  drawdown >= threshold: {drawdown >= threshold}")
                print(f"  回撤 >= 8%: {drawdown >= 0.08}")
                
                if drawdown >= threshold:
                    print(f"  ✅ 应该触发止损！")
                else:
                    print(f"  ❌ 不触发止损")

if __name__ == "__main__":
    debug_check_stop_loss()