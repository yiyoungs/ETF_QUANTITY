import pandas as pd
import numpy as np
from portfolio_manager import PortfolioManager

def test_stop_loss_unit():
    """单元测试：直接验证止损机制"""
    print("=" * 80)
    print("           止损机制单元测试")
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
    
    print("\n初始状态:")
    print(f"  持仓: {pm.positions}")
    print(f"  买入日期: {pm.bought_dates}")
    print(f"  最高价记录: {pm.highest_prices}")
    
    print("\n测试数据概览:")
    print(f"  日期范围: {dates[0]} 到 {dates[-1]}")
    print(f"  价格序列: {test_data['hfq']['510050']['close'].tolist()[:15]}...")
    
    print("\n测试不同日期的止损检查:")
    print("-" * 60)
    
    test_indices = [9, 10, 11, 12, 13]
    
    for idx in test_indices:
        date_str = dates[idx].strftime('%Y-%m-%d')
        print(f"\n【{date_str} (第{idx+1}天)】")
        
        df = test_data['hfq']['510050']
        price = df['close'].iloc[idx]
        highest_price = pm.highest_prices['510050']
        drawdown = (highest_price - price) / highest_price
        can_sell = pm.can_sell('510050', date_str)
        
        print(f"  当前价: {price:.4f}, 最高价: {highest_price:.4f}")
        print(f"  回撤: {drawdown:.2%}, 可卖出: {can_sell}")
        
        stop_list = pm.check_stop_loss(test_data, date_str, 0.08)
        print(f"  止损列表: {stop_list}")
        
        if drawdown >= 0.08 and not stop_list:
            print(f"  ⚠️ 问题！回撤 >= 8% 但未触发止损")
            print(f"  浮点数精度检查: drawdown={drawdown}, threshold=0.08")
            print(f"  drawdown >= 0.08: {drawdown >= 0.08}")
        
        pm.update_highest_price(test_data, date_str)
        print(f"  更新后最高价: {pm.highest_prices}")

if __name__ == "__main__":
    test_stop_loss_unit()