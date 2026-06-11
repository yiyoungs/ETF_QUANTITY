"""
获取2018-01-01 至 2025-12-31 的ETF数据并缓存
"""
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_history_data

CORE_ETFS = [
    '510050', '510300', '510500', '159915', '512100',
    '512660', '512480', '512690', '512010', '515030',
    '511010', '518880'
]

def main():
    print("=" * 60)
    print("获取ETF数据 (2018-01-01 ~ 2025-12-31)")
    print("=" * 60)
    
    start_date = '2018-01-01'
    end_date = '2025-12-31'
    
    for etf_code in CORE_ETFS:
        print(f"\n正在获取 {etf_code}...")
        try:
            data = fetch_history_data(etf_code, start_date, end_date)
            
            if data['qfq'] is not None and not data['qfq'].empty:
                qfq_dates = data['qfq']['date']
                hfq_dates = data['hfq']['date']
                
                print(f"  ✅ qfq数据: {len(data['qfq'])} 条, 日期范围: {qfq_dates.min()} ~ {qfq_dates.max()}")
                print(f"  ✅ hfq数据: {len(data['hfq'])} 条, 日期范围: {hfq_dates.min()} ~ {hfq_dates.max()}")
            else:
                print(f"  ❌ 获取失败")
        except Exception as e:
            print(f"  ❌ 错误: {e}")
    
    print("\n" + "=" * 60)
    print("数据获取完成！")
    print("=" * 60)

if __name__ == '__main__':
    main()