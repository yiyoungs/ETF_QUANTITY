"""
专门分析2021-05-18异常收益的调试脚本
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_history_data

def main():
    print("=" * 80)
    print("分析2021-05-18异常收益")
    print("=" * 80)
    
    # 获取所有ETF数据
    etf_codes = ['510050', '510300', '510500', '159915', '512100',
                 '512660', '512480', '512690', '512010', '515030',
                 '511010', '518880']
    
    print("\n检查各ETF在2021-05-17至2021-05-19的价格变化：")
    print("-" * 80)
    print(f"{'ETF代码':<10} {'名称':<10} {'5/17 HFQ':>10} {'5/18 HFQ':>10} {'5/19 HFQ':>10} {'日跌幅(%)':>12}")
    print("-" * 80)
    
    etf_names = {
        '510050': '50ETF',
        '510300': '沪深300',
        '510500': '中证500',
        '159915': '创业板',
        '512100': '中证100',
        '512660': '证券ETF',
        '512480': '半导体',
        '512690': '酒ETF',
        '512010': '医药ETF',
        '515030': '新能源',
        '511010': '国债ETF',
        '518880': '黄金ETF'
    }
    
    abnormal_etfs = []
    
    for code in etf_codes:
        data = fetch_history_data(code, '2021-05-15', '2021-05-20')
        if data['hfq'] is not None and not data['hfq'].empty:
            df = data['hfq']
            df['date'] = pd.to_datetime(df['date'])
            
            # 获取三天的收盘价
            prices = {}
            for date in ['2021-05-17', '2021-05-18', '2021-05-19']:
                mask = df['date'] == pd.to_datetime(date)
                if mask.any():
                    prices[date] = df[mask]['close'].iloc[0]
                else:
                    prices[date] = np.nan
            
            if not np.any(np.isnan(list(prices.values()))):
                pct_change = (prices['2021-05-18'] - prices['2021-05-17']) / prices['2021-05-17'] * 100
                print(f"{code:<10} {etf_names.get(code,''):<10} {prices['2021-05-17']:>10.4f} {prices['2021-05-18']:>10.4f} {prices['2021-05-19']:>10.4f} {pct_change:>12.2f}")
                
                if abs(pct_change) > 5:
                    abnormal_etfs.append((code, etf_names.get(code,''), pct_change))
    
    if abnormal_etfs:
        print("\n" + "=" * 80)
        print("发现异常ETF：")
        print("-" * 80)
        for code, name, pct in abnormal_etfs:
            print(f"{code} ({name}): {pct:.2f}%")
    
    # 检查QFQ数据
    print("\n" + "=" * 80)
    print("检查QFQ（前复权）数据：")
    print("-" * 80)
    for code in abnormal_etfs:
        code = code[0]
        data = fetch_history_data(code, '2021-05-15', '2021-05-20')
        if data['qfq'] is not None and not data['qfq'].empty:
            df = data['qfq']
            df['date'] = pd.to_datetime(df['date'])
            
            prices = {}
            for date in ['2021-05-17', '2021-05-18', '2021-05-19']:
                mask = df['date'] == pd.to_datetime(date)
                if mask.any():
                    prices[date] = df[mask]['close'].iloc[0]
            
            pct_change = (prices['2021-05-18'] - prices['2021-05-17']) / prices['2021-05-17'] * 100
            print(f"{code} QFQ: 5/17={prices['2021-05-17']:.4f}, 5/18={prices['2021-05-18']:.4f}, 跌幅={pct_change:.2f}%")

if __name__ == '__main__':
    main()