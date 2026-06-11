"""
打印每周四调仓的目标组合列表并输出为CSV
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BacktestConfig, StrategyConfig
from data_fetcher import fetch_history_data
from signal_engine import generate_target_portfolio, filter_liquidity

CORE_ETFS = [
    '510050', '510300', '510500', '159915', '512100',
    '512660', '512480', '512690', '512010', '515030',
    '511010', '518880'
]

def load_core_etf_data(start_date, end_date):
    all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    
    for etf_code in CORE_ETFS:
        try:
            data = fetch_history_data(etf_code, start_date, end_date)
            if data['qfq'] is not None and not data['qfq'].empty:
                all_data['qfq'][etf_code] = data['qfq']
                all_data['hfq'][etf_code] = data['hfq']
        except Exception as e:
            print(f"  ❌ {etf_code}: {str(e)}")
    
    return all_data

def create_etf_pool():
    etf_info = {
        '510050': {'name': '50ETF', 'type': '宽基'},
        '510300': {'name': '沪深300ETF', 'type': '宽基'},
        '510500': {'name': '中证500ETF', 'type': '宽基'},
        '159915': {'name': '创业板ETF', 'type': '宽基'},
        '512100': {'name': '中证100ETF', 'type': '宽基'},
        '512660': {'name': '券商ETF', 'type': '行业'},
        '512480': {'name': '半导体ETF', 'type': '行业'},
        '512690': {'name': '酒ETF', 'type': '行业'},
        '512010': {'name': '医药ETF', 'type': '行业'},
        '515030': {'name': '新能源ETF', 'type': '行业'},
        '511010': {'name': '国债ETF', 'type': '避险'},
        '518880': {'name': '黄金ETF', 'type': '避险'}
    }
    
    return pd.DataFrame([
        {'code': code, 'name': info['name'], 'type': info['type'], 
         'list_date': '2015-01-01', 'scale': 5000000000}
        for code, info in etf_info.items()
    ])

def get_trading_dates(all_data, start_date, end_date):
    dates = set()
    if 'qfq' in all_data:
        for etf_code, df in all_data['qfq'].items():
            if not df.empty:
                dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())
    dates = sorted(list(dates))
    
    start_idx = dates.index(start_date) if start_date in dates else 0
    end_idx = dates.index(end_date) + 1 if end_date in dates else len(dates)
    
    return dates[start_idx:end_idx]

def is_rebalance_day(date_str, rebalance_day=4):
    date = pd.to_datetime(date_str)
    return date.weekday() == rebalance_day

def main():
    print("="*80)
    print("每周四调仓目标组合列表 - 输出CSV")
    print("="*80)
    
    print("\n加载ETF数据...")
    all_data = load_core_etf_data('2018-01-01', '2025-12-31')
    etf_pool = create_etf_pool()
    
    if len(all_data['qfq']) == 0:
        print("未能获取任何ETF数据")
        return
    
    trading_dates = get_trading_dates(all_data, '2018-01-01', '2025-12-31')
    
    qfq_data = all_data.get('qfq', all_data)
    
    print(f"\n回测时间范围: {trading_dates[0]} ~ {trading_dates[-1]}")
    print(f"总交易天数: {len(trading_dates)}")
    
    portfolio_list = []
    
    for date_str in trading_dates:
        if not is_rebalance_day(date_str):
            continue
        
        target_portfolio = generate_target_portfolio(qfq_data, etf_pool, date_str)
        
        portfolio_list.append({
            'date': date_str,
            'portfolio': ','.join(target_portfolio),
            'count': len(target_portfolio)
        })
        
        print(f"处理: {date_str}")
    
    df = pd.DataFrame(portfolio_list)
    
    output_path = 'weekly_portfolio.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ CSV文件已保存: {output_path}")
    print(f"共 {len(df)} 个调仓日")

if __name__ == '__main__':
    main()