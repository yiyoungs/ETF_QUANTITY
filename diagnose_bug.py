"""
Bug诊断脚本 - 排查2018-10-12异常收益问题
"""
import pandas as pd
import numpy as np
import os

# 1. 检查511010原始数据
print("=" * 80)
print("1. 检查511010原始数据")
print("=" * 80)

cache_dir = 'data/cache'
file_path = os.path.join(cache_dir, '511010_qfq.csv')

if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df['pct_change'] = df['close'].pct_change()
    
    mask = (df['date'] >= '2018-10-01') & (df['date'] <= '2018-10-20')
    subset = df[mask]
    
    print("2018-10-01 ~ 2018-10-20 期间的价格数据:")
    print(subset[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'pct_change']].to_string())
    
    print("\n检查是否有异常值:")
    print(f"收盘价 <= 0 的数量: {(df['close'] <= 0).sum()}")
    print(f"单日涨跌幅超过±15%的数量: {(df['pct_change'].abs() > 0.15).sum()}")
    print(f"成交量为0的数量: {(df['volume'] == 0).sum()}")
    
else:
    print(f"文件不存在: {file_path}")

# 2. 检查日志中2018-10-12的交易记录
print("\n" + "=" * 80)
print("2. 检查2018-10-11和2018-10-12的交易日志")
print("=" * 80)

log_file = 'logs/full_test.log'
if os.path.exists(log_file):
    with open(log_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    relevant_lines = []
    for line in lines:
        if '2018-10-11' in line or '2018-10-12' in line:
            relevant_lines.append(line)
    
    print(f"找到 {len(relevant_lines)} 条相关日志:")
    print("-" * 80)
    for line in relevant_lines:
        print(line, end='')
else:
    print(f"日志文件不存在: {log_file}")

# 3. 检查所有ETF的数据质量
print("\n" + "=" * 80)
print("3. 数据质量报告")
print("=" * 80)

etf_codes = ['510050', '510300', '510500', '512100', '159915', '512880', '512480', '512690', '511010']

for code in etf_codes:
    qfq_file = os.path.join(cache_dir, f'{code}_qfq.csv')
    if os.path.exists(qfq_file):
        df = pd.read_csv(qfq_file)
        df['date'] = pd.to_datetime(df['date'])
        df['pct_change'] = df['close'].pct_change()
        
        close_gt_zero = (df['close'] <= 0).sum()
        extreme_change = (df['pct_change'].abs() > 0.15).sum()
        zero_volume = (df['volume'] == 0).sum()
        negative_amount = (df['amount'] < 0).sum() if 'amount' in df.columns else 0
        
        if close_gt_zero > 0 or extreme_change > 0 or zero_volume > 0 or negative_amount > 0:
            print(f"\n⚠️ {code} 存在异常:")
            print(f"   - 收盘价<=0: {close_gt_zero}")
            print(f"   - 单日涨跌幅>15%: {extreme_change}")
            print(f"   - 成交量为0: {zero_volume}")
            if 'amount' in df.columns:
                print(f"   - 成交额为负: {negative_amount}")
            
            if extreme_change > 0:
                extreme_dates = df[df['pct_change'].abs() > 0.15]
                print(f"   极端波动日期: {list(extreme_dates['date'].dt.strftime('%Y-%m-%d'))}")
        else:
            print(f"✅ {code} 数据正常")
    else:
        print(f"❌ {code} 数据文件不存在")
