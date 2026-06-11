import pandas as pd
import numpy as np

def generate_test_data():
    """生成测试数据并分析MA60"""
    dates = pd.date_range('2016-01-01', periods=1200, freq='B')
    n_days = len(dates)
    
    base_price = 3.0
    prices = np.ones(n_days) * base_price
    
    phase1_end = int(n_days * 0.2)
    prices[:phase1_end] = np.linspace(base_price, base_price * 1.3, phase1_end)
    
    crash_start = phase1_end
    crash_end = int(n_days * 0.5)
    prices[crash_start:crash_end] = np.linspace(base_price * 1.3, base_price * 0.6, crash_end - crash_start)
    
    recovery_start = crash_end
    recovery_end = int(n_days * 0.8)
    prices[recovery_start:recovery_end] = np.linspace(base_price * 0.6, base_price * 1.2, recovery_end - recovery_start)
    
    prices[recovery_end:] = np.linspace(base_price * 1.2, base_price * 1.4, n_days - recovery_end)
    
    prices += np.random.normal(0, 0.02, n_days)
    
    df = pd.DataFrame({
        'date': dates,
        'close': prices
    })
    
    df['ma60'] = df['close'].rolling(window=60).mean().shift(1)
    
    return df

def analyze_ma_condition():
    """分析MA60条件"""
    df = generate_test_data()
    
    print("=" * 80)
    print("           MA60 条件分析")
    print("=" * 80)
    
    start_2018 = df[df['date'] >= '2018-01-01'].iloc[0]
    print(f"\n2018年1月1日:")
    print(f"  收盘价: {start_2018['close']:.4f}")
    print(f"  MA60: {start_2018['ma60']:.4f}")
    print(f"  Price > MA60: {start_2018['close'] > start_2018['ma60']}")
    
    print("\n2018年各月首日情况:")
    print("-" * 60)
    
    months = ['2018-01-01', '2018-02-01', '2018-03-01', '2018-04-01', '2018-05-01', '2018-06-01']
    
    for month in months:
        row = df[df['date'] >= month].iloc[0]
        condition = row['close'] > row['ma60']
        print(f"{month}: close={row['close']:.4f}, ma60={row['ma60']:.4f}, 条件满足={condition}")
    
    print("\n2018年全年满足 Price > MA60 的天数:")
    year_2018 = df[(df['date'] >= '2018-01-01') & (df['date'] <= '2018-12-31')]
    satisfied = year_2018[year_2018['close'] > year_2018['ma60']]
    print(f"  总天数: {len(year_2018)}")
    print(f"  满足条件天数: {len(satisfied)} ({len(satisfied)/len(year_2018)*100:.1f}%)")
    
    print("\n2018年5月25日情况:")
    may_25 = df[df['date'] >= '2018-05-25'].iloc[0]
    print(f"  收盘价: {may_25['close']:.4f}")
    print(f"  MA60: {may_25['ma60']:.4f}")
    print(f"  Price > MA60: {may_25['close'] > may_25['ma60']}")

if __name__ == "__main__":
    analyze_ma_condition()