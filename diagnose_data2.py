import traceback

print("=== 测试更多 akshare 接口 ===")

import akshare as ak

# 测试不同的 ETF 数据接口
interfaces = [
    ("fund_etf_hist_sina", {"symbol": "sh510300"}),
    ("fund_etf_hist_sina", {"symbol": "sz159915"}),
    ("fund_etf_hist_em", {"symbol": "510300", "period": "daily", "start_date": "20180101", "end_date": "20220805"}),
    ("fund_etf_hist_em", {"symbol": "159915", "period": "daily", "start_date": "20180101", "end_date": "20220805"}),
    ("stock_zh_a_hist", {"symbol": "510300", "period": "daily", "start_date": "20180101", "end_date": "20220805", "adjust": "qfq"}),
    ("stock_zh_a_hist", {"symbol": "510300", "period": "daily", "start_date": "20180101", "end_date": "20220805", "adjust": "hfq"}),
]

for func_name, params in interfaces:
    print(f"\n尝试 {func_name}(**{params})...")
    try:
        func = getattr(ak, func_name)
        df = func(**params)
        if df is not None and len(df) > 0:
            print(f"✅ 成功！{len(df)} 条数据")
            print(f"列名: {list(df.columns)[:5]}...")
            if '日期' in df.columns or 'date' in df.columns:
                date_col = '日期' if '日期' in df.columns else 'date'
                print(f"日期范围: {df.iloc[0][date_col]} ~ {df.iloc[-1][date_col]}")
        else:
            print("❌ 返回空数据")
    except Exception as e:
        print(f"❌ 失败: {type(e).__name__}: {e}")
        traceback.print_exc()

# 测试新浪 ETF 列表接口
print("\n=== 测试 ETF 列表接口 ===")
try:
    print("尝试 fund_etf_category_sina...")
    df = ak.fund_etf_category_sina()
    if df is not None and len(df) > 0:
        print(f"✅ 成功！{len(df)} 条")
        print(f"列名: {list(df.columns)}")
        print(f"前5条代码: {list(df['代码'].head())}")
    else:
        print("❌ 空数据")
except Exception as e:
    print(f"❌ 失败: {e}")
    traceback.print_exc()
