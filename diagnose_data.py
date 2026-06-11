import traceback

# 1. 检查 akshare 是否安装
print("=== Step 1: 检查 akshare ===")
try:
    import akshare
    print(f"✅ akshare 已安装，版本: {akshare.__version__}")
except ImportError:
    print("❌ akshare 未安装！请运行: pip install akshare")

# 2. 测试网络连通性
print("\n=== Step 2: 测试网络 ===")
try:
    import requests
    r = requests.get("https://www.baidu.com", timeout=5)
    print(f"✅ 网络连通，状态码: {r.status_code}")
except Exception as e:
    print(f"❌ 网络不通: {e}")

# 3. 测试 AkShare 获取单只 ETF 数据
print("\n=== Step 3: 测试 AkShare 获取数据 ===")
try:
    import akshare as ak
    print("尝试获取 510300 数据...")
    df = ak.fund_etf_hist_sina(symbol="sz510300")
    if df is not None and len(df) > 0:
        print(f"✅ 成功！获取到 {len(df)} 条数据")
        print(f"日期范围: {df.iloc[0]['date']} ~ {df.iloc[-1]['date']}")
        print(f"列名: {list(df.columns)}")
    else:
        print("❌ 返回空数据")
except Exception as e:
    print(f"❌ 获取失败: {e}")
    traceback.print_exc()
    # 尝试备用接口
    try:
        print("\n尝试备用接口 fund_etf_hist_em...")
        df = ak.fund_etf_hist_em(symbol="510300", period="daily", 
                                  start_date="20180101", end_date="20220805")
        if df is not None and len(df) > 0:
            print(f"✅ 备用接口成功！获取到 {len(df)} 条数据")
            print(f"列名: {list(df.columns)}")
        else:
            print("❌ 备用接口也返回空")
    except Exception as e2:
        print(f"❌ 备用接口也失败: {e2}")
        traceback.print_exc()

# 4. 检查项目 data_fetcher 的实际调用
print("\n=== Step 4: 测试项目 data_fetcher ===")
try:
    from data_fetcher import fetch_etf_data
    print("尝试通过项目 data_fetcher 获取数据...")
    df = fetch_etf_data("510300", "20180101", "20220805")
    if df is not None and len(df) > 0:
        print(f"✅ 成功！{len(df)} 条")
    else:
        print("❌ 返回空")
except Exception as e:
    print(f"❌ 项目 data_fetcher 失败: {e}")
    traceback.print_exc()
