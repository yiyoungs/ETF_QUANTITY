import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from data_fetcher import get_etf_pool, fetch_all_etf_data, load_cached_data
from config import BacktestConfig
import logging

logging.basicConfig(level=logging.WARNING)

def main():
    """测试真实数据回测"""
    print("=" * 80)
    print("           真实数据回测调试")
    print("=" * 80)
    
    try:
        print("\n【阶段1：尝试加载缓存数据】")
        all_data = load_cached_data()
        
        if all_data is None:
            print("缓存数据不存在，尝试获取真实数据...")
            etf_pool = get_etf_pool()
            all_data = fetch_all_etf_data(
                etf_pool,
                start_date='2018-01-01',
                end_date='2020-12-31'  # 包含2018年熊市
            )
        
        if all_data is None or len(all_data) == 0:
            print("无法获取数据，使用测试数据")
            return
        
        print(f"获取到 {len(all_data['qfq'])} 只 ETF 数据")
        
        print("\n【阶段2：运行默认参数回测 (60/0.08)】")
        params = {
            'lookback_days': 60,
            'trailing_stop_pct': 0.08
        }
        
        backtester = Backtester(all_data, etf_pool, params=params)
        results_df = backtester.run()
        
        print("\n【阶段3：分析结果】")
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        print("\n" + "=" * 60)
        print("                    绩效指标")
        print("=" * 60)
        print(f"年化收益率: {metrics.get('annualized_return', 0) * 100:.2f}%")
        print(f"年化波动率: {metrics.get('annualized_volatility', 0) * 100:.2f}%")
        print(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        print(f"盈亏比: {metrics.get('profit_factor', 0):.2f}")
        print(f"风险资产占比: {metrics.get('risk_asset_ratio', 0) * 100:.2f}%")
        print("=" * 60)
        
        print("\n【阶段4：分析持仓情况】")
        positions_data = []
        for idx, row in results_df.iterrows():
            date = row['date']
            positions = row['positions']
            risk_assets = sum(1 for code in positions.keys() if code != '511010') if isinstance(positions, dict) else 0
            total_assets = row['total_value'] + row['cash']
            positions_data.append({
                'date': date,
                'risk_assets': risk_assets,
                'total_assets': total_assets
            })
        
        positions_df = pd.DataFrame(positions_data)
        positions_df['date'] = pd.to_datetime(positions_df['date'])
        
        bear_2018 = positions_df[(positions_df['date'] >= '2018-01-01') & (positions_df['date'] <= '2018-12-31')]
        print(f"\n2018年熊市期间:")
        print(f"  平均风险资产数量: {bear_2018['risk_assets'].mean():.1f}")
        print(f"  最大风险资产数量: {bear_2018['risk_assets'].max()}")
        print(f"  最小风险资产数量: {bear_2018['risk_assets'].min()}")
        print(f"  持有风险资产天数: {len(bear_2018[bear_2018['risk_assets'] > 0])}/{len(bear_2018)}")
        
        print(f"\n全回测期:")
        print(f"  平均风险资产数量: {positions_df['risk_assets'].mean():.1f}")
        print(f"  持有风险资产天数: {len(positions_df[positions_df['risk_assets'] > 0])}/{len(positions_df)}")
        
        print("\n【调试完成】")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()