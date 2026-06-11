"""
详细分析2021-05-18异常收益的调试脚本
追踪当天的每一步操作和资产变化
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import Backtester
from config import BacktestConfig, StrategyConfig
from data_fetcher import fetch_history_data

def load_data():
    """加载数据"""
    core_etfs = [
        '510050', '510300', '510500', '159915', '512100',
        '512660', '512480', '512690', '512010', '515030',
        '511010', '518880'
    ]
    
    all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    for etf_code in core_etfs:
        data = fetch_history_data(etf_code, '2018-01-01', '2021-06-01')
        if data['qfq'] is not None and not data['qfq'].empty:
            all_data['qfq'][etf_code] = data['qfq']
            all_data['hfq'][etf_code] = data['hfq']
            all_data['dividend'][etf_code] = data['dividend']
    return all_data

def create_etf_pool():
    """创建ETF池"""
    return pd.DataFrame({
        'code': ['510050', '510300', '510500', '159915', '512100',
                 '512660', '512480', '512690', '512010', '515030',
                 '511010', '518880'],
        'name': ['50ETF', '沪深300ETF', '中证500ETF', '创业板ETF', '中证100ETF',
                 '证券ETF', '半导体ETF', '酒ETF', '医药ETF', '新能源ETF',
                 '国债ETF', '黄金ETF'],
        'type': ['ETF'] * 12,
        'list_date': ['2015-02-16', '2015-05-19', '2013-03-15', '2012-01-09', '2014-12-15',
                      '2013-05-21', '2019-06-12', '2019-05-06', '2013-06-14', '2020-03-04',
                      '2015-01-21', '2013-07-29'],
        'scale': [5000000000] * 12
    })

def main():
    print("=" * 80)
    print("详细分析2021-05-18异常收益")
    print("=" * 80)
    
    # 加载数据
    print("\n加载数据...")
    all_data = load_data()
    etf_pool = create_etf_pool()
    
    # 创建回测器（只运行到2021-05-19）
    config = BacktestConfig(
        start_date='2018-01-02',
        end_date='2021-05-19',
        rebalance_day=4
    )
    
    params = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08,
        'periods_weights': [(20, 0.5), (60, 0.3), (120, 0.2)]
    }
    
    # 自定义回测器，添加详细日志（只关注2021-05-17至2021-05-19）
    target_dates = ['2021-05-17', '2021-05-18', '2021-05-19']
    
    class DebugBacktester(Backtester):
        def _daily_process(self, date_str):
            if date_str in target_dates:
                print(f"\n{'='*60}")
                print(f"日期: {date_str}")
                print(f"{'='*60}")
                
                # 记录处理前的状态
                print(f"处理前 - 现金: {self.portfolio_manager.cash:.2f}")
                print(f"处理前 - 持仓: {dict(self.portfolio_manager.positions)}")
            
            # 调用父类方法
            result = super()._daily_process(date_str)
            
            if date_str in target_dates:
                # 记录处理后的状态
                print(f"处理后 - 现金: {self.portfolio_manager.cash:.2f}")
                print(f"处理后 - 持仓: {dict(self.portfolio_manager.positions)}")
                
                # 计算当日资产价值
                total_value = self.portfolio_manager.get_total_value(self.all_data, date_str)
                print(f"当日总资产: {total_value:.2f}")
            
            return result
        
        def _weekly_rebalance(self, date_str, exclude=None):
            if date_str in target_dates:
                print(f"\n{'='*60}")
                print(f"周度调仓: {date_str}")
                print(f"{'='*60}")
                print(f"调仓前现金: {self.portfolio_manager.cash:.2f}")
                print(f"调仓前持仓: {dict(self.portfolio_manager.positions)}")
            
            super()._weekly_rebalance(date_str, exclude)
            
            if date_str in target_dates:
                print(f"调仓后现金: {self.portfolio_manager.cash:.2f}")
                print(f"调仓后持仓: {dict(self.portfolio_manager.positions)}")
    
    backtester = DebugBacktester(
        all_data=all_data,
        etf_pool=etf_pool,
        params=params,
        backtest_config=config
    )
    
    # 运行回测
    print("\n开始回测...")
    results_df = backtester.run()
    
    # 输出2021-05-17至2021-05-19的详细数据
    print("\n" + "=" * 80)
    print("2021-05-17至2021-05-19详细数据")
    print("=" * 80)
    mask = (results_df['date'] >= '2021-05-17') & (results_df['date'] <= '2021-05-19')
    print(results_df[mask])
    
    # 分析持仓变化
    print("\n" + "=" * 80)
    print("持仓变化分析")
    print("=" * 80)
    
    # 获取每日持仓
    for date in ['2021-05-17', '2021-05-18', '2021-05-19']:
        print(f"\n{date} 持仓:")
        hfq_data = all_data.get('hfq', all_data)
        
        # 从结果中获取当日持仓
        daily_record = results_df[results_df['date'] == date]
        if not daily_record.empty:
            positions_str = daily_record['positions'].iloc[0]
            # 解析positions字符串
            import ast
            try:
                positions = ast.literal_eval(positions_str)
                for etf_code, shares in positions.items():
                    if etf_code in hfq_data:
                        df = hfq_data[etf_code]
                        df['date'] = pd.to_datetime(df['date'])
                        mask_date = df['date'] == pd.to_datetime(date)
                        if mask_date.any():
                            price = df[mask_date]['close'].iloc[0]
                            value = shares * price
                            print(f"  {etf_code}: {shares} 股 @ {price:.4f} = {value:.2f}")
            except:
                print(f"  无法解析持仓: {positions_str}")

if __name__ == '__main__':
    main()