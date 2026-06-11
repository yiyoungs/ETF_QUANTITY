"""
测试方案A（趋势强度分级）和方案B（缩短动量周期）的效果
"""
import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtester import Backtester
from config import BacktestConfig, StrategyConfig
from data_fetcher import fetch_history_data
from signal_engine import generate_target_portfolio, calc_multi_period_momentum

# ETF池
CORE_ETFS = [
    '510050', '510300', '510500', '159915', '512100',
    '512660', '512480', '512690', '512010', '515030',
    '511010', '518880'
]

def load_data():
    """加载完整数据"""
    all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    for etf_code in CORE_ETFS:
        data = fetch_history_data(etf_code, '2018-01-01', '2025-12-31')
        if data['qfq'] is not None and not data['qfq'].empty:
            all_data['qfq'][etf_code] = data['qfq']
            all_data['hfq'][etf_code] = data['hfq']
            all_data['dividend'][etf_code] = data['dividend']
    return all_data

def create_etf_pool():
    """创建ETF池"""
    return pd.DataFrame({
        'code': CORE_ETFS,
        'name': ['50ETF', '沪深300ETF', '中证500ETF', '创业板ETF', '中证100ETF',
                 '证券ETF', '半导体ETF', '酒ETF', '医药ETF', '新能源ETF',
                 '国债ETF', '黄金ETF'],
        'type': ['ETF'] * 12,
        'list_date': ['2015-02-16', '2015-05-19', '2013-03-15', '2012-01-09', '2014-12-15',
                      '2013-05-21', '2019-06-12', '2019-05-06', '2013-06-14', '2020-03-04',
                      '2015-01-21', '2013-07-29'],
        'scale': [5000000000] * 12
    })

def run_backtest(all_data, etf_pool, params, config_suffix=""):
    """运行回测"""
    config = BacktestConfig(
        start_date='2018-01-02',
        end_date='2025-12-31',
        rebalance_day=4
    )
    
    backtester = Backtester(
        all_data=all_data,
        etf_pool=etf_pool,
        params=params,
        backtest_config=config
    )
    
    results_df = backtester.run()
    
    # 计算绩效指标
    nav = results_df['total_value']
    daily_returns = nav.pct_change().dropna()
    
    total_return = (nav.iloc[-1] / nav.iloc[0] - 1) * 100
    years = len(results_df) / 252
    annualized_return = ((nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1) * 100
    annualized_vol = daily_returns.std() * np.sqrt(252) * 100
    sharpe = (annualized_return/100 - 0.03) / (annualized_vol/100) if annualized_vol > 0 else 0
    cumulative_max = nav.cummax()
    drawdown = (nav - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'config': config_suffix,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'results_df': results_df
    }

def main():
    print("=" * 80)
    print("策略对比测试（2018-01-01 ~ 2025-12-31）")
    print("=" * 80)
    
    # 加载数据
    print("\n加载ETF数据...")
    all_data = load_data()
    etf_pool = create_etf_pool()
    print(f"✅ 数据加载完成，共 {len(all_data['qfq'])} 只ETF")
    
    results = []
    
    # 基准版本：原始多周期动量 (20+60+120)
    print("\n" + "=" * 60)
    print("测试基准版本（20+60+120天动量）")
    print("=" * 60)
    params_base = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08,
        'periods_weights': [(20, 0.5), (60, 0.3), (120, 0.2)]  # (周期, 权重)
    }
    result_base = run_backtest(all_data, etf_pool, params_base, "基准(20+60+120)")
    results.append(result_base)
    
    # 方案A：趋势强度分级（基于沪深300偏离度调节仓位）
    print("\n" + "=" * 60)
    print("测试方案A（趋势强度分级）")
    print("=" * 60)
    params_a = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08,
        'periods_weights': [(20, 0.5), (60, 0.3), (120, 0.2)],
        'trend_strength_enabled': True,
        'trend_weak_threshold': 0.02,   # 最大偏离<2%为弱趋势
        'trend_strong_threshold': 0.05   # 最大偏离>5%为强趋势
    }
    result_a = run_backtest(all_data, etf_pool, params_a, "方案A(趋势强度)")
    results.append(result_a)
    
    # 方案B：缩短动量周期 (10+20+60)
    print("\n" + "=" * 60)
    print("测试方案B（缩短动量周期 10+20+60）")
    print("=" * 60)
    params_b = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08,
        'periods_weights': [(10, 0.4), (20, 0.3), (60, 0.3)]  # (周期, 权重)
    }
    result_b = run_backtest(all_data, etf_pool, params_b, "方案B(短周期)")
    results.append(result_b)
    
    # 方案A+B：组合方案
    print("\n" + "=" * 60)
    print("测试方案A+B（趋势强度 + 短周期动量）")
    print("=" * 60)
    params_ab = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08,
        'periods_weights': [(10, 0.4), (20, 0.3), (60, 0.3)],
        'trend_strength_enabled': True,
        'trend_weak_threshold': 0.02,
        'trend_strong_threshold': 0.05
    }
    result_ab = run_backtest(all_data, etf_pool, params_ab, "方案A+B(组合)")
    results.append(result_ab)
    
    # 输出对比表格
    print("\n" + "=" * 80)
    print("策略对比结果")
    print("=" * 80)
    print(f"{'策略':<20} {'总收益(%)':>12} {'年化收益(%)':>12} {'年化波动(%)':>12} {'夏普比率':>10} {'最大回撤(%)':>12}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['config']:<20} {r['total_return']:>12.2f} {r['annualized_return']:>12.2f} {r['annualized_vol']:>12.2f} {r['sharpe']:>10.2f} {r['max_drawdown']:>12.2f}")
    
    # 保存详细结果
    print("\n保存详细结果...")
    for r in results:
        filename = f"reports/backtest_{r['config'].replace('(', '_').replace(')', '_').replace('+', '_')}.csv"
        r['results_df'].to_csv(filename, index=False)
        print(f"  ✅ {filename}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == '__main__':
    main()