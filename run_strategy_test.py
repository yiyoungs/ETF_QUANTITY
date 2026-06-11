"""
ETF动量轮动策略 - 方案A/B测试脚本
用于测试不同策略配置的效果
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BacktestConfig, StrategyConfig, DATE_FORMAT
from data_fetcher import fetch_history_data
from backtester import Backtester

CORE_ETFS = [
    '510050', '510300', '510500', '159915', '512100',
    '512660', '512480', '512690', '512010', '515030',
    '511010', '518880'
]

def load_core_etf_data(start_date, end_date):
    logger = logging.getLogger(__name__)
    logger.info("开始加载核心ETF数据...")
    
    all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    
    for etf_code in CORE_ETFS:
        try:
            data = fetch_history_data(etf_code, start_date, end_date)
            if data['qfq'] is not None and not data['qfq'].empty:
                all_data['qfq'][etf_code] = data['qfq']
                all_data['hfq'][etf_code] = data['hfq']
                all_data['dividend'][etf_code] = data['dividend']
                logger.info(f"  ✅ {etf_code}")
        except Exception as e:
            logger.warning(f"  ❌ {etf_code}: {str(e)}")
    
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

def calculate_metrics(results_df):
    df = results_df.copy()
    
    total_return = (df['nav'].iloc[-1] - 1) * 100
    years = (df['date'].iloc[-1] - df['date'].iloc[0]).days / 365.25
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    
    daily_vol = df['daily_return'].std()
    annualized_vol = daily_vol * np.sqrt(252) * 100
    
    risk_free_rate = 0.03
    sharpe = (annualized_return/100 - risk_free_rate) / (annualized_vol/100) if annualized_vol > 0 else 0
    
    cumulative_max = df['nav'].cummax()
    drawdown = (df['nav'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'years': years,
        'days': len(df)
    }

def run_backtest(all_data, etf_pool, test_name, params):
    """运行回测并返回结果"""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*60}")
    logger.info(f"  测试: {test_name}")
    logger.info(f"  参数: {params}")
    logger.info(f"{'='*60}")
    
    config = BacktestConfig(
        start_date='2018-01-01',
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
    metrics = calculate_metrics(results_df)
    
    print(f"\n【{test_name}】")
    print(f"总收益率: {metrics['total_return']:.2f}%")
    print(f"年化收益率: {metrics['annualized_return']:.2f}%")
    print(f"年化波动率: {metrics['annualized_vol']:.2f}%")
    print(f"夏普比率: {metrics['sharpe']:.2f}")
    print(f"最大回撤: {metrics['max_drawdown']:.2f}%")
    
    return results_df, metrics

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    
    logger.info("加载ETF数据...")
    all_data = load_core_etf_data('2018-01-01', '2025-12-31')
    etf_pool = create_etf_pool()
    
    if len(all_data['qfq']) == 0:
        logger.error("未能获取任何ETF数据")
        return
    
    print("\n" + "="*70)
    print("              策略方案对比测试")
    print("="*70)
    
    all_results = {}
    all_metrics = {}
    
    # 测试1：基准版本（多周期20+60+120）
    results_base, metrics_base = run_backtest(all_data, etf_pool, 
        "基准版本 - 多周期(20/60/120)",
        {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    )
    all_results['基准'] = results_base
    all_metrics['基准'] = metrics_base
    
    # 测试2：方案A - 趋势强度分级
    results_a, metrics_a = run_backtest(all_data, etf_pool,
        "方案A - 趋势强度分级",
        {'lookback_days': 60, 'trailing_stop_pct': 0.08, 'trend_strength_mode': True}
    )
    all_results['方案A'] = results_a
    all_metrics['方案A'] = metrics_a
    
    # 测试3：方案B - 缩短动量周期（10+20+60）
    results_b, metrics_b = run_backtest(all_data, etf_pool,
        "方案B - 缩短动量周期(10/20/60)",
        {'lookback_days': 60, 'trailing_stop_pct': 0.08, 'periods_weights': [(10, 0.4), (20, 0.3), (60, 0.3)]}
    )
    all_results['方案B'] = results_b
    all_metrics['方案B'] = metrics_b
    
    # 测试4：方案A+B - 趋势强度分级 + 缩短动量周期
    results_ab, metrics_ab = run_backtest(all_data, etf_pool,
        "方案A+B - 趋势强度+短周期",
        {
            'lookback_days': 60, 
            'trailing_stop_pct': 0.08, 
            'trend_strength_mode': True,
            'periods_weights': [(10, 0.4), (20, 0.3), (60, 0.3)]
        }
    )
    all_results['方案A+B'] = results_ab
    all_metrics['方案A+B'] = metrics_ab
    
    # 输出汇总对比表
    print("\n" + "="*70)
    print("              测试结果汇总")
    print("="*70)
    
    print(f"\n{'策略':<20} {'年化收益':>10} {'年化波动':>10} {'夏普比率':>10} {'最大回撤':>10}")
    print("-" * 60)
    
    for name, metrics in all_metrics.items():
        print(f"{name:<20} {metrics['annualized_return']:>10.2f}% {metrics['annualized_vol']:>10.2f}% {metrics['sharpe']:>10.2f} {metrics['max_drawdown']:>10.2f}%")
    
    # 绘制净值曲线对比图
    fig, ax = plt.subplots(figsize=(12, 6))
    
    for name, df in all_results.items():
        ax.plot(df['date'], df['nav'], label=name, alpha=0.8)
    
    ax.plot(results_base['date'], results_base['benchmark'], label='沪深300ETF', alpha=0.5, linestyle='--')
    
    ax.set_title('策略净值曲线对比', fontsize=14)
    ax.set_xlabel('日期')
    ax.set_ylabel('净值')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.savefig('strategy_comparison.png', dpi=150, bbox_inches='tight')
    logger.info("策略对比图已保存: strategy_comparison.png")
    
    logger.info("\n" + "="*70)
    logger.info("              测试完成")
    logger.info("="*70)

if __name__ == '__main__':
    main()