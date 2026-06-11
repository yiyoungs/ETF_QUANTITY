import sys
import logging
import os
import copy
from datetime import datetime
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from config import LOG_DIR, BacktestConfig, DEFAULT_BACKTEST_CONFIG, PARAM_GRID
from data_fetcher import get_etf_pool, fetch_all_etf_data, load_cached_data
from backtester import Backtester
from analyzer import Analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'main.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
# 为各子模块添加独立的 FileHandler
for module_name in ['backtester', 'portfolio_manager', 'signal_engine', 'analyzer', 'data_fetcher']:
    _handler = logging.FileHandler(os.path.join(LOG_DIR, f'{module_name}.log'), encoding='utf-8')
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(name)s - %(message)s'))
    logging.getLogger(module_name).addHandler(_handler)
    logging.getLogger(module_name).setLevel(logging.INFO)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("     ETF 双重动量量化系统 - 主程序入口")
    logger.info("=" * 60)
    
    try:
        logger.info("\n【阶段1：数据获取】")
        logger.info("获取 ETF 池...")
        etf_pool = get_etf_pool()
        logger.info(f"成功获取 {len(etf_pool)} 只 ETF")
        
        logger.info("\n下载历史数据...")
        bt_config = DEFAULT_BACKTEST_CONFIG
        all_data = fetch_all_etf_data(
            etf_pool, 
            start_date=bt_config.start_date,
            end_date=bt_config.end_date
        )
        logger.info(f"成功下载 {len(all_data)} 只 ETF 的历史数据")
        
        if len(all_data) == 0:
            logger.error("未能获取任何 ETF 数据，请检查网络连接或数据源")
            return
        
        logger.info("\n【阶段2：回测执行】")
        logger.info(f"回测时间范围: {bt_config.start_date} ~ {bt_config.end_date}")
        
        backtester = Backtester(all_data, etf_pool, backtest_config=bt_config)
        results_df = backtester.run()
        
        logger.info(f"回测完成，共 {len(results_df)} 个交易日")
        
        logger.info("\n【阶段3：绩效分析】")
        analyzer = Analyzer(results_df)
        metrics = analyzer.generate_report()
        
        analyzer.print_summary()
        
        logger.info("\n【执行完成】")
        logger.info("策略报告已保存至 reports/ 目录")
        
    except Exception as e:
        logger.error(f"程序执行异常: {str(e)}", exc_info=True)
        logger.error("程序终止")
        sys.exit(1)

def run_sensitivity_test(all_data, etf_pool):
    """
    参数敏感性测试
    遍历所有参数组合，运行回测并收集核心指标
    """
    logger.info("=" * 60)
    logger.info("     参数敏感性测试")
    logger.info("=" * 60)
    
    lookback_days_list = PARAM_GRID['lookback_days']
    trailing_stop_list = PARAM_GRID['trailing_stop_pct']
    
    results = []
    
    total_combinations = len(lookback_days_list) * len(trailing_stop_list)
    current_combination = 0
    
    for lookback_days, trailing_stop_pct in itertools.product(lookback_days_list, trailing_stop_list):
        current_combination += 1
        print(f"\n【组合 {current_combination}/{total_combinations}】")
        print(f"Running with lookback={lookback_days}, stop={trailing_stop_pct*100:.0f}%")
        
        params = {
            'lookback_days': lookback_days,
            'trailing_stop_pct': trailing_stop_pct
        }
        
        backtester = Backtester(all_data, etf_pool, params=params)
        print(f"  Backtester lookback_days: {backtester.lookback_days}")
        print(f"  Backtester trailing_stop_pct: {backtester.trailing_stop_pct}")
        
        results_df = backtester.run()
        
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        results.append({
            'lookback_days': lookback_days,
            'trailing_stop_pct': trailing_stop_pct,
            'annualized_return': metrics.get('annualized_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'profit_factor': metrics.get('profit_factor', 0)
        })
        
        print(f"  年化收益: {metrics.get('annualized_return', 0) * 100:.2f}%")
        print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        print(f"  盈亏比: {metrics.get('profit_factor', 0):.2f}")
    
    results_df = pd.DataFrame(results)
    
    pivot_sharpe = results_df.pivot(
        index='lookback_days', 
        columns='trailing_stop_pct', 
        values='sharpe_ratio'
    )
    
    pivot_return = results_df.pivot(
        index='lookback_days', 
        columns='trailing_stop_pct', 
        values='annualized_return'
    )
    
    os.makedirs('reports', exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(pivot_sharpe.values, cmap='RdYlGn', interpolation='nearest')
    
    ax.set_xticks(np.arange(len(trailing_stop_list)))
    ax.set_yticks(np.arange(len(lookback_days_list)))
    ax.set_xticklabels([f'{p*100:.0f}%' for p in trailing_stop_list])
    ax.set_yticklabels(lookback_days_list)
    
    for i in range(len(lookback_days_list)):
        for j in range(len(trailing_stop_list)):
            text = ax.text(j, i, f'{pivot_sharpe.values[i, j]:.2f}',
                           ha='center', va='center', color='black')
    
    ax.set_title('夏普比率参数热力图')
    ax.set_xlabel('移动止损比例')
    ax.set_ylabel('回溯天数')
    plt.colorbar(im, ax=ax, label='夏普比率')
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'sensitivity_heatmap.png'), dpi=150)
    plt.close()
    logger.info("热力图已保存至: reports/sensitivity_heatmap.png")
    
    results_df.to_csv(os.path.join('reports', 'sensitivity_results.csv'), index=False, encoding='utf-8-sig')
    logger.info("参数测试结果已保存至: reports/sensitivity_results.csv")
    
    print("\n" + "=" * 80)
    print("                    参数敏感性测试结果汇总")
    print("=" * 80)
    print(results_df.to_string(index=False))
    print("=" * 80)
    
    return results_df

def run_split_test(all_data, etf_pool):
    """
    分段回测
    将回测期等分为前半段和后半段，分别运行回测
    """
    logger.info("=" * 60)
    logger.info("     分段回测")
    logger.info("=" * 60)
    
    dates = set()
    if 'qfq' in all_data:
        for etf_code, df in all_data['qfq'].items():
            if not df.empty:
                dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())
    dates = sorted(list(dates))
    
    mid_idx = len(dates) // 2
    split_date = dates[mid_idx]
    
    logger.info(f"回测时间分割点: {split_date}")
    logger.info(f"前半段: {dates[0]} ~ {split_date}")
    logger.info(f"后半段: {split_date} ~ {dates[-1]}")
    
    config1 = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    config1.start_date = dates[0]
    config1.end_date = split_date
    
    logger.info("\n【前半段回测】")
    backtester1 = Backtester(all_data, etf_pool, backtest_config=config1)
    results_df1 = backtester1.run()
    analyzer1 = Analyzer(results_df1)
    metrics1 = analyzer1.calculate_metrics()
    
    config2 = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    config2.start_date = split_date
    config2.end_date = dates[-1]
    
    logger.info("\n【后半段回测】")
    backtester2 = Backtester(all_data, etf_pool, backtest_config=config2)
    results_df2 = backtester2.run()
    analyzer2 = Analyzer(results_df2)
    metrics2 = analyzer2.calculate_metrics()
    
    print("\n" + "=" * 80)
    print("                    分段回测结果对比")
    print("=" * 80)
    print(f"{'指标':<15} {'前半段':<15} {'后半段':<15} {'稳定性':<10}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('年化收益率', 'annualized_return', lambda x: f'{x * 100:.2f}%'),
        ('夏普比率', 'sharpe_ratio', lambda x: f'{x:.2f}'),
        ('最大回撤', 'max_drawdown', lambda x: f'{x * 100:.2f}%'),
        ('盈亏比', 'profit_factor', lambda x: f'{x:.2f}')
    ]
    
    stability_scores = []
    for name, key, formatter in metrics_to_compare:
        val1 = metrics1.get(key, 0)
        val2 = metrics2.get(key, 0)
        
        if key == 'max_drawdown':
            diff = abs(val1 - val2)
            stable = diff < 0.10
        else:
            diff = abs(val1 - val2) / max(abs(val1), abs(val2), 0.0001)
            stable = diff < 0.30
        
        stability_scores.append(stable)
        
        print(f"{name:<15} {formatter(val1):<15} {formatter(val2):<15} {'稳定' if stable else '不稳定':<10}")
    
    overall_stable = all(stability_scores)
    print("-" * 80)
    print(f"整体稳定性评估: {'策略表现稳定' if overall_stable else '策略表现不稳定'}")
    print("=" * 80)
    
    return metrics1, metrics2

def worst_day_analysis(results_df, all_data):
    """
    最差交易日复盘
    找到总资产回撤最大的5个交易日并输出详细信息
    """
    logger.info("=" * 60)
    logger.info("     最差交易日复盘")
    logger.info("=" * 60)
    
    df = results_df.copy()
    df['daily_return'] = df['nav'].pct_change().fillna(0)
    
    worst_days = df.sort_values('daily_return').head(5)
    
    print("\n" + "=" * 80)
    print("                    最差5个交易日复盘")
    print("=" * 80)
    
    hfq_data = all_data.get('hfq', all_data)
    
    for idx, row in worst_days.iterrows():
        date = row['date']
        total_value = row['total_value']
        positions = row['positions']
        is_rebalance = row['is_rebalance']
        
        print(f"\n【日期】{date.strftime('%Y-%m-%d')}")
        print(f"【当日总资产】{total_value:,.2f} 元")
        print(f"【当日收益率】{row['daily_return'] * 100:.2f}%")
        print(f"【是否调仓日】{'是' if is_rebalance else '否'}")
        print("【持仓明细】")
        
        for etf_code, shares in positions.items():
            if shares > 0:
                if etf_code in hfq_data:
                    etf_df = hfq_data[etf_code]
                    price_data = etf_df[etf_df['date'] <= date]
                    if not price_data.empty:
                        current_price = price_data['close'].iloc[-1]
                        print(f"  - {etf_code}: {shares} 份 @ {current_price:.2f}")
                    else:
                        print(f"  - {etf_code}: {shares} 份 (价格数据缺失)")
                else:
                    print(f"  - {etf_code}: {shares} 份")
        
        print("-" * 60)
    
    print("=" * 80)
    
    return worst_days

if __name__ == '__main__':
    main()