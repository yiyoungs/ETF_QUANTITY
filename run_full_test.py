"""
综合测试套件 - 使用真实A股ETF历史数据
已修复数据获取接口，现在使用 ak.fund_etf_hist_sina 获取真实数据
"""
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
from data_fetcher import get_etf_pool, fetch_history_data
from backtester import Backtester
from analyzer import Analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'full_test.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

TEST_START_DATE = '2018-01-01'
TEST_END_DATE = '2022-08-05'

MAJOR_ETFS = [
    '510050',  # 上证50ETF
    '510300',  # 沪深300ETF
    '510500',  # 中证500ETF
    '512100',  # 中证1000ETF
    '159915',  # 创业板ETF
    '512880',  # 证券ETF
    '512480',  # 医药ETF
    '512690',  # 酒ETF
    '511010',  # 国债ETF
]

def get_major_etf_pool():
    etf_info = [
        {'code': '510050', 'name': '上证50ETF', 'type': 'ETF', 'list_date': '2005-02-23'},
        {'code': '510300', 'name': '沪深300ETF', 'type': 'ETF', 'list_date': '2012-05-04'},
        {'code': '510500', 'name': '中证500ETF', 'type': 'ETF', 'list_date': '2013-02-06'},
        {'code': '512100', 'name': '中证1000ETF', 'type': 'ETF', 'list_date': '2015-06-26'},
        {'code': '159915', 'name': '创业板ETF', 'type': 'ETF', 'list_date': '2011-09-20'},
        {'code': '512880', 'name': '证券ETF', 'type': 'ETF', 'list_date': '2016-03-11'},
        {'code': '512480', 'name': '医药ETF', 'type': 'ETF', 'list_date': '2015-04-16'},
        {'code': '512690', 'name': '酒ETF', 'type': 'ETF', 'list_date': '2016-05-20'},
        {'code': '511010', 'name': '国债ETF', 'type': 'ETF', 'list_date': '2013-03-05'},
    ]
    return pd.DataFrame(etf_info)

def fetch_major_etf_data():
    """下载主流ETF真实数据"""
    all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    
    print("正在下载主流ETF真实数据...")
    for etf_code in MAJOR_ETFS:
        try:
            data = fetch_history_data(etf_code, TEST_START_DATE, TEST_END_DATE)
            if data['qfq'] is not None and not data['qfq'].empty:
                all_data['qfq'][etf_code] = data['qfq']
                all_data['hfq'][etf_code] = data['hfq']
                all_data['dividend'][etf_code] = data['dividend']
                print(f"  ✓ {etf_code} 下载成功 ({len(data['qfq'])} 条)")
            else:
                print(f"  ✗ {etf_code} 下载失败")
        except Exception as e:
            print(f"  ✗ {etf_code} 下载异常: {str(e)}")
    
    return all_data

def run_benchmark_test(all_data, etf_pool):
    """测试1：默认参数基准回测"""
    print("\n" + "=" * 80)
    print("  📊 测试1：默认参数基准回测")
    print("=" * 80)
    print(f"参数：lookback_days=60, trailing_stop_pct=0.08")
    print(f"时间范围：{TEST_START_DATE} ~ {TEST_END_DATE}")
    print("=" * 80)
    
    bt_config = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    bt_config.start_date = TEST_START_DATE
    bt_config.end_date = TEST_END_DATE
    
    params = {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params, backtest_config=bt_config)
    results_df = backtester.run()
    
    analyzer = Analyzer(results_df)
    metrics = analyzer.generate_report()
    
    analyzer.print_summary()
    
    print("\n【持仓暴露度分析】")
    print(f"  风险资产仓位占比: {metrics.get('risk_exposure_ratio', 0) * 100:.2f}%")
    print(f"  平均持仓数量: {metrics.get('avg_position_count', 0):.1f} 只")
    print(f"  最大连续空仓天数: {metrics.get('max_cash_streak', 0)} 天")
    print(f"  最大回撤恢复天数: {metrics.get('max_recovery_days', 0)} 天")
    print(f"  平均回撤恢复天数: {metrics.get('avg_recovery_days', 0):.1f} 天")
    
    return results_df, metrics

def run_sensitivity_test(all_data, etf_pool):
    """测试2：参数敏感性测试"""
    print("\n" + "=" * 80)
    print("  📈 测试2：参数敏感性测试")
    print("=" * 80)
    print(f"时间范围：{TEST_START_DATE} ~ {TEST_END_DATE}")
    print("=" * 80)
    
    lookback_days_list = PARAM_GRID['lookback_days']
    trailing_stop_list = PARAM_GRID['trailing_stop_pct']
    
    print(f"测试网格：lookback_days={lookback_days_list}")
    print(f"         trailing_stop_pct={trailing_stop_list}")
    print(f"总组合数：{len(lookback_days_list) * len(trailing_stop_list)}")
    print("-" * 80)
    
    results = []
    
    total_combinations = len(lookback_days_list) * len(trailing_stop_list)
    current_combination = 0
    
    for lookback_days, trailing_stop_pct in itertools.product(lookback_days_list, trailing_stop_list):
        current_combination += 1
        
        bt_config = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
        bt_config.start_date = TEST_START_DATE
        bt_config.end_date = TEST_END_DATE
        
        params = {
            'lookback_days': lookback_days,
            'trailing_stop_pct': trailing_stop_pct
        }
        
        print(f"\n【组合 {current_combination}/{total_combinations}】")
        print(f"参数: lookback_days={lookback_days}, trailing_stop_pct={trailing_stop_pct*100:.0f}%")
        
        backtester = Backtester(all_data, etf_pool, params=params, backtest_config=bt_config)
        
        print(f"  ✓ Backtester.lookback_days = {backtester.lookback_days}")
        print(f"  ✓ Backtester.trailing_stop_pct = {backtester.trailing_stop_pct}")
        
        results_df = backtester.run()
        
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        results.append({
            'lookback_days': lookback_days,
            'trailing_stop_pct': trailing_stop_pct,
            'annualized_return': metrics.get('annualized_return', 0),
            'sharpe_ratio': metrics.get('sharpe_ratio', 0),
            'max_drawdown': metrics.get('max_drawdown', 0),
            'profit_factor': metrics.get('profit_factor', 0),
            'max_recovery_days': metrics.get('max_recovery_days', 0)
        })
        
        print(f"  结果: 年化={metrics.get('annualized_return', 0)*100:.2f}%, 夏普={metrics.get('sharpe_ratio', 0):.2f}, 回撤={metrics.get('max_drawdown', 0)*100:.2f}%, 盈亏比={metrics.get('profit_factor', 0):.2f}")
    
    print("\n" + "=" * 80)
    print("  参数敏感性测试结果汇总")
    print("=" * 80)
    
    results_df = pd.DataFrame(results)
    
    print(f"{'lookback':<10} {'stop':<8} {'年化收益':<10} {'夏普':<8} {'最大回撤':<10} {'盈亏比':<8} {'恢复天数':<8}")
    print("-" * 80)
    for _, row in results_df.iterrows():
        print(f"{row['lookback_days']:<10} {row['trailing_stop_pct']*100:<8.0f}% {row['annualized_return']*100:<10.2f}% {row['sharpe_ratio']:<8.2f} {row['max_drawdown']*100:<10.2f}% {row['profit_factor']:<8.2f} {int(row['max_recovery_days']):<8}")
    
    os.makedirs('reports', exist_ok=True)
    
    pivot_sharpe = results_df.pivot(
        index='lookback_days', 
        columns='trailing_stop_pct', 
        values='sharpe_ratio'
    )
    
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

    ax.set_title('夏普比率参数热力图', fontsize=14)
    ax.set_xlabel('移动止损比例', fontsize=12)
    ax.set_ylabel('回溯天数', fontsize=12)
    plt.colorbar(im, ax=ax, label='夏普比率')
    plt.tight_layout()
    plt.savefig(os.path.join('reports', 'sensitivity_heatmap_real.png'), dpi=150)
    plt.close()
    
    print(f"\n✅ 热力图已保存")
    results_df.to_csv(os.path.join('reports', 'sensitivity_results_real.csv'), index=False, encoding='utf-8-sig')
    print(f"✅ 参数测试结果已保存")
    
    return results_df

def run_split_test(all_data, etf_pool):
    """测试3：分段回测"""
    print("\n" + "=" * 80)
    print("  📉 测试3：分段回测")
    print("=" * 80)
    
    period1_start = '2018-01-01'
    period1_end = '2020-03-31'
    period2_start = '2020-04-01'
    period2_end = TEST_END_DATE
    
    print(f"前半段（2018熊市+2020疫情）: {period1_start} ~ {period1_end}")
    print(f"后半段（2021结构牛+2022回调）: {period2_start} ~ {period2_end}")
    print("=" * 80)
    
    print("\n【前半段回测】")
    config1 = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    config1.start_date = period1_start
    config1.end_date = period1_end
    
    backtester1 = Backtester(all_data, etf_pool, backtest_config=config1)
    results1 = backtester1.run()
    analyzer1 = Analyzer(results1)
    metrics1 = analyzer1.calculate_metrics()
    
    print("前半段核心指标:")
    print(f"  年化收益: {metrics1.get('annualized_return', 0)*100:.2f}%")
    print(f"  夏普比率: {metrics1.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics1.get('max_drawdown', 0)*100:.2f}%")
    print(f"  盈亏比: {metrics1.get('profit_factor', 0):.2f}")
    
    print("\n【后半段回测】")
    config2 = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    config2.start_date = period2_start
    config2.end_date = period2_end
    
    backtester2 = Backtester(all_data, etf_pool, backtest_config=config2)
    results2 = backtester2.run()
    analyzer2 = Analyzer(results2)
    metrics2 = analyzer2.calculate_metrics()
    
    print("后半段核心指标:")
    print(f"  年化收益: {metrics2.get('annualized_return', 0)*100:.2f}%")
    print(f"  夏普比率: {metrics2.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics2.get('max_drawdown', 0)*100:.2f}%")
    print(f"  盈亏比: {metrics2.get('profit_factor', 0):.2f}")
    
    print("\n【两段对比】")
    print(f"{'指标':<12} {'前半段':<12} {'后半段':<12} {'变化':<12}")
    print("-" * 50)
    
    ret1, ret2 = metrics1.get('annualized_return', 0), metrics2.get('annualized_return', 0)
    print(f"年化收益   {ret1*100:>10.2f}%    {ret2*100:>10.2f}%    {'+' if ret2 > ret1 else ''}{(ret2-ret1)*100:>10.2f}%")
    
    sharpe1, sharpe2 = metrics1.get('sharpe_ratio', 0), metrics2.get('sharpe_ratio', 0)
    print(f"夏普比率   {sharpe1:>10.2f}     {sharpe2:>10.2f}     {'+' if sharpe2 > sharpe1 else ''}{(sharpe2-sharpe1):>10.2f}")
    
    dd1, dd2 = metrics1.get('max_drawdown', 0), metrics2.get('max_drawdown', 0)
    print(f"最大回撤   {dd1*100:>10.2f}%    {dd2*100:>10.2f}%    {'+' if dd2 > dd1 else ''}{(dd2-dd1)*100:>10.2f}%")
    
    pf1, pf2 = metrics1.get('profit_factor', 0), metrics2.get('profit_factor', 0)
    print(f"盈亏比     {pf1:>10.2f}     {pf2:>10.2f}     {'+' if pf2 > pf1 else ''}{(pf2-pf1):>10.2f}")
    
    return metrics1, metrics2, results1, results2

def worst_day_analysis(results_df, all_data):
    """测试4：最差交易日复盘"""
    print("\n" + "=" * 80)
    print("  🔥 测试4：最差交易日复盘")
    print("=" * 80)
    
    df = results_df.copy()
    df['daily_return'] = df['nav'].pct_change().fillna(0)
    
    worst_days = df.sort_values('daily_return').head(5)
    
    print("\n最差5个交易日:")
    print("-" * 60)
    
    hfq_data = all_data.get('hfq', all_data)
    
    for idx, row in worst_days.iterrows():
        date = pd.to_datetime(row['date'])
        total_value = row['total_value']
        positions = row['positions']
        
        print(f"\n【日期】{date.strftime('%Y-%m-%d')}")
        print(f"【当日总资产】{total_value:,.2f} 元")
        print(f"【当日收益率】{row['daily_return'] * 100:.2f}%")
        print(f"【是否调仓日】{'是' if row['is_rebalance'] else '否'}")
        
        has_stop_loss = False
        if idx > 0:
            prev_positions = df.iloc[idx-1]['positions']
            for code in prev_positions:
                if code not in positions:
                    has_stop_loss = True
                    break
        
        print(f"【是否有止损】{'是' if has_stop_loss else '否'}")
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
                        print(f"  - {etf_code}: {shares} 份")
                else:
                    print(f"  - {etf_code}: {shares} 份")
        
        print("-" * 60)
    
    return worst_days

def main():
    """主测试入口"""
    print("=" * 80)
    print("      ETF 双重动量量化系统 - 综合测试套件")
    print("=" * 80)
    print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"测试范围: {TEST_START_DATE} ~ {TEST_END_DATE}")
    print("=" * 80)
    
    try:
        print("\n【阶段1：获取真实ETF数据】")
        etf_pool = get_major_etf_pool()
        print(f"ETF池: {len(etf_pool)} 只主流ETF")
        
        all_data = fetch_major_etf_data()
        print(f"成功下载: {len(all_data['qfq'])} 只ETF")
        
        if len(all_data['qfq']) == 0:
            print("❌ 未能获取任何ETF数据")
            return
        
        print("\n【阶段2：运行测试】")
        
        benchmark_results, benchmark_metrics = run_benchmark_test(all_data, etf_pool)
        
        sensitivity_results = run_sensitivity_test(all_data, etf_pool)
        
        metrics1, metrics2, results1, results2 = run_split_test(all_data, etf_pool)
        
        worst_days = worst_day_analysis(benchmark_results, all_data)
        
        print("\n" + "=" * 80)
        print("      🎉 综合测试完成")
        print("=" * 80)
        print("\n📁 输出文件:")
        print("  - reports/sensitivity_results_real.csv    # 参数敏感性结果")
        print("  - reports/sensitivity_heatmap_real.png    # 夏普比率热力图")
        print("  - logs/full_test.log                     # 测试日志")
        
        print("\n📊 基准测试核心指标:")
        print(f"  年化收益: {benchmark_metrics.get('annualized_return', 0)*100:.2f}%")
        print(f"  夏普比率: {benchmark_metrics.get('sharpe_ratio', 0):.2f}")
        print(f"  最大回撤: {benchmark_metrics.get('max_drawdown', 0)*100:.2f}%")
        print(f"  盈亏比: {benchmark_metrics.get('profit_factor', 0):.2f}")
        
        logger.info("综合测试套件执行完成")
        
    except Exception as e:
        print(f"\n❌ 测试执行异常: {str(e)}")
        logger.error(f"测试执行异常: {str(e)}", exc_info=True)
        sys.exit(1)

if __name__ == '__main__':
    main()
