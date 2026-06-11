"""
ETF双重动量策略 - 完整7年周期回测
覆盖2018-2025年，输出详细绩效报告
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

# 核心ETF列表
CORE_ETFS = [
    # 宽基
    '510050', '510300', '510500', '159915', '512100',
    # 行业
    '512660', '512480', '512690', '512010', '515030',
    # 避险
    '511010', '518880'
]

def load_core_etf_data(start_date, end_date):
    """加载核心ETF数据"""
    logger = logging.getLogger(__name__)
    logger.info("开始加载核心ETF数据...")
    
    all_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    for etf_code in CORE_ETFS:
        try:
            data = fetch_history_data(etf_code, start_date, end_date)
            if data['qfq'] is not None and not data['qfq'].empty:
                all_data['qfq'][etf_code] = data['qfq']
                all_data['hfq'][etf_code] = data['hfq']
                all_data['dividend'][etf_code] = data['dividend']
                logger.info(f"  ✅ {etf_code}: {len(data['qfq'])} 条数据")
            else:
                logger.warning(f"  ❌ {etf_code}: 数据获取失败")
        except Exception as e:
            logger.error(f"  ❌ {etf_code}: {str(e)}")
    
    logger.info(f"数据加载完成，共 {len(all_data['qfq'])} 只ETF")
    return all_data

def create_etf_pool():
    """创建ETF池"""
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

def calculate_annual_metrics(results_df):
    """按年度计算绩效指标"""
    df = results_df.copy()
    df['year'] = df['date'].dt.year
    
    annual_metrics = []
    
    for year in range(2018, 2026):
        year_df = df[df['year'] == year]
        
        if year_df.empty:
            continue
        
        # 年度收益
        year_return = (year_df['nav'].iloc[-1] / year_df['nav'].iloc[0] - 1) * 100
        
        # 年度最大回撤
        year_df['cummax'] = year_df['nav'].cummax()
        year_df['drawdown'] = (year_df['nav'] - year_df['cummax']) / year_df['cummax']
        max_drawdown = year_df['drawdown'].min() * 100
        
        # 权益仓位占比（非国债ETF持仓天数比例）
        cash_etf = StrategyConfig.CASH_ETF_CODE
        
        def has_risk_assets(positions):
            if not positions:
                return False
            for code, shares in positions.items():
                if code != cash_etf and shares > 0:
                    return True
            return False
        
        year_df['has_risk'] = year_df['positions'].apply(has_risk_assets)
        equity_ratio = year_df['has_risk'].mean() * 100
        
        annual_metrics.append({
            '年份': year,
            '年度收益': year_return,
            '最大回撤': max_drawdown,
            '权益仓位占比': equity_ratio
        })
    
    return pd.DataFrame(annual_metrics)

def analyze_period(results_df, start_date, end_date, period_name):
    """分析特定时间段"""
    mask = (results_df['date'] >= pd.to_datetime(start_date)) & \
           (results_df['date'] <= pd.to_datetime(end_date))
    period_df = results_df[mask]
    
    if period_df.empty:
        logger.warning(f"{period_name} 期间无数据")
        return None
    
    period_return = (period_df['nav'].iloc[-1] / period_df['nav'].iloc[0] - 1) * 100
    
    period_df['cummax'] = period_df['nav'].cummax()
    period_df['drawdown'] = (period_df['nav'] - period_df['cummax']) / period_df['cummax']
    max_drawdown = period_df['drawdown'].min() * 100
    
    cash_etf = StrategyConfig.CASH_ETF_CODE
    
    def count_risk_assets(positions):
        count = 0
        if positions:
            for code, shares in positions.items():
                if code != cash_etf and shares > 0:
                    count += 1
        return count
    
    period_df['risk_count'] = period_df['positions'].apply(count_risk_assets)
    avg_risk_count = period_df['risk_count'].mean()
    
    return {
        'period': period_name,
        'return': period_return,
        'max_drawdown': max_drawdown,
        'avg_risk_count': avg_risk_count,
        'days': len(period_df),
        'first_nav': period_df['nav'].iloc[0],
        'last_nav': period_df['nav'].iloc[-1],
        'positions': period_df[['date', 'positions']]
    }

def find_max_drawdown_period(results_df):
    """找到最大回撤发生的时间段"""
    df = results_df.copy()
    df['cummax'] = df['nav'].cummax()
    df['drawdown'] = (df['nav'] - df['cummax']) / df['cummax']
    
    max_drawdown_idx = df['drawdown'].idxmin()
    max_drawdown_val = df['drawdown'].min()
    
    # 找到回撤开始点（创新高之后开始下跌）
    peak_before = df.loc[:max_drawdown_idx]
    peak_idx = peak_before['cummax'].idxmax()
    
    # 找到回撤恢复点
    recovery_df = df.loc[max_drawdown_idx:]
    recovery_mask = recovery_df['nav'] >= df.loc[peak_idx, 'cummax']
    
    if recovery_mask.any():
        recovery_idx = recovery_mask.idxmax()
    else:
        recovery_idx = df.index[-1]
    
    return {
        'peak_date': df.loc[peak_idx, 'date'],
        'trough_date': df.loc[max_drawdown_idx, 'date'],
        'recovery_date': df.loc[recovery_idx, 'date'],
        'max_drawdown_pct': max_drawdown_val * 100,
        'peak_nav': df.loc[peak_idx, 'nav'],
        'trough_nav': df.loc[max_drawdown_idx, 'nav'],
        'recovery_nav': df.loc[recovery_idx, 'nav'],
        'duration_days': (df.loc[recovery_idx, 'date'] - df.loc[peak_idx, 'date']).days
    }

def plot_nav_comparison(results_df, hfq_data):
    """绘制策略净值与沪深300ETF对比图"""
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.plot(results_df['date'], results_df['nav'], label='策略净值', linewidth=2)
    
    if '510300' in hfq_data:
        bench_df = hfq_data['510300'].copy()
        bench_df = bench_df[bench_df['date'] >= results_df['date'].iloc[0]]
        bench_df = bench_df[bench_df['date'] <= results_df['date'].iloc[-1]]
        bench_df['benchmark_nav'] = bench_df['close'] / bench_df['close'].iloc[0]
        ax.plot(bench_df['date'], bench_df['benchmark_nav'], label='沪深300ETF(买入持有)', linewidth=2, alpha=0.7)
    
    ax.set_title('策略净值 vs 沪深300ETF基准', fontsize=16)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('净值', fontsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs('reports', exist_ok=True)
    plt.savefig('reports/nav_comparison_7years.png', dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("策略净值对比图已保存: reports/nav_comparison_7years.png")

def main():
    global logger
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("    ETF 双重动量策略 - 完整7年周期回测")
    logger.info("=" * 60)
    
    # 回测配置
    start_date = '2018-01-01'
    end_date = '2025-12-31'
    
    logger.info(f"\n回测时间范围: {start_date} ~ {end_date}")
    logger.info(f"回测参数: lookback_days=60, trailing_stop_pct=0.08")
    
    # 加载核心ETF数据
    all_data = load_core_etf_data(start_date, end_date)
    
    if len(all_data['qfq']) == 0:
        logger.error("未能获取任何ETF数据")
        return
    
    # 创建ETF池
    etf_pool = create_etf_pool()
    
    # 创建回测配置
    config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_day=4
    )
    
    # 创建回测器
    params = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08
    }
    
    backtester = Backtester(
        all_data=all_data,
        etf_pool=etf_pool,
        params=params,
        backtest_config=config
    )
    
    # 运行回测
    logger.info("\n开始回测...")
    results_df = backtester.run()
    
    # 计算整体绩效
    logger.info("\n" + "=" * 60)
    logger.info("【整体绩效指标】")
    logger.info("=" * 60)
    
    total_return = (results_df['nav'].iloc[-1] - 1) * 100
    years = (results_df['date'].iloc[-1] - results_df['date'].iloc[0]).days / 365.25
    annualized_return = ((1 + total_return / 100) ** (1 / years) - 1) * 100
    
    daily_vol = results_df['daily_return'].std()
    annualized_vol = daily_vol * np.sqrt(252) * 100
    
    risk_free_rate = 0.03
    sharpe = (annualized_return/100 - risk_free_rate) / (annualized_vol/100) if annualized_vol > 0 else 0
    
    cumulative_max = results_df['nav'].cummax()
    drawdown = (results_df['nav'] - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100
    
    print(f"\n总收益率: {total_return:.2f}%")
    print(f"年化收益率: {annualized_return:.2f}%")
    print(f"年化波动率: {annualized_vol:.2f}%")
    print(f"夏普比率: {sharpe:.2f}")
    print(f"最大回撤: {max_drawdown:.2f}%")
    print(f"回测天数: {len(results_df)} 天")
    print(f"回测年限: {years:.2f} 年")
    
    # 年度分年度收益表
    logger.info("\n" + "=" * 60)
    logger.info("【年度分年度收益表】")
    logger.info("=" * 60)
    
    annual_df = calculate_annual_metrics(results_df)
    
    print(f"\n{'年份':<8} {'年度收益':<12} {'最大回撤':<12} {'权益仓位占比':<12}")
    print("-" * 50)
    for _, row in annual_df.iterrows():
        print(f"{row['年份']:<8} {row['年度收益']:>8.2f}% {row['最大回撤']:>10.2f}% {row['权益仓位占比']:>10.2f}%")
    
    # 绘制净值对比图
    plot_nav_comparison(results_df, all_data['hfq'])
    
    # 关键时段复盘 - 2024年9-10月
    logger.info("\n" + "=" * 60)
    logger.info("【关键时段复盘 - 2024年9-10月】")
    logger.info("=" * 60)
    
    period_result = analyze_period(results_df, '2024-09-01', '2024-10-31', '2024年9-10月')
    
    if period_result:
        print(f"\n期间收益: {period_result['return']:.2f}%")
        print(f"期间最大回撤: {period_result['max_drawdown']:.2f}%")
        print(f"平均权益持仓数量: {period_result['avg_risk_count']:.1f} 只")
        print(f"期间天数: {period_result['days']} 天")
        
        # 输出关键日期的仓位变化
        positions_df = period_result['positions']
        important_dates = ['2024-09-01', '2024-09-15', '2024-10-01', '2024-10-15', '2024-10-28']
        
        print("\n关键日期仓位:")
        for date_str in important_dates:
            date_mask = positions_df['date'] == pd.to_datetime(date_str)
            if date_mask.any():
                pos = positions_df[date_mask]['positions'].iloc[0]
                pos_str = ", ".join([f"{k}:{v}" for k, v in pos.items() if v > 0])
                print(f"  {date_str}: {pos_str}")
    
    # 最大回撤时间段分析
    logger.info("\n" + "=" * 60)
    logger.info("【最大回撤时间段分析】")
    logger.info("=" * 60)
    
    drawdown_info = find_max_drawdown_period(results_df)
    
    print(f"\n最大回撤幅度: {drawdown_info['max_drawdown_pct']:.2f}%")
    print(f"峰值日期: {drawdown_info['peak_date'].strftime('%Y-%m-%d')} (净值: {drawdown_info['peak_nav']:.2f})")
    print(f"谷底日期: {drawdown_info['trough_date'].strftime('%Y-%m-%d')} (净值: {drawdown_info['trough_nav']:.2f})")
    print(f"恢复日期: {drawdown_info['recovery_date'].strftime('%Y-%m-%d')} (净值: {drawdown_info['recovery_nav']:.2f})")
    print(f"持续天数: {drawdown_info['duration_days']} 天")
    
    # 保存结果
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/backtest_results_7years.csv', index=False, encoding='utf-8-sig')
    annual_df.to_csv('reports/annual_metrics_7years.csv', index=False, encoding='utf-8-sig')
    
    logger.info("\n回测完成！结果已保存至 reports/ 目录")

if __name__ == '__main__':
    main()