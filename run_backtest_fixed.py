"""
验证回测修复 - 运行默认参数回测
"""
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime
import matplotlib.pyplot as plt

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG, StrategyConfig
from data_fetcher import fetch_history_data
from backtester import Backtester

# 主要ETF列表
MAJOR_ETFS = [
    '510050', '510300', '510500', '512100',
    '159915', '512880', '512480', '512690', '511010'
]

def load_etf_data():
    """加载ETF数据"""
    logger.info("开始加载ETF数据...")
    
    all_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    for etf_code in MAJOR_ETFS:
        data = fetch_history_data(etf_code, '2018-01-01', '2025-12-31')
        if data['qfq'] is not None and not data['qfq'].empty:
            all_data['qfq'][etf_code] = data['qfq']
            all_data['hfq'][etf_code] = data['hfq']
            all_data['dividend'][etf_code] = data['dividend']
            logger.info(f"  ✅ {etf_code}: {len(data['qfq'])} 条数据")
    
    logger.info(f"数据加载完成，共 {len(all_data['qfq'])} 只ETF")
    return all_data

def create_etf_pool():
    """创建ETF池"""
    return pd.DataFrame({
        'code': MAJOR_ETFS,
        'name': ['50ETF', '沪深300ETF', '中证500ETF', '中证100ETF',
                 '创业板ETF', '证券ETF', '半导体ETF', '酒ETF', '国债ETF'],
        'type': ['ETF'] * 9,
        'list_date': ['2015-02-16', '2015-05-19', '2013-03-15', '2014-12-15',
                      '2012-01-09', '2013-05-21', '2019-06-12', '2019-05-06', '2015-01-21'],
        'scale': [5000000000] * 9
    })

def analyze_results(results_df):
    """分析回测结果"""
    # 检查是否有异常单日收益
    daily_returns = results_df['daily_return']
    abnormal_mask = daily_returns.abs() > 0.12
    abnormal_count = abnormal_mask.sum()
    
    logger.info("\n" + "=" * 60)
    logger.info("回测结果分析")
    logger.info("=" * 60)
    
    if abnormal_count > 0:
        logger.warning(f"⚠️ 发现 {abnormal_count} 个异常单日收益（>12%）:")
        abnormal_days = results_df[abnormal_mask][['date', 'daily_return', 'total_value', 'positions']]
        for _, row in abnormal_days.iterrows():
            logger.warning(f"  {row['date'].strftime('%Y-%m-%d')}: {row['daily_return']:.2%}")
    else:
        logger.info("✅ 没有发现异常单日收益")
    
    # 2018-10-12 专项检查
    oct_12 = results_df[results_df['date'] == '2018-10-12']
    if not oct_12.empty:
        oct_12_return = oct_12.iloc[0]['daily_return']
        oct_12_value = oct_12.iloc[0]['total_value']
        logger.info(f"\n2018-10-12 专项检查:")
        logger.info(f"  当日收益: {oct_12_return:.2%}")
        logger.info(f"  当日总资产: {oct_12_value:,.2f}")
        if abs(oct_12_return) > 0.12:
            logger.warning(f"  ⚠️ 该日收益异常!")
        else:
            logger.info(f"  ✅ 该日收益正常")
    
    # 计算绩效指标
    nav = results_df['nav']
    total_return = (nav.iloc[-1] - 1) * 100
    years = (results_df['date'].iloc[-1] - results_df['date'].iloc[0]).days / 365.25
    annualized_return = ((nav.iloc[-1] / nav.iloc[0]) ** (1/years) - 1) * 100
    
    # 年化波动率
    daily_vol = daily_returns.std()
    annualized_vol = daily_vol * np.sqrt(252) * 100
    
    # 夏普比率
    risk_free_rate = 0.03
    sharpe = (annualized_return/100 - risk_free_rate) / (annualized_vol/100) if annualized_vol > 0 else 0
    
    # 最大回撤
    cumulative_max = nav.cummax()
    drawdown = (nav - cumulative_max) / cumulative_max
    max_drawdown = drawdown.min() * 100
    
    logger.info("\n" + "=" * 60)
    logger.info("绩效指标")
    logger.info("=" * 60)
    logger.info(f"回测时间范围: {results_df['date'].iloc[0].strftime('%Y-%m-%d')} ~ {results_df['date'].iloc[-1].strftime('%Y-%m-%d')}")
    logger.info(f"回测天数: {len(results_df)} 天")
    logger.info(f"总收益率: {total_return:.2f}%")
    logger.info(f"年化收益率: {annualized_return:.2f}%")
    logger.info(f"年化波动率: {annualized_vol:.2f}%")
    logger.info(f"夏普比率: {sharpe:.2f}")
    logger.info(f"最大回撤: {max_drawdown:.2f}%")
    
    return {
        'total_return': total_return,
        'annualized_return': annualized_return,
        'annualized_vol': annualized_vol,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown,
        'abnormal_days': abnormal_count
    }

def plot_results(results_df):
    """绘制结果图表"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 净值曲线
    ax1 = axes[0]
    ax1.plot(results_df['date'], results_df['nav'], label='策略净值')
    ax1.plot(results_df['date'], results_df['benchmark'], label='沪深300', alpha=0.7)
    ax1.set_title('净值曲线', fontsize=14)
    ax1.set_xlabel('日期')
    ax1.set_ylabel('净值')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 回撤曲线
    ax2 = axes[1]
    nav = results_df['nav']
    cumulative_max = nav.cummax()
    drawdown = (nav - cumulative_max) / cumulative_max * 100
    ax2.fill_between(results_df['date'], drawdown, 0, alpha=0.3, color='red')
    ax2.plot(results_df['date'], drawdown, color='red', linewidth=1)
    ax2.set_title('回撤曲线', fontsize=14)
    ax2.set_xlabel('日期')
    ax2.set_ylabel('回撤 (%)')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('reports/backtest_result_fixed.png', dpi=150, bbox_inches='tight')
    logger.info("图表已保存: reports/backtest_result_fixed.png")

def main():
    logger.info("=" * 60)
    logger.info("ETF 双重动量策略回测 (修复版)")
    logger.info("=" * 60)
    
    # 加载数据（清除缓存）
    if hasattr(load_etf_data, 'cached_data'):
        delattr(load_etf_data, 'cached_data')
    all_data = load_etf_data()
    
    # 创建ETF池
    etf_pool = create_etf_pool()
    
    # 创建回测配置
    config = BacktestConfig(
        start_date='2018-01-02',
        end_date='2025-12-31',
        rebalance_day=4
    )
    
    # 创建回测器
    params = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08
    }
    
    logger.info("\n回测参数:")
    logger.info(f"  回溯天数: {params['lookback_days']}")
    logger.info(f"  移动止损: {params['trailing_stop_pct']:.0%}")
    logger.info(f"  回测时间: {config.start_date} ~ {config.end_date}")
    
    backtester = Backtester(
        all_data=all_data,
        etf_pool=etf_pool,
        params=params,
        backtest_config=config
    )
    
    # 运行回测
    logger.info("\n开始回测...")
    results_df = backtester.run()
    
    # 分析结果
    metrics = analyze_results(results_df)
    
    # 绘制图表
    plot_results(results_df)
    
    # 保存结果
    results_df.to_csv('reports/backtest_results_fixed.csv', index=False)
    logger.info("\n结果已保存: reports/backtest_results_fixed.csv")
    
    return metrics

if __name__ == '__main__':
    metrics = main()
