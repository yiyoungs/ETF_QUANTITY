import os
import sys
import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/high_elastic_backtest.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

from config import (
    CACHE_DIR, StrategyConfig, BacktestConfig, 
    HIGH_ELASTIC_ETF_POOL, get_all_high_elastic_etfs
)
from data_fetcher import fetch_history_data, load_cached_data
from backtester import Backtester

def prepare_high_elastic_etf_pool():
    """准备高弹性ETF池数据"""
    all_codes = get_all_high_elastic_etfs()
    
    etf_pool_data = []
    for code in all_codes:
        etf_pool_data.append({
            'code': code,
            'name': get_etf_name(code),
            'type': 'ETF',
            'list_date': '2015-01-01',
            'scale': 5000000000
        })
    
    return pd.DataFrame(etf_pool_data)

def get_etf_name(code):
    """根据代码获取ETF名称"""
    name_map = {
        '512480': '半导体ETF',
        '159995': '芯片ETF',
        '512760': '半导体行业ETF',
        '515050': '5G ETF',
        '512000': '券商ETF',
        '515030': '新能源车ETF',
        '159875': '光伏ETF',
        '512660': '军工ETF',
        '512690': '酒ETF',
        '159928': '消费ETF',
        '512010': '医药ETF',
        '159801': '恒生科技ETF',
        '512890': '红利ETF',
        '510880': '红利低波ETF',
        '168204': '煤炭LOF',
        '512400': '有色ETF',
        '511010': '国债ETF',
        '510300': '沪深300ETF'
    }
    return name_map.get(code, '未知ETF')

def load_all_etf_data(etf_codes, start_date, end_date):
    """加载所有ETF数据"""
    all_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    # 必须包含沪深300用于市场模式判断
    required_codes = set(etf_codes + ['510300', '511010'])
    
    for code in required_codes:
        logger.info(f"加载 ETF {code} 数据...")
        cache_data = load_cached_data(code)
        
        if cache_data['qfq'] is not None and not cache_data['qfq'].empty:
            qfq_df = cache_data['qfq']
            qfq_df = qfq_df[(qfq_df['date'] >= start_date) & (qfq_df['date'] <= end_date)]
            all_data['qfq'][code] = qfq_df
        else:
            # 尝试重新获取
            data = fetch_history_data(code, start_date=start_date, end_date=end_date)
            if data['qfq'] is not None:
                all_data['qfq'][code] = data['qfq']
        
        if cache_data['hfq'] is not None and not cache_data['hfq'].empty:
            hfq_df = cache_data['hfq']
            hfq_df = hfq_df[(hfq_df['date'] >= start_date) & (hfq_df['date'] <= end_date)]
            all_data['hfq'][code] = hfq_df
        else:
            data = fetch_history_data(code, start_date=start_date, end_date=end_date)
            if data['hfq'] is not None:
                all_data['hfq'][code] = data['hfq']
        
        if cache_data['dividend'] is not None:
            all_data['dividend'][code] = cache_data['dividend']
    
    return all_data

def main():
    logger.info("=== 高弹性行业/主题 ETF 策略回测 ===")
    
    # 回测配置
    start_date = '2018-01-01'
    end_date = '2025-12-31'
    
    # 获取高弹性ETF池
    etf_pool = prepare_high_elastic_etf_pool()
    logger.info(f"ETF池规模: {len(etf_pool)} 只")
    logger.info(f"ETF列表: {etf_pool['code'].tolist()}")
    
    # 加载数据
    logger.info(f"加载数据: {start_date} ~ {end_date}")
    all_data = load_all_etf_data(get_all_high_elastic_etfs(), start_date, end_date)
    
    logger.info(f"QFQ数据覆盖: {len(all_data['qfq'])} 只ETF")
    logger.info(f"HFQ数据覆盖: {len(all_data['hfq'])} 只ETF")
    
    # 策略参数
    params = {
        'lookback_days': 60,
        'trailing_stop_pct': 0.08,
        'periods_weights': [(20, 0.5), (60, 0.3), (120, 0.2)]
    }
    
    backtest_config = BacktestConfig(
        start_date=start_date,
        end_date=end_date,
        rebalance_day=4  # 周四调仓
    )
    
    # 创建回测器
    backtester = Backtester(
        all_data=all_data,
        etf_pool=etf_pool,
        params=params,
        backtest_config=backtest_config
    )
    
    # 运行回测
    logger.info("启动回测...")
    results_df = backtester.run()
    
    # 计算绩效指标
    total_return = (results_df['nav'].iloc[-1] - 1) * 100
    annual_return = (results_df['nav'].iloc[-1] ** (252 / len(results_df)) - 1) * 100
    annual_vol = results_df['daily_return'].std() * np.sqrt(252) * 100
    sharpe_ratio = results_df['daily_return'].mean() / results_df['daily_return'].std() * np.sqrt(252)
    
    # 计算最大回撤
    cum_return = (1 + results_df['daily_return']).cumprod()
    running_max = cum_return.cummax()
    drawdown = (cum_return - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    # 输出绩效指标
    logger.info("\n" + "="*80)
    logger.info("绩效指标")
    logger.info("="*80)
    logger.info(f"回测时间范围: {start_date} ~ {end_date}")
    logger.info(f"回测天数: {len(results_df)} 天")
    logger.info(f"总收益率: {total_return:.2f}%")
    logger.info(f"年化收益率: {annual_return:.2f}%")
    logger.info(f"年化波动率: {annual_vol:.2f}%")
    logger.info(f"夏普比率: {sharpe_ratio:.2f}")
    logger.info(f"最大回撤: {max_drawdown:.2f}%")
    
    # 保存结果
    os.makedirs('reports', exist_ok=True)
    results_df.to_csv('reports/high_elastic_backtest_results.csv', index=False)
    logger.info(f"\n结果已保存: reports/high_elastic_backtest_results.csv")
    
    # 绘制净值曲线
    plt.figure(figsize=(12, 6))
    plt.plot(results_df['date'], results_df['nav'], label='策略净值', linewidth=2)
    plt.plot(results_df['date'], results_df['benchmark'], label='沪深300基准', linewidth=2, linestyle='--')
    plt.title('高弹性行业/主题ETF策略净值曲线')
    plt.xlabel('日期')
    plt.ylabel('净值')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('reports/high_elastic_backtest_result.png', dpi=150, bbox_inches='tight')
    logger.info(f"图表已保存: reports/high_elastic_backtest_result.png")
    
    # 年度收益统计
    results_df['year'] = results_df['date'].dt.year
    annual_stats = results_df.groupby('year').agg({
        'daily_return': lambda x: (1 + x).prod() - 1
    }).reset_index()
    annual_stats.columns = ['年份', '收益率']
    annual_stats['收益率'] = annual_stats['收益率'] * 100
    
    logger.info("\n" + "="*80)
    logger.info("年度收益统计")
    logger.info("="*80)
    print(annual_stats.to_string(index=False))
    
    annual_stats.to_csv('reports/high_elastic_annual_stats.csv', index=False)
    logger.info(f"\n年度统计已保存: reports/high_elastic_annual_stats.csv")
    
    # 检查异常收益
    extreme_mask = results_df['daily_return'].abs() > 0.10
    if extreme_mask.sum() > 0:
        logger.warning(f"\n发现 {extreme_mask.sum()} 天单日收益超过±10%")
        print(results_df[extreme_mask][['date', 'daily_return']])
    else:
        logger.info("\n✓ 无极端单日收益，数据正常")

if __name__ == '__main__':
    main()