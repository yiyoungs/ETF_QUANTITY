import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from data_fetcher import fetch_all_etf_data, get_etf_pool
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/debug_backtest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=" * 60)
    logger.info("     调试默认参数回测 (60/0.08)")
    logger.info("=" * 60)
    
    try:
        logger.info("\n【阶段1：获取真实数据】")
        etf_pool = get_etf_pool()
        logger.info(f"获取到 {len(etf_pool)} 只 ETF")
        all_data = fetch_all_etf_data(etf_pool)
        logger.info(f"获取到 {len(all_data['qfq'])} 只 ETF 的历史数据")
        
        logger.info("\n【阶段2：运行默认参数回测】")
        params = {
            'lookback_days': 60,
            'trailing_stop_pct': 0.08
        }
        backtester = Backtester(all_data, etf_pool, params=params)
        logger.info(f"参数: lookback_days={backtester.lookback_days}, trailing_stop_pct={backtester.trailing_stop_pct}")
        
        results_df = backtester.run()
        
        logger.info("\n【阶段3：分析结果】")
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        logger.info("\n" + "=" * 60)
        logger.info("                    绩效指标")
        logger.info("=" * 60)
        logger.info(f"年化收益率: {metrics.get('annualized_return', 0) * 100:.2f}%")
        logger.info(f"年化波动率: {metrics.get('annualized_volatility', 0) * 100:.2f}%")
        logger.info(f"夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
        logger.info(f"最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
        logger.info(f"盈亏比: {metrics.get('profit_factor', 0):.2f}")
        logger.info(f"胜率: {metrics.get('win_rate', 0) * 100:.2f}%")
        logger.info(f"风险资产占比: {metrics.get('risk_asset_ratio', 0) * 100:.2f}%")
        logger.info("=" * 60)
        
        analyzer.plot_nav(output_dir='reports')
        analyzer.plot_drawdown(output_dir='reports')
        analyzer.generate_report(output_dir='reports')
        
        logger.info("\n【阶段4：输出2018年熊市交易日志】")
        bear_market_df = results_df[(results_df['date'] >= '2018-01-01') & (results_df['date'] <= '2018-12-31')]
        logger.info(f"2018年交易日数: {len(bear_market_df)}")
        
        monthly_stats = bear_market_df.resample('M', on='date').agg({
            'total_assets': ['first', 'last'],
            'daily_return': 'sum'
        })
        logger.info("\n2018年每月表现:")
        for idx, row in monthly_stats.iterrows():
            start_assets = row['total_assets']['first']
            end_assets = row['total_assets']['last']
            ret = row['daily_return']['sum']
            logger.info(f"  {idx.strftime('%Y-%m')}: 期初 {start_assets:.0f} -> 期末 {end_assets:.0f}, 月收益 {ret*100:.2f}%")
        
        logger.info("\n【调试完成】")
        
    except Exception as e:
        logger.error(f"程序执行异常: {str(e)}", exc_info=True)
        logger.error("程序终止")

if __name__ == "__main__":
    main()