import sys
import logging
import os
from datetime import datetime

from config import LOG_DIR, BacktestConfig
from data_fetcher import get_etf_pool, fetch_all_etf_data, load_cached_data
from backtester import Backtester
from analyzer import Analyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'main.log')),
        logging.StreamHandler()
    ]
)
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
        all_data = fetch_all_etf_data(
            etf_pool, 
            start_date=BacktestConfig.START_DATE,
            end_date=BacktestConfig.END_DATE
        )
        logger.info(f"成功下载 {len(all_data)} 只 ETF 的历史数据")
        
        if len(all_data) == 0:
            logger.error("未能获取任何 ETF 数据，请检查网络连接或数据源")
            return
        
        logger.info("\n【阶段2：回测执行】")
        logger.info(f"回测时间范围: {BacktestConfig.START_DATE} ~ {BacktestConfig.END_DATE}")
        
        backtester = Backtester(all_data, etf_pool)
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

if __name__ == '__main__':
    main()