import sys
import logging
import os
import pandas as pd
import numpy as np

from config import LOG_DIR, BacktestConfig
from backtester import Backtester
from analyzer import Analyzer
from main import run_sensitivity_test, run_split_test, worst_day_analysis

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'stress_test.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def generate_test_data(n_days=1200):
    """生成测试数据，包含足够的上涨期让策略持有风险资产"""
    dates = pd.date_range('2016-01-01', periods=n_days, freq='B')
    
    etf_list = ['510050', '510300', '510500', '512000', '512880', '511010']
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    base_prices = {
        '510050': (2.0, 3.5),
        '510300': (4.0, 6.0),
        '510500': (5.0, 7.0),
        '512000': (2.0, 4.0),
        '512880': (1.5, 3.0),
        '511010': (100.0, 101.0)
    }
    
    for etf_code in etf_list:
        if etf_code == '511010':
            start_price, end_price = base_prices[etf_code]
            trend = np.linspace(start_price, end_price, n_days)
            noise = np.random.normal(0, 0.001 * end_price, n_days)
            qfq_close = trend + noise
        else:
            start_price, end_price = base_prices[etf_code]
            
            phase1_end = int(n_days * 0.4)
            phase2_end = int(n_days * 0.55)
            phase3_end = int(n_days * 0.75)
            phase4_end = n_days
            
            phase1 = np.linspace(start_price, start_price * 1.4, phase1_end)
            phase2 = np.linspace(start_price * 1.4, start_price * 1.1, phase2_end - phase1_end)
            phase3 = np.linspace(start_price * 1.1, start_price * 1.5, phase3_end - phase2_end)
            phase4 = np.linspace(start_price * 1.5, end_price, phase4_end - phase3_end)
            
            trend = np.concatenate([phase1, phase2, phase3, phase4])
            noise = np.random.normal(0, 0.01 * start_price, n_days)
            qfq_close = trend + noise
        
        hfq_close = qfq_close * 1.01
        
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': qfq_close * (1 + np.random.uniform(-0.01, 0.01, n_days)),
            'high': qfq_close * (1 + np.random.uniform(0, 0.015, n_days)),
            'low': qfq_close * (1 + np.random.uniform(-0.015, 0, n_days)),
            'close': qfq_close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        test_data['hfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'close': hfq_close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        if etf_code != '511010':
            dividend_dates = [dates[200], dates[400], dates[600], dates[800], dates[1000]]
            dividend_amounts = [0.10 + np.random.uniform(-0.02, 0.03) for _ in range(5)]
            test_data['dividend'][etf_code] = pd.DataFrame({
                'ex_date': dividend_dates,
                'dividend_per_share': dividend_amounts
            })
        else:
            test_data['dividend'][etf_code] = pd.DataFrame({
                'ex_date': [],
                'dividend_per_share': []
            })
    
    etf_pool = pd.DataFrame({
        'code': etf_list[:-1],
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF', '券商ETF', '医药ETF'],
        'type': ['ETF'] * 5,
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    return test_data, etf_pool

def run_single_backtest(all_data, etf_pool, lookback_days, trailing_stop_pct):
    """单参数回测"""
    print(f"\nRunning with lookback={lookback_days}, stop={trailing_stop_pct*100:.0f}%")
    
    params = {
        'lookback_days': lookback_days,
        'trailing_stop_pct': trailing_stop_pct
    }
    
    backtester = Backtester(all_data, etf_pool, params=params)
    print(f"  Backtester lookback_days: {backtester.lookback_days}")
    print(f"  Backtester trailing_stop_pct: {backtester.trailing_stop_pct}")
    print(f"  PortfolioManager lookback_days: {backtester.portfolio_manager.lookback_days}")
    
    results_df = backtester.run()
    
    analyzer = Analyzer(results_df)
    metrics = analyzer.calculate_metrics()
    
    print(f"  年化收益: {metrics.get('annualized_return', 0) * 100:.2f}%")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"  盈亏比: {metrics.get('profit_factor', 0):.2f}")
    
    return metrics

def main():
    logger.info("=" * 60)
    logger.info("     ETF 双重动量量化系统 - 压力测试")
    logger.info("=" * 60)
    
    try:
        logger.info("\n【阶段1：生成测试数据】")
        all_data, etf_pool = generate_test_data(n_days=1200)
        logger.info(f"生成 {len(all_data['qfq'])} 只 ETF 的测试数据")
        
        logger.info("\n【阶段2：单参数测试（默认参数 60/0.08）】")
        default_metrics = run_single_backtest(all_data, etf_pool, 60, 0.08)
        
        logger.info("\n【阶段3：参数敏感性测试】")
        sensitivity_results = run_sensitivity_test(all_data, etf_pool)
        
        logger.info("\n【阶段3：分段回测】")
        metrics1, metrics2 = run_split_test(all_data, etf_pool)
        
        logger.info("\n【阶段4：最差交易日复盘】")
        backtester = Backtester(all_data, etf_pool)
        results_df = backtester.run()
        worst_days = worst_day_analysis(results_df, all_data)
        
        logger.info("\n【压力测试完成】")
        logger.info("测试报告已保存至 reports/ 目录")
        
    except Exception as e:
        logger.error(f"程序执行异常: {str(e)}", exc_info=True)
        logger.error("程序终止")
        sys.exit(1)

if __name__ == '__main__':
    main()