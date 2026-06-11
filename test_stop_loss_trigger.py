import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from config import BacktestConfig
import logging

logging.basicConfig(level=logging.DEBUG)

def generate_stop_loss_test_data():
    """生成能触发止损的测试数据"""
    dates = pd.date_range('2017-01-01', periods=500, freq='B')
    n_days = len(dates)
    
    test_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    
    np.random.seed(42)
    
    prices = np.linspace(3.0, 4.0, 150)
    prices = np.concatenate([prices, np.linspace(4.0, 3.68, 50)])
    prices = np.concatenate([prices, np.linspace(3.68, 4.2, n_days - 200)])
    
    prices += np.random.normal(0, 0.005, n_days)
    
    for etf_code in ['510050', '510300', '510500']:
        close = prices.copy()
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': close * (1 + np.random.uniform(-0.005, 0.005, n_days)),
            'high': close * (1 + np.random.uniform(0, 0.008, n_days)),
            'low': close * (1 + np.random.uniform(-0.008, 0, n_days)),
            'close': close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        test_data['hfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'close': close,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        
        test_data['dividend'][etf_code] = pd.DataFrame({
            'ex_date': [],
            'dividend_per_share': []
        })
    
    test_data['qfq']['511010'] = pd.DataFrame({
        'date': dates,
        'open': 100.01 + np.random.uniform(-0.01, 0.01, n_days),
        'high': 100.02 + np.random.uniform(0, 0.01, n_days),
        'low': 100.00 + np.random.uniform(-0.01, 0, n_days),
        'close': 100.01 + np.random.uniform(-0.005, 0.005, n_days),
        'volume': np.ones(n_days) * 1000000,
        'amount': np.ones(n_days) * 100000000
    })
    
    test_data['hfq']['511010'] = test_data['qfq']['511010'][['date', 'close', 'volume', 'amount']].copy()
    test_data['dividend']['511010'] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    
    etf_pool = pd.DataFrame({
        'code': ['510050', '510300', '510500'],
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF'],
        'type': ['ETF'] * 3,
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    return test_data, etf_pool

def test_stop_loss_trigger():
    """测试止损是否被触发"""
    print("=" * 80)
    print("           止损触发测试")
    print("=" * 80)
    
    all_data, etf_pool = generate_stop_loss_test_data()
    
    original_start = BacktestConfig.START_DATE
    original_end = BacktestConfig.END_DATE
    
    BacktestConfig.START_DATE = '2017-07-01'
    BacktestConfig.END_DATE = '2018-06-30'
    
    params = {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params)
    
    print(f"\n参数: lookback_days={backtester.lookback_days}, trailing_stop_pct={backtester.trailing_stop_pct*100:.0f}%")
    print(f"PortfolioManager参数: lookback_days={backtester.portfolio_manager.lookback_days}, trailing_stop_pct={backtester.portfolio_manager.trailing_stop_pct*100:.0f}%")
    
    results_df = backtester.run()
    
    analyzer = Analyzer(results_df)
    metrics = analyzer.calculate_metrics()
    
    print(f"\n回测结果:")
    print(f"  年化收益: {metrics.get('annualized_return', 0) * 100:.2f}%")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    
    print("\n价格走势分析:")
    print("-" * 60)
    
    qfq_data = all_data['qfq']['510050']
    print(f"2017-07-01 价格: {qfq_data.iloc[0]['close']:.2f}")
    print(f"最高点价格: {qfq_data['high'].max():.2f}")
    print(f"回撤阶段价格从 {qfq_data.iloc[149]['close']:.2f} 跌到 {qfq_data.iloc[199]['close']:.2f}")
    print(f"理论回撤幅度: {(qfq_data.iloc[149]['close'] - qfq_data.iloc[199]['close']) / qfq_data.iloc[149]['close'] * 100:.2f}%")
    
    BacktestConfig.START_DATE = original_start
    BacktestConfig.END_DATE = original_end

if __name__ == "__main__":
    test_stop_loss_trigger()