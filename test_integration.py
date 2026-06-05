import numpy as np
import pandas as pd
from backtester import Backtester
from analyzer import Analyzer
from config import StrategyConfig

def create_test_data():
    n_days = 1200
    dates = pd.date_range('2018-01-01', periods=n_days, freq='B')
    
    etf_list = ['510050', '510300', '510500', '512000', '512880', '511010']
    base_prices = {
        '510050': (2.0, 3.0),
        '510300': (4.0, 5.5),
        '510500': (5.0, 6.5),
        '512000': (1.5, 2.2),
        '512880': (1.8, 2.6),
        '511010': (100.0, 100.8)
    }
    
    qfq_data = {}
    hfq_data = {}
    
    for etf_code in etf_list:
        start_price, end_price = base_prices[etf_code]
        qfq_data[etf_code] = pd.DataFrame({
            'date': dates,
            'open': np.linspace(start_price, end_price, n_days) + np.random.normal(0, 0.02, n_days),
            'high': np.linspace(start_price * 1.01, end_price * 1.01, n_days) + np.random.normal(0, 0.02, n_days),
            'low': np.linspace(start_price * 0.99, end_price * 0.99, n_days) + np.random.normal(0, 0.02, n_days),
            'close': np.linspace(start_price, end_price, n_days) + np.random.normal(0, 0.015, n_days),
            'volume': np.ones(n_days) * 10000000,
            'amount': np.ones(n_days) * 200000000
        })
        
        hfq_data[etf_code] = pd.DataFrame({
            'date': dates,
            'close': np.linspace(start_price, end_price, n_days) + np.random.normal(0, 0.015, n_days),
            'volume': np.ones(n_days) * 10000000,
            'amount': np.ones(n_days) * 200000000
        })
    
    dividend_data = {
        '510050': pd.DataFrame({
            'ex_date': [dates[120], dates[360], dates[600], dates[880]],
            'dividend_per_share': [0.15, 0.12, 0.18, 0.16]
        }),
        '510300': pd.DataFrame({
            'ex_date': [dates[200], dates[450], dates[720]],
            'dividend_per_share': [0.10, 0.14, 0.12]
        }),
        '510500': pd.DataFrame({
            'ex_date': [dates[280], dates[520]],
            'dividend_per_share': [0.08, 0.10]
        }),
        '512000': pd.DataFrame({
            'ex_date': [dates[350], dates[650]],
            'dividend_per_share': [0.06, 0.07]
        }),
        '512880': pd.DataFrame({
            'ex_date': [dates[400], dates[700]],
            'dividend_per_share': [0.09, 0.08]
        }),
        '511010': pd.DataFrame({
            'ex_date': [dates[300], dates[550], dates[800]],
            'dividend_per_share': [0.30, 0.35, 0.40]
        })
    }
    
    test_data = {
        'qfq': qfq_data,
        'hfq': hfq_data,
        'dividend': dividend_data
    }
    
    test_pool = pd.DataFrame({
        'code': etf_list,
        'name': ['上证50ETF', '沪深300ETF', '中证500ETF', '券商ETF', '银行ETF', '国债ETF'],
        'type': ['ETF', 'ETF', 'ETF', 'ETF', 'ETF', 'ETF'],
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    return test_data, test_pool

if __name__ == '__main__':
    print("=" * 70)
    print("          ETF 双重动量量化系统 V1.2 - 集成测试")
    print("=" * 70)
    
    print("\n【步骤1：创建测试数据】")
    test_data, test_pool = create_test_data()
    print(f"  创建了 {len(test_data['qfq'])} 只 ETF 的模拟数据")
    
    print("\n【步骤2：运行回测】")
    backtester = Backtester(test_data, test_pool)
    results_df = backtester.run()
    print(f"  回测完成，共 {len(results_df)} 个交易日")
    
    print("\n【步骤3：绩效分析】")
    analyzer = Analyzer(results_df)
    analyzer.calculate_metrics()
    analyzer.generate_report()
    analyzer.plot_nav('reports')
    analyzer.plot_drawdown('reports')
    
    print("\n【绩效摘要】")
    print("=" * 60)
    print("          ETF 双重动量策略绩效摘要")
    print("=" * 60)
    print(f"\n回测时间: {results_df['date'].min().strftime('%Y-%m-%d')} ~ {results_df['date'].max().strftime('%Y-%m-%d')}")
    print(f"初始资金: {StrategyConfig.INITIAL_CAPITAL:,} 元")
    
    print("\n【核心指标】")
    final_nav = results_df['nav'].iloc[-1]
    total_return = (final_nav - 1) * 100
    annualized_return = analyzer.metrics.get('annualized_return', 0) * 100
    sharpe_ratio = analyzer.metrics.get('sharpe_ratio', 0)
    max_drawdown = analyzer.metrics.get('max_drawdown', 0) * 100
    win_rate = analyzer.metrics.get('win_rate', 0) * 100
    profit_factor = analyzer.metrics.get('profit_factor', 0)
    
    print(f"  总收益率: {total_return:.2f}%")
    print(f"  年化收益率: {annualized_return:.2f}%")
    print(f"  年化波动率: {analyzer.metrics.get('annualized_volatility', 0) * 100:.2f}%")
    print(f"  夏普比率: {sharpe_ratio:.2f}")
    print(f"  最大回撤: {max_drawdown:.2f}%")
    print(f"  胜率: {win_rate:.2f}%")
    print(f"  盈亏比: {profit_factor:.2f}")
    
    print("\n" + "=" * 70)
    print("                    测试完成")
    print("=" * 70)
