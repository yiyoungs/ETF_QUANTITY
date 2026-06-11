"""
验证修复效果的集成测试：
1. 默认参数(60/0.08)回测
2. 参数敏感性测试（验证不同参数产生不同结果）
3. BacktestConfig 状态隔离验证
"""
import pandas as pd
import numpy as np
from backtester import Backtester
from analyzer import Analyzer
from config import BacktestConfig, DEFAULT_BACKTEST_CONFIG, StrategyConfig
import copy
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
)
logger = logging.getLogger(__name__)

def generate_test_data():
    """生成包含明显趋势变化和崩盘的测试数据"""
    dates = pd.date_range('2016-01-01', periods=1200, freq='B')
    n_days = len(dates)
    np.random.seed(42)
    
    test_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    
    # 3只权益ETF，走势有显著差异
    etf_profiles = [
        ('510050', 3.0, 0.0),   # 基准
        ('510300', 4.0, 0.15),  # 波动更大
        ('510500', 2.5, -0.10), # 走势相反
    ]
    
    for etf_code, base, offset in etf_profiles:
        prices = np.ones(n_days) * base
        # 上涨阶段
        prices[:200] = np.linspace(base, base * 1.5, 200)
        # 崩盘阶段
        crash_depth = 0.30 + offset  # 不同ETF崩盘深度不同
        prices[200:280] = np.linspace(base * 1.5, base * (1.5 - crash_depth), 80)
        # 恢复阶段
        prices[280:500] = np.linspace(base * (1.5 - crash_depth), base * 1.6, 220)
        # 第二次下跌
        prices[500:560] = np.linspace(base * 1.6, base * 1.2, 60)
        # 再恢复
        prices[560:] = np.linspace(base * 1.2, base * 1.8, n_days - 560)
        
        prices += np.random.normal(0, 0.015, n_days)
        
        test_data['qfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'open': prices * (1 + np.random.uniform(-0.005, 0.005, n_days)),
            'high': prices * (1 + np.random.uniform(0, 0.01, n_days)),
            'low': prices * (1 + np.random.uniform(-0.01, 0, n_days)),
            'close': prices,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        test_data['hfq'][etf_code] = pd.DataFrame({
            'date': dates,
            'close': prices,
            'volume': np.ones(n_days) * 50000000,
            'amount': np.ones(n_days) * 150000000
        })
        test_data['dividend'][etf_code] = pd.DataFrame({'ex_date': [], 'dividend_per_share': []})
    
    # 现金ETF
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

def test_default_backtest():
    """测试1：默认参数(60/0.08)回测"""
    print("=" * 80)
    print("  测试1：默认参数(60/0.08)回测")
    print("=" * 80)
    
    all_data, etf_pool = generate_test_data()
    
    bt_config = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    bt_config.start_date = '2018-01-01'
    bt_config.end_date = '2020-12-31'
    
    params = {'lookback_days': 60, 'trailing_stop_pct': 0.08}
    backtester = Backtester(all_data, etf_pool, params=params, backtest_config=bt_config)
    results_df = backtester.run()
    
    analyzer = Analyzer(results_df)
    metrics = analyzer.calculate_metrics()
    
    print(f"\n回测结果 (默认参数 60/0.08):")
    print(f"  年化收益: {metrics.get('annualized_return', 0) * 100:.2f}%")
    print(f"  夏普比率: {metrics.get('sharpe_ratio', 0):.2f}")
    print(f"  最大回撤: {metrics.get('max_drawdown', 0) * 100:.2f}%")
    print(f"  盈亏比: {metrics.get('profit_factor', 0):.2f}")
    print(f"  最大回撤恢复天数: {metrics.get('max_recovery_days', 0)}")
    print(f"  平均回撤恢复天数: {metrics.get('avg_recovery_days', 0):.1f}")
    print(f"  风险资产仓位占比: {metrics.get('risk_exposure_ratio', 0) * 100:.2f}%")
    
    return metrics

def test_sensitivity():
    """测试2：参数敏感性测试（验证不同参数产生不同结果）"""
    print("\n" + "=" * 80)
    print("  测试2：参数敏感性测试")
    print("=" * 80)
    
    all_data, etf_pool = generate_test_data()
    
    param_combinations = [
        {'lookback_days': 40, 'trailing_stop_pct': 0.05},
        {'lookback_days': 40, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.08},
        {'lookback_days': 60, 'trailing_stop_pct': 0.15},
        {'lookback_days': 80, 'trailing_stop_pct': 0.08},
        {'lookback_days': 80, 'trailing_stop_pct': 0.15},
    ]
    
    results = []
    
    for params in param_combinations:
        lb = params['lookback_days']
        ts = params['trailing_stop_pct']
        
        # 每次创建新的配置实例，确保状态隔离
        bt_config = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
        bt_config.start_date = '2018-01-01'
        bt_config.end_date = '2020-12-31'
        
        backtester = Backtester(all_data, etf_pool, params=params, backtest_config=bt_config)
        results_df = backtester.run()
        
        analyzer = Analyzer(results_df)
        metrics = analyzer.calculate_metrics()
        
        result = {
            'lookback': lb,
            'stop': ts,
            'return': metrics.get('annualized_return', 0),
            'sharpe': metrics.get('sharpe_ratio', 0),
            'drawdown': metrics.get('max_drawdown', 0),
            'recovery': metrics.get('max_recovery_days', 0)
        }
        results.append(result)
        
        print(f"  lookback={lb}, stop={ts*100:.0f}%: "
              f"return={result['return']*100:.2f}%, "
              f"sharpe={result['sharpe']:.2f}, "
              f"drawdown={result['drawdown']*100:.2f}%, "
              f"recovery={result['recovery']}d")
    
    # 检查结果是否有差异
    returns = [r['return'] for r in results]
    drawdowns = [r['drawdown'] for r in results]
    
    return_diff = max(returns) - min(returns)
    drawdown_diff = max(drawdowns) - min(drawdowns)
    
    print(f"\n  收益率差异范围: {return_diff*100:.2f}%")
    print(f"  回撤差异范围: {drawdown_diff*100:.2f}%")
    
    if return_diff > 0.001 or drawdown_diff > 0.001:
        print("  ✅ 参数敏感性测试通过：不同参数产生不同结果")
    else:
        print("  ⚠️ 参数敏感性测试警告：不同参数结果差异很小")
    
    return results

def test_config_isolation():
    """测试3：BacktestConfig 状态隔离验证"""
    print("\n" + "=" * 80)
    print("  测试3：BacktestConfig 状态隔离验证")
    print("=" * 80)
    
    all_data, etf_pool = generate_test_data()
    
    # 保存原始配置
    original_config = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    
    # 第一次回测：修改配置
    config1 = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    config1.start_date = '2018-01-01'
    config1.end_date = '2019-06-30'
    
    backtester1 = Backtester(all_data, etf_pool, backtest_config=config1)
    results1 = backtester1.run()
    
    # 验证原始配置未被修改
    assert DEFAULT_BACKTEST_CONFIG.start_date == original_config.start_date, \
        f"DEFAULT_BACKTEST_CONFIG.start_date 被污染: {DEFAULT_BACKTEST_CONFIG.start_date}"
    assert DEFAULT_BACKTEST_CONFIG.end_date == original_config.end_date, \
        f"DEFAULT_BACKTEST_CONFIG.end_date 被污染: {DEFAULT_BACKTEST_CONFIG.end_date}"
    
    # 第二次回测：使用不同时间段
    config2 = copy.deepcopy(DEFAULT_BACKTEST_CONFIG)
    config2.start_date = '2019-07-01'
    config2.end_date = '2020-12-31'
    
    backtester2 = Backtester(all_data, etf_pool, backtest_config=config2)
    results2 = backtester2.run()
    
    # 验证两次回测的时间范围不同
    range1_start = results1['date'].iloc[0].strftime('%Y-%m-%d')
    range1_end = results1['date'].iloc[-1].strftime('%Y-%m-%d')
    range2_start = results2['date'].iloc[0].strftime('%Y-%m-%d')
    range2_end = results2['date'].iloc[-1].strftime('%Y-%m-%d')
    
    print(f"  第一次回测范围: {range1_start} ~ {range1_end}")
    print(f"  第二次回测范围: {range2_start} ~ {range2_end}")
    
    if range1_start != range2_start:
        print("  ✅ 状态隔离验证通过：两次回测使用不同时间范围")
    else:
        print("  ❌ 状态隔离验证失败：两次回测时间范围相同")
    
    # 验证原始配置仍然未被修改
    assert DEFAULT_BACKTEST_CONFIG.start_date == original_config.start_date
    assert DEFAULT_BACKTEST_CONFIG.end_date == original_config.end_date
    print("  ✅ DEFAULT_BACKTEST_CONFIG 未被污染")

if __name__ == '__main__':
    test_default_backtest()
    test_sensitivity()
    test_config_isolation()
    print("\n" + "=" * 80)
    print("  所有测试完成")
    print("=" * 80)
