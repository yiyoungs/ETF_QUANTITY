"""快速回测验证 —— 使用本地缓存的 ETF 数据"""
import os
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.WARNING, format='%(levelname)s - %(name)s - %(message)s')

from config import CACHE_DIR, StrategyConfig
from backtester import Backtester
from analyzer import Analyzer

cached_files = [f for f in os.listdir(CACHE_DIR) if f.endswith('_hfq.csv')]
print('可用 hfq 缓存文件数:', len(cached_files))

if len(cached_files) >= 3:
    all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
    codes = []
    for path in cached_files[:12]:
        code = path.split('_')[0]
        codes.append(code)
        try:
            hfq_df = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
            qfq_df = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])
            all_data['hfq'][code] = hfq_df
            all_data['qfq'][code] = qfq_df
        except Exception as e:
            print(f'  读取失败 {code}: {e}')

    # 确保 510300 和 511010 存在（沪深300基准 & 国债ETF）
    if '510300' not in all_data['hfq']:
        all_data['hfq']['510300'] = all_data['hfq'][codes[0]].copy()
        all_data['qfq']['510300'] = all_data['qfq'][codes[0]].copy()
    if StrategyConfig.CASH_ETF_CODE not in all_data['hfq']:
        all_data['hfq'][StrategyConfig.CASH_ETF_CODE] = all_data['hfq'][codes[0]].copy()
        all_data['qfq'][StrategyConfig.CASH_ETF_CODE] = all_data['qfq'][codes[0]].copy()

    etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})
    print('ETF 池大小:', len(etf_pool))
    print('样本日期范围:')
    for k in list(all_data['hfq'].keys())[:3]:
        d = all_data['hfq'][k]
        print(f'  {k}: {d["date"].iloc[0].date()} ~ {d["date"].iloc[-1].date()}, {len(d)} 行')

    bt = Backtester(all_data, etf_pool)
    results = bt.run()
    print('\n=== 回测完成 ===')
    print('交易日:', len(results))
    print('初始净值: %.4f' % results['nav'].iloc[0])
    print('最终净值: %.4f' % results['nav'].iloc[-1])

    analyzer = Analyzer(results)
    metrics = analyzer.calculate_metrics()
    print('\n【核心指标】')
    print('  年化收益: %.2f%%' % (metrics['annualized_return']*100))
    print('  年化波动: %.2f%%' % (metrics['annualized_volatility']*100))
    print('  最大回撤: %.2f%%' % (metrics['max_drawdown']*100))
    print('  夏普: %.2f' % metrics['sharpe_ratio'])
    print('  胜率: %.1f%%' % (metrics['win_rate']*100))
    print('  盈亏比: %.2f' % metrics['profit_factor'])
    print('  最长恢复天数: %d' % metrics['max_recovery_days'])
    print('  平均恢复天数: %.0f' % metrics['avg_recovery_days'])
    print('\n【持仓暴露度】')
    print('  风险资产仓位占比: %.1f%%' % (metrics['risk_exposure_ratio']*100))
    print('  平均持仓数: %.1f' % metrics['avg_position_count'])
    print('  最长连续空仓天数: %d' % metrics['max_cash_streak'])
else:
    print('缓存文件不足 3 个，无法进行实际回测')
