import pandas as pd
import numpy as np
from config import StrategyConfig
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'signal_engine.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def calculate_rolling_liquidity(df, window=20):
    """
    计算滚动流动性指标（基于 T-1 日数据）
    :param df: ETF 历史数据 DataFrame
    :param window: 滚动窗口大小
    :return: 添加了 rolling_amount 和 listing_days 列的 DataFrame
    """
    df = df.copy()
    df['rolling_amount'] = df['amount'].rolling(window=window).mean().shift(1)
    df['listing_days'] = (df['date'] - df['date'].iloc[0]).dt.days
    return df

def filter_liquidity(all_data, etf_pool, target_date):
    """
    根据流动性规则过滤 ETF 池（基于 T-1 日数据）
    :param all_data: 所有 ETF 的历史数据字典 {code: df}
    :param etf_pool: ETF 池 DataFrame
    :param target_date: 目标日期
    :return: 符合流动性要求的 ETF 代码列表
    """
    qualified = []
    
    for etf_code in etf_pool['code'].unique():
        if etf_code not in all_data:
            continue
        
        df = all_data[etf_code]
        df = calculate_rolling_liquidity(df)
        
        mask = df['date'] <= pd.to_datetime(target_date)
        if mask.sum() == 0:
            continue
        
        latest_data = df[mask].iloc[-1]
        
        if pd.isna(latest_data['rolling_amount']):
            continue
        
        if latest_data['rolling_amount'] >= StrategyConfig.MIN_AMOUNT and \
           latest_data['listing_days'] >= StrategyConfig.MIN_LISTING_DAYS:
            qualified.append(etf_code)
    
    return qualified

def calc_cross_section_momentum(all_data, qualified_codes, target_date, lookback_days=60):
    """
    计算截面动量（基于 T-1 日数据）
    :param all_data: 所有 ETF 的历史数据字典
    :param qualified_codes: 符合流动性要求的 ETF 代码列表
    :param target_date: 目标日期
    :param lookback_days: 回溯天数
    :return: 按动量排序的 DataFrame
    """
    momentum_results = []
    target_dt = pd.to_datetime(target_date)
    
    for etf_code in qualified_codes:
        if etf_code not in all_data:
            continue
        
        df = all_data[etf_code].copy()
        df = df[df['date'] <= target_dt].copy()
        df.reset_index(drop=True, inplace=True)
        
        if len(df) < lookback_days + 1:
            continue
        
        end_idx = len(df) - 1
        start_idx = max(0, end_idx - lookback_days)
        
        start_price = df['close'].iloc[start_idx]
        end_price = df['close'].iloc[end_idx]
        
        momentum = (end_price - start_price) / start_price
        
        momentum_results.append({
            'code': etf_code,
            'momentum': momentum,
            'lookback_days': lookback_days
        })
    
    if not momentum_results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(momentum_results)
    result_df = result_df.sort_values('momentum', ascending=False).reset_index(drop=True)
    
    return result_df

def calc_time_series_momentum(all_data, etf_code, target_date):
    """
    计算时序动量（基于 T-1 日数据）
    :param all_data: 所有 ETF 的历史数据字典
    :param etf_code: ETF 代码
    :param target_date: 目标日期
    :return: 包含 MA60、MA60_slope 和 signal 的字典
    """
    if etf_code not in all_data:
        return {'code': etf_code, 'ma60': np.nan, 'ma60_slope': np.nan, 'signal': False}
    
    df = all_data[etf_code].copy()
    df = df[df['date'] <= pd.to_datetime(target_date)]
    
    if len(df) < 70:
        return {'code': etf_code, 'ma60': np.nan, 'ma60_slope': np.nan, 'signal': False}
    
    df.loc[:, 'ma60'] = df['close'].rolling(window=60).mean().shift(1)
    df.loc[:, 'ma60_5d_ago'] = df['ma60'].shift(5)
    
    latest_data = df.iloc[-1]
    
    price_above_ma60 = latest_data['close'] > latest_data['ma60']
    ma60_upward = latest_data['ma60'] > latest_data['ma60_5d_ago']
    
    return {
        'code': etf_code,
        'ma60': latest_data['ma60'],
        'ma60_slope': latest_data['ma60'] - latest_data['ma60_5d_ago'],
        'signal': price_above_ma60 and ma60_upward
    }

def generate_target_portfolio(all_data, etf_pool, target_date):
    """
    生成目标投资组合（基于 T-1 日数据）
    :param all_data: 所有 ETF 的历史数据字典
    :param etf_pool: ETF 池 DataFrame
    :param target_date: 目标日期
    :return: 目标持仓列表（ETF 代码，无重复）
    """
    logger.info(f"生成 {target_date} 的目标持仓")
    
    qualified_codes = filter_liquidity(all_data, etf_pool, target_date)
    
    if not qualified_codes:
        logger.warning(f"{target_date} 没有符合流动性要求的 ETF")
        return [StrategyConfig.CASH_ETF_CODE]
    
    momentum_df = calc_cross_section_momentum(
        all_data, 
        qualified_codes, 
        target_date,
        lookback_days=StrategyConfig.LOOKBACK_DAYS
    )
    
    if momentum_df.empty:
        logger.warning(f"{target_date} 无法计算截面动量")
        return [StrategyConfig.CASH_ETF_CODE]
    
    candidates = momentum_df.head(StrategyConfig.MAX_CANDIDATES)['code'].tolist()
    
    selected = []
    for etf_code in candidates:
        if etf_code in selected:
            continue
        
        ts_result = calc_time_series_momentum(all_data, etf_code, target_date)
        
        if ts_result['signal']:
            selected.append(etf_code)
            
        if len(selected) >= StrategyConfig.TARGET_HOLDINGS:
            break
    
    while len(selected) < StrategyConfig.TARGET_HOLDINGS:
        selected.append(StrategyConfig.CASH_ETF_CODE)
    
    logger.info(f"{target_date} 目标持仓: {selected}")
    return selected

def is_stop_loss_triggered(all_data, etf_code, current_date):
    """
    判断是否触发止损（基于 T-1 日 MA60 和 T 日收盘价）
    :param all_data: 所有 ETF 的历史数据字典
    :param etf_code: ETF 代码
    :param current_date: 当前日期
    :return: 是否触发止损
    """
    if etf_code not in all_data:
        return False
    
    df = all_data[etf_code].copy()
    df = df[df['date'] <= pd.to_datetime(current_date)]
    
    if len(df) < 61:
        return False
    
    df.loc[:, 'ma60'] = df['close'].rolling(window=60).mean().shift(1)
    
    latest_data = df.iloc[-1]
    
    if pd.isna(latest_data['ma60']):
        return False
    
    drawdown = (latest_data['close'] - latest_data['ma60']) / latest_data['ma60']
    
    return drawdown <= StrategyConfig.STOP_LOSS_THRESHOLD

if __name__ == '__main__':
    logger.info("=== 信号引擎模块测试 ===")
    
    test_data = {
        '510050': pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=800, freq='D'),
            'open': np.linspace(2.0, 2.8, 800),
            'high': np.linspace(2.05, 2.85, 800),
            'low': np.linspace(1.95, 2.75, 800),
            'close': np.linspace(2.0, 2.8, 800),
            'volume': np.ones(800) * 10000000,
            'amount': np.ones(800) * 200000000
        }),
        '510300': pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=800, freq='D'),
            'open': np.linspace(4.0, 5.2, 800),
            'high': np.linspace(4.05, 5.25, 800),
            'low': np.linspace(3.95, 5.15, 800),
            'close': np.linspace(4.0, 5.2, 800),
            'volume': np.ones(800) * 15000000,
            'amount': np.ones(800) * 600000000
        }),
        '511010': pd.DataFrame({
            'date': pd.date_range('2021-01-01', periods=800, freq='D'),
            'open': np.linspace(100.0, 101.0, 800),
            'high': np.linspace(100.02, 101.02, 800),
            'low': np.linspace(99.98, 100.98, 800),
            'close': np.linspace(100.0, 101.0, 800),
            'volume': np.ones(800) * 5000000,
            'amount': np.ones(800) * 500000000
        })
    }
    
    test_pool = pd.DataFrame({
        'code': ['510050', '510300', '511010'],
        'name': ['上证50ETF', '沪深300ETF', '国债ETF'],
        'type': ['ETF', 'ETF', 'ETF'],
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    target_date = '2023-04-15'
    portfolio = generate_target_portfolio(test_data, test_pool, target_date)
    logger.info(f"测试目标持仓: {portfolio}")
    
    stop_loss_test = is_stop_loss_triggered(test_data, '510050', target_date)
    logger.info(f"止损测试结果: {stop_loss_test}")
    
    logger.info("=== 测试完成 ===")