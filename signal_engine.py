import pandas as pd
import numpy as np
from config import StrategyConfig
import logging

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
    target_dt = pd.to_datetime(target_date)
    
    for etf_code in etf_pool['code'].unique():
        if etf_code not in all_data:
            continue
        
        df = all_data[etf_code]
        
        mask = df['date'] <= target_dt
        if mask.sum() == 0:
            continue
        
        latest_data = df[mask].iloc[-1]
        
        if 'amount' in df.columns and pd.isna(latest_data.get('amount')):
            continue
        
        if len(df[mask]) >= 20:
            df_temp = df[mask].copy()
            df_temp['rolling_amount'] = df_temp['amount'].rolling(window=20).mean().shift(1)
            if not pd.isna(df_temp['rolling_amount'].iloc[-1]):
                if df_temp['rolling_amount'].iloc[-1] >= StrategyConfig.MIN_AMOUNT:
                    qualified.append(etf_code)
            else:
                qualified.append(etf_code)
        else:
            qualified.append(etf_code)
    
    return qualified

def calc_multi_period_momentum(all_data, qualified_codes, target_date, periods_weights=None):
    """
    计算多周期动量融合评分（基于 T-1 日数据）
    
    支持参数化周期配置：
    - 默认：20(0.5)+60(0.3)+120(0.2)
    - 方案B：10(0.4)+20(0.3)+60(0.3)
    
    momentum_score = Σ(weight_i × rank(period_i收益率))
    
    :param all_data: 所有 ETF 的历史数据字典
    :param qualified_codes: 符合流动性要求的 ETF 代码列表
    :param target_date: 目标日期
    :param periods_weights: 周期和权重列表，如 [(20, 0.5), (60, 0.3), (120, 0.2)]
    :return: 按动量评分排序的 DataFrame，包含 code, momentum_score, 各周期收益
    """
    if periods_weights is None:
        periods_weights = [(20, 0.5), (60, 0.3), (120, 0.2)]
    
    momentum_results = []
    target_dt = pd.to_datetime(target_date)
    max_period = max(p[0] for p in periods_weights)
    
    for etf_code in qualified_codes:
        if etf_code not in all_data:
            continue
        
        df = all_data[etf_code].copy()
        df = df[df['date'] <= target_dt].copy()
        df.reset_index(drop=True, inplace=True)
        
        if len(df) < max_period + 1:
            continue
        
        end_idx = len(df) - 1
        record = {'code': etf_code}
        
        for period, _ in periods_weights:
            start_idx = max(0, end_idx - period)
            ret = (df['close'].iloc[end_idx] - df['close'].iloc[start_idx]) / df['close'].iloc[start_idx]
            record[f'mom_{period}'] = ret
        
        momentum_results.append(record)
    
    if not momentum_results:
        return pd.DataFrame()
    
    result_df = pd.DataFrame(momentum_results)
    
    # 使用rank计算加权动量评分
    total_weight = sum(w for _, w in periods_weights)
    result_df['momentum_score'] = 0.0
    
    for period, weight in periods_weights:
        result_df[f'rank_{period}'] = result_df[f'mom_{period}'].rank(pct=True)
        result_df['momentum_score'] += (weight / total_weight) * result_df[f'rank_{period}']
    
    result_df = result_df.sort_values('momentum_score', ascending=False).reset_index(drop=True)
    
    return result_df


def calculate_market_trend_strength(all_data, qualified_codes, target_date, ma_period=60):
    """
    方案A：计算市场趋势强度
    
    计算每个标的(Price-MA60)/MA60的偏离度，取最大值作为市场趋势强度指标
    
    :param all_data: 所有ETF数据
    :param qualified_codes: 符合流动性要求的ETF代码列表
    :param target_date: 目标日期
    :param ma_period: MA周期，默认60
    :return: 趋势强度指标（最大偏离度%）
    """
    hfq_data = all_data.get('hfq', all_data)
    deviations = []
    
    for etf_code in qualified_codes:
        if etf_code == StrategyConfig.CASH_ETF_CODE:
            continue
        
        df = hfq_data.get(etf_code)
        if df is None or df.empty:
            continue
        
        df_filtered = df[df['date'] <= pd.to_datetime(target_date)]
        if len(df_filtered) < ma_period + 1:
            continue
        
        df_filtered['ma60'] = df_filtered['close'].rolling(window=ma_period).mean().shift(1)
        latest = df_filtered.iloc[-1]
        
        if not pd.isna(latest['ma60']) and latest['ma60'] > 0:
            deviation = (latest['close'] - latest['ma60']) / latest['ma60'] * 100
            deviations.append(deviation)
    
    if not deviations:
        return 0.0
    
    return max(deviations)


def calc_cross_section_momentum(all_data, qualified_codes, target_date, lookback_days=60, periods_weights=None):
    """
    计算截面动量（基于 T-1 日数据）
    :param all_data: 所有 ETF 的历史数据字典
    :param qualified_codes: 符合流动性要求的 ETF 代码列表
    :param target_date: 目标日期
    :param lookback_days: 回溯天数（仅用于兼容）
    :param periods_weights: 周期和权重配置
    :return: 按动量评分排序的 DataFrame
    """
    return calc_multi_period_momentum(all_data, qualified_codes, target_date, periods_weights)

def calc_time_series_momentum(all_data, etf_code, target_date, lookback_days=60):
    """
    计算时序动量（基于 T-1 日数据，使用不复权数据避免前复权跳空）
    :param all_data: 所有 ETF 的历史数据字典
    :param etf_code: ETF 代码
    :param target_date: 目标日期
    :param lookback_days: 均线周期（默认60天，可动态配置）
    :return: 包含 MA 和 signal 的字典
    """
    if 'hfq' in all_data and etf_code in all_data['hfq']:
        df = all_data['hfq'][etf_code].copy()
    elif etf_code in all_data:
        df = all_data[etf_code].copy()
    else:
        return {'code': etf_code, 'ma': np.nan, 'signal': False}
    
    df = df[df['date'] <= pd.to_datetime(target_date)]
    
    if len(df) < lookback_days + 1:
        return {'code': etf_code, 'ma': np.nan, 'signal': False}
    
    ma_col = f'ma{lookback_days}'
    df[ma_col] = df['close'].rolling(window=lookback_days).mean().shift(1)
    
    latest_data = df.iloc[-1]
    
    price_above_ma = latest_data['close'] > latest_data[ma_col]
    
    return {
        'code': etf_code,
        'ma': latest_data[ma_col],
        'signal': price_above_ma
    }

def get_top_momentum_stocks(all_data, etf_pool, target_date, top_n=10, periods_weights=None):
    """
    获取前 N 名动量股票（使用多周期动量评分）
    :param all_data: 所有 ETF 的历史数据字典
    :param etf_pool: ETF 池 DataFrame
    :param target_date: 目标日期
    :param top_n: 返回前 N 名
    :param periods_weights: 周期和权重配置
    :return: 前 N 名动量股票列表
    """
    qualified_codes = filter_liquidity(all_data, etf_pool, target_date)
    
    if not qualified_codes:
        return []
    
    momentum_df = calc_multi_period_momentum(all_data, qualified_codes, target_date, periods_weights)
    
    if momentum_df.empty:
        return []
    
    return momentum_df.head(top_n)['code'].tolist()


def generate_target_portfolio(all_data, etf_pool, target_date, periods_weights=None, trend_strength_mode=False, lookback_days=60, skip_ts_filter=False):
    """
    生成目标投资组合（基于 T-1 日数据）
    
    :param all_data: 所有 ETF 的历史数据字典
    :param etf_pool: ETF 池 DataFrame
    :param target_date: 目标日期
    :param periods_weights: 周期和权重配置
    :param trend_strength_mode: 是否启用趋势强度分级模式
    :param lookback_days: 时序动量均线周期（默认60天）
    :param skip_ts_filter: 是否跳过时序动量过滤（直接选择截面动量Top N）
    :return: 目标持仓列表（ETF 代码，无重复）
    """
    logger.info(f"生成 {target_date} 的目标持仓 (skip_ts_filter={skip_ts_filter})")
    
    qualified_codes = filter_liquidity(all_data, etf_pool, target_date)
    
    if not qualified_codes:
        logger.warning(f"{target_date} 没有符合流动性要求的 ETF")
        return [StrategyConfig.CASH_ETF_CODE]
    
    # 方案A：趋势强度分级
    equity_ratio = 1.0  # 默认满仓权益
    if trend_strength_mode:
        trend_strength = calculate_market_trend_strength(all_data, qualified_codes, target_date)
        logger.info(f"{target_date} 趋势强度: {trend_strength:.2f}%")
        
        if trend_strength < 2.0:
            equity_ratio = 0.6  # 弱趋势：降仓至60%
            logger.info(f"{target_date} 弱趋势，权益仓位降至60%")
        elif trend_strength > 5.0:
            equity_ratio = 1.0  # 强趋势：满仓
            logger.info(f"{target_date} 强趋势，保持满仓")
        else:
            equity_ratio = 0.8  # 中等趋势：80%权益
    
    # 使用多周期动量评分
    momentum_df = calc_multi_period_momentum(all_data, qualified_codes, target_date, periods_weights)
    
    if momentum_df.empty:
        logger.warning(f"{target_date} 无法计算截面动量")
        return [StrategyConfig.CASH_ETF_CODE]
    
    # 根据趋势强度调整目标持仓数量
    target_count = int(StrategyConfig.TARGET_HOLDINGS * equity_ratio)
    target_count = max(1, target_count)
    
    # 跳过时序动量过滤：直接选择截面动量最高的标的
    if skip_ts_filter:
        selected = momentum_df.head(target_count)['code'].tolist()
    else:
        # 原始逻辑：使用时序动量过滤
        candidates = momentum_df.head(StrategyConfig.MAX_CANDIDATES)['code'].tolist()
        
        selected = []
        for etf_code in candidates:
            if etf_code in selected:
                continue
            
            ts_result = calc_time_series_momentum(all_data, etf_code, target_date, lookback_days)
            
            if ts_result['signal']:
                selected.append(etf_code)
                
            if len(selected) >= target_count:
                break
    
    # 确保国债ETF在组合中（无论是否有动量信号）
    if StrategyConfig.CASH_ETF_CODE not in selected:
        selected.append(StrategyConfig.CASH_ETF_CODE)
    
    logger.info(f"{target_date} 目标持仓: {selected}")
    return selected

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
    
    logger.info("=== 测试完成 ===")