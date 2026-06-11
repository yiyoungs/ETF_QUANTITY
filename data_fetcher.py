import os
import time
import pandas as pd
import akshare as ak
from config import (
    CACHE_DIR, DATE_FORMAT, ETF_PREFIX_MAP, 
    DataConfig, StrategyConfig, LOG_DIR
)
import logging

logger = logging.getLogger(__name__)

def validate_data_quality(df, code, data_type='qfq'):
    """
    验证数据质量，记录异常情况
    :param df: 行情数据 DataFrame
    :param code: ETF 代码
    :param data_type: 数据类型 (qfq/hfq)
    :return: (is_valid, issues)
    """
    issues = []
    is_valid = True
    
    if df is None or df.empty:
        issues.append(f"数据为空")
        is_valid = False
        return (is_valid, issues)
    
    # 检查必需列
    required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"缺少必需列: {', '.join(missing_cols)}")
        is_valid = False
    
    # 检查价格是否合理
    if 'close' in df.columns:
        # 检查负数价格
        neg_prices = (df['close'] < 0).sum()
        if neg_prices > 0:
            issues.append(f"发现 {neg_prices} 条负数价格")
            is_valid = False
        
        # 检查价格为0
        zero_prices = (df['close'] == 0).sum()
        if zero_prices > 0:
            issues.append(f"发现 {zero_prices} 条零价格")
            is_valid = False
    
    # 检查涨跌幅异常（清洗前）
    if 'close' in df.columns and len(df) > 1:
        pct_change = df['close'].pct_change().abs()
        extreme_changes = pct_change[pct_change > 0.12].count()
        if extreme_changes > 0:
            issues.append(f"发现 {extreme_changes} 条极端涨跌幅(>12%)")
    
    if issues:
        logger.warning(f"⚠️ {code} ({data_type}) 数据质量检查发现问题: {'; '.join(issues)}")
    
    return (is_valid, issues)


def clean_price_data(df, code):
    """
    清洗价格数据，处理前复权异常
    :param df: 原始行情数据 DataFrame
    :param code: ETF 代码
    :return: 清洗后的 DataFrame
    """
    df = df.copy()
    
    # 1. 过滤单日涨跌幅超过 ±12% 的记录（ETF涨跌停为±10%，
    #    超过的基本都是前复权除权导致的假信号）
    if 'close' in df.columns and len(df) > 1:
        df['pct_change'] = df['close'].pct_change()
        abnormal_mask = df['pct_change'].abs() > 0.12
        
        if abnormal_mask.sum() > 0:
            logger.warning(f"⚠️ {code} 发现 {abnormal_mask.sum()} 条异常涨跌幅，将用前一日收盘价替换")
            for idx in df[abnormal_mask].index:
                prev_idx = idx - 1
                if prev_idx >= 0:
                    prev_close = df.loc[prev_idx, 'close']
                    df.loc[idx, 'open'] = prev_close
                    df.loc[idx, 'high'] = prev_close
                    df.loc[idx, 'low'] = prev_close
                    df.loc[idx, 'close'] = prev_close
        
        df.drop('pct_change', axis=1, inplace=True)
    
    # 2. 过滤成交量为0的停牌日，标记为不可交易
    if 'volume' in df.columns:
        df['suspended'] = df['volume'] == 0
    else:
        df['suspended'] = False
    
    return df

def format_etf_code(code, with_suffix=True):
    code = str(code).strip()
    
    if '.' in code:
        pure_code = code.split('.')[0].zfill(6)
        if with_suffix:
            return code
        else:
            return pure_code
    else:
        pure_code = code.zfill(6)
        if with_suffix:
            for prefix, suffix in ETF_PREFIX_MAP.items():
                if pure_code.startswith(prefix):
                    return f"{pure_code}.{suffix}"
            logger.warning(f"无法识别 ETF 代码前缀: {pure_code}，默认使用 .SH")
            return f"{pure_code}.SH"
        else:
            return pure_code

def get_etf_pool(include_delisted=True):
    logger.info("开始获取全市场 ETF 列表...")
    
    try:
        etf_info = ak.fund_etf_category_sina()
        
        def parse_code(code_str):
            code_str = str(code_str).strip().lower()
            if code_str.startswith('sz'):
                return code_str[2:].zfill(6)
            elif code_str.startswith('sh'):
                return code_str[2:].zfill(6)
            else:
                return code_str.zfill(6)
        
        df = pd.DataFrame({
            'code': etf_info['代码'].apply(parse_code),
            'name': etf_info['名称'],
            'type': 'ETF',
            'list_date': None,
            'scale': None
        })
        
        df = df.drop_duplicates('code')
        
        logger.info(f"成功获取 {len(df)} 只 ETF")
        return df
    
    except Exception as e:
        logger.error(f"获取 ETF 列表失败: {str(e)}")
        raise

def fetch_history_data(etf_code, start_date='2015-01-01', end_date=None, force_refresh=False):
    if end_date is None:
        end_date = pd.Timestamp.now().strftime(DATE_FORMAT)
    
    etf_code_6digit = format_etf_code(etf_code, with_suffix=False)
    etf_code_suffix = format_etf_code(etf_code, with_suffix=True)
    
    cache_path_qfq = os.path.join(CACHE_DIR, f"{etf_code_6digit}_qfq.csv")
    cache_path_hfq = os.path.join(CACHE_DIR, f"{etf_code_6digit}_hfq.csv")
    
    result = {
        'qfq': None,
        'hfq': None,
        'dividend': None
    }
    
    need_fetch_qfq = force_refresh or not os.path.exists(cache_path_qfq)
    need_fetch_hfq = force_refresh or not os.path.exists(cache_path_hfq)
    start_date_qfq = None
    start_date_hfq = None
    
    if not need_fetch_qfq:
        logger.debug(f"从缓存读取 ETF {etf_code_6digit} 的前复权数据")
        df_qfq = pd.read_csv(cache_path_qfq, parse_dates=['date'])
        last_date = df_qfq['date'].max().strftime(DATE_FORMAT)
        if last_date < end_date:
            need_fetch_qfq = True
            start_date_qfq = last_date
        else:
            df_qfq = df_qfq[(df_qfq['date'] >= start_date) & (df_qfq['date'] <= end_date)]
            # 强制清洗：即使从缓存读取也要清洗，确保数据质量
            df_qfq = clean_price_data(df_qfq, etf_code_6digit)
            result['qfq'] = df_qfq
    
    if not need_fetch_hfq:
        logger.debug(f"从缓存读取 ETF {etf_code_6digit} 的不复权数据")
        df_hfq = pd.read_csv(cache_path_hfq, parse_dates=['date'])
        last_date = df_hfq['date'].max().strftime(DATE_FORMAT)
        if last_date < end_date:
            need_fetch_hfq = True
            start_date_hfq = last_date
        else:
            df_hfq = df_hfq[(df_hfq['date'] >= start_date) & (df_hfq['date'] <= end_date)]
            # 强制清洗：即使从缓存读取也要清洗，确保数据质量
            df_hfq = clean_price_data(df_hfq, etf_code_6digit)
            result['hfq'] = df_hfq
    
    if need_fetch_qfq or need_fetch_hfq:
        logger.info(f"开始下载 ETF {etf_code_6digit} 的历史数据 ({start_date} ~ {end_date})")
        
        for retry in range(DataConfig.MAX_RETRY):
            try:
                time.sleep(DataConfig.REQUEST_INTERVAL)
                
                if etf_code_suffix.endswith('.SH'):
                    sina_symbol = f"sh{etf_code_6digit}"
                else:
                    sina_symbol = f"sz{etf_code_6digit}"
                
                if need_fetch_qfq:
                    df_qfq_new = ak.fund_etf_hist_sina(symbol=sina_symbol)
                    
                    df_qfq_new = df_qfq_new.rename(columns={
                        'date': 'date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume'
                    })
                    
                    if 'amount' not in df_qfq_new.columns:
                        df_qfq_new['amount'] = df_qfq_new['close'] * df_qfq_new['volume']
                    
                    df_qfq_new['date'] = pd.to_datetime(df_qfq_new['date'])
                    df_qfq_new = df_qfq_new.sort_values('date').reset_index(drop=True)
                    
                    # 数据质量验证
                    validate_data_quality(df_qfq_new, etf_code_6digit, 'qfq')
                    
                    if os.path.exists(cache_path_qfq) and not force_refresh:
                        existing_df = pd.read_csv(cache_path_qfq, parse_dates=['date'])
                        df_qfq = pd.concat([existing_df, df_qfq_new]).drop_duplicates('date').sort_values('date')
                    else:
                        df_qfq = df_qfq_new
                    
                    # 保存前先清洗数据
                    df_qfq = clean_price_data(df_qfq, etf_code_6digit)
                    df_qfq.to_csv(cache_path_qfq, index=False)
                
                if need_fetch_hfq:
                    df_hfq_new = ak.fund_etf_hist_sina(symbol=sina_symbol)
                    
                    df_hfq_new = df_hfq_new.rename(columns={
                        'date': 'date',
                        'open': 'open',
                        'high': 'high',
                        'low': 'low',
                        'close': 'close',
                        'volume': 'volume'
                    })
                    
                    if 'amount' not in df_hfq_new.columns:
                        df_hfq_new['amount'] = df_hfq_new['close'] * df_hfq_new['volume']
                    
                    df_hfq_new['date'] = pd.to_datetime(df_hfq_new['date'])
                    df_hfq_new = df_hfq_new.sort_values('date').reset_index(drop=True)
                    
                    # 数据质量验证
                    validate_data_quality(df_hfq_new, etf_code_6digit, 'hfq')
                    
                    if os.path.exists(cache_path_hfq) and not force_refresh:
                        existing_df = pd.read_csv(cache_path_hfq, parse_dates=['date'])
                        df_hfq = pd.concat([existing_df, df_hfq_new]).drop_duplicates('date').sort_values('date')
                    else:
                        df_hfq = df_hfq_new
                    
                    # 保存前先清洗数据
                    df_hfq = clean_price_data(df_hfq, etf_code_6digit)
                    df_hfq.to_csv(cache_path_hfq, index=False)
                
                # 应用数据清洗
                df_qfq_clean = clean_price_data(df_qfq[(df_qfq['date'] >= start_date) & (df_qfq['date'] <= end_date)], etf_code_6digit)
                df_hfq_clean = clean_price_data(df_hfq[(df_hfq['date'] >= start_date) & (df_hfq['date'] <= end_date)], etf_code_6digit)
                
                result['qfq'] = df_qfq_clean
                result['hfq'] = df_hfq_clean
                logger.info(f"成功下载 ETF {etf_code_6digit} 的历史数据（已清洗）")
                
                break
                
            except Exception as e:
                logger.warning(f"下载 ETF {etf_code_6digit} 失败 (尝试 {retry+1}/{DataConfig.MAX_RETRY}): {str(e)}")
                time.sleep(1)
        else:
            logger.error(f"ETF {etf_code_6digit} 下载失败，已达最大重试次数")
    
    dividend_df = fetch_dividend_data(etf_code)
    result['dividend'] = dividend_df
    
    return result

def fetch_dividend_data(etf_code):
    """
    获取 ETF 的历史分红数据
    :param etf_code: ETF 代码（6位纯数字）
    :return: DataFrame，包含 ex_date（除权除息日）、dividend_per_share（每股派息）
    """
    etf_code_6digit = format_etf_code(etf_code, with_suffix=False)
    cache_path = os.path.join(CACHE_DIR, f"{etf_code_6digit}_dividend.csv")
    
    if os.path.exists(cache_path):
        logger.debug(f"从缓存读取 ETF {etf_code_6digit} 的分红数据")
        return pd.read_csv(cache_path, parse_dates=['ex_date'])
    
    logger.info(f"开始获取 ETF {etf_code_6digit} 的分红数据")
    
    try:
        etf_code_suffix = format_etf_code(etf_code, with_suffix=True)
        
        df = ak.fund_dividend_detail(symbol=etf_code_suffix)
        
        if df.empty:
            logger.warning(f"ETF {etf_code_6digit} 无分红数据")
            return pd.DataFrame(columns=['ex_date', 'dividend_per_share'])
        
        df = df.rename(columns={
            '除权除息日': 'ex_date',
            '每股派息': 'dividend_per_share'
        })
        
        df['ex_date'] = pd.to_datetime(df['ex_date'])
        df = df[['ex_date', 'dividend_per_share']].drop_duplicates().sort_values('ex_date').reset_index(drop=True)
        
        df.to_csv(cache_path, index=False)
        logger.info(f"成功获取 ETF {etf_code_6digit} 的分红数据，共 {len(df)} 条")
        
        return df
    
    except Exception as e:
        logger.warning(f"获取 ETF {etf_code_6digit} 分红数据失败: {str(e)}")
        return pd.DataFrame(columns=['ex_date', 'dividend_per_share'])

def fetch_all_etf_data(etf_pool, start_date='2015-01-01', end_date=None):
    logger.info("开始批量下载全市场 ETF 历史数据...")
    
    all_data = {
        'qfq': {},
        'hfq': {},
        'dividend': {}
    }
    success_count = 0
    fail_count = 0
    
    for idx, row in etf_pool.iterrows():
        etf_code = row['code']
        
        try:
            data = fetch_history_data(etf_code, start_date, end_date)
            
            if data['qfq'] is not None and not data['qfq'].empty:
                all_data['qfq'][etf_code] = data['qfq']
                all_data['hfq'][etf_code] = data['hfq']
                all_data['dividend'][etf_code] = data['dividend']
                success_count += 1
            else:
                fail_count += 1
                
        except Exception as e:
            logger.error(f"处理 ETF {etf_code} 时发生异常: {str(e)}")
            fail_count += 1
        
        if (idx + 1) % 10 == 0:
            logger.info(f"已处理 {idx + 1}/{len(etf_pool)} 只 ETF")
    
    logger.info(f"批量下载完成，成功: {success_count} 只，失败: {fail_count} 只")
    return all_data

def load_cached_data(etf_code):
    etf_code_6digit = format_etf_code(etf_code, with_suffix=False)
    cache_path_qfq = os.path.join(CACHE_DIR, f"{etf_code_6digit}_qfq.csv")
    cache_path_hfq = os.path.join(CACHE_DIR, f"{etf_code_6digit}_hfq.csv")
    cache_path_dividend = os.path.join(CACHE_DIR, f"{etf_code_6digit}_dividend.csv")
    
    result = {
        'qfq': None,
        'hfq': None,
        'dividend': None
    }
    
    if os.path.exists(cache_path_qfq):
        result['qfq'] = pd.read_csv(cache_path_qfq, parse_dates=['date'])
    
    if os.path.exists(cache_path_hfq):
        result['hfq'] = pd.read_csv(cache_path_hfq, parse_dates=['date'])
    
    if os.path.exists(cache_path_dividend):
        result['dividend'] = pd.read_csv(cache_path_dividend, parse_dates=['ex_date'])
    
    return result

if __name__ == '__main__':
    logger.info("=== 数据获取模块测试 ===")
    
    etf_pool = get_etf_pool()
    logger.info(f"ETF 池大小: {len(etf_pool)}")
    
    sample_etf = StrategyConfig.CASH_ETF_CODE
    logger.info(f"测试下载样本 ETF: {sample_etf}")
    
    data = fetch_history_data(sample_etf)
    if data['qfq'] is not None and not data['qfq'].empty:
        logger.info(f"前复权数据预览:")
        logger.info(data['qfq'].head())
        logger.info(f"数据时间范围: {data['qfq']['date'].min().strftime(DATE_FORMAT)} ~ {data['qfq']['date'].max().strftime(DATE_FORMAT)}")
    
    if data['dividend'] is not None and not data['dividend'].empty:
        logger.info(f"分红数据预览:")
        logger.info(data['dividend'].head())
    
    logger.info("=== 测试完成 ===")
