import os
import time
import pandas as pd
import akshare as ak
from config import (
    CACHE_DIR, DATE_FORMAT, ETF_PREFIX_MAP, 
    DataConfig, StrategyConfig, LOG_DIR
)
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'data_fetcher.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

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
    
    if not need_fetch_qfq:
        logger.debug(f"从缓存读取 ETF {etf_code_6digit} 的前复权数据")
        df_qfq = pd.read_csv(cache_path_qfq, parse_dates=['date'])
        last_date = df_qfq['date'].max().strftime(DATE_FORMAT)
        if last_date < end_date:
            need_fetch_qfq = True
            start_date_qfq = last_date
        else:
            df_qfq = df_qfq[(df_qfq['date'] >= start_date) & (df_qfq['date'] <= end_date)]
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
            result['hfq'] = df_hfq
    
    if need_fetch_qfq or need_fetch_hfq:
        logger.info(f"开始下载 ETF {etf_code_6digit} 的历史数据 ({start_date} ~ {end_date})")
        
        for retry in range(DataConfig.MAX_RETRY):
            try:
                time.sleep(DataConfig.REQUEST_INTERVAL)
                
                if need_fetch_qfq:
                    df_qfq_new = ak.fund_etf_hist_em(
                        symbol=etf_code_suffix,
                        period='daily',
                        start_date=start_date_qfq if 'start_date_qfq' in locals() else start_date,
                        end_date=end_date,
                        adjust='qfq'
                    )
                    
                    df_qfq_new = df_qfq_new.rename(columns={
                        '日期': 'date',
                        '开盘价': 'open',
                        '最高价': 'high',
                        '最低价': 'low',
                        '收盘价': 'close',
                        '成交量': 'volume',
                        '成交额': 'amount'
                    })
                    df_qfq_new['date'] = pd.to_datetime(df_qfq_new['date'])
                    df_qfq_new = df_qfq_new.sort_values('date').reset_index(drop=True)
                    
                    if os.path.exists(cache_path_qfq) and not force_refresh:
                        existing_df = pd.read_csv(cache_path_qfq, parse_dates=['date'])
                        df_qfq = pd.concat([existing_df, df_qfq_new]).drop_duplicates('date').sort_values('date')
                    else:
                        df_qfq = df_qfq_new
                    
                    df_qfq.to_csv(cache_path_qfq, index=False)
                
                if need_fetch_hfq:
                    df_hfq_new = ak.fund_etf_hist_em(
                        symbol=etf_code_suffix,
                        period='daily',
                        start_date=start_date_hfq if 'start_date_hfq' in locals() else start_date,
                        end_date=end_date,
                        adjust='hfq'
                    )
                    
                    df_hfq_new = df_hfq_new.rename(columns={
                        '日期': 'date',
                        '开盘价': 'open',
                        '最高价': 'high',
                        '最低价': 'low',
                        '收盘价': 'close',
                        '成交量': 'volume',
                        '成交额': 'amount'
                    })
                    df_hfq_new['date'] = pd.to_datetime(df_hfq_new['date'])
                    df_hfq_new = df_hfq_new.sort_values('date').reset_index(drop=True)
                    
                    if os.path.exists(cache_path_hfq) and not force_refresh:
                        existing_df = pd.read_csv(cache_path_hfq, parse_dates=['date'])
                        df_hfq = pd.concat([existing_df, df_hfq_new]).drop_duplicates('date').sort_values('date')
                    else:
                        df_hfq = df_hfq_new
                    
                    df_hfq.to_csv(cache_path_hfq, index=False)
                
                result['qfq'] = df_qfq[(df_qfq['date'] >= start_date) & (df_qfq['date'] <= end_date)]
                result['hfq'] = df_hfq[(df_hfq['date'] >= start_date) & (df_hfq['date'] <= end_date)]
                logger.info(f"成功下载 ETF {etf_code_6digit} 的历史数据")
                
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
