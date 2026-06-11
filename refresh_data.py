"""
重新下载并清洗所有ETF数据
"""
import pandas as pd
import numpy as np
import os
import sys
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_fetcher import fetch_history_data, format_etf_code, clean_price_data
from config import CACHE_DIR, DATE_FORMAT

# 主要ETF列表
MAJOR_ETFS = [
    ('510050', '50ETF'),
    ('510300', '沪深300ETF'),
    ('510500', '中证500ETF'),
    ('512100', '中证100ETF'),
    ('159915', '创业板ETF'),
    ('512880', '证券ETF'),
    ('512480', '半导体ETF'),
    ('512690', '酒ETF'),
    ('511010', '国债ETF'),
]

def re_download_all_etfs():
    """强制重新下载并清洗所有主要ETF数据"""
    logger.info("=" * 60)
    logger.info("开始强制重新下载 ETF 数据（应用数据清洗）")
    logger.info("=" * 60)
    
    # 重新下载每只ETF
    for etf_code, etf_name in MAJOR_ETFS:
        logger.info(f"\n正在处理: {etf_code} ({etf_name})")
        
        # 删除旧缓存
        cache_path_qfq = os.path.join(CACHE_DIR, f"{etf_code}_qfq.csv")
        cache_path_hfq = os.path.join(CACHE_DIR, f"{etf_code}_hfq.csv")
        
        if os.path.exists(cache_path_qfq):
            os.remove(cache_path_qfq)
            logger.info(f"  已删除旧缓存: {cache_path_qfq}")
        if os.path.exists(cache_path_hfq):
            os.remove(cache_path_hfq)
            logger.info(f"  已删除旧缓存: {cache_path_hfq}")
        
        # 重新下载
        try:
            data = fetch_history_data(etf_code, '2015-01-01', '2025-12-31', force_refresh=True)
            
            if data['qfq'] is not None and not data['qfq'].empty:
                df_qfq = data['qfq']
                df_hfq = data['hfq']
                
                logger.info(f"  前复权数据: {len(df_qfq)} 条")
                logger.info(f"  不复权数据: {len(df_hfq)} 条")
                
                # 检查数据质量
                if 'suspended' in df_qfq.columns:
                    suspended_count = df_qfq['suspended'].sum()
                    if suspended_count > 0:
                        logger.info(f"  停牌日数量: {suspended_count}")
                
                logger.info(f"  ✅ {etf_code} 数据下载并清洗完成")
            else:
                logger.error(f"  ❌ {etf_code} 数据为空")
                
        except Exception as e:
            logger.error(f"  ❌ {etf_code} 下载失败: {str(e)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("所有 ETF 数据重新下载完成")
    logger.info("=" * 60)

def verify_data_quality():
    """验证数据质量"""
    logger.info("\n" + "=" * 60)
    logger.info("数据质量验证")
    logger.info("=" * 60)
    
    for etf_code, etf_name in MAJOR_ETFS:
        cache_path = os.path.join(CACHE_DIR, f"{etf_code}_qfq.csv")
        
        if not os.path.exists(cache_path):
            logger.warning(f"  ⚠️ {etf_code} 缓存文件不存在")
            continue
        
        df = pd.read_csv(cache_path, parse_dates=['date'])
        df['pct_change'] = df['close'].pct_change()
        
        # 检查异常涨跌幅
        abnormal = (df['pct_change'].abs() > 0.12).sum()
        suspended = ((df['volume'] == 0) | (df.get('suspended', False) == True)).sum()
        
        if abnormal > 0 or suspended > 0:
            logger.info(f"  ⚠️ {etf_code}: 异常涨跌幅 {abnormal} 条, 停牌 {suspended} 天")
        else:
            logger.info(f"  ✅ {etf_code}: 数据正常")

if __name__ == '__main__':
    re_download_all_etfs()
    verify_data_quality()
