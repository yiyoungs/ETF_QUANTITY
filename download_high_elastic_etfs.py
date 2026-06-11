import os
import pandas as pd
import logging
from data_fetcher import fetch_history_data, format_etf_code

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

NEW_ETF_POOL = {
    '科技主线': [
        '512480',  # 半导体ETF
        '159995',  # 芯片ETF
        '512760',  # 半导体行业ETF
        '515050',  # 5G ETF
        '512000',  # 券商ETF
    ],
    '高端制造/新能源': [
        '515030',  # 新能源车ETF
        '159875',  # 光伏ETF
        '512660',  # 军工ETF
    ],
    '消费/医药': [
        '512690',  # 酒ETF
        '159928',  # 消费ETF
        '512010',  # 医药ETF
        '159801',  # 恒生科技ETF
    ],
    '周期/红利': [
        '512890',  # 红利ETF
        '510880',  # 红利低波ETF
        '168204',  # 煤炭LOF (替代511170，511170是货币基金)
        '512400',  # 有色ETF
    ]
}

def download_all_high_elastic_etfs(start_date='2018-01-01', end_date='2025-12-31'):
    logger.info("=== 开始下载高弹性行业/主题 ETF 数据 ===")
    
    all_codes = []
    for category, codes in NEW_ETF_POOL.items():
        all_codes.extend(codes)
    
    success_count = 0
    fail_count = 0
    failed_codes = []
    
    for code in all_codes:
        try:
            logger.info(f"正在下载 {code}...")
            data = fetch_history_data(code, start_date=start_date, end_date=end_date)
            
            if data['qfq'] is not None and not data['qfq'].empty:
                logger.info(f"✓ {code} 下载成功，数据条数: {len(data['qfq'])}")
                success_count += 1
            else:
                logger.warning(f"✗ {code} 数据为空")
                fail_count += 1
                failed_codes.append(code)
            
        except Exception as e:
            logger.error(f"✗ {code} 下载失败: {str(e)}")
            fail_count += 1
            failed_codes.append(code)
    
    logger.info("\n=== 下载完成 ===")
    logger.info(f"成功: {success_count} 只")
    logger.info(f"失败: {fail_count} 只")
    if failed_codes:
        logger.info(f"失败代码: {', '.join(failed_codes)}")
    
    return success_count, fail_count, failed_codes

if __name__ == '__main__':
    download_all_high_elastic_etfs()