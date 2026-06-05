import os
from datetime import datetime

# ==================== 路径配置 ====================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
CACHE_DIR = os.path.join(DATA_DIR, 'cache')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# ==================== 数据格式规范 ====================
DATE_FORMAT = '%Y-%m-%d'
COLUMN_MAP = {
    'date': 'date',
    'open': 'open',
    'high': 'high',
    'low': 'low',
    'close': 'close',
    'volume': 'volume',
    'amount': 'amount'
}

# ==================== ETF 代码前缀映射（6位纯数字 → 交易所后缀） ====================
ETF_PREFIX_MAP = {
    '510': 'SH',
    '511': 'SH',
    '512': 'SH',
    '513': 'SH',
    '515': 'SH',
    '516': 'SH',
    '518': 'SH',
    '519': 'SH',
    '501': 'SH',
    '502': 'SH',
    '159': 'SZ',
    '1599': 'SZ',
    '16': 'SZ',
    '514': 'SH',
    '517': 'SH',
}

# ==================== 策略参数配置 ====================
class StrategyConfig:
    INITIAL_CAPITAL = 1000000
    TRANSACTION_COST = 0.001
    LOOKBACK_DAYS = 60
    SHORT_LOOKBACK_DAYS = 20
    MAX_CANDIDATES = 10
    TARGET_HOLDINGS = 5
    STOP_LOSS_THRESHOLD = -0.05
    ROLLING_WINDOW = 20
    MIN_AMOUNT = 50000000
    MIN_SCALE = 200000000
    MIN_LISTING_DAYS = 365
    CASH_ETF_CODE = '511010'

# ==================== 回测时间配置 ====================
class BacktestConfig:
    START_DATE = '2018-01-01'
    END_DATE = datetime.now().strftime(DATE_FORMAT)
    REBALANCE_DAY = 4

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ==================== 数据获取配置 ====================
class DataConfig:
    REQUEST_INTERVAL = 0.5
    MAX_RETRY = 3
    TIMEOUT = 30