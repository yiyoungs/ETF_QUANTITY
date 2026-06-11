import os
from datetime import datetime
from dataclasses import dataclass, field

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

# ==================== 高弹性行业/主题 ETF 池 ====================
HIGH_ELASTIC_ETF_POOL = {
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
        '168204',  # 煤炭LOF
        '512400',  # 有色ETF
    ]
}

def get_all_high_elastic_etfs():
    """获取所有高弹性ETF代码列表"""
    all_codes = []
    for category, codes in HIGH_ELASTIC_ETF_POOL.items():
        all_codes.extend(codes)
    return all_codes

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
    
    # 市场模式切换配置
    MARKET_MODE_CONFIRM_DAYS = 2  # 连续调仓日确认次数
    MA200_WINDOW = 200  # MA200周期

# ==================== 回测时间配置 ====================
@dataclass
class BacktestConfig:
    start_date: str = '2018-01-01'
    end_date: str = field(default_factory=lambda: datetime.now().strftime(DATE_FORMAT))
    rebalance_day: int = 4

# ==================== 日志配置 ====================
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# ==================== 数据获取配置 ====================
class DataConfig:
    REQUEST_INTERVAL = 0.5
    MAX_RETRY = 3
    TIMEOUT = 30

# ==================== 参数敏感性测试配置 ====================
PARAM_GRID = {
    'lookback_days': [40, 50, 60, 70, 80],
    'trailing_stop_pct': [0.08, 0.10, 0.12, 0.15]
}

# 全局默认回测配置实例
DEFAULT_BACKTEST_CONFIG = BacktestConfig()