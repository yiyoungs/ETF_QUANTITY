import pandas as pd
import numpy as np
from config import StrategyConfig, BacktestConfig
from portfolio_manager import PortfolioManager
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'backtester.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, all_data, etf_pool):
        self.all_data = all_data
        self.etf_pool = etf_pool
        self.portfolio_manager = PortfolioManager()
        self.trading_dates = self._get_trading_dates()
        self.daily_results = []
    
    def _get_trading_dates(self):
        dates = set()
        if 'qfq' in self.all_data:
            for etf_code, df in self.all_data['qfq'].items():
                if not df.empty:
                    dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())
        else:
            for etf_code, df in self.all_data.items():
                if not df.empty:
                    dates.update(df['date'].dt.strftime('%Y-%m-%d').tolist())
        
        dates = sorted(list(dates))
        if not dates:
            return []
        
        start_idx = dates.index(BacktestConfig.START_DATE) if BacktestConfig.START_DATE in dates else 0
        end_idx = dates.index(BacktestConfig.END_DATE) + 1 if BacktestConfig.END_DATE in dates else len(dates)
        
        return dates[start_idx:end_idx]
    
    def is_rebalance_day(self, date_str):
        date = pd.to_datetime(date_str)
        return date.weekday() == BacktestConfig.REBALANCE_DAY
    
    def run(self):
        logger.info(f"开始回测: {self.trading_dates[0]} ~ {self.trading_dates[-1]}")
        logger.info(f"初始资金: {StrategyConfig.INITIAL_CAPITAL:,}")
        
        first_date = self.trading_dates[0]
        logger.info(f"{first_date} 执行初始调仓")
        self.portfolio_manager.rebalance(self.all_data, self.etf_pool, first_date)
        
        for date_str in self.trading_dates:
            date = pd.to_datetime(date_str)
            
            self._daily_process(date_str)
            
            if self.is_rebalance_day(date_str):
                self._weekly_rebalance(date_str)
            
            total_value = self.portfolio_manager.get_total_value(self.all_data, date_str)
            self.portfolio_manager.record_daily_status(date_str, total_value)
            
            daily_record = {
                'date': date_str,
                'total_value': total_value,
                'cash': self.portfolio_manager.cash,
                'positions': dict(self.portfolio_manager.positions),
                'is_rebalance': self.is_rebalance_day(date_str)
            }
            self.daily_results.append(daily_record)
            
            if (self.trading_dates.index(date_str) + 1) % 50 == 0:
                logger.info(f"已完成 {self.trading_dates.index(date_str) + 1}/{len(self.trading_dates)} 个交易日")
        
        logger.info("回测完成")
        return self.get_results_df()
    
    def _daily_process(self, date_str):
        self.portfolio_manager.check_dividends(self.all_data, date_str)
        
        stop_loss_list = self.portfolio_manager.check_stop_loss(self.all_data, date_str)
        
        for etf_code in stop_loss_list:
            self.portfolio_manager.execute_stop_loss(self.all_data, etf_code, date_str)
            
            if StrategyConfig.CASH_ETF_CODE not in self.portfolio_manager.positions:
                hfq_data = self.all_data.get('hfq', self.all_data)
                df = hfq_data.get(StrategyConfig.CASH_ETF_CODE)
                if df is not None and not df.empty:
                    price = df[df['date'] <= pd.to_datetime(date_str)]['close'].iloc[-1]
                    cost = self.portfolio_manager.cash * StrategyConfig.TRANSACTION_COST
                    available_cash = self.portfolio_manager.cash - cost
                    shares = int(available_cash / price / 100) * 100
                    
                    if shares > 0:
                        self.portfolio_manager.positions[StrategyConfig.CASH_ETF_CODE] = shares
                        self.portfolio_manager.bought_dates[StrategyConfig.CASH_ETF_CODE] = date_str
                        self.portfolio_manager.cash -= shares * price + cost
                        
                        logger.info(f"{date_str} 止损后换入 {StrategyConfig.CASH_ETF_CODE}: {shares} 份 @ {price:.2f}")
    
    def _weekly_rebalance(self, date_str):
        logger.info(f"{date_str} 执行周度调仓")
        self.portfolio_manager.rebalance(self.all_data, self.etf_pool, date_str)
    
    def get_results_df(self):
        df = pd.DataFrame(self.daily_results)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        
        df['nav'] = df['total_value'] / StrategyConfig.INITIAL_CAPITAL
        df['daily_return'] = df['nav'].pct_change().fillna(0)
        
        df['benchmark'] = 1.0
        hfq_data = self.all_data.get('hfq', self.all_data)
        if '510300' in hfq_data:
            benchmark_df = hfq_data['510300']
            merged = df.merge(benchmark_df[['date', 'close']], on='date', how='left')
            df['benchmark'] = merged['close'] / merged['close'].iloc[0]
        
        return df

if __name__ == '__main__':
    logger.info("=== 回测引擎测试 ===")
    
    n_days = 1200
    dates = pd.date_range('2018-01-01', periods=n_days, freq='B')
    
    test_data = {
        'qfq': {
            '510050': pd.DataFrame({
                'date': dates,
                'open': np.linspace(2.0, 3.0, n_days) + np.random.normal(0, 0.02, n_days),
                'high': np.linspace(2.02, 3.02, n_days) + np.random.normal(0, 0.02, n_days),
                'low': np.linspace(1.98, 2.98, n_days) + np.random.normal(0, 0.02, n_days),
                'close': np.linspace(2.0, 3.0, n_days) + np.random.normal(0, 0.015, n_days),
                'volume': np.ones(n_days) * 10000000,
                'amount': np.ones(n_days) * 200000000
            }),
            '510300': pd.DataFrame({
                'date': dates,
                'open': np.linspace(4.0, 5.5, n_days) + np.random.normal(0, 0.04, n_days),
                'high': np.linspace(4.02, 5.52, n_days) + np.random.normal(0, 0.04, n_days),
                'low': np.linspace(3.98, 5.48, n_days) + np.random.normal(0, 0.04, n_days),
                'close': np.linspace(4.0, 5.5, n_days) + np.random.normal(0, 0.03, n_days),
                'volume': np.ones(n_days) * 15000000,
                'amount': np.ones(n_days) * 600000000
            }),
            '511010': pd.DataFrame({
                'date': dates,
                'close': np.linspace(100.0, 100.8, n_days) + np.random.normal(0, 0.01, n_days),
                'volume': np.ones(n_days) * 5000000,
                'amount': np.ones(n_days) * 500000000
            })
        },
        'hfq': {
            '510050': pd.DataFrame({
                'date': dates,
                'close': np.linspace(2.0, 3.0, n_days) + np.random.normal(0, 0.015, n_days),
                'volume': np.ones(n_days) * 10000000,
                'amount': np.ones(n_days) * 200000000
            }),
            '510300': pd.DataFrame({
                'date': dates,
                'close': np.linspace(4.0, 5.5, n_days) + np.random.normal(0, 0.03, n_days),
                'volume': np.ones(n_days) * 15000000,
                'amount': np.ones(n_days) * 600000000
            }),
            '511010': pd.DataFrame({
                'date': dates,
                'close': np.linspace(100.0, 100.8, n_days) + np.random.normal(0, 0.01, n_days),
                'volume': np.ones(n_days) * 5000000,
                'amount': np.ones(n_days) * 500000000
            })
        },
        'dividend': {
            '510050': pd.DataFrame({
                'ex_date': [dates[120], dates[360], dates[600]],
                'dividend_per_share': [0.15, 0.12, 0.18]
            }),
            '510300': pd.DataFrame({
                'ex_date': [dates[200], dates[450]],
                'dividend_per_share': [0.10, 0.14]
            }),
            '511010': pd.DataFrame({
                'ex_date': [dates[300], dates[550]],
                'dividend_per_share': [0.30, 0.35]
            })
        }
    }
    
    test_pool = pd.DataFrame({
        'code': ['510050', '510300', '511010'],
        'name': ['上证50ETF', '沪深300ETF', '国债ETF'],
        'type': ['ETF', 'ETF', 'ETF'],
        'list_date': '2015-01-01',
        'scale': 5000000000
    })
    
    backtester = Backtester(test_data, test_pool)
    results_df = backtester.run()
    
    logger.info(f"回测完成，共 {len(results_df)} 个交易日")
    logger.info(f"初始净值: {results_df['nav'].iloc[0]:.4f}")
    logger.info(f"最终净值: {results_df['nav'].iloc[-1]:.4f}")
    logger.info(f"总收益率: {(results_df['nav'].iloc[-1] - 1) * 100:.2f}%")
    
    logger.info("=== 测试完成 ===")
