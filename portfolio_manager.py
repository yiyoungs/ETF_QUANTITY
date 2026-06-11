import pandas as pd
import numpy as np
from config import StrategyConfig
from signal_engine import is_stop_loss_triggered, generate_target_portfolio
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'portfolio_manager.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self):
        self.positions = {}
        self.cash = StrategyConfig.INITIAL_CAPITAL
        self.bought_dates = {}
        self.daily_records = []
        self.dividend_records = []
    
    def get_current_holdings(self):
        return list(self.positions.keys())
    
    def get_position_size(self, etf_code):
        return self.positions.get(etf_code, 0)
    
    def can_sell(self, etf_code, current_date):
        if etf_code not in self.bought_dates:
            return False
        
        bought_date = pd.to_datetime(self.bought_dates[etf_code])
        current_date = pd.to_datetime(current_date)
        
        days_diff = (current_date - bought_date).days
        return days_diff >= 1
    
    def update_bought_date(self, etf_code, date):
        self.bought_dates[etf_code] = date
    
    def check_stop_loss(self, all_data, current_date):
        stop_loss_list = []
        
        qfq_data = all_data.get('qfq', all_data)
        
        for etf_code in list(self.positions.keys()):
            if etf_code == StrategyConfig.CASH_ETF_CODE:
                continue
            
            if not self.can_sell(etf_code, current_date):
                continue
            
            if is_stop_loss_triggered(qfq_data, etf_code, current_date):
                stop_loss_list.append(etf_code)
        
        return stop_loss_list
    
    def execute_stop_loss(self, all_data, etf_code, current_date):
        if etf_code not in self.positions:
            return 0
        
        hfq_data = all_data.get('hfq', all_data)
        df = hfq_data.get(etf_code)
        if df is None or df.empty:
            logger.warning(f"{current_date} 止损失败: {etf_code} 不在 hfq 数据中")
            return 0
        
        price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
        
        shares = self.positions[etf_code]
        amount = shares * price
        cost = amount * StrategyConfig.TRANSACTION_COST
        
        self.cash += amount - cost
        del self.positions[etf_code]
        del self.bought_dates[etf_code]
        
        logger.info(f"{current_date} 止损卖出 {etf_code}: {shares} 份 @ {price:.2f}, 获得 {amount - cost:.2f}")
        
        return amount - cost
    
    def rebalance(self, all_data, etf_pool, current_date):
        qfq_data = all_data.get('qfq', all_data)
        hfq_data = all_data.get('hfq', all_data)
        
        target_portfolio = generate_target_portfolio(qfq_data, etf_pool, current_date)
        
        current_holdings = set(self.get_current_holdings())
        target_holdings = set(target_portfolio)
        
        to_sell = current_holdings - target_holdings
        to_buy = target_holdings - current_holdings
        to_hold = current_holdings & target_holdings
        
        logger.info(f"{current_date} 调仓计划 - 卖出: {list(to_sell)}, 买入: {list(to_buy)}, 持有: {list(to_hold)}")
        
        for etf_code in to_sell:
            if etf_code in self.positions:
                if not self.can_sell(etf_code, current_date):
                    logger.info(f"{current_date} 无法卖出 {etf_code}: T+1 限制，推迟到下一个交易日")
                    continue
                
                df = hfq_data.get(etf_code)
                if df is None or df.empty:
                    logger.warning(f"{current_date} 卖出失败: {etf_code} 不在 hfq 数据中")
                    continue
                
                price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
                shares = self.positions[etf_code]
                
                shares_to_sell = shares
                if shares_to_sell <= 0:
                    continue
                
                amount = shares_to_sell * price
                cost = amount * StrategyConfig.TRANSACTION_COST
                
                self.cash += amount - cost
                del self.positions[etf_code]
                del self.bought_dates[etf_code]
                
                logger.info(f"{current_date} 清仓卖出 {etf_code}: {shares_to_sell} 份 @ {price:.2f}, 获得 {amount - cost:.2f}")
        
        if to_buy:
            available_cash = self.cash
            allocation_per_stock = available_cash / len(to_buy)
            
            for etf_code in to_buy:
                df = hfq_data.get(etf_code)
                if df is None or df.empty:
                    logger.warning(f"{current_date} 买入失败: {etf_code} 不在 hfq 数据中")
                    continue
                
                price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
                
                max_shares = int(allocation_per_stock / price / (1 + StrategyConfig.TRANSACTION_COST))
                shares_to_buy = (max_shares // 100) * 100
                
                if shares_to_buy <= 0:
                    continue
                
                actual_cost = shares_to_buy * price * StrategyConfig.TRANSACTION_COST
                total_spent = shares_to_buy * price + actual_cost
                
                if total_spent <= self.cash:
                    self.positions[etf_code] = shares_to_buy
                    self.bought_dates[etf_code] = current_date
                    self.cash -= total_spent
                    
                    logger.info(f"{current_date} 全仓买入 {etf_code}: {shares_to_buy} 份 @ {price:.2f}")

    def get_total_value(self, all_data, current_date):
        total = self.cash
        
        hfq_data = all_data.get('hfq', all_data)
        
        for etf_code, shares in self.positions.items():
            df = hfq_data.get(etf_code)
            if df is not None and not df.empty:
                price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
                total += shares * price
        
        return total
    
    def check_dividends(self, all_data, current_date):
        dividend_data = all_data.get('dividend', {})
        
        current_dt = pd.to_datetime(current_date)
        
        for etf_code, shares in list(self.positions.items()):
            dividend_df = dividend_data.get(etf_code)
            if dividend_df is None or dividend_df.empty:
                continue
            
            mask = dividend_df['ex_date'] == current_dt
            if not mask.any():
                continue
            
            row = dividend_df[mask].iloc[0]
            dividend_per_share = row.get('dividend_per_share', 0)
            
            if dividend_per_share > 0:
                total_dividend = shares * dividend_per_share
                
                self.cash += total_dividend
                self.dividend_records.append({
                    'date': current_date,
                    'etf_code': etf_code,
                    'shares': shares,
                    'dividend_per_share': dividend_per_share,
                    'total_dividend': total_dividend
                })
                
                logger.info(f"{current_date} 收到 {etf_code} 现金分红: {total_dividend:.2f}")
    
    def record_daily_status(self, date, total_value):
        record = {
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions': dict(self.positions)
        }
        self.daily_records.append(record)
    
    def get_records_df(self):
        df = pd.DataFrame(self.daily_records)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
        return df

if __name__ == '__main__':
    logger.info("=== 组合管理器测试 ===")
    
    n_days = 200
    dates = pd.date_range('2018-01-01', periods=n_days, freq='B')
    
    test_data = {
        'qfq': {
            '510050': pd.DataFrame({
                'date': dates,
                'close': np.linspace(2.0, 2.5, n_days),
                'amount': np.ones(n_days) * 200000000
            }),
            '510300': pd.DataFrame({
                'date': dates,
                'close': np.linspace(4.0, 5.0, n_days),
                'amount': np.ones(n_days) * 600000000
            }),
            '511010': pd.DataFrame({
                'date': dates,
                'close': np.linspace(100.0, 100.5, n_days),
                'amount': np.ones(n_days) * 500000000
            })
        },
        'hfq': {
            '510050': pd.DataFrame({
                'date': dates,
                'close': np.linspace(2.0, 2.5, n_days)
            }),
            '510300': pd.DataFrame({
                'date': dates,
                'close': np.linspace(4.0, 5.0, n_days)
            }),
            '511010': pd.DataFrame({
                'date': dates,
                'close': np.linspace(100.0, 100.5, n_days)
            })
        },
        'dividend': {
            '510050': pd.DataFrame({
                'ex_date': [dates[100]],
                'dividend_per_share': [0.15]
            })
        }
    }
    
    pm = PortfolioManager()
    print(f"初始现金: {pm.cash}")
    
    pm.rebalance(test_data, pd.DataFrame({'code': ['510050', '510300']}), '2018-01-01')
    print(f"调仓后现金: {pm.cash}")
    print(f"持仓: {pm.positions}")
    
    total_value = pm.get_total_value(test_data, '2018-01-02')
    print(f"第二天总价值: {total_value}")
    
    pm.check_dividends(test_data, dates[100].strftime('%Y-%m-%d'))
    print(f"分红后现金: {pm.cash}")
    
    logger.info("=== 测试完成 ===")
