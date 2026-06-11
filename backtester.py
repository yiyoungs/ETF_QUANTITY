import pandas as pd
import numpy as np
from config import StrategyConfig, BacktestConfig, DEFAULT_BACKTEST_CONFIG
from portfolio_manager import PortfolioManager
import logging
import os

logger = logging.getLogger(__name__)

class Backtester:
    def __init__(self, all_data, etf_pool, params=None, backtest_config=None):
        self.all_data = all_data
        self.etf_pool = etf_pool
        self.params = params or {}
        self.backtest_config = backtest_config or DEFAULT_BACKTEST_CONFIG
        self.lookback_days = self.params.get('lookback_days', StrategyConfig.LOOKBACK_DAYS)
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.08)
        
        self.periods_weights = self.params.get('periods_weights', [(20, 0.5), (60, 0.3), (120, 0.2)])
        
        self.portfolio_manager = PortfolioManager(params=self.params)
        self.trading_dates = self._get_trading_dates()
        self.daily_results = []
        self.initial_rebalance_done = False
        
        # 双动量策略：市场模式判断
        self.market_mode = 'bear'  # 'bull' or 'bear'，初始为熊市
        self.bull_signal_count = 0  # 连续牛市信号计数（调仓日）
        self.bear_signal_count = 0  # 连续熊市信号计数（调仓日）
        self.MA200_WINDOW = StrategyConfig.MA200_WINDOW
        self.MARKET_MODE_CONFIRM_DAYS = StrategyConfig.MARKET_MODE_CONFIRM_DAYS
        
        # 记录每日市场信号（用于检查）
        self.daily_market_signals = []
        
        # 风险控制：动态权益仓位比例
        self.equity_exposure = 0.0  # 当前权益仓位比例，牛市模式下可动态调整
        
        # 风险控制：跟踪沪深300前期高点，用于回撤控制
        self.sp500_high_water_mark = 0.0  # 沪深300ETF前期高点
        
        # 风险控制：跟踪组合净值高点
        self.nav_high_water_mark = 0.0  # 组合净值前期高点
    
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
        
        start_idx = dates.index(self.backtest_config.start_date) if self.backtest_config.start_date in dates else 0
        end_idx = dates.index(self.backtest_config.end_date) + 1 if self.backtest_config.end_date in dates else len(dates)
        
        return dates[start_idx:end_idx]
    
    def is_rebalance_day(self, date_str):
        date = pd.to_datetime(date_str)
        return date.weekday() == self.backtest_config.rebalance_day
    
    def _check_market_mode(self, date_str):
        """
        检查市场模式：基于沪深300 MA200判断（使用前复权QFQ数据）
        - 每日检查沪深300指数 vs MA200
        - 连续2个调仓日确认才切换模式（防震荡市频繁切换）
        - 牛市模式下动态调整权益仓位：当从高点回撤过大时降仓
        :return: 当前市场模式 'bull' 或 'bear'
        """
        # 使用QFQ（前复权）数据计算MA200，避免除权导致的均线失真
        qfq_data = self.all_data.get('qfq', self.all_data)
        
        if '510300' not in qfq_data:
            logger.warning("无法获取沪深300ETF数据，使用当前市场模式")
            return self.market_mode
        
        df = qfq_data['510300']
        df = df[df['date'] <= pd.to_datetime(date_str)].copy()
        
        if len(df) < self.MA200_WINDOW:
            return self.market_mode
        
        df.loc[:, 'ma200'] = df['close'].rolling(window=self.MA200_WINDOW).mean().shift(1)
        latest_data = df.iloc[-1]
        
        close_price = latest_data['close']
        ma200_price = latest_data['ma200']
        
        # 更新前期高点
        if close_price > self.sp500_high_water_mark:
            self.sp500_high_water_mark = close_price
        
        # 计算价格相对于MA200的偏离度
        if ma200_price > 0:
            deviation = (close_price - ma200_price) / ma200_price * 100
        else:
            deviation = 0.0
        
        # 计算从前期高点的回撤比例
        if self.sp500_high_water_mark > 0:
            drawdown = (self.sp500_high_water_mark - close_price) / self.sp500_high_water_mark * 100
        else:
            drawdown = 0.0
        
        if close_price > ma200_price:
            current_signal = 'bull'
        else:
            current_signal = 'bear'
        
        # 记录每日信号
        self.daily_market_signals.append({
            'date': date_str,
            'signal': current_signal,
            'close': close_price,
            'ma200': ma200_price,
            'deviation': deviation,
            'drawdown': drawdown,
            'high_water_mark': self.sp500_high_water_mark
        })
        
        # 只有调仓日才计数和调整
        if self.is_rebalance_day(date_str):
            logger.info(f"{date_str} 市场模式检查 - 当前信号: {current_signal}, 沪深300: {close_price:.2f}, MA200: {ma200_price:.2f}, 偏离度: {deviation:.2f}%, 回撤: {drawdown:.2f}%")
            
            if current_signal == 'bull':
                self.bull_signal_count += 1
                self.bear_signal_count = 0
            else:
                self.bear_signal_count += 1
                self.bull_signal_count = 0
            
            # 连续N个调仓日确认才切换模式
            if self.bull_signal_count >= self.MARKET_MODE_CONFIRM_DAYS and self.market_mode != 'bull':
                logger.info(f"{date_str} 市场模式切换：熊市 -> 牛市 (连续{self.MARKET_MODE_CONFIRM_DAYS}个调仓日信号确认)")
                self.market_mode = 'bull'
                self.equity_exposure = 1.0
                # 重置高点记录
                self.sp500_high_water_mark = close_price
            elif self.bear_signal_count >= self.MARKET_MODE_CONFIRM_DAYS and self.market_mode != 'bear':
                logger.info(f"{date_str} 市场模式切换：牛市 -> 熊市 (连续{self.MARKET_MODE_CONFIRM_DAYS}个调仓日信号确认)")
                self.market_mode = 'bear'
                self.equity_exposure = 0.0
        
        # 牛市模式下：持续更新高点并检查回撤
        if self.market_mode == 'bull':
            # 持续更新沪深300高点
            if close_price > self.sp500_high_water_mark:
                self.sp500_high_water_mark = close_price
                # 重新计算回撤（基于新高点）
                drawdown = 0.0
            
            # 双重风险控制：回撤 + 价格与MA200的偏离度
            # 逐步降仓策略
            price_to_ma200_ratio = close_price / ma200_price if ma200_price > 0 else 1.0
            
            # 根据回撤程度逐步降低权益仓位
            mode_text = ""
            if drawdown > 12.0 or price_to_ma200_ratio < 1.01:
                new_exposure = 0.0
                mode_text = "切换到熊市"
            elif drawdown > 10.0:
                new_exposure = 0.25
                mode_text = "降仓至25%"
            elif drawdown > 8.0:
                new_exposure = 0.5
                mode_text = "降仓至50%"
            elif drawdown > 6.0:
                new_exposure = 0.75
                mode_text = "降仓至75%"
            elif price_to_ma200_ratio < 1.03:
                new_exposure = 0.85
                mode_text = "接近MA200，降仓至85%"
            else:
                new_exposure = 1.0
            
            if new_exposure != self.equity_exposure:
                if new_exposure == 0.0:
                    self.market_mode = 'bear'
                    self.bull_signal_count = 0
                    self.bear_signal_count = self.MARKET_MODE_CONFIRM_DAYS
                if mode_text:
                    logger.info(f"{date_str} 牛市模式风险控制 - 回撤 {drawdown:.2f}%, 价格/MA200: {price_to_ma200_ratio:.3f}, {mode_text}")
                self.equity_exposure = new_exposure
        
        return self.market_mode
    
    def run(self):
        logger.info(f"开始回测: {self.trading_dates[0]} ~ {self.trading_dates[-1]}")
        logger.info(f"初始资金: {StrategyConfig.INITIAL_CAPITAL:,}")
        logger.info(f"动量周期配置: {self.periods_weights}")
        logger.info(f"初始市场模式: {self.market_mode}")
        logger.info(f"模式切换确认天数: {self.MARKET_MODE_CONFIRM_DAYS} 个调仓日")
        
        for date_str in self.trading_dates:
            date = pd.to_datetime(date_str)
            
            # 每日检查市场模式（但只在调仓日才会触发模式切换）
            self._check_market_mode(date_str)
            
            # 止损检查和执行（每日执行）
            stop_loss_result = self._daily_process(date_str)
            stop_loss_list = stop_loss_result['list']
            
            # 只有在调仓日才执行调仓（每周一次）
            if self.is_rebalance_day(date_str):
                if self.initial_rebalance_done:
                    self._weekly_rebalance(date_str, exclude=set(stop_loss_list))
                else:
                    logger.info(f"{date_str} 执行初始调仓，当前市场模式: {self.market_mode}")
                    self.portfolio_manager.rebalance(
                        self.all_data, self.etf_pool, date_str,
                        periods_weights=self.periods_weights,
                        market_mode=self.market_mode,
                        equity_exposure=self.equity_exposure
                    )
                    self.initial_rebalance_done = True
            
            total_value = self.portfolio_manager.get_total_value(self.all_data, date_str)
            self.portfolio_manager.record_daily_status(date_str, total_value, self.all_data)
            
            daily_record = {
                'date': date_str,
                'total_value': total_value,
                'cash': self.portfolio_manager.cash,
                'positions': dict(self.portfolio_manager.positions),
                'is_rebalance': self.is_rebalance_day(date_str),
                'market_mode': self.market_mode
            }
            self.daily_results.append(daily_record)
            
            if (self.trading_dates.index(date_str) + 1) % 50 == 0:
                logger.info(f"已完成 {self.trading_dates.index(date_str) + 1}/{len(self.trading_dates)} 个交易日")
        
        logger.info("回测完成")
        return self.get_results_df()
    
    def _daily_process(self, date_str):
        """每日处理：分红检查、止损检查和执行
        返回：字典 {'triggered': bool, 'list': list}
        """
        self.portfolio_manager.check_dividends(self.all_data, date_str)
        
        logger.debug(f"{date_str} 止损检查: 使用 trailing_stop_pct={self.trailing_stop_pct:.2%}")
        stop_loss_list = self.portfolio_manager.check_stop_loss(
            self.all_data, date_str, self.trailing_stop_pct
        )
        logger.debug(f"{date_str} 止损列表: {stop_loss_list}")
        
        self.portfolio_manager.update_highest_price(self.all_data, date_str)
        
        stop_loss_triggered = len(stop_loss_list) > 0
        
        for etf_code in stop_loss_list:
            self.portfolio_manager.execute_stop_loss(self.all_data, etf_code, date_str)
        
        # 优化3：取消止损后自动换入国债ETF，改为在周度调仓时统一处理
        # 这样可以让止损释放的现金在调仓时更高效地分配
        
        return {'triggered': stop_loss_triggered, 'list': stop_loss_list}
    
    def _weekly_rebalance(self, date_str, exclude=None):
        logger.info(f"{date_str} 执行周度调仓，当前市场模式: {self.market_mode}, 权益仓位: {self.equity_exposure*100:.0f}%")
        self.portfolio_manager.rebalance(
            self.all_data, self.etf_pool, date_str, 
            exclude=exclude,
            periods_weights=self.periods_weights,
            market_mode=self.market_mode,
            equity_exposure=self.equity_exposure
        )
    
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
