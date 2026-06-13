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
        
        self.periods_weights = self.params.get('periods_weights', None)
        
        self.portfolio_manager = PortfolioManager(params=self.params)
        self.trading_dates = self._get_trading_dates()
        self.daily_results = []
        self.initial_rebalance_done = False

        # 双动量策略：市场模式判断
        self.market_mode = 'bear'  # 'bull' or 'bear'，初始为熊市
        self.bull_signal_count = 0  # 连续牛市信号计数
        self.bear_signal_count = 0  # 连续熊市信号计数
        self.MA200_WINDOW = 200
        self.equity_ratio = 0.0  # 风险资产仓位比例 (0.0 ~ 1.0)，用于平滑仓位
    
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

        start_str = self.backtest_config.start_date
        end_str = self.backtest_config.end_date

        # 找到第一个 >= start_str 的交易日（如果 start_str 不是交易日）
        start_idx = 0
        for i, d in enumerate(dates):
            if d >= start_str:
                start_idx = i
                break
        # 找到最后一个 <= end_str 的交易日
        end_idx = len(dates)
        for i in range(len(dates) - 1, -1, -1):
            if dates[i] <= end_str:
                end_idx = i + 1
                break

        return dates[start_idx:end_idx]
    
    def is_rebalance_day(self, date_str):
        date = pd.to_datetime(date_str)
        return date.weekday() == self.backtest_config.rebalance_day
    
    def _check_market_mode(self, date_str):
        """
        检查市场模式：基于沪深300 MA200判断（使用前复权QFQ数据）

        [Bug修复 v2]
        - 消除 lookahead bias：使用"昨日 close vs 昨日 MA200"判断，避免使用当日收盘
        - 引入趋势强度打分 equity_ratio ∈ [0, 1]，用于平滑仓位调节
        - 增加连续信号确认阈值（MIN_CONFIRM_WEEKS），减少震荡市 whipsaw
        """
        # 使用QFQ（前复权）数据计算MA200，避免除权导致的均线失真
        qfq_data = self.all_data.get('qfq', self.all_data)

        if '510300' not in qfq_data:
            logger.warning("无法获取沪深300ETF数据，使用当前市场模式")
            self.equity_ratio = 0.0
            return self.market_mode

        df = qfq_data['510300']
        df = df[df['date'] <= pd.to_datetime(date_str)].copy()

        # 需要至少 MA200 + 1 天数据，才能计算"昨日 MA200"和"昨日 close"
        if len(df) < self.MA200_WINDOW + 1:
            self.equity_ratio = 0.0
            return self.market_mode

        df.loc[:, 'ma200'] = df['close'].rolling(window=self.MA200_WINDOW).mean()
        # 取倒数第二行：昨日 close vs 昨日 MA200（避免使用当日尚未发生的 close）
        latest_data = df.iloc[-2]

        prev_close = float(latest_data['close'])
        prev_ma200 = float(latest_data['ma200'])

        if pd.isna(prev_ma200) or prev_ma200 <= 0:
            self.equity_ratio = 0.0
            return self.market_mode

        # 趋势偏离度：(close - MA200) / MA200
        deviation = (prev_close - prev_ma200) / prev_ma200

        # --- 趋势强度打分（平滑过渡，降低震荡回撤） ---
        # 偏离度 < -0.02：空仓防守
        # 偏离度 -0.02 ~ +0.02：半仓中性
        # 偏离度 > +0.05：满仓进攻
        if deviation < -0.02:
            self.equity_ratio = 0.0
            current_signal = 'bear'
        elif deviation < 0.02:
            self.equity_ratio = 0.5
            current_signal = 'neutral'
        elif deviation < 0.05:
            self.equity_ratio = 0.8
            current_signal = 'bull'
        else:
            self.equity_ratio = 1.0
            current_signal = 'bull'

        # --- 信号计数 / 延迟确认 ---
        MIN_CONFIRM_WEEKS = 2  # 需要连续 2 周确认，减少 whipsaw
        if current_signal == 'bull':
            self.bull_signal_count += 1
            self.bear_signal_count = 0
        else:
            self.bear_signal_count += 1
            self.bull_signal_count = 0

        # 从空仓切到满仓需要 MIN_CONFIRM_WEEKS 周连续 bull
        if current_signal == 'bull' and self.market_mode != 'bull':
            if self.bull_signal_count >= MIN_CONFIRM_WEEKS:
                logger.info(
                    f"{date_str} 市场模式切换：{self.market_mode} -> bull "
                    f"(连续 {MIN_CONFIRM_WEEKS} 周确认, 偏离度={deviation:.4f})"
                )
                self.market_mode = 'bull'
        elif current_signal == 'bear' and self.market_mode != 'bear':
            if self.bear_signal_count >= 1:
                logger.info(
                    f"{date_str} 市场模式切换：{self.market_mode} -> bear "
                    f"(偏离度={deviation:.4f})"
                )
                self.market_mode = 'bear'
        # neutral 不主动切换 market_mode，但 equity_ratio 会影响调仓

        logger.debug(
            f"{date_str} 市场模式判断: close={prev_close:.4f}, "
            f"MA200={prev_ma200:.4f}, 偏离度={deviation:.4f}, "
            f"equity_ratio={self.equity_ratio:.2f}, mode={self.market_mode}"
        )
        return self.market_mode
    
    def run(self):
        logger.info(f"开始回测: {self.trading_dates[0]} ~ {self.trading_dates[-1]}")
        logger.info(f"初始资金: {StrategyConfig.INITIAL_CAPITAL:,}")
        logger.info(f"动量周期配置: {self.periods_weights}")
        logger.info(f"初始市场模式: {self.market_mode}")
        
        for date_str in self.trading_dates:
            date = pd.to_datetime(date_str)
            
            # 只有在调仓日才检查市场模式
            if self.is_rebalance_day(date_str):
                self._check_market_mode(date_str)
            
            # 止损检查和执行（每日执行）
            stop_loss_result = self._daily_process(date_str)
            stop_loss_list = stop_loss_result['list']
            
            # 只有在调仓日才执行调仓（每周一次）
            if self.is_rebalance_day(date_str):
                if self.initial_rebalance_done:
                    self._weekly_rebalance(
                        date_str, exclude=set(stop_loss_list),
                        equity_ratio=getattr(self, 'equity_ratio', 1.0)
                    )
                else:
                    logger.info(f"{date_str} 执行初始调仓，当前市场模式: {self.market_mode}")
                    self.portfolio_manager.rebalance(
                        self.all_data, self.etf_pool, date_str,
                        periods_weights=self.periods_weights,
                        market_mode=self.market_mode,
                        equity_ratio=getattr(self, 'equity_ratio', 1.0)
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
    
    def _weekly_rebalance(self, date_str, exclude=None, equity_ratio=1.0):
        logger.info(
            f"{date_str} 执行周度调仓，当前市场模式: {self.market_mode}, "
            f"equity_ratio={equity_ratio:.2f}"
        )
        self.portfolio_manager.rebalance(
            self.all_data, self.etf_pool, date_str,
            exclude=exclude,
            periods_weights=self.periods_weights,
            market_mode=self.market_mode,
            equity_ratio=equity_ratio
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
