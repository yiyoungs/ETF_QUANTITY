import pandas as pd
import numpy as np
from config import StrategyConfig
from signal_engine import generate_target_portfolio, calc_time_series_momentum
import logging

logger = logging.getLogger(__name__)

class PortfolioManager:
    def __init__(self, params=None):
        self.positions = {}
        self.cash = StrategyConfig.INITIAL_CAPITAL
        self.bought_dates = {}
        self.daily_records = []
        self.dividend_records = []
        self.highest_prices = {}  # 记录每个持仓标的买入以来的最高价
        self.params = params or {}
        self.lookback_days = self.params.get('lookback_days', StrategyConfig.LOOKBACK_DAYS)
        self.trailing_stop_pct = self.params.get('trailing_stop_pct', 0.08)
    
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
    
    def update_highest_price(self, all_data, current_date):
        """更新每个持仓标的的最高价记录"""
        hfq_data = all_data.get('hfq', all_data)
        
        for etf_code, shares in list(self.positions.items()):
            if etf_code == StrategyConfig.CASH_ETF_CODE:
                continue
            
            df = hfq_data.get(etf_code)
            if df is None or df.empty:
                continue
            
            price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
            
            if etf_code not in self.highest_prices:
                self.highest_prices[etf_code] = price
            else:
                self.highest_prices[etf_code] = max(self.highest_prices[etf_code], price)
    
    def check_stop_loss(self, all_data, current_date, stop_loss_threshold=0.08):
        """
        V1.3 修改：移动止盈/止损机制
        当价格从买入以来的最高点回落超过阈值时触发止损
        :param stop_loss_threshold: 止损回撤阈值，默认 8%
        """
        stop_loss_list = []
        
        hfq_data = all_data.get('hfq', all_data)
        
        for etf_code in list(self.positions.keys()):
            if etf_code == StrategyConfig.CASH_ETF_CODE:
                continue
            
            if not self.can_sell(etf_code, current_date):
                continue
            
            df = hfq_data.get(etf_code)
            if df is None or df.empty:
                continue
            
            price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
            highest_price = self.highest_prices.get(etf_code, price)
            
            if highest_price > 0:
                drawdown = (highest_price - price) / highest_price
                logger.debug(f"{current_date} {etf_code} 止损检查: 当前价 {price:.2f}, 最高价 {highest_price:.2f}, 回撤 {drawdown:.6f}, 阈值 {stop_loss_threshold:.6f}")
                if drawdown >= stop_loss_threshold - 1e-9:
                    stop_loss_list.append(etf_code)
                    logger.info(f"{current_date} {etf_code} 触发移动止损: 当前价 {price:.2f}, 最高价 {highest_price:.2f}, 回撤 {drawdown:.2%}")
        
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
        if etf_code in self.highest_prices:
            del self.highest_prices[etf_code]
        
        logger.info(f"{current_date} 止损卖出 {etf_code}: {shares} 份 @ {price:.2f}, 获得 {amount - cost:.2f}")
        
        return amount - cost
    
    def _calculate_volatility_weights(self, all_data, target_portfolio, current_date, max_weight=0.25):
        """
        计算目标组合中各标的的波动率权重（反向波动率加权）
        
        优化2实现：
        - 仅对目标组合中的标的做波动率加权（已过滤出高动量标的）
        - 权重 = (1/vol_i) / Σ(1/vol_j)
        - 单标的上限 max_weight（默认25%）
        - vol_i = 过去60天日收益率标准差 × √252
        
        :param all_data: 所有ETF数据
        :param target_portfolio: 目标组合列表
        :param current_date: 当前日期
        :param max_weight: 单标的权重上限
        :return: 权重字典 {etf_code: weight}
        """
        hfq_data = all_data.get('hfq', all_data)
        vol_dict = {}
        
        for etf_code in target_portfolio:
            df = hfq_data.get(etf_code)
            if df is None or df.empty:
                continue
            
            df_filtered = df[df['date'] <= pd.to_datetime(current_date)]
            if len(df_filtered) < 61:
                continue
            
            df_filtered['return'] = df_filtered['close'].pct_change()
            daily_std = df_filtered['return'].iloc[-60:].std()
            annual_vol = daily_std * np.sqrt(252)
            
            if annual_vol > 0:
                vol_dict[etf_code] = annual_vol
        
        if not vol_dict:
            equal_weight = 1.0 / len(target_portfolio)
            return {code: equal_weight for code in target_portfolio}
        
        # 反向波动率加权
        inv_vol_sum = sum(1.0 / vol for vol in vol_dict.values())
        weights = {code: (1.0 / vol) / inv_vol_sum for code, vol in vol_dict.items()}
        
        # 应用权重上限并重新分配
        total_excess = 0.0
        capped_weights = {}
        
        for code, weight in weights.items():
            if weight > max_weight:
                capped_weights[code] = max_weight
                total_excess += weight - max_weight
            else:
                capped_weights[code] = weight
        
        if total_excess > 0 and len(capped_weights) > 1:
            non_capped_sum = sum(w for w in capped_weights.values() if w < max_weight)
            if non_capped_sum > 0:
                for code in capped_weights:
                    if capped_weights[code] < max_weight:
                        capped_weights[code] += total_excess * (capped_weights[code] / non_capped_sum)
        
        return capped_weights
    
    def rebalance(self, all_data, etf_pool, current_date, exclude=None, periods_weights=None, market_mode='bear', equity_exposure=1.0):
        """
        双动量策略调仓逻辑
        
        牛市模式（bull）：沪深300 > MA200，选Top5动量行业ETF等权持仓，根据equity_exposure调整权益仓位
        熊市模式（bear）：沪深300 < MA200，100%持有511010，不选行业ETF
        
        :param exclude: 排除的标的集合（用于止损后调仓，避免刚卖又买回）
        :param periods_weights: 动量周期配置，如 [(20, 0.5), (60, 0.3), (120, 0.2)]
        :param market_mode: 市场模式 'bull' 或 'bear'
        :param equity_exposure: 权益仓位比例（0.0-1.0），牛市模式下有效
        """
        qfq_data = all_data.get('qfq', all_data)
        hfq_data = all_data.get('hfq', all_data)
        
        if market_mode == 'bull':
            target_portfolio = self._get_bull_portfolio(qfq_data, etf_pool, current_date, periods_weights)
            # 牛市模式下确保不包含国债ETF
            target_portfolio = [code for code in target_portfolio if code != StrategyConfig.CASH_ETF_CODE]
            
            # 如果权益仓位小于100%，剩余资金买入国债ETF
            if equity_exposure < 1.0 and StrategyConfig.CASH_ETF_CODE not in target_portfolio:
                target_portfolio.append(StrategyConfig.CASH_ETF_CODE)
        else:
            target_portfolio = [StrategyConfig.CASH_ETF_CODE]
        
        if exclude:
            target_portfolio = [code for code in target_portfolio if code not in exclude]
        
        # 保存权益仓位比例供后续买入时使用
        self.current_equity_exposure = equity_exposure
        
        current_holdings = set(self.get_current_holdings())
        target_holdings = set(target_portfolio)
        
        to_sell = current_holdings - target_holdings
        to_buy = target_holdings - current_holdings
        
        logger.info(f"{current_date} [{market_mode}] 调仓计划 - 卖出: {list(to_sell)}, 买入: {list(to_buy)}")
        
        for etf_code in to_sell:
            if etf_code in self.positions:
                if not self.can_sell(etf_code, current_date):
                    logger.info(f"{current_date} 无法卖出 {etf_code}: T+1 限制")
                    continue
                
                df = hfq_data.get(etf_code)
                if df is None or df.empty:
                    logger.warning(f"{current_date} 卖出失败: {etf_code} 不在 hfq 数据中")
                    continue
                
                price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
                shares = self.positions[etf_code]
                
                amount = shares * price
                cost = amount * StrategyConfig.TRANSACTION_COST
                
                self.cash += amount - cost
                del self.positions[etf_code]
                del self.bought_dates[etf_code]
                
                logger.info(f"{current_date} 清仓卖出 {etf_code}: {shares} 份 @ {price:.2f}, 获得 {amount - cost:.2f}")
        
        if to_buy:
            self._execute_buy(hfq_data, to_buy, current_date, market_mode)
    
    def _get_bull_portfolio(self, qfq_data, etf_pool, current_date, periods_weights):
        """
        牛市模式：获取Top5行业ETF（不含国债ETF）
        取消时序MA60过滤，只要动量排名前5就买入
        """
        from signal_engine import generate_target_portfolio
        
        equity_pool = etf_pool[etf_pool['code'] != StrategyConfig.CASH_ETF_CODE]
        
        # 牛市模式下跳过时序动量过滤，直接选择截面动量最高的标的
        top_stocks = generate_target_portfolio(
            qfq_data, equity_pool, current_date, 
            periods_weights=periods_weights,
            skip_ts_filter=True  # 牛市模式不使用MA60过滤
        )
        
        # 确保不包含国债ETF
        top_stocks = [code for code in top_stocks if code != StrategyConfig.CASH_ETF_CODE]
        
        return top_stocks
    
    def _execute_buy(self, hfq_data, to_buy, current_date, market_mode):
        """
        执行买入操作
        牛市模式：根据equity_exposure分配资金，权益部分等权分配给行业ETF，剩余买入国债ETF
        熊市模式：100%资金买入国债ETF
        """
        remaining_cash = self.cash
        
        buy_info = {}
        for etf_code in to_buy:
            df = hfq_data.get(etf_code)
            if df is None or df.empty:
                logger.warning(f"{current_date} 买入失败: {etf_code} 不在 hfq 数据中")
                continue
            
            date_df = df[df['date'] == pd.to_datetime(current_date)]
            if not date_df.empty and date_df.iloc[0].get('suspended', False):
                logger.info(f"{current_date} 跳过停牌标的: {etf_code}")
                continue
            
            price = df[df['date'] <= pd.to_datetime(current_date)]['close'].iloc[-1]
            buy_info[etf_code] = {'price': price}
        
        if not buy_info:
            logger.warning(f"{current_date} 没有可买入的标的")
            return
        
        sorted_etfs = list(buy_info.keys())
        
        # 牛市模式下根据权益仓位比例分配资金
        if market_mode == 'bull' and hasattr(self, 'current_equity_exposure'):
            equity_exposure = self.current_equity_exposure
            
            # 分离股票ETF和国债ETF
            equity_etfs = [code for code in sorted_etfs if code != StrategyConfig.CASH_ETF_CODE]
            n_equity = len(equity_etfs)
            
            logger.info(f"{current_date} [{market_mode}] 买入顺序: {sorted_etfs}, 权益仓位: {equity_exposure*100:.0f}%")
            
            for etf_code in sorted_etfs:
                info = buy_info[etf_code]
                price = info['price']
                
                if etf_code == StrategyConfig.CASH_ETF_CODE:
                    allocation = remaining_cash * (1 - equity_exposure)
                else:
                    if n_equity > 0:
                        allocation = remaining_cash * equity_exposure / n_equity
                    else:
                        allocation = 0.0
                
                max_shares = int(allocation / price / (1 + StrategyConfig.TRANSACTION_COST))
                shares_to_buy = (max_shares // 100) * 100
                
                if shares_to_buy <= 0:
                    continue
                
                total_spent = shares_to_buy * price * (1 + StrategyConfig.TRANSACTION_COST)
                
                if total_spent <= remaining_cash:
                    self.positions[etf_code] = shares_to_buy
                    self.bought_dates[etf_code] = current_date
                    self.highest_prices[etf_code] = price
                    remaining_cash -= total_spent
                    if etf_code == StrategyConfig.CASH_ETF_CODE:
                        logger.info(f"{current_date} 买入 {etf_code}: {shares_to_buy} 份 @ {price:.2f}, 权重 {(1-equity_exposure):.1%}")
                    else:
                        logger.info(f"{current_date} 买入 {etf_code}: {shares_to_buy} 份 @ {price:.2f}, 权重 {equity_exposure/n_equity:.1%}")
        else:
            # 熊市模式或没有权益仓位设置，等权分配
            n_stocks = len(buy_info)
            equal_weight = 1.0 / n_stocks
            
            logger.info(f"{current_date} [{market_mode}] 买入顺序: {sorted_etfs}")
            
            for etf_code in sorted_etfs:
                info = buy_info[etf_code]
                price = info['price']
                
                allocation = remaining_cash * equal_weight
                max_shares = int(allocation / price / (1 + StrategyConfig.TRANSACTION_COST))
                shares_to_buy = (max_shares // 100) * 100
                
                if shares_to_buy <= 0:
                    continue
                
                total_spent = shares_to_buy * price * (1 + StrategyConfig.TRANSACTION_COST)
                
                if total_spent <= remaining_cash:
                    self.positions[etf_code] = shares_to_buy
                    self.bought_dates[etf_code] = current_date
                    self.highest_prices[etf_code] = price
                    remaining_cash -= total_spent
                    logger.info(f"{current_date} 买入 {etf_code}: {shares_to_buy} 份 @ {price:.2f}, 权重 {equal_weight:.1%}")
        
        self.cash = remaining_cash

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
    
    def record_daily_status(self, date, total_value, all_data=None):
        record = {
            'date': date,
            'total_value': total_value,
            'cash': self.cash,
            'positions': dict(self.positions)
        }
        
        # ======== 2021-05-18 Debug 日志 ========
        debug_dates = ['2021-05-17', '2021-05-18', '2021-05-19']
        if date in debug_dates:
            logger.info(f"\n{'='*80}")
            logger.info(f"DEBUG {date} - 组合持仓明细")
            logger.info(f"{'='*80}")
            logger.info(f"当日总资产: {total_value:.2f}")
            logger.info(f"可用现金: {self.cash:.2f}")
            logger.info(f"持仓数量: {len(self.positions)} 只")
            
            if all_data is not None:
                hfq_data = all_data.get('hfq', all_data)
                qfq_data = all_data.get('qfq', all_data)
                
                for etf_code, shares in sorted(self.positions.items()):
                    # 获取 HFQ 价格
                    hfq_price = None
                    if etf_code in hfq_data:
                        df = hfq_data[etf_code]
                        df_filtered = df[df['date'] <= pd.to_datetime(date)]
                        if not df_filtered.empty:
                            hfq_price = df_filtered['close'].iloc[-1]
                    
                    # 获取 QFQ 价格
                    qfq_price = None
                    if etf_code in qfq_data:
                        df = qfq_data[etf_code]
                        df_filtered = df[df['date'] <= pd.to_datetime(date)]
                        if not df_filtered.empty:
                            qfq_price = df_filtered['close'].iloc[-1]
                    
                    # 计算市值
                    if hfq_price is not None:
                        market_value = shares * hfq_price
                    else:
                        market_value = 0.0
                    
                    logger.info(f"  [{etf_code}] 股数: {shares}, HFQ价格: {hfq_price:.4f}, QFQ价格: {qfq_price:.4f}, 市值: {market_value:.2f}")
            
            logger.info(f"{'='*80}\n")
        # ======== End Debug 日志 ========
        
        if all_data is not None and total_value > 0:
            hfq_data = all_data.get('hfq', all_data)
            
            bond_etf_code = StrategyConfig.CASH_ETF_CODE
            bond_value = 0.0
            equity_value = 0.0
            
            for etf_code, shares in self.positions.items():
                df = hfq_data.get(etf_code)
                if df is None or df.empty:
                    continue
                
                price = df[df['date'] <= pd.to_datetime(date)]['close'].iloc[-1]
                position_value = shares * price
                
                if etf_code == bond_etf_code:
                    bond_value = position_value
                else:
                    equity_value += position_value
            
            bond_weight = (bond_value / total_value) * 100
            equity_weight = (equity_value / total_value) * 100
            cash_weight = (self.cash / total_value) * 100
            
            record['bond_weight'] = bond_weight
            record['equity_weight'] = equity_weight
            record['cash_weight'] = cash_weight
            
            logger.info(f"{date} 资产权重 - 国债: {bond_weight:.1f}%, 股票ETF: {equity_weight:.1f}%, 现金: {cash_weight:.1f}%")
            
            if len(self.positions) > 0 and equity_weight > 0 and equity_weight < 5.0:
                logger.warning(f"⚠️ {date} 警告：股票型ETF权重仅 {equity_weight:.1f}%，远低于预期！请检查下单股数计算逻辑是否正确")
        
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
