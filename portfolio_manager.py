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
    
    def rebalance(self, all_data, etf_pool, current_date, exclude=None, periods_weights=None, market_mode='bear', equity_ratio=1.0):
        """
        双动量策略调仓逻辑（v2 改进版）

        - 引入 equity_ratio [0,1]：平滑控制风险资产仓位，介于空仓与满仓之间
        - 使用波动率加权（替代等权），降低高波动标的对组合回撤的贡献
        - 在执行买入时使用"总资金一次性分配"，避免因分批扣减导致的递减分配

        :param exclude: 排除的标的集合（用于止损后调仓，避免刚卖又买回）
        :param periods_weights: 动量周期配置
        :param market_mode: 市场模式 'bull' 或 'bear'
        :param equity_ratio: 风险资产仓位比例 (0.0 ~ 1.0)，用于平滑调节
        """
        qfq_data = all_data.get('qfq', all_data)
        hfq_data = all_data.get('hfq', all_data)

        # --- 决定目标组合 ---
        if market_mode == 'bull':
            target_portfolio = self._get_bull_portfolio(qfq_data, etf_pool, current_date, periods_weights)
        else:
            target_portfolio = []

        if exclude:
            target_portfolio = [code for code in target_portfolio if code not in exclude]

        # 始终保留国债 ETF 作为安全垫
        if StrategyConfig.CASH_ETF_CODE not in target_portfolio:
            target_portfolio.append(StrategyConfig.CASH_ETF_CODE)

        current_holdings = set(self.get_current_holdings())
        target_holdings = set(target_portfolio)

        to_sell = current_holdings - target_holdings

        logger.info(
            f"{current_date} [{market_mode}] 调仓计划 - "
            f"equity_ratio={equity_ratio:.2f}, 卖出: {list(to_sell)}, 候选买入: {target_portfolio}"
        )

        # --- 卖出不在目标池的持仓 ---
        proceeds_cash = 0.0
        for etf_code in to_sell:
            if etf_code not in self.positions:
                continue
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

            self.cash += (amount - cost)
            proceeds_cash += (amount - cost)
            del self.positions[etf_code]
            if etf_code in self.bought_dates:
                del self.bought_dates[etf_code]

            logger.info(f"{current_date} 清仓卖出 {etf_code}: {shares} 份 @ {price:.2f}, 获得 {amount - cost:.2f}")

        # --- 执行买入（使用波动率加权 + equity_ratio 控制） ---
        self._execute_buy(
            all_data, target_portfolio, current_date, market_mode,
            equity_ratio=equity_ratio
        )
    
    def _get_bull_portfolio(self, qfq_data, etf_pool, current_date, periods_weights):
        """
        牛市模式：获取Top5行业ETF（不含国债ETF）
        取消时序MA60过滤，只要动量排名前5就买入
        """
        from signal_engine import get_top_momentum_stocks
        
        equity_pool = etf_pool[etf_pool['code'] != StrategyConfig.CASH_ETF_CODE]
        
        top_stocks = get_top_momentum_stocks(
            qfq_data, equity_pool, current_date, 
            top_n=5, periods_weights=periods_weights
        )
        
        return top_stocks
    
    def _execute_buy(self, all_data, target_portfolio, current_date, market_mode, equity_ratio=1.0):
        """
        执行买入操作（v2 改进版）

        关键改进：
        1. 总资金在调仓前一次性决定（避免分批扣减导致的递减分配 bug）
        2. 风险资产使用反向波动率加权（替代等权），降低高波动标的权重
        3. 通过 equity_ratio 控制风险资产总仓位，平滑市场切换的回撤
        4. 保留单标的权重上限（默认 25%），避免集中风险
        5. 使用 T-1 收盘价作为执行价格，避免 lookahead bias

        :param all_data: 完整的 ETF 数据字典（包含 hfq/qfq/dividend）
        :param target_portfolio: 目标持仓列表（包含国债 ETF）
        :param equity_ratio: 风险资产仓位比例 [0, 1]
        """
        hfq_data = all_data.get('hfq', all_data)
        cash_etf = StrategyConfig.CASH_ETF_CODE

        # 划分风险资产 vs 现金安全标的
        risk_etfs = [code for code in target_portfolio if code != cash_etf]

        # 预先获取所有标的价格 / 停牌检测
        price_info = {}
        for etf_code in target_portfolio:
            df = hfq_data.get(etf_code)
            if df is None or df.empty:
                logger.warning(f"{current_date} 买入失败: {etf_code} 不在 hfq 数据中")
                continue

            # 停牌检测（若最近一天成交量为 0 则认为停牌）
            date_df = df[df['date'] <= pd.to_datetime(current_date)]
            if len(date_df) < 2:
                continue
            last_row = date_df.iloc[-1]
            if 'volume' in last_row and pd.notna(last_row['volume']) and last_row['volume'] == 0:
                logger.info(f"{current_date} 跳过停牌标的: {etf_code}")
                continue

            # 使用 T-1 close（避免使用当日未实现收盘）
            price = float(date_df['close'].iloc[-2])
            if pd.isna(price) or price <= 0:
                continue

            price_info[etf_code] = {'price': price}

        if not price_info:
            logger.warning(f"{current_date} 没有可买入的标的")
            return

        # --- 权重分配 ---
        # 风险资产：根据可用资金 × equity_ratio 分配；使用反向波动率加权
        # 国债资产：剩余资金 × (1 - equity_ratio) + 风险资产买不进后剩下的资金
        total_cash = float(self.cash)
        if total_cash <= 0:
            logger.info(f"{current_date} 无可用现金用于买入")
            return

        # 约束 equity_ratio 范围
        equity_ratio = max(0.0, min(1.0, float(equity_ratio)))

        # 仅在 bull 模式且 equity_ratio > 0 时买入风险资产
        eligible_risk_etfs = []
        if market_mode == 'bull' and equity_ratio > 0 and risk_etfs:
            eligible_risk_etfs = [code for code in risk_etfs if code in price_info]

        risk_weights = {}
        if eligible_risk_etfs:
            # 对风险资产计算波动率权重（用完整 all_data，支持 hfq/qfq 访问）
            risk_weights = self._calculate_volatility_weights(
                all_data=all_data,
                target_portfolio=eligible_risk_etfs,
                current_date=current_date,
                max_weight=0.25
            )
            w_sum = sum(risk_weights.values())
            if w_sum > 0:
                risk_weights = {k: v / w_sum for k, v in risk_weights.items()}
            else:
                risk_weights = {k: 1.0 / len(eligible_risk_etfs) for k in eligible_risk_etfs}

        # 分配总资金
        risk_cash_total = total_cash * equity_ratio
        cash_etf_cash_total = total_cash * (1.0 - equity_ratio)

        logger.info(
            f"{current_date} 资金分配 - 总现金: {total_cash:.2f}, "
            f"风险资产资金: {risk_cash_total:.2f} (equity_ratio={equity_ratio:.2f}), "
            f"国债资金: {cash_etf_cash_total:.2f}"
        )

        # --- 风险资产买入（固定预算 × 波动率权重，不随前次买入递减） ---
        total_risk_spent = 0.0
        for etf_code in eligible_risk_etfs:
            if etf_code not in risk_weights or risk_weights[etf_code] <= 0:
                continue
            if etf_code not in price_info:
                continue

            price = price_info[etf_code]['price']
            allocation = risk_cash_total * risk_weights[etf_code]
            max_shares = int(allocation / price / (1 + StrategyConfig.TRANSACTION_COST))
            shares_to_buy = (max_shares // 100) * 100

            if shares_to_buy <= 0:
                continue

            total_spent = shares_to_buy * price * (1 + StrategyConfig.TRANSACTION_COST)
            if total_spent > self.cash:
                continue

            self.positions[etf_code] = shares_to_buy
            self.bought_dates[etf_code] = current_date
            self.highest_prices[etf_code] = price
            self.cash -= total_spent
            total_risk_spent += total_spent
            logger.info(
                f"{current_date} 买入 {etf_code}: {shares_to_buy} 份 @ {price:.4f}, "
                f"权重 {risk_weights[etf_code]:.2%}, 花费 {total_spent:.2f}"
            )

        # --- 最后买入国债（用配置的国债资金 + 风险资产未用完部分） ---
        if cash_etf in price_info:
            price = price_info[cash_etf]['price']
            risk_unused = max(0.0, risk_cash_total - total_risk_spent)
            # 取"配置预算 + 风险资产未用完部分"和"当前现金的99%"中的较小值
            available_for_cash_etf = min(
                cash_etf_cash_total + risk_unused,
                self.cash * 0.99
            )

            if available_for_cash_etf > price * 100:
                max_shares = int(available_for_cash_etf / price / (1 + StrategyConfig.TRANSACTION_COST))
                shares_to_buy = (max_shares // 100) * 100

                if shares_to_buy > 0:
                    total_spent = shares_to_buy * price * (1 + StrategyConfig.TRANSACTION_COST)
                    self.positions[cash_etf] = self.positions.get(cash_etf, 0) + shares_to_buy
                    self.bought_dates[cash_etf] = current_date
                    if cash_etf not in self.highest_prices:
                        self.highest_prices[cash_etf] = price
                    else:
                        self.highest_prices[cash_etf] = max(self.highest_prices[cash_etf], price)
                    self.cash -= total_spent
                    logger.info(
                        f"{current_date} 买入国债 {cash_etf}: {shares_to_buy} 份 @ {price:.4f}, "
                        f"花费 {total_spent:.2f}"
                    )

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
