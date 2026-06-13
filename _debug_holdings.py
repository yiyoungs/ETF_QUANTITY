import pandas as pd
import numpy as np
import os

from config import CACHE_DIR, StrategyConfig, BacktestConfig
from backtester import Backtester

# 1. 先看有哪些 ETF 以及每只的历史收益
print("=" * 80)
print("ETF 池各标的 2018-2025 全期表现")
print("=" * 80)

etf_summary = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if not fname.endswith('_hfq.csv'):
        continue
    code = fname.split('_')[0]
    df = pd.read_csv(os.path.join(CACHE_DIR, fname), parse_dates=['date'])
    df = df[(df.date >= '2018-01-02') & (df.date <= '2025-12-31')].reset_index(drop=True)
    if len(df) < 50:
        continue
    ret = df.close.iloc[-1] / df.close.iloc[0] - 1
    cummax = df.close.cummax()
    maxdd = ((df.close - cummax) / cummax).min()
    vol = df.close.pct_change().std() * np.sqrt(252)
    etf_summary.append((code, ret, vol, maxdd, len(df)))

etf_summary.sort(key=lambda x: x[1], reverse=True)
print("| 代码    | 总收益(%) | 年化波动(%) | 最大回撤(%) | 交易日 |")
print("|---------|-----------|-------------|-------------|--------|")
for code, ret, vol, maxdd, nd in etf_summary:
    print(f"| {code:>7} | {ret*100:>9.2f} | {vol*100:>11.2f} | {maxdd*100:>11.2f} | {nd:>6} |")

# 2. 再看 511010（国债ETF）的情况
print()
print("=" * 80)
print("511010 国债 ETF 年化近似（无风险资产基准）")
print("=" * 80)
bond = pd.read_csv(os.path.join(CACHE_DIR, '511010_hfq.csv'), parse_dates=['date'])
bond = bond[(bond.date >= '2018-01-02') & (bond.date <= '2025-12-31')].reset_index(drop=True)
bond_ret = bond.close.iloc[-1] / bond.close.iloc[0] - 1
years = (bond.date.iloc[-1] - bond.date.iloc[0]).days / 365.25
bond_ann = (1 + bond_ret) ** (1 / years) - 1
print(f"国债总收益: {bond_ret*100:.2f}%, 年化: {bond_ann*100:.2f}%, {years:.2f}年")

# 3. 看策略实际持仓与模式切换分布
print()
print("=" * 80)
print("策略各年的持仓结构（市场模式 vs 实际风险资产持有比例）")
print("=" * 80)

all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
codes = []
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        codes.append(code)
        hfq = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        qfq = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])
        all_data['hfq'][code] = hfq
        all_data['qfq'][code] = qfq

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})

# 运行回测，记录每日持仓细节
class DebugBacktester(Backtester):
    def run(self):
        records = []
        for date_str in self.trading_dates:
            if self.is_rebalance_day(date_str):
                self._check_market_mode(date_str)
            # 止损
            hfq_data = all_data.get('hfq', all_data)
            if hasattr(self.portfolio_manager, 'check_stop_loss'):
                stop_loss_list = self.portfolio_manager.check_stop_loss(
                    hfq_data, date_str, self.params.get('trailing_stop_pct', 0.08)
                )
                for etf_code in stop_loss_list:
                    self.portfolio_manager.execute_stop_loss(hfq_data, etf_code, date_str)
            if hasattr(self.portfolio_manager, 'update_highest_price'):
                self.portfolio_manager.update_highest_price(all_data, date_str)

            # 调仓
            if self.is_rebalance_day(date_str):
                if self.initial_rebalance_done:
                    self._weekly_rebalance(
                        date_str, exclude=set(stop_loss_list) if stop_loss_list else set(),
                        equity_ratio=getattr(self, 'equity_ratio', 1.0)
                    )
                else:
                    self.portfolio_manager.rebalance(
                        self.all_data, self.etf_pool, date_str,
                        periods_weights=self.periods_weights,
                        market_mode=self.market_mode,
                        equity_ratio=getattr(self, 'equity_ratio', 1.0)
                    )
                    self.initial_rebalance_done = True

            total_val = self.portfolio_manager.get_total_value(self.all_data, date_str)
            positions = dict(self.portfolio_manager.positions)

            # 统计风险资产 vs 债券
            risk_value = 0.0
            bond_value = 0.0
            for code, shares in positions.items():
                if code == StrategyConfig.CASH_ETF_CODE:
                    if code in hfq_data:
                        price = hfq_data[code][hfq_data[code]['date'] <= pd.to_datetime(date_str)]['close'].iloc[-1]
                        bond_value += shares * price
                else:
                    if code in hfq_data:
                        price = hfq_data[code][hfq_data[code]['date'] <= pd.to_datetime(date_str)]['close'].iloc[-1]
                        risk_value += shares * price

            records.append({
                'date': date_str,
                'nav': total_val / StrategyConfig.INITIAL_CAPITAL,
                'market_mode': self.market_mode,
                'equity_ratio': getattr(self, 'equity_ratio', 1.0),
                'risk_value': risk_value,
                'bond_value': bond_value,
                'cash': self.portfolio_manager.cash,
                'positions': positions.copy() if positions else {},
            })

        return pd.DataFrame(records)

bt_config = BacktestConfig(start_date='2018-01-01', end_date='2025-12-31')
bt = DebugBacktester(all_data, etf_pool, backtest_config=bt_config)
records = bt.run()
records['year'] = pd.to_datetime(records['date']).dt.year

# 汇总每年
for year, group in records.groupby('year'):
    bull_days = (group['market_mode'] == 'bull').sum()
    bear_days = (group['market_mode'] == 'bear').sum()
    avg_eq = group['equity_ratio'].mean()
    risk_pct = (group['risk_value'] / (group['risk_value'] + group['bond_value'] + group['cash'] + 1e-12)).mean()
    bond_pct = (group['bond_value'] / (group['risk_value'] + group['bond_value'] + group['cash'] + 1e-12)).mean()
    cash_pct = (group['cash'] / (group['risk_value'] + group['bond_value'] + group['cash'] + 1e-12)).mean()
    # 风险资产持仓名单（取非空且非国债的集合）
    risk_codes = set()
    for pos in group['positions']:
        for c in pos:
            if c != StrategyConfig.CASH_ETF_CODE:
                risk_codes.add(c)
    print(f"{year}: bull={bull_days} days, bear={bear_days} days, "
          f"avg_eq_ratio={avg_eq:.2f}, 风险仓位={risk_pct*100:.1f}%, "
          f"国债仓位={bond_pct*100:.1f}%, 现金={cash_pct*100:.1f}%, "
          f"涉及标的={sorted(risk_codes)}")

# 全期汇总
print()
total_risk = (records['risk_value'].sum() / (records['risk_value'] + records['bond_value'] + records['cash'] + 1e-12)).mean()
print(f"全期平均风险资产仓位: {total_risk*100:.1f}%")
print(f"牛市天数占比: {(records['market_mode']=='bull').mean()*100:.1f}%")

# 看被选中的动量 Top 代码分布
print()
print("=" * 80)
print("Top 动量 ETF 被选中的频次")
print("=" * 80)

from collections import Counter
code_counter = Counter()
for pos in records['positions']:
    for code in pos:
        if code != StrategyConfig.CASH_ETF_CODE:
            code_counter[code] += 1
for code, cnt in code_counter.most_common(15):
    print(f"{code}: {cnt} 天")
