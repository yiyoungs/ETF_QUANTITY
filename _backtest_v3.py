"""改进版策略 v3 (fix): 
- ETF级别的时序动量（价格 > MA60）+ 截面动量（Top 3）
- 月度调仓（每月第一个交易日）
- 按日期查找价格（解决索引不匹配问题）
"""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from config import StrategyConfig, CACHE_DIR

# 加载数据
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

# 建立每个ETF的日期 -> 价格字典（方便按日期查找）
etf_price_map = {}
for code, df in all_data['hfq'].items():
    etf_price_map[code] = dict(zip(df['date'], df['close']))

etf_codes = [c for c in all_data['hfq'].keys() if c != StrategyConfig.CASH_ETF_CODE]
cash_etf = StrategyConfig.CASH_ETF_CODE

# 生成交易日历（使用沪深300的日期）
hs300_dates = set(all_data['hfq'].get('510300', pd.DataFrame()).get('date', []))
dates = sorted([d for d in hs300_dates if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

# 策略参数
LOOKBACK_20 = 20
LOOKBACK_60 = 60
TOP_N = 3
TRAILING_STOP = 0.10  # 10% 移动止损
TXN_COST = 0.001  # 0.1% 单边交易成本
RISK_RATIO = 0.7  # 风险资产目标比例

# 回测状态
cash = float(StrategyConfig.INITIAL_CAPITAL)
positions = {}  # code -> shares
highest_prices = {}
nav_history = []
rebalance_count = 0

def get_price_hfq(code, date):
    """获取某日收盘价格"""
    m = etf_price_map.get(code, {})
    return m.get(date)

def get_close_n_days_ago(code, date, n_days):
    """获取某ETF n个交易日之前的收盘价"""
    df = all_data['hfq'].get(code)
    if df is None:
        return None
    sub = df[df['date'] <= date]
    if len(sub) <= n_days:
        return None
    return float(sub['close'].iloc[-n_days - 1])

def calc_momentum_and_trend(code, date):
    """计算动量 + 时序动量检查"""
    df = all_data['hfq'].get(code)
    if df is None:
        return -np.inf, False
    sub = df[df['date'] <= date]
    if len(sub) <= LOOKBACK_60 + 1:
        return -np.inf, False

    prev_close = float(sub['close'].iloc[-1])
    ret_20 = prev_close / float(sub['close'].iloc[-LOOKBACK_20 - 1]) - 1
    ret_60 = prev_close / float(sub['close'].iloc[-LOOKBACK_60 - 1]) - 1

    # 时序动量：价格是否在MA60之上
    ma60 = float(sub['close'].iloc[-LOOKBACK_60:].mean())
    above_ma = prev_close > ma60

    score = 0.5 * ret_20 + 0.5 * ret_60
    return score, above_ma

def get_total_value(date):
    total = cash
    for code, shares in positions.items():
        price = get_price_hfq(code, date)
        if price is not None and price > 0:
            total += shares * price
    return total

def is_first_trading_of_month(date):
    idx = dates.index(date) if date in dates else -1
    if idx <= 0:
        return True
    return dates[idx - 1].month != date.month or dates[idx - 1].year != date.year

print("开始回测 v3: ETF级别时序+截面双动量，月度调仓")
print(f"参数: TOP_N={TOP_N}, MA_TREND={LOOKBACK_60}, 止损={TRAILING_STOP*100:.0f}%, 风险比={RISK_RATIO*100:.0f}%")
print("=" * 80)

for i, date in enumerate(dates):
    current_total = get_total_value(date)

    # 止损检查
    if i > 0 and positions:
        stop_loss_codes = []
        for code in list(positions.keys()):
            if code == cash_etf:
                continue
            price = get_price_hfq(code, date)
            if price is None or price <= 0:
                continue
            if code not in highest_prices or price > highest_prices[code]:
                highest_prices[code] = price
            dd = (price - highest_prices[code]) / highest_prices[code]
            if dd <= -TRAILING_STOP:
                stop_loss_codes.append(code)

        for code in stop_loss_codes:
            price = get_price_hfq(code, date)
            if price is None or price <= 0:
                continue
            shares = positions[code]
            amount = shares * price
            cost = amount * TXN_COST
            cash += amount - cost
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]
            logging.info(f" 止损 {date.strftime('%Y-%m-%d')} 止损 {code}: {shares}股 @ {price:.3f}")

    # 月度调仓
    should_rebalance = is_first_trading_of_month(date)

    if should_rebalance and i >= 80:
        candidates = []
        for code in etf_codes:
            mom, above_ma = calc_momentum_and_trend(code, date)
            if mom > -np.inf and above_ma and mom > 0:
                candidates.append((code, mom))

        candidates.sort(key=lambda x: x[1], reverse=True)
        target_risk = [c[0] for c in candidates[:TOP_N]]

        total_value = get_total_value(date)

        # 卖出所有现有风险资产
        for code in list(positions.keys()):
            if code == cash_etf:
                continue
            price = get_price_hfq(code, date)
            if price is None or price <= 0:
                continue
            shares = positions[code]
            amount = shares * price
            cost = amount * TXN_COST
            cash += amount - cost
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 调整国债ETF
        target_risk_value = total_value * RISK_RATIO
        target_cash_etf_value = total_value * (1 - RISK_RATIO) - total_value * 0.02

        if cash_etf in positions:
            price = get_price_hfq(cash_etf, date)
            if price is not None and price > 0:
                current_value = positions[cash_etf] * price
                if current_value > target_cash_etf_value:
                    excess_shares = int((current_value - target_cash_etf_value) / price / 100) * 100
                    if excess_shares > 0:
                        proceeds = excess_shares * price * (1 - TXN_COST)
                        cash += proceeds
                        positions[cash_etf] -= excess_shares
                        if positions[cash_etf] <= 0:
                            del positions[cash_etf]

        # 买入目标风险ETF
        if target_risk and cash > total_value * 0.05:
            per_etf_budget = target_risk_value / TOP_N
            for code in target_risk:
                price = get_price_hfq(code, date)
                if price is None or price <= 0:
                    continue
                max_shares = int(per_etf_budget / price / (1 + TXN_COST) / 100) * 100
                if max_shares <= 0:
                    continue
                cost = max_shares * price * (1 + TXN_COST)
                if cash < cost:
                    max_shares = int(cash / price / (1 + TXN_COST) / 100) * 100
                    if max_shares <= 0:
                        break
                    cost = max_shares * price * (1 + TXN_COST)
                positions[code] = max_shares
                cash -= cost
                highest_prices[code] = price

        # 剩余现金买入国债ETF
        remaining_cash = cash * 0.98
        if remaining_cash > 1000:
            price = get_price_hfq(cash_etf, date)
            if price is not None and price > 0:
                max_shares = int(remaining_cash / price / (1 + TXN_COST) / 100) * 100
                if max_shares > 0:
                    cost = max_shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = positions.get(cash_etf, 0) + max_shares
                        cash -= cost

        rebalance_count += 1
        pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
        logging.info(f"调仓#{rebalance_count} {date.strftime('%Y-%m-%d')}: 总={current_total:.0f}, 现金={cash:.0f}, {pos_detail}")

    nav_history.append({
        'date': date,
        'total_value': current_total,
        'cash': cash,
        'positions': len(positions),
    })

print(f"\n回测完成: 共 {rebalance_count} 次调仓, {len(nav_history)} 交易日")

results = pd.DataFrame(nav_history)
nav = results['total_value'].values
initial = nav[0]
final = nav[-1]
years = (dates[-1] - dates[0]).days / 365.25
total_return = final / initial - 1

daily_returns = pd.Series(nav).pct_change().dropna().values
annualized_return = (final / initial) ** (1 / years) - 1
annualized_vol = daily_returns.std() * np.sqrt(252)
sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

cummax = pd.Series(nav).cummax()
drawdown = (nav - cummax) / cummax
max_dd = drawdown.min()
win_rate = (daily_returns > 0).mean()

hs300_dates_list = sorted(dates)
hs300_prices = [etf_price_map.get('510300', {}).get(d) for d in hs300_dates_list]
hs300_nav = np.array([p for p in hs300_prices if p is not None], dtype=float)
if len(hs300_nav) > 1:
    hs300_total = hs300_nav[-1] / hs300_nav[0] - 1
    hs300_daily = pd.Series(hs300_nav).pct_change().dropna().values
    hs300_years = (hs300_dates_list[-1] - hs300_dates_list[0]).days / 365.25
    hs300_ann = (hs300_nav[-1] / hs300_nav[0]) ** (1 / hs300_years) - 1
    hs300_vol = hs300_daily.std() * np.sqrt(252)
    hs300_sharpe = hs300_ann / hs300_vol if hs300_vol > 0 else 0
    hs300_cummax = pd.Series(hs300_nav).cummax()
    hs300_dd = (hs300_nav - hs300_cummax) / hs300_cummax
    hs300_maxdd = hs300_dd.min()
else:
    hs300_total = hs300_ann = hs300_vol = hs300_sharpe = hs300_maxdd = 0

print("\n" + "=" * 70)
print(f"{'指标':<20} {'v3 动量策略':>15} {'沪深300':>15}")
print("=" * 70)
print(f"{'总资产 (最终)':<20} {final:>15,.0f} {hs300_nav[-1]*initial/hs300_nav[0]:>15,.0f}")
print(f"{'总收益':<20} {total_return*100:>14.2f}% {hs300_total*100:>14.2f}%")
print(f"{'年化收益':<20} {annualized_return*100:>14.2f}% {hs300_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {annualized_vol*100:>14.2f}% {hs300_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {hs300_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {hs300_maxdd*100:>14.2f}%")
print(f"{'日胜率':<20} {win_rate*100:>14.2f}%")
print("=" * 70)

results['year'] = results['date'].dt.year
results['daily_ret'] = results['total_value'].pct_change()

print(f"\n{'年份':<8} {'交易日':>7} {'年度收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'日胜率':>8}")
print("-" * 85)
for year in sorted(results['year'].unique()):
    sub = results[results['year'] == year].reset_index(drop=True)
    if len(sub) < 10:
        continue
    ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
    vol = sub['daily_ret'].std() * np.sqrt(252)
    sharpe_y = ret / vol if vol > 0 else 0
    cummax = sub['total_value'].cummax()
    dd = ((sub['total_value'] - cummax) / cummax).min()
    wr = (sub['daily_ret'] > 0).mean()
    print(f"{year:<8} {len(sub):>7} {ret*100:>9.2f}% {vol*100:>9.2f}% {sharpe_y:>8.3f} {dd*100:>9.2f}% {wr*100:>7.2f}%")

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v3.csv'), index=False)
