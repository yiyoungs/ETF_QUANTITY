"""v9: 宏观择时增强版
- 核心: 510300 (固定50%)
- 卫星: Top N 动量行业ETF (动态)
- 宏观过滤: 510300 > MA200时50%卫星，<MA200时20%卫星
- 动量: 60日 > 0，趋势: 价格 > MA60
- 月度调仓 + 15%止损
"""
import pandas as pd
import numpy as np
import os

from config import StrategyConfig, CACHE_DIR

etf_price_map = {}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        etf_price_map[code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])

cash_etf = StrategyConfig.CASH_ETF_CODE
core_etf = '510300'
sector_codes = [c for c in etf_price_map.keys() if c not in [cash_etf, core_etf]]

hs300_df = etf_price_map[core_etf]
dates = sorted([d for d in hs300_df['date'] if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

MOM_WINDOW = 60
MA_TREND = 60
TOP_N = 3
STOP_LOSS = 0.15
TXN_COST = 0.001
CORE_RATIO = 0.50
SAT_BULL = 0.50   # 牛市50%卫星
SAT_BEAR = 0.20   # 熊市20%卫星（剩余30%国债）

cash = float(StrategyConfig.INITIAL_CAPITAL)
positions = {}
highest_prices = {}
nav_history = []

def get_current_price(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    p = df[df['date'] == date]['close']
    if len(p) == 0: return None
    return float(p.iloc[0])

def get_total(date):
    total = cash
    for code, shares in positions.items():
        price = get_current_price(code, date)
        if price is not None: total += shares * price
    return total

def is_first_trading(date):
    if date not in dates: return False
    idx = dates.index(date)
    if idx <= 0: return True
    return dates[idx - 1].month != date.month

print("v9: 50% 510300 + 动态卫星(牛市50%/熊市20%) | MA60过滤 | 月度调仓")
print("=" * 80)

for i, date in enumerate(dates):
    current_total = get_total(date)

    if i > 0 and positions:
        for code in list(positions.keys()):
            if code in [cash_etf, core_etf]: continue
            price = get_current_price(code, date)
            if price is None: continue
            if code not in highest_prices or price > highest_prices[code]:
                highest_prices[code] = price
            dd = (price - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
            if dd <= -STOP_LOSS:
                shares = positions[code]
                cash += shares * price * (1 - TXN_COST)
                del positions[code]
                if code in highest_prices:
                    del highest_prices[code]

    if is_first_trading(date) and i >= MOM_WINDOW + 10:
        # 宏观择时判断
        core_sub = hs300_df[hs300_df['date'] <= date]
        sat_ratio = SAT_BULL
        if len(core_sub) >= 200:
            ma200 = float(core_sub['close'].iloc[-200:].mean())
            core_price = float(core_sub['close'].iloc[-1])
            if core_price < ma200:
                sat_ratio = SAT_BEAR

        # 动量筛选
        candidates = []
        for code in sector_codes:
            df = etf_price_map.get(code)
            if df is None: continue
            sub = df[df['date'] <= date]
            if len(sub) < MOM_WINDOW + 1: continue
            t1 = float(sub['close'].iloc[-1])
            hist = float(sub['close'].iloc[-MOM_WINDOW - 1])
            ma60 = float(sub['close'].iloc[-MA_TREND:].mean())
            mom = t1 / hist - 1
            if t1 > ma60 and mom > 0:
                candidates.append((code, mom))

        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [c[0] for c in candidates[:TOP_N]]
        total_value = get_total(date)

        # 卖出所有卫星持仓
        for code in list(positions.keys()):
            if code == cash_etf or code == core_etf: continue
            price = get_current_price(code, date)
            if price is None: continue
            shares = positions[code]
            cash += shares * price * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices: del highest_prices[code]

        # 调整核心到 50%
        core_target = total_value * CORE_RATIO
        if core_etf in positions:
            price = get_current_price(core_etf, date)
            if price and price > 0:
                current_value = positions[core_etf] * price
                diff = current_value - core_target
                if abs(diff) / core_target > 0.10:
                    if diff > 0:
                        shares_sell = int(diff / price / 100) * 100
                        if shares_sell > 0:
                            cash += shares_sell * price * (1 - TXN_COST)
                            positions[core_etf] -= shares_sell
                            if positions[core_etf] <= 0: del positions[core_etf]
                    else:
                        buy_value = -diff
                        shares_buy = int(buy_value / price / 100) * 100
                        if shares_buy > 0:
                            cost = shares_buy * price * (1 + TXN_COST)
                            if cash >= cost:
                                positions[core_etf] += shares_buy
                                cash -= cost
        else:
            price = get_current_price(core_etf, date)
            if price and price > 0:
                shares = int(core_target / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[core_etf] = shares
                        cash -= cost

        # 卖出多余国债ETF
        sat_target = total_value * sat_ratio
        cash_etf_target = total_value * (1 - CORE_RATIO - sat_ratio)
        if cash_etf in positions:
            price = get_current_price(cash_etf, date)
            if price and price > 0:
                current_value = positions[cash_etf] * price
                if current_value > cash_etf_target * 1.05:
                    excess_shares = int((current_value - cash_etf_target) / price / 100) * 100
                    if excess_shares > 0:
                        cash += excess_shares * price * (1 - TXN_COST)
                        positions[cash_etf] -= excess_shares
                        if positions[cash_etf] <= 0: del positions[cash_etf]

        # 买入卫星ETF
        if targets and cash > 1000:
            per_budget = sat_target / len(targets)
            for code in targets:
                price = get_current_price(code, date)
                if price is None or price <= 0: continue
                shares = int(per_budget / price / 100) * 100
                if shares <= 0: continue
                cost = shares * price * (1 + TXN_COST)
                if cash < cost:
                    shares = int(cash / price / 100) * 100
                    if shares <= 0: break
                    cost = shares * price * (1 + TXN_COST)
                positions[code] = shares
                cash -= cost
                highest_prices[code] = price

        # 剩余现金 -> 国债ETF
        if cash > 1000:
            price = get_current_price(cash_etf, date)
            if price and price > 0:
                shares = int(cash * 0.98 / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = positions.get(cash_etf, 0) + shares
                        cash -= cost

    nav_history.append({'date': date, 'total_value': current_total})

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

hs300_prices = [float(hs300_df[hs300_df['date'] == d]['close'].iloc[0]) if len(hs300_df[hs300_df['date'] == d]) > 0 else None for d in dates]
hs300_nav_clean = [p for p in hs300_prices if p is not None]
hs300_total = hs300_nav_clean[-1] / hs300_nav_clean[0] - 1
hs300_daily = pd.Series(hs300_nav_clean).pct_change().dropna().values
hs300_years = (dates[-1] - dates[0]).days / 365.25
hs300_ann = (hs300_nav_clean[-1] / hs300_nav_clean[0]) ** (1 / hs300_years) - 1
hs300_vol = hs300_daily.std() * np.sqrt(252)
hs300_sharpe = hs300_ann / hs300_vol if hs300_vol > 0 else 0
hs300_cummax = pd.Series(hs300_nav_clean).cummax()
hs300_dd = (hs300_nav_clean - hs300_cummax) / hs300_cummax
hs300_maxdd = hs300_dd.min()

print(f"\n\n{'指标':<20} {'v9 宏观择时':>15} {'沪深300':>15}")
print("=" * 70)
print(f"{'总资产 (最终)':<20} {final:>15,.0f} {hs300_nav_clean[-1]*initial/hs300_nav_clean[0]:>15,.0f}")
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
    if len(sub) < 10: continue
    ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
    vol = sub['daily_ret'].std() * np.sqrt(252)
    sharpe_y = ret / vol if vol > 0 else 0
    cummax = sub['total_value'].cummax()
    dd = ((sub['total_value'] - cummax) / cummax).min()
    wr = (sub['daily_ret'] > 0).mean()
    print(f"{year:<8} {len(sub):>7} {ret*100:>9.2f}% {vol*100:>9.2f}% {sharpe_y:>8.3f} {dd*100:>9.2f}% {wr*100:>7.2f}%")
