"""v13c: v13优化版 - 60%核心 + 放宽止损 + 牛熊自适应
=====================================================
改进:
  1. 核心提高到60% (v13是50%), 卫星40%
  2. 止损放宽到20% (v13是15%), 减少被反复止损
  3. 牛熊判断: MA20(快) + MA60(慢) 双确认
     - 牛市: 510300 > MA20 → 100%权益
     - 中性: MA20~MA60之间 → 70%权益
     - 熊市: 510300 < MA60 → 40%权益(60%债基)
  4. 卫星: Top3 纯截面动量(无过滤)
"""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from config import StrategyConfig, CACHE_DIR

etf_price_map = {}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        etf_price_map[code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])

CASH_ETF = StrategyConfig.CASH_ETF_CODE
CORE_ETF = '510300'
sector_codes = [c for c in etf_price_map.keys() if c not in [CASH_ETF, CORE_ETF]]

hs300_df = etf_price_map[CORE_ETF]
dates = sorted([d for d in hs300_df['date']
                if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

MOM_WINDOW = 60
TOP_N = 3
STOP_LOSS = 0.20       # 放宽到20%
TXN_COST = 0.001
CORE_RATIO = 0.60      # 60%核心
SAT_RATIO = 0.40        # 40%卫星

cash = 1_000_000.0
positions = {}
highest_prices = {}
nav_history = []
last_eq = 0.0

def get_price(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    m = df[df['date'] == date]['close']
    return float(m.iloc[0]) if len(m) > 0 else None

def calc_momentum(code, date, window):
    df = etf_price_map.get(code)
    if df is None: return None
    sub = df[df['date'] <= date]
    if len(sub) < window + 1: return None
    return float(sub['close'].iloc[-1]) / float(sub['close'].iloc[-window - 1]) - 1

def calc_equity_ratio(date):
    """牛熊判断: MA20 + MA60 双确认"""
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 61: return 0.40  # 数据不足默认熊市
    price = float(sub['close'].iloc[-1])
    ma20 = float(sub['close'].iloc[-20:].mean())
    ma60 = float(sub['close'].iloc[-60:].mean())
    if price > ma20:
        return 1.00   # 牛市: 满仓
    elif price > ma60:
        return 0.70   # 中性: 70%权益
    else:
        return 0.40   # 熊市: 40%权益(60%债基)

def total_value(date):
    total = cash
    for code, shares in positions.items():
        p = get_price(code, date)
        if p: total += shares * p
    return total

def is_first_trading(date):
    if date not in dates: return False
    idx = dates.index(date)
    return idx <= 0 or dates[idx - 1].month != date.month

print("v13c: 60%核心 + 牛熊自适应(MA20/MA60) + 20%止损")
print(f"    牛市(>MA20): 100%权益 | 中性(MA20~MA60): 70% | 熊市(<MA60): 40%权益")
print(f"    核心{CORE_RATIO*100:.0f}% 510300 + 卫星{SAT_RATIO*100:.0f}% Top{TOP_N} | 月度调仓")
print("=" * 80)

for i, date in enumerate(dates):
    tv = total_value(date)

    # 止损
    if i > 0 and positions:
        for code in list(positions.keys()):
            if code in [CASH_ETF, CORE_ETF]: continue
            p = get_price(code, date)
            if not p: continue
            if code not in highest_prices or p > highest_prices[code]:
                highest_prices[code] = p
            dd = (p - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
            if dd <= -STOP_LOSS:
                cash += positions[code] * p * (1 - TXN_COST)
                del positions[code]
                if code in highest_prices:
                    del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= 80:
        tv = total_value(date)

        # 1. 牛熊判断
        eq_ratio = calc_equity_ratio(date)
        equity_value = tv * eq_ratio
        bond_value = tv * (1 - eq_ratio)

        core_value = equity_value * CORE_RATIO
        sat_value = equity_value * SAT_RATIO

        # 2. 选股 (纯截面动量)
        candidates = []
        for code in sector_codes:
            mom = calc_momentum(code, date, MOM_WINDOW)
            if mom is not None:
                candidates.append((code, mom))
        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [c[0] for c in candidates[:TOP_N]]

        # 3. 卖出所有风险资产
        for code in list(positions.keys()):
            if code == CASH_ETF: continue
            p = get_price(code, date)
            if not p: continue
            cash += positions[code] * p * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 4. 调整国债ETF
        if CASH_ETF in positions:
            p = get_price(CASH_ETF, date)
            if p and p > 0:
                cur = positions[CASH_ETF] * p
                if cur > bond_value * 1.05:
                    excess = int((cur - bond_value) / p / 100) * 100
                    if excess > 0:
                        cash += excess * p * (1 - TXN_COST)
                        positions[CASH_ETF] -= excess
                        if positions[CASH_ETF] <= 0:
                            del positions[CASH_ETF]
                elif cur < bond_value * 0.90:
                    need = bond_value - cur
                    buy_shares = int(need / p / 100) * 100
                    if buy_shares > 0:
                        cost = buy_shares * p * (1 + TXN_COST)
                        if cash >= cost:
                            positions[CASH_ETF] = positions.get(CASH_ETF, 0) + buy_shares
                            cash -= cost
        elif bond_value > 1000:
            p = get_price(CASH_ETF, date)
            if p and p > 0:
                shares = int(bond_value * 0.98 / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CASH_ETF] = shares
                        cash -= cost

        # 5. 买入核心ETF
        if core_value > 1000 and cash > 1000:
            p = get_price(CORE_ETF, date)
            if p and p > 0:
                shares = int(core_value / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CORE_ETF] = shares
                        cash -= cost
                        highest_prices[CORE_ETF] = p

        # 6. 买入卫星ETF
        if targets and sat_value > 1000 and cash > 1000:
            per_budget = sat_value / len(targets)
            for code in targets:
                p = get_price(code, date)
                if not p or p <= 0: continue
                shares = int(per_budget / p / 100) * 100
                if shares <= 0: continue
                cost = shares * p * (1 + TXN_COST)
                if cash < cost:
                    shares = int(cash / p / 100) * 100
                    if shares <= 0: break
                    cost = shares * p * (1 + TXN_COST)
                positions[code] = shares
                cash -= cost
                highest_prices[code] = p

        # 7. 剩余现金 -> 国债
        if cash > 1000:
            p = get_price(CASH_ETF, date)
            if p and p > 0:
                shares = int(cash * 0.98 / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CASH_ETF] = positions.get(CASH_ETF, 0) + shares
                        cash -= cost

        if i < 200 or i % 400 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            state = "牛" if eq_ratio >= 1.0 else ("中" if eq_ratio >= 0.7 else "熊")
            print(f"{date.strftime('%Y-%m-%d')} [{state}] 权益{eq_ratio*100:.0f}%: 总={tv:.0f}, 现金={cash:.0f}, {pos_detail}")
        last_eq = eq_ratio

    nav_history.append({'date': date, 'total_value': tv, 'eq': last_eq})

# ============ 报告 ============
results = pd.DataFrame(nav_history)
nav = results['total_value'].values
initial = nav[0]; final_nav = nav[-1]
years = (dates[-1] - dates[0]).days / 365.25
total_ret = final_nav / initial - 1

daily_ret = pd.Series(nav).pct_change().dropna().values
ann_ret = (final_nav / initial) ** (1 / years) - 1
ann_vol = daily_ret.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

cummax = pd.Series(nav).cummax()
dd = (nav - cummax) / cummax
max_dd = dd.min()
win_rate = (daily_ret > 0).mean()

# 沪深300基准
hs300 = []
for d in dates:
    p = get_price('510300', d)
    if p: hs300.append(p)
hs_total = hs300[-1] / hs300[0] - 1
hs_daily = pd.Series(hs300).pct_change().dropna().values
hs_ann = (hs300[-1] / hs300[0]) ** (1 / years) - 1
hs_vol = hs_daily.std() * np.sqrt(252)
hs_sharpe = hs_ann / hs_vol if hs_vol > 0 else 0
hs_cummax = pd.Series(hs300).cummax()
hs_dd = (hs300 - hs_cummax) / hs_cummax
hs_maxdd = hs_dd.min()

print(f"\n\n{'='*90}")
print(f"{' v13c 回测报告 (60%核心 + 牛熊自适应 + 20%止损) ':=^90}")
print(f"{'='*90}")
print(f"{'指标':<20} {'v13c':>12} {'v13':>12} {'v13b':>12} {'v10c':>12} {'沪深300':>12}")
print("-" * 90)
print(f"{'总收益':<20} {total_ret*100:>11.2f}% {'69.23%':>12} {'2.29%':>12} {'28.98%':>12} {hs_total*100:>11.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>11.2f}% {'6.44%':>12} {'0.27%':>12} {'3.06%':>12} {hs_ann*100:>11.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>11.2f}% {'27.10%':>12} {'21.39%':>12} {'9.42%':>12} {hs_vol*100:>11.2f}%")
print(f"{'夏普比率':<20} {sharpe:>12.3f} {'0.238':>12} {'0.013':>12} {'0.325':>12} {hs_sharpe:>12.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>11.2f}% {'-48.06%':>12} {'-46.71%':>12} {'-19.67%':>12} {hs_maxdd*100:>11.2f}%")
print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>12.3f} {'1.441':>12} {'0.049':>12} {'1.473':>12} {abs(hs_total/hs_maxdd):>12.3f}")
print("=" * 90)

results['year'] = results['date'].dt.year
results['daily_ret'] = results['total_value'].pct_change()

print(f"\n{'年份':<8} {'交易日':>7} {'年度收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8}")
print("-" * 85)
for year in sorted(results['year'].unique()):
    sub = results[results['year'] == year].reset_index(drop=True)
    if len(sub) < 10: continue
    ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
    vol = sub['daily_ret'].std() * np.sqrt(252)
    s = ret / vol if vol > 0 else 0
    cm = sub['total_value'].cummax()
    mdd = ((sub['total_value'] - cm) / cm).min()
    wr = (sub['daily_ret'] > 0).mean()
    m = " ★" if ret > 0 else ""
    print(f"{year:<8} {len(sub):>7} {ret*100:>9.2f}% {vol*100:>9.2f}% {s:>8.3f} {mdd*100:>9.2f}% {wr*100:>7.2f}%{m}")

# 牛熊切换时间线
print(f"\n牛熊切换时间线:")
print(f"{'日期':<12} {'权益%':>8} {'状态':<6}")
print("-" * 30)
prev_state = None
for idx in range(0, len(nav_history)):
    row = nav_history[idx]
    eq = row['eq']
    state = "牛" if eq >= 1.0 else ("中" if eq >= 0.7 else "熊")
    if state != prev_state:
        date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
        print(f"{date_str:<12} {eq*100:>7.0f}% {state:<6}")
        prev_state = state

# 最终持仓
print(f"\n最终持仓:")
for code, shares in sorted(positions.items()):
    p = get_price(code, dates[-1]) or 0
    val = shares * p
    pct = val / final_nav * 100
    print(f"  {code}: {shares}股 @ {p:.3f} = {val:,.0f} ({pct:.1f}%)")
print(f"  现金: {cash:,.0f} ({cash/final_nav*100:.1f}%)")

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v13c.csv'), index=False)
