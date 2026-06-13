"""v10d: v10c激进版 - 卫星用MA10+20日动量
=====================================
在v10c基础上改进:
- 卫星ETF: 必须 > MA10 (v10c用的是>MA20)
- 动量窗口: 20日 (v10c用的是60日)
- 评分: 0.4*多MA趋势分 + 0.6*20日动量分
这样卫星能更快捕捉牛市启动

核心: 50% 510300 (不变)
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

# v10d: 激进版卫星
MOM_WINDOW = 20          # 20日动量 (vs v10c的60日)
SAT_MA_MIN = 10         # 必须>MA10 (vs v10c的MA20)
TOP_N = 3
STOP_LOSS = 0.15
TXN_COST = 0.001
CORE_RATIO = 0.50

cash = 1_000_000.0
positions = {}
highest_prices = {}
nav_history = []

def get_price(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    m = df[df['date'] == date]['close']
    return float(m.iloc[0]) if len(m) > 0 else None

def calc_satellite_score(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    sub = df[df['date'] <= date]
    if len(sub) < max(MOM_WINDOW + 1, SAT_MA_MIN + 1): return None

    price = float(sub['close'].iloc[-1])

    # 必须 > MA10
    ma10 = float(sub['close'].iloc[-SAT_MA_MIN:].mean())
    if price <= ma10: return None

    # 20日动量
    hist = float(sub['close'].iloc[-MOM_WINDOW - 1])
    momentum = price / hist - 1
    if momentum <= 0: return None

    # 多MA趋势分
    trend_score = 0.0
    for ma_w, w in [(10, 1.0), (20, 0.5), (60, 0.3)]:
        if len(sub) >= ma_w:
            ma = float(sub['close'].iloc[-ma_w:].mean())
            if price > ma: trend_score += w

    return 0.4 * trend_score + 0.6 * momentum

def total_value(date):
    total = cash
    for code, shares in positions.items():
        p = get_price(code, date)
        if p: total += shares * p
    return total

def is_first_trading(date):
    if date not in dates: return False
    idx = dates.index(date)
    return idx <= 0 or dates[idx-1].month != date.month

print("v10d: v10c激进版 | 卫星>MA10 + 20日动量 (vs v10c>MA20+60日)")
print(f"    核心50% + 卫星50% | Top{TOP_N} | >MA10 | 20日动量>0 | 止损{STOP_LOSS*100:.0f}%")
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
                if code in highest_prices: del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= 30:
        tv = total_value(date)

        # 选卫星
        candidates = []
        for code in sector_codes:
            score = calc_satellite_score(code, date)
            if score is not None:
                candidates.append((code, score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [c[0] for c in candidates[:TOP_N]]

        # 卖出卫星
        for code in list(positions.keys()):
            if code == CASH_ETF or code == CORE_ETF: continue
            p = get_price(code, date)
            if not p: continue
            cash += positions[code] * p * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices: del highest_prices[code]

        # 调整核心
        core_target = tv * CORE_RATIO
        if CORE_ETF in positions:
            p = get_price(CORE_ETF, date)
            if p and p > 0:
                cur = positions[CORE_ETF] * p
                diff = cur - core_target
                if abs(diff) / core_target > 0.10:
                    if diff > 0:
                        shares = int(diff / p / 100) * 100
                        if shares > 0:
                            cash += shares * p * (1 - TXN_COST)
                            positions[CORE_ETF] -= shares
                            if positions[CORE_ETF] <= 0: del positions[CORE_ETF]
                    else:
                        shares = int(-diff / p / 100) * 100
                        if shares > 0:
                            cost = shares * p * (1 + TXN_COST)
                            if cash >= cost:
                                positions[CORE_ETF] += shares
                                cash -= cost
        else:
            p = get_price(CORE_ETF, date)
            if p and p > 0:
                shares = int(core_target / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CORE_ETF] = shares
                        cash -= cost

        # 买入卫星
        if targets and cash > 1000:
            sat_budget = tv * (1 - CORE_RATIO) / len(targets)
            for code in targets:
                p = get_price(code, date)
                if not p or p <= 0: continue
                shares = int(sat_budget / p / 100) * 100
                if shares <= 0: continue
                cost = shares * p * (1 + TXN_COST)
                if cash < cost:
                    shares = int(cash / p / 100) * 100
                    if shares <= 0: break
                    cost = shares * p * (1 + TXN_COST)
                positions[code] = shares
                cash -= cost
                highest_prices[code] = p

        # 剩余现金 -> 国债
        if cash > 1000:
            p = get_price(CASH_ETF, date)
            if p and p > 0:
                shares = int(cash * 0.98 / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CASH_ETF] = positions.get(CASH_ETF, 0) + shares
                        cash -= cost

    nav_history.append({'date': date, 'total_value': tv})

# 报告
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

hs300 = [float(hs300_df[hs300_df['date'] == d]['close'].iloc[0])
         if len(hs300_df[hs300_df['date'] == d]) > 0 else None for d in dates]
hs_clean = [p for p in hs300 if p is not None]
hs_total = hs_clean[-1] / hs_clean[0] - 1
hs_daily = pd.Series(hs_clean).pct_change().dropna().values
hs_ann = (hs_clean[-1] / hs_clean[0]) ** (1 / years) - 1
hs_vol = hs_daily.std() * np.sqrt(252)
hs_sharpe = hs_ann / hs_vol if hs_vol > 0 else 0
hs_cummax = pd.Series(hs_clean).cummax()
hs_dd = (hs_clean - hs_cummax) / hs_cummax
hs_maxdd = hs_dd.min()

print(f"\n\n{'指标':<20} {'v10d 激进':>14} {'v10c':>15} {'沪深300':>15}")
print("=" * 75)
print(f"{'总收益':<20} {total_ret*100:>13.2f}% {'28.98%':>15} {hs_total*100:>14.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>13.2f}% {'3.06%':>15} {hs_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>13.2f}% {'9.42%':>15} {hs_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>14.3f} {'0.325':>15} {hs_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>13.2f}% {'-19.67%':>15} {hs_maxdd*100:>14.2f}%")
print("=" * 75)

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

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v10d.csv'), index=False)
