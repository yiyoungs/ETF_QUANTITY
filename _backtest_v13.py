"""v13: 终极版 - 核心+卫星 + 纯截面动量(无过滤)
================================================
策略:
  - 核心 50%: 510300 (固定持有，不择时)
  - 卫星 50%: Top3 60日动量最高的行业ETF（不做任何趋势过滤）
  - 15%止损
  - 月度调仓
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
STOP_LOSS = 0.15
CORE_RATIO = 0.50
TXN_COST = 0.001

cash = 1_000_000.0
positions = {}
highest_prices = {}
nav_history = []

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

print("v13: 核心+卫星 | 50% 510300 + 50% Top3 60日动量(无过滤) | 15%止损 | 月度")
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

        # 1. 排序所有行业ETF取Top3（不做任何过滤）
        candidates = []
        for code in sector_codes:
            mom = calc_momentum(code, date, MOM_WINDOW)
            if mom is not None:
                candidates.append((code, mom))

        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [c[0] for c in candidates[:TOP_N]]

        # 2. 卖出所有风险资产
        for code in list(positions.keys()):
            if code in [CASH_ETF, CORE_ETF]: continue
            p = get_price(code, date)
            if not p: continue
            cash += positions[code] * p * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 3. 调整核心仓位到 50%
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
                            if positions[CORE_ETF] <= 0:
                                del positions[CORE_ETF]
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

        # 4. 买入卫星ETF (即使动量为负也买入"相对强"的)
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

        # 5. 剩余现金 -> 国债ETF
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
            print(f"{date.strftime('%Y-%m-%d')}: 总={tv:.0f}, 现金={cash:.0f}, {pos_detail}")

    nav_history.append({'date': date, 'total_value': tv})

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

print(f"\n\n{'='*80}")
print(f"{' 策略对比 v13 (终极版) ':=^80}")
print(f"{'='*80}")
print(f"{'指标':<20} {'v13 核心+动量':>15} {'v10c 基准':>15} {'沪深300':>15}")
print("-" * 80)
print(f"{'总收益':<20} {total_ret*100:>14.2f}% {'28.98%':>15} {hs_total*100:>14.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>14.2f}% {'3.06%':>15} {hs_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>14.2f}% {'9.42%':>15} {hs_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {'0.325':>15} {hs_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {'-19.67%':>15} {hs_maxdd*100:>14.2f}%")
print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>15.3f} {'1.473':>15} {abs(hs_total/hs_maxdd):>15.3f}")
print(f"{'日胜率':<20} {win_rate*100:>14.2f}% {'53%':>15}")
print("=" * 80)

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

# 最终持仓
print(f"\n最终持仓:")
for code, shares in sorted(positions.items()):
    p = get_price(code, dates[-1]) or 0
    val = shares * p
    pct = val / final_nav * 100
    print(f"  {code}: {shares}股 @ {p:.3f} = {val:,.0f} ({pct:.1f}%)")
print(f"  现金: {cash:,.0f} ({cash/final_nav*100:.1f}%)")

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v13.csv'), index=False)
