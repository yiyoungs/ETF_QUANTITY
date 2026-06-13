"""v12d: 极简双动量
=====================================
最简化版:
  - 动量窗口: 60日
  - 过滤: 只要求动量>0 (无MA过滤)
  - 持仓: Top 5 动量最高ETF
  - 止损: 15%
  - 调仓: 月度
  - 100%权益 (无固定核心)
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
sector_codes = [c for c in etf_price_map.keys() if c != CASH_ETF]

hs300_df = etf_price_map['510300']
dates = sorted([d for d in hs300_df['date']
                if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

MOM_WINDOW = 60
TOP_N = 5
STOP_LOSS = 0.15
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

print(f"v12d: 极简双动量 | 60日动量>0 | Top{TOP_N} | {STOP_LOSS*100:.0f}%止损 | 月度")
print("=" * 80)

for i, date in enumerate(dates):
    tv = total_value(date)

    # 止损
    if i > 0 and positions:
        for code in list(positions.keys()):
            if code == CASH_ETF: continue
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

        # 只筛选动量>0，不加MA过滤
        candidates = []
        for code in sector_codes:
            mom = calc_momentum(code, date, MOM_WINDOW)
            if mom is not None and mom > 0:
                candidates.append((code, mom))

        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [c[0] for c in candidates[:TOP_N]]

        # 卖出所有风险资产
        for code in list(positions.keys()):
            if code == CASH_ETF: continue
            p = get_price(code, date)
            if not p: continue
            cash += positions[code] * p * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 买入目标
        if targets and cash > 1000:
            per_budget = tv * 1.0 / len(targets)
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

        if i < 300 or i % 500 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"{date.strftime('%Y-%m-%d')}: 总={tv:.0f}, 现金={cash:.0f}, 目标={len(targets)}, {pos_detail}")

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

print(f"\n\n{'指标':<20} {'v12d 极简':>15} {'v10c':>15} {'沪深300':>15}")
print("=" * 80)
print(f"{'总收益':<20} {total_ret*100:>14.2f}% {'28.98%':>15} {hs_total*100:>14.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>14.2f}% {'3.06%':>15} {hs_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>14.2f}% {'9.42%':>15} {hs_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {'0.325':>15} {hs_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {'-19.67%':>15} {hs_maxdd*100:>14.2f}%")
print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>15.3f} {'1.473':>15} {abs(hs_total/hs_maxdd):>15.3f}")
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

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v12d.csv'), index=False)
