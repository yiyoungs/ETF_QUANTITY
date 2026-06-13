"""v11b: 100%择时 + 修正权益比例逻辑
=====================================
v11问题: 权益比例在0/30/50/80/100频繁跳变，MA10太敏感
修正:
  - 权益比例 = 最高满足条件的MA层的权重（非累加）
  - MA60>: 100% 权益（满仓）
  - MA20> only: 50% 权益
  - MA10> only: 30% 权益
  - MA10以下: 0% 权益（全仓国债）
  这样只有三个状态: 0%, 50%, 100%

选股: Top 5 | >MA20 且 60日动量>0
止损: 15%
调仓: 月度
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
hs300_df = etf_price_map['510300']
sector_codes = [c for c in etf_price_map.keys() if c != CASH_ETF]

dates = sorted([d for d in hs300_df['date']
                if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

# 权益比例: 非累加，取最高满足条件的层的权重
# MA60>: 100%, MA20>: 50%, MA10>: 30%, else: 0%
LAYERS = [
    (60, 1.00),  # MA60 以上 → 满仓
    (20, 0.50),  # MA20 以上 → 半仓
    (10, 0.30),  # MA10 以上 → 30%
]

TOP_N = 5
MOM_WINDOW = 60
SAT_MA_MIN = 20
STOP_LOSS = 0.15
TXN_COST = 0.001

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

def calc_equity_ratio(date):
    """非累加权益比例: 取最高满足条件的层的权重"""
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 11: return 0.0
    price = float(sub['close'].iloc[-1])
    ratio = 0.0
    for ma_win, weight in LAYERS:
        if len(sub) >= ma_win:
            ma = float(sub['close'].iloc[-ma_win:].mean())
            if price > ma:
                ratio = weight  # 非累加: 取当前最高层的权重
                break
    return ratio

def is_trend_up(code, date, ma_win):
    df = etf_price_map.get(code)
    if df is None: return False
    sub = df[df['date'] <= date]
    if len(sub) < ma_win: return False
    return float(sub['close'].iloc[-1]) > float(sub['close'].iloc[-ma_win:].mean())

def calc_momentum(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    sub = df[df['date'] <= date]
    if len(sub) < MOM_WINDOW + 1: return None
    t1 = float(sub['close'].iloc[-1])
    return t1 / float(sub['close'].iloc[-MOM_WINDOW - 1]) - 1

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

print("v11b: 100%择时 | MA60>→100%, MA20>→50%, MA10>→30% | 非累加")
print(f"    Top{TOP_N} | >MA20 | 60日动量>0 | 止损{STOP_LOSS*100:.0f}%")
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
                if code in highest_prices: del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= 80:
        tv = total_value(date)
        eq_ratio = calc_equity_ratio(date)
        equity_value = tv * eq_ratio
        bond_value = tv * (1 - eq_ratio)

        # 选股
        candidates = []
        for code in sector_codes:
            mom = calc_momentum(code, date)
            if mom is None or mom <= 0: continue
            if not is_trend_up(code, date, SAT_MA_MIN): continue
            # 多MA趋势评分
            score = 0.0
            for ma_w, w in [(10, 1.0), (20, 0.5), (60, 0.3)]:
                if is_trend_up(code, date, ma_w):
                    score += w
            candidates.append((code, mom + 0.01 * score))
        candidates.sort(key=lambda x: x[1], reverse=True)
        targets = [c[0] for c in candidates[:TOP_N]]

        # 卖出风险资产
        for code in list(positions.keys()):
            if code == CASH_ETF: continue
            p = get_price(code, date)
            if not p: continue
            cash += positions[code] * p * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices: del highest_prices[code]

        # 调整国债ETF
        if CASH_ETF in positions:
            p = get_price(CASH_ETF, date)
            if p:
                cur = positions[CASH_ETF] * p
                if cur > bond_value * 1.05:
                    excess = int((cur - bond_value) / p / 100) * 100
                    if excess > 0:
                        cash += excess * p * (1 - TXN_COST)
                        positions[CASH_ETF] -= excess
                        if positions[CASH_ETF] <= 0: del positions[CASH_ETF]
        elif bond_value > 1000:
            p = get_price(CASH_ETF, date)
            if p:
                shares = int(bond_value * 0.98 / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CASH_ETF] = shares
                        cash -= cost

        # 买入目标ETF
        if targets and cash > 1000:
            per_budget = equity_value / len(targets)
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
            if p:
                shares = int(cash * 0.98 / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CASH_ETF] = positions.get(CASH_ETF, 0) + shares
                        cash -= cost

        if i < 200 or i % 400 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"{date.strftime('%Y-%m-%d')} 权益{eq_ratio*100:.0f}%: 总={tv:.0f}, 现金={cash:.0f}, {pos_detail}")
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

print(f"\n\n{'指标':<20} {'v11b 非累加':>15} {'v11':>15} {'v10c':>15} {'沪深300':>15}")
print("=" * 85)
print(f"{'总收益':<20} {total_ret*100:>14.2f}% {'-41.33%':>15} {'28.98%':>15} {hs_total*100:>14.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>14.2f}% {'-6.13%':>15} {'3.06%':>15} {hs_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>14.2f}% {'22.36%':>15} {'9.42%':>15} {hs_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {'-0.274':>15} {'0.325':>15} {hs_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {'-55.88%':>15} {'-19.67%':>15} {hs_maxdd*100:>14.2f}%")
print("=" * 85)

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

# 权益比例
print(f"\n权益比例变化:")
for idx in range(0, len(nav_history), max(1, len(nav_history) // 15)):
    row = nav_history[idx]
    date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
    labels = {0.0: '空仓', 0.3: '30%', 0.5: '50%', 1.0: '满仓'}
    label = labels.get(row['eq'], f"{row['eq']*100:.0f}%")
    print(f"  {date_str}: {label}")

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v11b.csv'), index=False)
