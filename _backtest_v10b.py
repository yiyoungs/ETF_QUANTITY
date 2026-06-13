"""v10b: 改进多时间框架 - 渐变权重 + 平滑过渡
==============================================
v10的问题: MA10太敏感，导致0%/100%频繁跳变
改进:
1. 用渐变而非阶跃: 价格偏离MA的程度决定权重
2. 增加底仓: 即使全部不通过也保留20%底仓
3. 短周期权重降低: MA10→30%, MA20→30%, MA60→40%
4. 核心+卫星结构保持

渐变逻辑:
  - 价格 > MA × (1 + buffer) → 满分
  - 价格在 MA 附近 → 部分分
  - 价格 < MA × (1 - buffer) → 0分
  buffer = 2%
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

# ============ 渐变多时间框架参数 ============
LAYERS = [
    (10, 0.30, 0.02),   # MA10: 30%权重, 2%缓冲带
    (20, 0.30, 0.02),   # MA20: 30%权重, 2%缓冲带
    (60, 0.40, 0.03),   # MA60: 40%权重, 3%缓冲带
]
FLOOR_RATIO = 0.20      # 底仓20% (即使全不通过也保留)
CEIL_RATIO = 1.00       # 最大100%

MOM_WINDOW = 60
TOP_N = 3
STOP_LOSS = 0.15
TXN_COST = 0.001
CORE_BASE = 0.50
SAT_BASE = 0.50

cash = 1_000_000.0
positions = {}
highest_prices = {}
nav_history = []
trade_log = []
last_eq_ratio = 0.0

def get_price(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    match = df[df['date'] == date]['close']
    if len(match) == 0: return None
    return float(match.iloc[0])

def calc_layer_score(price, ma, buffer):
    """渐变打分: 0~1之间"""
    if ma <= 0: return 0.0
    deviation = (price - ma) / ma
    if deviation > buffer:
        return 1.0
    elif deviation > -buffer:
        # 在缓冲带内: 线性插值
        return (deviation + buffer) / (2 * buffer)
    else:
        return 0.0

def calc_equity_ratio(code, date):
    """渐变多时间框架权益比例"""
    df = etf_price_map.get(code)
    if df is None: return FLOOR_RATIO
    sub = df[df['date'] <= date]
    if len(sub) < 61: return FLOOR_RATIO
    current_price = float(sub['close'].iloc[-1])
    score = 0.0
    for ma_window, weight, buffer in LAYERS:
        if len(sub) >= ma_window:
            ma = float(sub['close'].iloc[-ma_window:].mean())
            s = calc_layer_score(current_price, ma, buffer)
            score += weight * s
    # 映射到 [FLOOR_RATIO, CEIL_RATIO]
    ratio = FLOOR_RATIO + score * (CEIL_RATIO - FLOOR_RATIO)
    return min(max(ratio, FLOOR_RATIO), CEIL_RATIO)

def calc_momentum(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    sub = df[df['date'] <= date]
    if len(sub) < MOM_WINDOW + 1: return None
    t1 = float(sub['close'].iloc[-1])
    hist = float(sub['close'].iloc[-MOM_WINDOW - 1])
    return t1 / hist - 1

def total_value(date):
    total = cash
    for code, shares in positions.items():
        p = get_price(code, date)
        if p is not None: total += shares * p
    return total

def is_first_trading(date):
    if date not in dates: return False
    idx = dates.index(date)
    if idx <= 0: return True
    return dates[idx - 1].month != date.month

print("v10b: 渐变多时间框架动量策略")
print(f"    MA10→30% | MA20→30% | MA60→40% (渐变+缓冲带)")
print(f"    底仓{FLOOR_RATIO*100:.0f}% | 核心{CORE_BASE*100:.0f}% 510300 + 卫星{SAT_BASE*100:.0f}% Top{TOP_N}")
print("=" * 80)

for i, date in enumerate(dates):
    tv = total_value(date)

    # 止损
    if i > 0 and positions:
        for code in list(positions.keys()):
            if code in [CASH_ETF, CORE_ETF]: continue
            p = get_price(code, date)
            if p is None: continue
            if code not in highest_prices or p > highest_prices[code]:
                highest_prices[code] = p
            dd = (p - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
            if dd <= -STOP_LOSS:
                shares = positions[code]
                cash += shares * p * (1 - TXN_COST)
                del positions[code]
                if code in highest_prices: del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= 70:
        tv = total_value(date)

        eq_ratio = calc_equity_ratio(CORE_ETF, date)
        equity_value = tv * eq_ratio
        cash_etf_value = tv * (1 - eq_ratio)

        # 动量筛选
        candidates = []
        if eq_ratio > FLOOR_RATIO:
            for code in sector_codes:
                mom = calc_momentum(code, date)
                if mom is not None and mom > 0:
                    sat_score = calc_equity_ratio(code, date)
                    if sat_score > FLOOR_RATIO:
                        candidates.append((code, mom, sat_score))
            candidates.sort(key=lambda x: x[1] * 0.6 + x[2] * 0.4, reverse=True)
            targets = [c[0] for c in candidates[:TOP_N]]
        else:
            targets = []

        # 卖出所有风险资产
        for code in list(positions.keys()):
            if code == CASH_ETF: continue
            p = get_price(code, date)
            if p is None: continue
            shares = positions[code]
            cash += shares * p * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices: del highest_prices[code]

        # 卖出多余国债ETF
        if CASH_ETF in positions:
            p = get_price(CASH_ETF, date)
            if p and p > 0:
                cur_val = positions[CASH_ETF] * p
                if cur_val > cash_etf_value * 1.05:
                    excess = int((cur_val - cash_etf_value) / p / 100) * 100
                    if excess > 0:
                        cash += excess * p * (1 - TXN_COST)
                        positions[CASH_ETF] -= excess
                        if positions[CASH_ETF] <= 0: del positions[CASH_ETF]

        # 买入核心
        if cash > 1000:
            core_budget = equity_value * CORE_BASE
            p = get_price(CORE_ETF, date)
            if p and p > 0:
                shares = int(core_budget / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CORE_ETF] = shares
                        cash -= cost
                        highest_prices[CORE_ETF] = p

        # 买入卫星
        if targets and cash > 1000:
            sat_budget = equity_value * SAT_BASE / len(targets)
            for code in targets:
                p = get_price(code, date)
                if p is None or p <= 0: continue
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

        if i < 300 or i % 500 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"{date.strftime('%Y-%m-%d')} 权益{eq_ratio*100:.0f}%: 总={tv:.0f}, 现金={cash:.0f}, {pos_detail}")
        last_eq_ratio = eq_ratio

    nav_history.append({'date': date, 'total_value': tv, 'equity_ratio': last_eq_ratio})

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

print(f"\n\n{'指标':<20} {'v10b 渐变':>15} {'v8 基准':>15} {'沪深300':>15}")
print("=" * 75)
print(f"{'总收益':<20} {total_ret*100:>14.2f}% {'28.05%':>15} {hs_total*100:>14.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>14.2f}% {'2.98%':>15} {hs_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>14.2f}% {'10.00%':>15} {hs_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {'0.297':>15} {hs_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {'-23.53%':>15} {hs_maxdd*100:>14.2f}%")
print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>15.3f} {'1.192':>15} {abs(hs_total/hs_maxdd):>15.3f}")
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

# 权益比例变化
print(f"\n权益比例变化:")
for idx in [0, 30, 60, 120, 180, 240, 300, 400, 500, 700, 900, 1200, 1500, 1800, 2000]:
    if idx < len(nav_history):
        row = nav_history[idx]
        date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
        print(f"  {date_str}: 权益 {row['equity_ratio']*100:.0f}%")
