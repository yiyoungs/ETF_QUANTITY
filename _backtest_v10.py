"""v10: 多时间框架动量策略
=====================================
核心思路: 不同MA周期代表不同趋势强度，短周期给更高仓位
- MA10 以上: 50% 仓位 (短期趋势启动，牛市初期信号)
- MA20 以上: 30% 仓位 (中期趋势确认)
- MA60 以上: 20% 仓位 (长期趋势确认)
- 满仓条件: 三层都通过 → 50%+30%+20% = 100% 权益
- 空仓条件: 三层都不通过 → 0% 权益，全仓国债

配置:
- 核心 50%: 510300 沪深300 (用多时间框架决定是否持有)
- 卫星 50%: Top3 动量行业ETF (同样用多时间框架过滤)
- 月度调仓 + 15% 止损
"""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from config import StrategyConfig, CACHE_DIR

# 加载数据
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

# ============ 多时间框架参数 ============
# 每层: (MA周期, 通过时的权益比例)
LAYERS = [
    (10, 0.50),   # 短期: 价格 > MA10 → 50% 权益
    (20, 0.30),   # 中期: 价格 > MA20 → 30% 权益
    (60, 0.20),   # 长期: 价格 > MA60 → 20% 权益
]
# 最大权益比例 = 0.50 + 0.30 + 0.20 = 1.00 (100%)
# 最小权益比例 = 0 (全部不通过)

MOM_WINDOW = 60      # 动量计算窗口
TOP_N = 3             # 卫星Top N
STOP_LOSS = 0.15      # 15% 止损
TXN_COST = 0.001      # 0.1% 手续费
CORE_BASE = 0.50      # 核心基础比例（在权益中的占比）
SAT_BASE = 0.50       # 卫星基础比例

cash = 1_000_000.0
positions = {}
highest_prices = {}
nav_history = []
trade_log = []
last_eq_ratio = 0.0  # 记录上一次调仓的权益比例

def get_price(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    match = df[df['date'] == date]['close']
    if len(match) == 0: return None
    return float(match.iloc[0])

def calc_equity_ratio(code, date):
    """多时间框架趋势判断: 返回权益目标比例"""
    df = etf_price_map.get(code)
    if df is None: return 0.0
    sub = df[df['date'] <= date]
    if len(sub) < 61:  # 至少需要MA60的数据
        return 0.0
    current_price = float(sub['close'].iloc[-1])
    ratio = 0.0
    for ma_window, weight in LAYERS:
        if len(sub) >= ma_window:
            ma = float(sub['close'].iloc[-ma_window:].mean())
            if current_price > ma:
                ratio += weight
    return min(ratio, 1.0)

def calc_momentum(code, date):
    """60日动量"""
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

print("v10: 多时间框架动量策略")
print(f"    MA10> → 50%权益, MA20> → +30%权益, MA60> → +20%权益")
print(f"    核心{CORE_BASE*100:.0f}% 510300 + 卫星{SAT_BASE*100:.0f}% Top{TOP_N}动量行业")
print(f"    止损{STOP_LOSS*100:.0f}% | 月度调仓")
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
                trade_log.append((date, 'STOP', code, shares, p))
                del positions[code]
                if code in highest_prices: del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= 70:
        tv = total_value(date)

        # 1) 多时间框架判断权益比例
        eq_ratio = calc_equity_ratio(CORE_ETF, date)
        equity_value = tv * eq_ratio
        cash_etf_value = tv * (1 - eq_ratio)

        # 2) 动量筛选卫星候选 (只在权益>0时才选)
        candidates = []
        if eq_ratio > 0:
            for code in sector_codes:
                mom = calc_momentum(code, date)
                if mom is not None and mom > 0:
                    # 卫星ETF也用多时间框架判断趋势
                    sat_eq = calc_equity_ratio(code, date)
                    if sat_eq > 0:
                        candidates.append((code, mom, sat_eq))

            # 综合排序: 动量分 + 趋势分
            candidates.sort(key=lambda x: x[1] * 0.6 + x[2] * 0.4, reverse=True)
            targets = [c[0] for c in candidates[:TOP_N]]
        else:
            targets = []

        # 3) 卖出所有风险资产
        for code in list(positions.keys()):
            if code == CASH_ETF: continue
            p = get_price(code, date)
            if p is None: continue
            shares = positions[code]
            cash += shares * p * (1 - TXN_COST)
            trade_log.append((date, 'SELL', code, shares, p))
            del positions[code]
            if code in highest_prices: del highest_prices[code]

        # 4) 卖出多余国债ETF
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

        # 5) 买入核心ETF
        if eq_ratio > 0 and cash > 1000:
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
                        trade_log.append((date, 'BUY-CORE', CORE_ETF, shares, p))

        # 6) 买入卫星ETF
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
                trade_log.append((date, 'BUY-SAT', code, shares, p))

        # 7) 剩余现金 -> 国债ETF
        if cash > 1000:
            p = get_price(CASH_ETF, date)
            if p and p > 0:
                shares = int(cash * 0.98 / p / 100) * 100
                if shares > 0:
                    cost = shares * p * (1 + TXN_COST)
                    if cash >= cost:
                        positions[CASH_ETF] = positions.get(CASH_ETF, 0) + shares
                        cash -= cost

        # 打印关键调仓信息
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

# 沪深300基准
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

print(f"\n\n{'指标':<20} {'v10 多时间框架':>18} {'v8 基准':>15} {'沪深300':>15}")
print("=" * 75)
print(f"{'总资产 (最终)':<20} {final_nav:>18,.0f} {'1,280,461':>15} {hs_clean[-1]*initial/hs_clean[0]:>15,.0f}")
print(f"{'总收益':<20} {total_ret*100:>17.2f}% {'28.05%':>15} {hs_total*100:>14.2f}%")
print(f"{'年化收益':<20} {ann_ret*100:>17.2f}% {'2.98%':>15} {hs_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {ann_vol*100:>17.2f}% {'10.00%':>15} {hs_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>18.3f} {'0.297':>15} {hs_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>17.2f}% {'-23.53%':>15} {hs_maxdd*100:>14.2f}%")
print(f"{'日胜率':<20} {win_rate*100:>17.2f}% {'48.95%':>15}")
print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>18.3f} {'1.192':>15} {abs(hs_total/hs_maxdd):>15.3f}")
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

# 打印权益比例变化（关键看牛市初期是否快速加仓）
print(f"\n权益比例变化 (关键时期):")
print(f"{'日期':<12} {'权益%':>8} {'说明'}")
print("-" * 50)
for idx in [0, 30, 60, 120, 180, 240, 300, 400, 500, 700, 900, 1200, 1500, 1800, 2000]:
    if idx < len(nav_history):
        row = nav_history[idx]
        eq = row.get('equity_ratio', np.nan)
        if not np.isnan(eq):
            date_str = pd.Timestamp(row['date']).strftime('%Y-%m-%d')
            print(f"{date_str:<12} {eq*100:>7.0f}%")

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v10.csv'), index=False)
