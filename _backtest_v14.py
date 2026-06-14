"""v14: 多种择时策略对比
=====================================
对比以下择时方案 (均基于v13架构: 50%核心 + 50%卫星):
  A. 无择时 (v13基准)
  B. MA200 二值 (v13d)
  C. 趋势偏离度平滑: equity = 0.4 + 0.6 * clamp((price/MA200-1)*10, 0, 1)
  D. 波动率目标: equity = clamp(target_vol / realized_vol_20d, 0.4, 1.0)
  E. 指数动量: equity = 0.4 + 0.6 * clamp((250日收益+5%)/20%, 0, 1)
  F. 多信号投票: MA200(40%) + 60日动量(30%) + MA60(30%)
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
TXN_COST = 0.001
CORE_IN_EQ = 0.50
SAT_IN_EQ = 0.50
EQ_MIN = 0.40
EQ_MAX = 1.00

# ============ 择时函数 ============

def timing_none(date):
    """A: 无择时"""
    return 1.00

def timing_ma200_binary(date):
    """B: MA200 二值"""
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 201: return EQ_MIN
    price = float(sub['close'].iloc[-1])
    ma200 = float(sub['close'].iloc[-200:].mean())
    return EQ_MAX if price > ma200 else EQ_MIN

def timing_trend_deviation(date):
    """C: 趋势偏离度平滑
    price/MA200 = 1.0 → equity = 0.4
    price/MA200 = 1.10 → equity = 1.0
    中间线性插值
    """
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 201: return EQ_MIN
    price = float(sub['close'].iloc[-1])
    ma200 = float(sub['close'].iloc[-200:].mean())
    deviation = (price / ma200 - 1.0) * 10  # 放大10倍
    equity = EQ_MIN + (EQ_MAX - EQ_MIN) * np.clip(deviation, 0, 1)
    return equity

def timing_vol_target(date, target_vol=0.15):
    """D: 波动率目标
    当市场波动率低时加仓, 高时减仓
    """
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 25: return EQ_MIN
    returns = sub['close'].pct_change().iloc[-20:]
    realized_vol = returns.std() * np.sqrt(252)
    if realized_vol < 0.01: realized_vol = 0.01
    equity = np.clip(target_vol / realized_vol, EQ_MIN, EQ_MAX)
    return equity

def timing_index_momentum(date):
    """E: 指数250日动量
    250日收益 > +5% → 100%权益
    250日收益 < -15% → 40%权益
    中间线性
    """
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 251: return EQ_MIN
    mom_250 = float(sub['close'].iloc[-1]) / float(sub['close'].iloc[-250]) - 1
    # -15% → 0.4, +5% → 1.0, range = 20%
    equity = EQ_MIN + (EQ_MAX - EQ_MIN) * np.clip((mom_250 + 0.15) / 0.20, 0, 1)
    return equity

def timing_multi_signal(date):
    """F: 多信号投票
    MA200信号(40%权重) + 60日动量信号(30%) + MA60信号(30%)
    每个信号输出0~1, 加权平均
    """
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 201: return EQ_MIN
    price = float(sub['close'].iloc[-1])

    # 信号1: MA200偏离度 (0~1)
    ma200 = float(sub['close'].iloc[-200:].mean())
    s1 = np.clip((price / ma200 - 0.95) / 0.15, 0, 1)

    # 信号2: 60日动量 (0~1)
    if len(sub) >= 61:
        mom60 = price / float(sub['close'].iloc[-60]) - 1
        s2 = np.clip((mom60 + 0.10) / 0.20, 0, 1)
    else:
        s2 = 0.0

    # 信号3: MA60偏离度 (0~1)
    ma60 = float(sub['close'].iloc[-60:].mean())
    s3 = np.clip((price / ma60 - 0.95) / 0.15, 0, 1)

    combined = 0.4 * s1 + 0.3 * s2 + 0.3 * s3
    return EQ_MIN + (EQ_MAX - EQ_MIN) * np.clip(combined, 0, 1)

TIMING_FUNCS = {
    'A.无择时':       timing_none,
    'B.MA200二值':    timing_ma200_binary,
    'C.趋势偏离度':   timing_trend_deviation,
    'D.波动率目标':   timing_vol_target,
    'E.指数动量':     timing_index_momentum,
    'F.多信号投票':   timing_multi_signal,
}

# ============ 回测引擎 ============

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

def is_first_trading(date):
    if date not in dates: return False
    idx = dates.index(date)
    return idx <= 0 or dates[idx - 1].month != date.month

def run_backtest(timing_func, timing_name):
    cash = 1_000_000.0
    positions = {}
    highest_prices = {}
    nav_history = []

    for i, date in enumerate(dates):
        tv = cash
        for code, shares in positions.items():
            p = get_price(code, date)
            if p: tv += shares * p

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
        if is_first_trading(date) and i >= 210:
            tv = cash
            for code, shares in positions.items():
                p = get_price(code, date)
                if p: tv += shares * p

            eq_ratio = timing_func(date)
            equity_value = tv * eq_ratio
            bond_value = tv * (1 - eq_ratio)
            core_value = equity_value * CORE_IN_EQ
            sat_value = equity_value * SAT_IN_EQ

            # 选股
            candidates = []
            for code in sector_codes:
                mom = calc_momentum(code, date, MOM_WINDOW)
                if mom is not None:
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

            # 调整国债ETF
            if CASH_ETF in positions:
                p = get_price(CASH_ETF, date)
                if p and p > 0:
                    cur = positions[CASH_ETF] * p
                    if cur > bond_value * 1.08:
                        excess = int((cur - bond_value) / p / 100) * 100
                        if excess > 0:
                            cash += excess * p * (1 - TXN_COST)
                            positions[CASH_ETF] -= excess
                            if positions[CASH_ETF] <= 0:
                                del positions[CASH_ETF]
                    elif cur < bond_value * 0.88:
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

            # 买入核心ETF
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

            # 买入卫星ETF
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

        # 记录净值
        tv = cash
        for code, shares in positions.items():
            p = get_price(code, date)
            if p: tv += shares * p
        nav_history.append(tv)

    return nav_history

# ============ 运行所有策略 ============
print("v14: 6种择时策略全面对比")
print("=" * 100)

all_results = {}
for name, func in TIMING_FUNCS.items():
    print(f"  回测中: {name}...", end=" ", flush=True)
    navs = run_backtest(func, name)
    all_results[name] = navs
    nav = np.array(navs)
    years = (dates[-1] - dates[0]).days / 365.25
    total_ret = nav[-1] / nav[0] - 1
    ann_ret = (nav[-1] / nav[0]) ** (1 / years) - 1
    daily_r = np.diff(nav) / nav[:-1]
    ann_vol = daily_r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = np.maximum.accumulate(nav)
    dd = (nav - cummax) / cummax
    max_dd = dd.min()
    wr = (daily_r > 0).mean()
    print(f"收益{total_ret*100:>7.2f}% 回撤{max_dd*100:>7.2f}% 夏普{sharpe:.3f}")

# 沪深300基准
hs300 = []
for d in dates:
    p = get_price('510300', d)
    if p: hs300.append(p)
hs = np.array(hs300)
years = (dates[-1] - dates[0]).days / 365.25
hs_ret = hs[-1] / hs[0] - 1
hs_ann = (hs[-1] / hs[0]) ** (1 / years) - 1
hs_vol = np.diff(hs) / hs[:-1]
hs_annvol = hs_vol.std() * np.sqrt(252)
hs_sharpe = hs_ann / hs_annvol if hs_annvol > 0 else 0
hs_cummax = np.maximum.accumulate(hs)
hs_maxdd = ((hs - hs_cummax) / hs_cummax).min()

# ============ 对比表 ============
print(f"\n\n{'='*110}")
print(f"{' 6种择时策略全面对比 (2018-2026) ':=^110}")
print(f"{'='*110}")
print(f"{'策略':<16} {'总收益':>10} {'年化':>8} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'收益/回撤':>10} {'日胜率':>8}")
print("-" * 110)

# 按收益/回撤比排序
ranked = []
for name, navs in all_results.items():
    nav = np.array(navs)
    total_ret = nav[-1] / nav[0] - 1
    ann_ret = (nav[-1] / nav[0]) ** (1 / years) - 1
    daily_r = np.diff(nav) / nav[:-1]
    ann_vol = daily_r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = np.maximum.accumulate(nav)
    dd = (nav - cummax) / cummax
    max_dd = dd.min()
    wr = (daily_r > 0).mean()
    ratio = abs(total_ret / max_dd) if max_dd != 0 else 0
    ranked.append((name, total_ret, ann_ret, ann_vol, sharpe, max_dd, ratio, wr))

ranked.sort(key=lambda x: x[6], reverse=True)

for name, tr, ar, av, sp, md, ratio, wr in ranked:
    print(f"{name:<16} {tr*100:>9.2f}% {ar*100:>7.2f}% {av*100:>9.2f}% {sp:>8.3f} {md*100:>9.2f}% {ratio:>10.3f} {wr*100:>7.2f}%")

print("-" * 110)
print(f"{'沪深300':<16} {hs_ret*100:>9.2f}% {hs_ann*100:>7.2f}% {hs_annvol*100:>9.2f}% {hs_sharpe:>8.3f} {hs_maxdd*100:>9.2f}% {abs(hs_ret/hs_maxdd):>10.3f}")
print("=" * 110)

# ============ 分年度对比 (只显示前3名) ============
print(f"\n分年度对比 (Top3策略):")
print(f"{'年份':<8}", end="")
for name, _, _, _, _, _, _, _ in ranked[:3]:
    print(f" {name:>14}", end="")
print(f" {'沪深300':>12}")
print("-" * 80)

for year in range(2018, 2027):
    start_idx = next((j for j, d in enumerate(dates) if d.year == year), None)
    end_idx = next((j for j, d in enumerate(dates) if d.year == year + 1), len(dates))
    if start_idx is None: continue
    sub_dates = dates[start_idx:end_idx]
    if len(sub_dates) < 10: continue

    print(f"{year:<8}", end="")
    for name, _, _, _, _, _, _, _ in ranked[:3]:
        navs = all_results[name]
        sub = navs[start_idx:end_idx]
        ret = sub[-1] / sub[0] - 1
        print(f" {ret*100:>13.2f}%", end="")
    # 沪深300
    hs_sub = hs[start_idx:end_idx]
    hs_yr = hs_sub[-1] / hs_sub[0] - 1
    print(f" {hs_yr*100:>11.2f}%")
print()

# ============ 择时信号分析 ============
print(f"\n择时信号分析 (2019-01 ~ 2026-06):")
print(f"{'策略':<16} {'平均权益%':>10} {'最低权益%':>10} {'最高权益%':>10} {'切换次数':>10} {'换手成本':>10}")
print("-" * 70)

for name, func in TIMING_FUNCS.items():
    eqs = []
    for d in dates[210:]:
        eqs.append(func(d))
    eqs = np.array(eqs)
    # 切换次数: 权益比例变化超过10个百分点
    switches = np.sum(np.abs(np.diff(eqs)) > 0.10)
    # 估算换手成本
    turnover = np.sum(np.abs(np.diff(eqs))) * 0.5  # 半边换手
    txn_cost_est = turnover * TXN_COST * 100  # 占总资金百分比
    print(f"{name:<16} {eqs.mean()*100:>9.1f}% {eqs.min()*100:>9.1f}% {eqs.max()*100:>9.1f}% {switches:>10} {txn_cost_est:>9.2f}%")

print("=" * 70)

# ============ 推荐 ============
best_name = ranked[0][0]
best_ratio = ranked[0][6]
best_ret = ranked[0][1]
best_dd = ranked[0][5]
print(f"\n推荐策略: {best_name}")
print(f"  收益/回撤比: {best_ratio:.3f} (沪深300: {abs(hs_ret/hs_maxdd):.3f})")
print(f"  总收益: {best_ret*100:.2f}%, 最大回撤: {best_dd*100:.2f}%")
