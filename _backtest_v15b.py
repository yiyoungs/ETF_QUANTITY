"""v15b: 状态自适应 + 最短持仓期
=====================================
在v15基础上增加:
  - 状态最短持仓期: 一旦确认状态, 至少持有N个交易日才允许切换
  - 测试不同持仓期: 20天(1月) / 60天(3月) / 120天(6月)
  - 切换时用"确认期": 新状态必须持续M天才确认切换(防假信号)
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

TXN_COST = 0.001

# ============ 市场状态识别 (原始信号) ============

def detect_regime_raw(date):
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 120:
        return 'BEAR'
    price = float(sub['close'].iloc[-1])
    closes = sub['close'].values
    ma20 = float(np.mean(closes[-20:])) if len(closes) >= 20 else price
    ma60 = float(np.mean(closes[-60:])) if len(closes) >= 60 else price
    daily_returns = np.diff(closes[-25:]) / closes[-25:-1] if len(closes) >= 25 else np.zeros(10)
    vol20 = daily_returns.std() * np.sqrt(252) if len(daily_returns) > 1 else 0.0
    recent_60 = closes[-60:] if len(closes) >= 60 else closes
    cummax_60 = np.maximum.accumulate(recent_60)
    dd_from_peak = (recent_60[-1] - cummax_60[-1]) / cummax_60[-1]
    trough_60 = float(np.min(recent_60))
    bounce_from_trough = (price - trough_60) / trough_60 if trough_60 > 0 else 0
    if len(closes) >= 30:
        ma20_current = float(np.mean(closes[-20:]))
        ma20_10d_ago = float(np.mean(closes[-30:-10]))
        ma20_slope = (ma20_current - ma20_10d_ago) / ma20_10d_ago
    else:
        ma20_slope = 0
    if price > ma60 and ma20 > ma60 and ma20_slope > 0.005:
        return 'BULL'
    if price < ma60 and (vol20 > 0.25 or dd_from_peak < -0.10):
        return 'BEAR'
    if bounce_from_trough > 0.15 and ma20_slope > 0.002:
        return 'RECOVERY'
    return 'RANGE'

# ============ 状态-参数映射 ============

REGIME_PARAMS = {
    'BULL':     {'name': '牛市',   'mom_window': 60,  'top_n': 5, 'stop_loss': 0.20, 'target_vol': 0.22, 'core_ratio': 0.40, 'equity_max': 1.00},
    'RECOVERY': {'name': '恢复期', 'mom_window': 45,  'top_n': 4, 'stop_loss': 0.18, 'target_vol': 0.18, 'core_ratio': 0.50, 'equity_max': 1.00},
    'RANGE':    {'name': '震荡市', 'mom_window': 90,  'top_n': 3, 'stop_loss': 0.15, 'target_vol': 0.15, 'core_ratio': 0.60, 'equity_max': 0.70},
    'BEAR':     {'name': '熊市',   'mom_window': 120, 'top_n': 2, 'stop_loss': 0.10, 'target_vol': 0.10, 'core_ratio': 0.80, 'equity_max': 0.40},
}

# ============ 辅助函数 ============

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

# ============ 带持仓期的状态序列 ============

def build_regime_series(min_hold_days, confirm_days=5):
    """构建带最短持仓期和确认期的状态序列"""
    raw_regimes = []
    for d in dates:
        raw_regimes.append(detect_regime_raw(d))

    # 确认期: 新状态必须持续confirm_days天才确认
    confirmed = raw_regimes[0]
    confirm_count = {confirmed: 1}
    confirmed_series = [confirmed]

    for i in range(1, len(raw_regimes)):
        raw = raw_regimes[i]
        confirm_count[raw] = confirm_count.get(raw, 0) + 1
        # 如果当前raw状态持续了confirm_days天, 且与已确认状态不同
        if raw != confirmed and confirm_count.get(raw, 0) >= confirm_days:
            confirmed = raw
            confirm_count = {confirmed: confirm_count.get(confirmed, 0)}
        confirmed_series.append(confirmed)

    # 最短持仓期: 确认后至少持有min_hold_days天
    final = [confirmed_series[0]]
    last_switch_idx = 0
    for i in range(1, len(confirmed_series)):
        if confirmed_series[i] != final[-1]:
            if (i - last_switch_idx) >= min_hold_days:
                final.append(confirmed_series[i])
                last_switch_idx = i
            else:
                final.append(final[-1])  # 保持当前状态
        else:
            final.append(confirmed_series[i])

    return final

# ============ 回测引擎 ============

def run_backtest(regime_series):
    cash = 1_000_000.0
    positions = {}
    highest_prices = {}
    nav_history = []

    for i, date in enumerate(dates):
        tv = cash
        for code, shares in positions.items():
            p = get_price(code, date)
            if p: tv += shares * p

        regime = regime_series[i]
        params = REGIME_PARAMS[regime]
        mom_w = params['mom_window']
        top_n = params['top_n']
        stop_loss = params['stop_loss']
        target_vol = params['target_vol']
        core_ratio = params['core_ratio']
        equity_max = params['equity_max']

        # 止损
        if i > 0 and positions:
            for code in list(positions.keys()):
                if code in [CASH_ETF, CORE_ETF]: continue
                p = get_price(code, date)
                if not p: continue
                if code not in highest_prices or p > highest_prices[code]:
                    highest_prices[code] = p
                dd = (p - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
                if dd <= -stop_loss:
                    cash += positions[code] * p * (1 - TXN_COST)
                    del positions[code]
                    if code in highest_prices:
                        del highest_prices[code]

        # 月度调仓
        if is_first_trading(date) and i >= mom_w + 10:
            tv = cash
            for code, shares in positions.items():
                p = get_price(code, date)
                if p: tv += shares * p

            sub = hs300_df[hs300_df['date'] <= date]
            if len(sub) >= 25:
                daily_r = sub['close'].pct_change().iloc[-20:]
                rv = daily_r.std() * np.sqrt(252)
                if rv < 0.01: rv = 0.01
                equity_ratio = np.clip(target_vol / rv, 0.40, equity_max)
            else:
                equity_ratio = equity_max

            equity_value = tv * equity_ratio
            bond_value = tv * (1 - equity_ratio)
            core_value = equity_value * core_ratio
            sat_value = equity_value * (1 - core_ratio)

            candidates = []
            for code in sector_codes:
                mom = calc_momentum(code, date, mom_w)
                if mom is not None:
                    candidates.append((code, mom))
            candidates.sort(key=lambda x: x[1], reverse=True)
            targets = [c[0] for c in candidates[:top_n]]

            for code in list(positions.keys()):
                if code == CASH_ETF: continue
                p = get_price(code, date)
                if not p: continue
                cash += positions[code] * p * (1 - TXN_COST)
                del positions[code]
                if code in highest_prices:
                    del highest_prices[code]

            if CASH_ETF in positions:
                p = get_price(CASH_ETF, date)
                if p and p > 0:
                    cur = positions[CASH_ETF] * p
                    if cur > bond_value * 1.10:
                        excess = int((cur - bond_value) / p / 100) * 100
                        if excess > 0:
                            cash += excess * p * (1 - TXN_COST)
                            positions[CASH_ETF] -= excess
                            if positions[CASH_ETF] <= 0:
                                del positions[CASH_ETF]
                    elif cur < bond_value * 0.85:
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

            if cash > 1000:
                p = get_price(CASH_ETF, date)
                if p and p > 0:
                    shares = int(cash * 0.98 / p / 100) * 100
                    if shares > 0:
                        cost = shares * p * (1 + TXN_COST)
                        if cash >= cost:
                            positions[CASH_ETF] = positions.get(CASH_ETF, 0) + shares
                            cash -= cost

        tv = cash
        for code, shares in positions.items():
            p = get_price(code, date)
            if p: tv += shares * p
        nav_history.append(tv)

    return nav_history

# ============ 运行 ============
print("v15b: 状态自适应 + 不同最短持仓期对比")
print("=" * 100)

configs = [
    ('无持仓期(原始)', 0, 0),
    ('持仓20天(1月)', 20, 5),
    ('持仓40天(2月)', 40, 5),
    ('持仓60天(3月)', 60, 5),
    ('持仓120天(半年)', 120, 5),
]

all_results = {}
all_regimes = {}

for label, min_hold, confirm in configs:
    print(f"\n  构建{label}状态序列...", end=" ", flush=True)
    regime_series = build_regime_series(min_hold, confirm)

    # 统计切换次数
    switches = 0
    for i in range(1, len(regime_series)):
        if regime_series[i] != regime_series[i-1]:
            switches += 1

    # 状态分布
    from collections import Counter
    dist = Counter(regime_series)
    dist_str = " ".join([f"{REGIME_PARAMS[r]['name']}:{c}" for r, c in sorted(dist.items(), key=lambda x: -x[1])])

    print(f"切换{switches}次, {dist_str}")

    print(f"  回测中...", end=" ", flush=True)
    navs = run_backtest(regime_series)
    all_results[label] = navs
    all_regimes[label] = regime_series

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
    print(f"收益{total_ret*100:>7.2f}% 回撤{max_dd*100:>7.2f}% 夏普{sharpe:.3f}")

# 沪深300基准
hs_prices = []
for d in dates:
    p = get_price('510300', d)
    if p: hs_prices.append(p)
hs = np.array(hs_prices)
years = (dates[-1] - dates[0]).days / 365.25
hs_total = hs[-1] / hs[0] - 1
hs_ann = (hs[-1] / hs[0]) ** (1 / years) - 1
hs_daily = np.diff(hs) / hs[:-1]
hs_vol = hs_daily.std() * np.sqrt(252)
hs_sharpe = hs_ann / hs_vol if hs_vol > 0 else 0
hs_cummax = np.maximum.accumulate(hs)
hs_maxdd = ((hs - hs_cummax) / hs_cummax).min()

# ============ 对比表 ============
print(f"\n\n{'='*110}")
print(f"{' 不同最短持仓期对比 (2018-2026) ':^110}")
print(f"{'='*110}")
print(f"{'持仓期':<18} {'总收益':>10} {'年化':>8} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'收益/回撤':>10}")
print("-" * 110)

ranked = []
for label, navs in all_results.items():
    nav = np.array(navs)
    total_ret = nav[-1] / nav[0] - 1
    ann_ret = (nav[-1] / nav[0]) ** (1 / years) - 1
    daily_r = np.diff(nav) / nav[:-1]
    ann_vol = daily_r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = np.maximum.accumulate(nav)
    dd = (nav - cummax) / cummax
    max_dd = dd.min()
    ratio = abs(total_ret / max_dd) if max_dd != 0 else 0
    ranked.append((label, total_ret, ann_ret, ann_vol, sharpe, max_dd, ratio))

ranked.sort(key=lambda x: x[6], reverse=True)

for label, tr, ar, av, sp, md, ratio in ranked:
    print(f"{label:<18} {tr*100:>9.2f}% {ar*100:>7.2f}% {av*100:>9.2f}% {sp:>8.3f} {md*100:>9.2f}% {ratio:>10.3f}")

print("-" * 110)
print(f"{'沪深300':<18} {hs_total*100:>9.2f}% {hs_ann*100:>7.2f}% {hs_vol*100:>9.2f}% {hs_sharpe:>8.3f} {hs_maxdd*100:>9.2f}% {abs(hs_total/hs_maxdd):>10.3f}")
print("=" * 110)

# ============ 分年度对比 (最佳 vs 最差 vs 原始) ============
print(f"\n分年度对比 (最佳持仓期 vs 原始 vs 沪深300):")
best_label = ranked[0][0]
worst_label = ranked[-1][0]
print(f"  最佳: {best_label}")
print(f"{'年份':<8} {best_label:>14} {'无持仓期(原始)':>14} {'沪深300':>12}")
print("-" * 60)

for year in range(2018, 2027):
    start_idx = next((j for j, d in enumerate(dates) if d.year == year), None)
    if start_idx is None: continue
    end_idx = next((j for j, d in enumerate(dates) if d.year == year + 1), len(dates))
    if end_idx - start_idx < 10: continue

    nav_best = np.array(all_results[best_label][start_idx:end_idx])
    nav_raw = np.array(all_results['无持仓期(原始)'][start_idx:end_idx])
    hs_sub = hs[start_idx:end_idx]

    ret_b = nav_best[-1] / nav_best[0] - 1
    ret_r = nav_raw[-1] / nav_raw[0] - 1
    ret_h = hs_sub[-1] / hs_sub[0] - 1

    print(f"{year:<8} {ret_b*100:>13.2f}% {ret_r*100:>13.2f}% {ret_h*100:>11.2f}%")

# ============ 切换次数 vs 收益的关系 ============
print(f"\n\n{'='*80}")
print(f"  持仓期 vs 切换次数 vs 收益 的关系")
print(f"{'='*80}")
print(f"{'持仓期':<18} {'切换次数':>10} {'总收益':>10} {'最大回撤':>10}")
print("-" * 50)

for label, min_hold, confirm in configs:
    regime_series = all_regimes[label]
    switches = sum(1 for i in range(1, len(regime_series)) if regime_series[i] != regime_series[i-1])
    navs = all_results[label]
    nav = np.array(navs)
    total_ret = nav[-1] / nav[0] - 1
    cummax = np.maximum.accumulate(nav)
    max_dd = ((nav - cummax) / cummax).min()
    print(f"{label:<18} {switches:>10} {total_ret*100:>9.2f}% {max_dd*100:>9.2f}%")

print(f"\n结论: 切换次数越少, 摩擦成本越低, 但可能错过状态变化")
print(f"  存在一个最优平衡点: 持仓期太短(频繁切换)和太长(反应迟钝)都不好")
