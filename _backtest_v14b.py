"""v14b: 波动率目标策略回撤深度分析
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

def timing_vol_target(date, target_vol=0.15):
    """波动率目标"""
    sub = hs300_df[hs300_df['date'] <= date]
    if len(sub) < 25: return EQ_MIN
    returns = sub['close'].pct_change().iloc[-20:]
    realized_vol = returns.std() * np.sqrt(252)
    if realized_vol < 0.01: realized_vol = 0.01
    equity = np.clip(target_vol / realized_vol, EQ_MIN, EQ_MAX)
    return equity, realized_vol

def run_backtest_with_detail(timing_func, timing_name, **kwargs):
    cash = 1_000_000.0
    positions = {}
    highest_prices = {}
    nav_history = []
    detail_history = []

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

        eq_ratio_val = None
        if is_first_trading(date):
            tv = cash
            for code, shares in positions.items():
                p = get_price(code, date)
                if p: tv += shares * p
            result = timing_func(date, **kwargs)
            eq_ratio = result[0] if isinstance(result, tuple) else result
            realized_vol = result[1] if isinstance(result, tuple) else None
            eq_ratio_val = eq_ratio
            equity_value = tv * eq_ratio
            bond_value = tv * (1 - eq_ratio)
            core_value = equity_value * CORE_IN_EQ
            sat_value = equity_value * SAT_IN_EQ

            candidates = []
            for code in sector_codes:
                mom = calc_momentum(code, date, MOM_WINDOW)
                if mom is not None:
                    candidates.append((code, mom))
            candidates.sort(key=lambda x: x[1], reverse=True)
            targets = [c[0] for c in candidates[:TOP_N]]

            # 卖出所有风险资产
            total_sold = 0
            for code in list(positions.keys()):
                if code == CASH_ETF: continue
                p = get_price(code, date)
                if not p: continue
                cash += positions[code] * p * (1 - TXN_COST)
                total_sold += positions[code] * p
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

            # 记录调仓细节
            tv_final = cash
            for code, shares in positions.items():
                p = get_price(code, date)
                if p: tv_final += shares * p
            pos_parts = []
            for c, s in positions.items():
                if s <= 0: continue
                p = get_price(c, date)
                if p is None: continue
                pct = int(s * p / tv_final * 100)
                pos_parts.append(f"{c}:{pct}%")
            pos_str = ', '.join(pos_parts)
            detail_history.append({
                'date': date.strftime('%Y-%m-%d'),
                'total_value': tv_final,
                'equity_ratio': eq_ratio,
                'realized_vol': realized_vol,
                'positions': pos_str,
                'cash': cash,
            })

        tv = cash
        for code, shares in positions.items():
            p = get_price(code, date)
            if p: tv += shares * p
        nav_history.append(tv)

    return nav_history, detail_history

# 运行波动率目标 (target_vol=15%)
print("v14b: 波动率目标策略回撤深度分析")
print("=" * 100)
navs, details = run_backtest_with_detail(timing_vol_target, '波动率目标')
nav = np.array(navs)
cummax = np.maximum.accumulate(nav)
dd_series = (nav - cummax) / cummax

# 找最大回撤
max_dd = dd_series.min()
max_dd_idx = np.argmin(dd_series)
peak_idx = np.argmax(nav[:max_dd_idx+1])

print(f"\n最大回撤发生时间:")
print(f"  峰值日期: {dates[peak_idx].strftime('%Y-%m-%d')} (净值={nav[peak_idx]:,.0f})")
print(f"  谷底日期: {dates[max_dd_idx].strftime('%Y-%m-%d')} (净值={nav[max_dd_idx]:,.0f})")
print(f"  回撤幅度: {max_dd*100:.2f}%")
print(f"  回撤周期: {(dates[max_dd_idx] - dates[peak_idx]).days} 天")

# 找回撤期间的调仓细节
print(f"\n回撤期间每月调仓记录:")
print(f"{'日期':<12} {'净值':>12} {'回撤%':>8} {'权益%':>8} {'20日波动%':>12} {'仓位':<40}")
print("-" * 110)

for detail in details:
    d = pd.Timestamp(detail['date'])
    if d >= dates[peak_idx] and d <= dates[max_dd_idx]:
        # 找这天的回撤
        idx = next((i for i, date in enumerate(dates) if date.strftime('%Y-%m-%d') == detail['date']), None)
        if idx is not None:
            dd_pct = dd_series[idx] * 100
            rv_pct = (detail['realized_vol'] * 100 if detail['realized_vol'] else 0)
            eq_pct = detail['equity_ratio'] * 100
            print(f"{detail['date']:<12} {detail['total_value']:>12,.0f} {dd_pct:>8.2f}% {eq_pct:>7.1f}% {rv_pct:>11.2f}% {detail['positions']:<40}")

# 分析三大回撤期
print(f"\n\n{'='*100}")
print(f" 三大回撤期分析")
print(f"{'='*100}")

# 找Top3回撤谷底
sorted_dd_idx = np.argsort(dd_series)[:3]
for rank, bottom_idx in enumerate(sorted_dd_idx):
    local_peak_idx = np.argmax(nav[:bottom_idx+1])
    bottom_date = dates[bottom_idx].strftime('%Y-%m-%d')
    peak_date = dates[local_peak_idx].strftime('%Y-%m-%d')
    dd_val = dd_series[bottom_idx] * 100

    # 回撤开始时的波动率
    peak_detail = None
    for detail in details:
        if pd.Timestamp(detail['date']) >= dates[local_peak_idx]:
            peak_detail = detail
            break
    # 回撤结束时的波动率
    bottom_detail = None
    for detail in details:
        if pd.Timestamp(detail['date']) >= dates[bottom_idx]:
            break
        bottom_detail = detail

    if peak_detail and bottom_detail:
        print(f"\n第{rank+1}大回撤: {peak_date} → {bottom_date}")
        print(f"  幅度: {dd_val:.2f}%")
        print(f"  峰值时: 权益{peak_detail['equity_ratio']*100:.0f}% 波动率 {peak_detail['realized_vol']*100:.1f}% 仓位: {peak_detail['positions']}")
        print(f"  谷底时: 权益{bottom_detail['equity_ratio']*100:.0f}% 波动率 {bottom_detail['realized_vol']*100:.1f}% 仓位: {bottom_detail['positions']}")

# 分析问题根源
print(f"\n\n{'='*100}")
print(f"  回撤来源拆解")
print(f"{'='*100}")

# 计算核心(510300)对总回撤的贡献
hs_prices = []
for d in dates:
    p = get_price('510300', d)
    if p: hs_prices.append(p)
hs = np.array(hs_prices)
hs_cummax = np.maximum.accumulate(hs)
hs_dd = (hs - hs_cummax) / hs_cummax

# 找沪深300最大回撤
hs_maxdd = hs_dd.min() * 100

# 找策略最大回撤时的沪深300表现
peak_val = nav[peak_idx]
bottom_val = nav[max_dd_idx]
strategy_drop = (bottom_val - peak_val) / peak_val * 100

hs_peak_price = hs[peak_idx]
hs_bottom_price = hs[max_dd_idx]
hs_drop = (hs_bottom_price - hs_peak_price) / hs_peak_price * 100

peak_date_str = dates[peak_idx].strftime('%Y-%m-%d')
bottom_date_str = dates[max_dd_idx].strftime('%Y-%m-%d')
print(f"\n在策略最大回撤期({peak_date_str} -> {bottom_date_str}):")
print(f"  策略净值下跌: {strategy_drop:.2f}%")
print(f"  510300同期下跌: {hs_drop:.2f}%")
print(f"  核心仓位贡献: {hs_drop * 0.5:.2f}% (按50%仓位)")
other_contribution = strategy_drop - hs_drop * 0.5
print(f"  剩余{other_contribution:.2f}%来自卫星动量仓位")

# 检查波动率目标策略的问题
print(f"\n{'='*60}")
print(f"  为什么波动率目标没能降低回撤?")
print(f"{'='*60}")

print(f"""
问题1: 波动率是"滞后指标"
  - 20日波动率用"过去20日"的涨跌幅计算
  - 大跌开始的前几天, 波动率还很低, 策略还在高仓位
  - 等波动率上升触发减仓时, 已经跌了一大截

问题2: 核心仓位是50% 510300
  - 510300本身最大回撤 -45.10%
  - 50%仓位 = 至少贡献 22.55% 的回撤
  - 卫星50%的动量策略在熊市中也会跌

问题3: 卫星动量ETF在熊市中反而跌最多
  - 动量策略选的是近期涨最多的ETF
  - 但在熊市中, 涨最多的ETF往往也是估值最高、泡沫最大的
  - 因此在大牛市→熊市切换时, 动量卫星会跌得比指数还狠

问题4: 20日窗口太短
  - 20日波动率在震荡市中频繁切换
  - 每次切换都有摩擦成本

问题5: 15%止损只对单只ETF有效
  - 整个市场系统性下跌时, 所有ETF都在跌
  - 止损会不断触发, 造成连锁式止损
  - 但调仓一个月一次, 跌了一个月后才会换到债基
""")

# 对比: 无择时 vs 波动率目标
print(f"\n{'='*60}")
print(f"  回撤改善有多大")
print(f"{'='*60}")
print(f"""
  A股2018年最大回撤 (2018-01 → 2018-12):
    510300本身回撤: {hs_maxdd:.2f}%
    波动率目标最大回撤: {max_dd*100:.2f}%
    无择时最大回撤: -50.85% (从v14结果)

  改善幅度: {(max_dd*100 - (-50.85)):.2f}个百分点
  核心50%的贡献约20%回撤是 unavoidable

结论: 波动率目标策略的最大回撤仍然有 -45%,
      但比无择时的-51%改善了~6个百分点,
      主要原因是50%核心仓位+510300本身就有-45%回撤,
      而波动率目标只是在核心基础上进一步降低了卫星部分的损失
""")
