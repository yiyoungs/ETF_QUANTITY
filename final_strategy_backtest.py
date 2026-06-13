"""
最终推荐策略: 核心-卫星 50/50 动量轮换
==============================================
策略说明:
  核心 (Core, 50%):   510300 (沪深300ETF) - 提供市场β
  卫星 (Satellite, 50%): Top3 动量行业ETF - 提供行业α
  动量过滤:            60日动量 > 0 AND 价格 > MA60
  止损:                单ETF 15%移动止损
  调仓频率:            月度

回测期: 2018-01 ~ 2026-06
预期表现: 年化 ~3%, 波动 ~10%, 最大回撤 ~-23%
          总收益 ~28% (vs 沪深300 ~17%)
"""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)

# ============ 配置 ============
try:
    from config import StrategyConfig, CACHE_DIR
    CASH_ETF = StrategyConfig.CASH_ETF_CODE
except ImportError:
    CASH_ETF = '511010'
    CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'etf_cache')

CORE_ETF = '510300'

PARAMS = {
    'core_ratio': 0.50,
    'satellite_ratio': 0.50,
    'satellite_top_n': 3,
    'momentum_window': 60,
    'trend_ma': 60,
    'stop_loss': 0.15,
    'txn_cost': 0.001,
}

# ============ 数据加载 ============
def load_data():
    etf_map = {}
    if not os.path.exists(CACHE_DIR):
        raise FileNotFoundError(f"数据目录不存在: {CACHE_DIR}")
    for fname in sorted(os.listdir(CACHE_DIR)):
        if fname.endswith('_hfq.csv'):
            code = fname.split('_')[0]
            etf_map[code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
    logging.info(f"已加载 {len(etf_map)} 只ETF数据")
    return etf_map

# ============ 工具函数 ============
def get_price(etf_map, code, date):
    df = etf_map.get(code)
    if df is None: return None
    match = df[df['date'] == date]['close']
    if len(match) == 0: return None
    return float(match.iloc[0])

def calc_momentum(etf_map, code, date, mom_win, ma_win):
    df = etf_map.get(code)
    if df is None: return None, False
    sub = df[df['date'] <= date]
    if len(sub) < mom_win + 1: return None, False
    t1 = float(sub['close'].iloc[-1])
    hist = float(sub['close'].iloc[-mom_win - 1])
    if len(sub) < ma_win: return None, False
    ma = float(sub['close'].iloc[-ma_win:].mean())
    return t1 / hist - 1, t1 > ma

def is_first_trading_month(dates, date):
    if date not in dates: return False
    idx = dates.index(date)
    if idx <= 0: return True
    return dates[idx - 1].month != date.month

# ============ 主回测 ============
def run_backtest():
    logging.info("=" * 60)
    logging.info(f"最终策略: 核心-卫星 50/50 | 核心={PARAMS['core_ratio']*100:.0f}% "
                 f"卫星={PARAMS['satellite_ratio']*100:.0f}% TopN={PARAMS['satellite_top_n']} "
                 f"动量={PARAMS['momentum_window']}日 MA={PARAMS['trend_ma']}日 "
                 f"止损={PARAMS['stop_loss']*100:.0f}%")
    logging.info("=" * 60)

    etf_map = load_data()
    sector_codes = [c for c in etf_map.keys() if c not in [CASH_ETF, CORE_ETF]]
    hs300_df = etf_map[CORE_ETF]

    dates = sorted([d for d in hs300_df['date']
                    if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])
    logging.info(f"交易日: {len(dates)} ({dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')})")

    # 状态
    cash = 1_000_000.0
    positions = {}
    highest_prices = {}
    nav_history = []
    trade_log = []

    def total_value(date):
        total = cash
        for code, shares in positions.items():
            p = get_price(etf_map, code, date)
            if p is not None: total += shares * p
        return total

    for i, date in enumerate(dates):
        tv = total_value(date)

        # 1) 止损检查
        if i > 0 and positions:
            for code in list(positions.keys()):
                if code in [CASH_ETF, CORE_ETF]: continue
                p = get_price(etf_map, code, date)
                if p is None: continue
                if code not in highest_prices or p > highest_prices[code]:
                    highest_prices[code] = p
                dd = (p - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
                if dd <= -PARAMS['stop_loss']:
                    shares = positions[code]
                    cash += shares * p * (1 - PARAMS['txn_cost'])
                    trade_log.append((date, 'STOP', code, shares, p))
                    del positions[code]
                    if code in highest_prices: del highest_prices[code]

        # 2) 月度调仓
        if is_first_trading_month(dates, date) and i >= PARAMS['momentum_window'] + 10:
            tv = total_value(date)

            # 2a) 动量筛选
            candidates = []
            for code in sector_codes:
                mom, trend = calc_momentum(etf_map, code, date,
                                           PARAMS['momentum_window'], PARAMS['trend_ma'])
                if mom is not None and mom > 0 and trend:
                    candidates.append((code, mom))
            candidates.sort(key=lambda x: x[1], reverse=True)
            targets = [c[0] for c in candidates[:PARAMS['satellite_top_n']]]

            # 2b) 卖出所有卫星持仓
            for code in list(positions.keys()):
                if code == CASH_ETF or code == CORE_ETF: continue
                p = get_price(etf_map, code, date)
                if p is None: continue
                shares = positions[code]
                cash += shares * p * (1 - PARAMS['txn_cost'])
                trade_log.append((date, 'SELL', code, shares, p))
                del positions[code]
                if code in highest_prices: del highest_prices[code]

            # 2c) 调整核心到 50% (偏离>10%才调)
            core_target = tv * PARAMS['core_ratio']
            if CORE_ETF in positions:
                p = get_price(etf_map, CORE_ETF, date)
                if p and p > 0:
                    cur_val = positions[CORE_ETF] * p
                    diff = cur_val - core_target
                    if abs(diff) / core_target > 0.10:
                        if diff > 0:
                            shares = int(diff / p / 100) * 100
                            if shares > 0:
                                cash += shares * p * (1 - PARAMS['txn_cost'])
                                positions[CORE_ETF] -= shares
                                trade_log.append((date, 'CORE-SELL', CORE_ETF, shares, p))
                                if positions[CORE_ETF] <= 0: del positions[CORE_ETF]
                        else:
                            shares = int(-diff / p / 100) * 100
                            if shares > 0:
                                cost = shares * p * (1 + PARAMS['txn_cost'])
                                if cash >= cost:
                                    positions[CORE_ETF] += shares
                                    cash -= cost
                                    trade_log.append((date, 'CORE-BUY', CORE_ETF, shares, p))
            else:
                p = get_price(etf_map, CORE_ETF, date)
                if p and p > 0:
                    shares = int(core_target / p / 100) * 100
                    if shares > 0:
                        cost = shares * p * (1 + PARAMS['txn_cost'])
                        if cash >= cost:
                            positions[CORE_ETF] = shares
                            cash -= cost
                            trade_log.append((date, 'CORE-BUY', CORE_ETF, shares, p))

            # 2d) 买入卫星ETF (注意: 不清掉 511010, 保留现金缓冲)
            if targets and cash > 1000:
                sat_budget = tv * PARAMS['satellite_ratio'] / len(targets)
                for code in targets:
                    p = get_price(etf_map, code, date)
                    if p is None or p <= 0: continue
                    shares = int(sat_budget / p / 100) * 100
                    if shares <= 0: continue
                    cost = shares * p * (1 + PARAMS['txn_cost'])
                    if cash < cost:
                        shares = int(cash / p / 100) * 100
                        if shares <= 0: break
                        cost = shares * p * (1 + PARAMS['txn_cost'])
                    positions[code] = shares
                    cash -= cost
                    highest_prices[code] = p
                    trade_log.append((date, 'BUY', code, shares, p))

            # 2e) 剩余现金 -> 国债ETF
            if cash > 1000:
                p = get_price(etf_map, CASH_ETF, date)
                if p and p > 0:
                    shares = int(cash * 0.98 / p / 100) * 100
                    if shares > 0:
                        cost = shares * p * (1 + PARAMS['txn_cost'])
                        if cash >= cost:
                            positions[CASH_ETF] = positions.get(CASH_ETF, 0) + shares
                            cash -= cost

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

    print("\n" + "=" * 70)
    print("  最终策略回测报告  |  核心-卫星 50/50 动量轮换")
    print("=" * 70)
    print(f"  回测期:   {dates[0].strftime('%Y-%m-%d')} ~ {dates[-1].strftime('%Y-%m-%d')} ({len(dates)}交易日)")
    print(f"  初始资金: {initial:,.0f}")
    print(f"  交易次数: {len(trade_log)}")
    print(f"  最终市值: {final_nav:,.0f}")
    print()
    print(f"  {'指标':<20} {'本策略':>15} {'沪深300':>15}")
    print("  " + "-" * 70)
    print(f"  {'总收益':<20} {total_ret*100:>14.2f}% {hs_total*100:>14.2f}%")
    print(f"  {'年化收益':<20} {ann_ret*100:>14.2f}% {hs_ann*100:>14.2f}%")
    print(f"  {'年化波动':<20} {ann_vol*100:>14.2f}% {hs_vol*100:>14.2f}%")
    print(f"  {'夏普比率':<20} {sharpe:>15.3f} {hs_sharpe:>15.3f}")
    print(f"  {'最大回撤':<20} {max_dd*100:>14.2f}% {hs_maxdd*100:>14.2f}%")
    print(f"  {'日胜率':<20} {win_rate*100:>14.2f}%")
    print(f"  {'收益/回撤比':<20} {abs(total_ret/max_dd):>15.3f} {abs(hs_total/hs_maxdd):>15.3f}")
    print("  " + "=" * 70)

    results['year'] = results['date'].dt.year
    results['daily_ret'] = results['total_value'].pct_change()

    print(f"\n  {'年份':<8} {'交易日':>7} {'年度收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8}")
    print("  " + "-" * 85)
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
        print(f"  {year:<8} {len(sub):>7} {ret*100:>9.2f}% {vol*100:>9.2f}% {s:>8.3f} {mdd*100:>9.2f}% {wr*100:>7.2f}%{m}")

    print(f"\n  最新持仓:")
    for code, shares in sorted(positions.items()):
        p = get_price(etf_map, code, dates[-1]) or 0
        v = shares * p
        pct = v / final_nav * 100
        print(f"    {code}: {shares}股 @ {p:.3f} = {v:,.0f} ({pct:.1f}%)")
    print(f"    现金: {cash:,.0f} ({cash/final_nav*100:.1f}%)")

    alpha_str = f"  +{abs(total_ret-hs_total)*100:.1f}%" if total_ret > hs_total else f"  {abs(total_ret-hs_total)*100:.1f}%"
    print(f"\n  ============= 策略评价 =============")
    print(f"  ★ 总收益: {total_ret*100:.1f}% (基准 {hs_total*100:.1f}%){alpha_str}")
    print(f"  ★ 最大回撤: {max_dd*100:.1f}% (基准 {hs_maxdd*100:.1f}%) 减少{abs((max_dd-hs_maxdd)/hs_maxdd*100):.1f}%")
    print(f"  ★ 夏普比率: {sharpe:.3f} (基准 {hs_sharpe:.3f}) {sharpe/hs_sharpe:.1f}倍" if hs_sharpe != 0 else "  ★ 夏普比率: N/A")

    # 保存
    out_csv = os.path.join(os.path.dirname(__file__), 'final_backtest_results.csv')
    results.to_csv(out_csv, index=False)
    trades_df = pd.DataFrame(trade_log, columns=['date', 'type', 'code', 'shares', 'price'])
    if len(trade_log) > 0:
        trades_df.to_csv(out_csv.replace('results', 'trades'), index=False)

    print(f"\n  结果已保存至: {out_csv}")
    print("\nDone.")
    return results, trades_df


if __name__ == '__main__':
    run_backtest()
