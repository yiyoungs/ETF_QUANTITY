"""_backtest_dual_momentum_v2.py
=====================================
双动量一刀切 v2:
  修复v1的问题:
  1. MA200代替MA120 (减少切换)
  2. 20天确认期 (防止假信号)
  3. 牛市保留30%核心510300 (不100%追动量)
  4. 无止损 (靠月度调仓风控)
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
        etf_price_map[code] = pd.read_csv(
            os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date']
        )

CASH_ETF = '511010'
BENCHMARK = '510300'
CORE_ETF = '510300'

AVAILABLE_CODES = set(etf_price_map.keys())
SECTOR_CODES = [
    '512480', '159995', '512760', '515050', '512000',
    '515030', '159875', '512660', '512690', '159928',
    '512010', '159801', '512890', '510880', '168204',
    '512400', '159915', '512100', '512880', '510050', '510500',
]
ETF_POOL = [c for c in SECTOR_CODES if c in AVAILABLE_CODES]
SEMI_LABELS = {'512480', '159995', '512760'}

MA_LOOKBACK = 200
MOM_PERIODS = [20, 60, 120]
MOM_WEIGHTS = [0.4, 0.3, 0.3]
TOP_N = 5
TXN_COST = 0.001
CONFIRM_DAYS = 20
CORE_RATIO = 0.30  # 牛市30%核心510300, 70%卫星动量

def get_price(code, date):
    df = etf_price_map.get(code)
    if df is None: return None
    m = df[df['date'] == date]['close']
    return float(m.iloc[0]) if len(m) > 0 else None

def check_market_mode(current_date):
    df = etf_price_map.get(BENCHMARK)
    if df is None: return "BEAR"
    sub = df[df['date'] <= current_date]
    if len(sub) < MA_LOOKBACK + 1: return "BEAR"
    price = float(sub['close'].iloc[-1])
    ma = float(sub['close'].iloc[-MA_LOOKBACK:].mean())
    return "BULL" if price > ma else "BEAR"

def generate_momentum_top5(current_date):
    scores = {}
    for code in ETF_POOL:
        df = etf_price_map.get(code)
        if df is None: continue
        sub = df[df['date'] <= current_date]
        if len(sub) < MOM_PERIODS[-1] + 1: continue
        current_price = float(sub['close'].iloc[-1])
        total_score = 0.0
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            past_price = float(sub['close'].iloc[-period - 1])
            ret = (current_price / past_price) - 1
            total_score += ret * weight
        scores[code] = total_score
    sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    final_top5 = []
    semi_count = 0
    for code, score in sorted_etfs:
        if len(final_top5) >= TOP_N: break
        if code in SEMI_LABELS:
            if semi_count >= 2: continue
            semi_count += 1
        final_top5.append(code)
    return final_top5

def run_backtest(start='2018-01-01', end='2026-06-30'):
    hs300_df = etf_price_map[BENCHMARK]
    dates = sorted([d for d in hs300_df['date']
                    if d >= pd.Timestamp(start) and d <= pd.Timestamp(end)])

    cash = 1_000_000.0
    holdings = {}
    nav_history = []

    confirmed_mode = "BEAR"
    pending_mode = None
    pending_count = 0

    def total_value(date):
        tv = cash
        for code, shares in holdings.items():
            p = get_price(code, date)
            if p: tv += shares * p
        return tv

    def is_rebalance_day(date):
        if date not in dates: return False
        idx = dates.index(date)
        return idx <= 0 or dates[idx - 1].month != date.month

    def sell_all(code, date):
        nonlocal cash
        if code not in holdings: return
        p = get_price(code, date)
        if not p: return
        cash += holdings[code] * p * (1 - TXN_COST)
        del holdings[code]

    def buy_value(code, value, date):
        nonlocal cash
        p = get_price(code, date)
        if not p or p <= 0: return
        shares = int(value / p / 100) * 100
        if shares <= 0: return
        cost = shares * p * (1 + TXN_COST)
        if cash < cost:
            shares = int(cash / p / 100) * 100
            if shares <= 0: return
            cost = shares * p * (1 + TXN_COST)
        if cash - cost < -1: return
        holdings[code] = holdings.get(code, 0) + shares
        cash -= cost

    print(f"双动量v2 | MA{MA_LOOKBACK}大盘开关 | {CONFIRM_DAYS}天确认 | 核心{CORE_RATIO*100:.0f}%+卫星{TOP_N}动量")
    print("=" * 90)

    for i, date in enumerate(dates):
        tv = total_value(date)

        # 状态确认
        raw_mode = check_market_mode(date)
        if raw_mode != confirmed_mode:
            if raw_mode == pending_mode:
                pending_count += 1
            else:
                pending_mode = raw_mode
                pending_count = 1
            if pending_count >= CONFIRM_DAYS:
                confirmed_mode = raw_mode
                pending_mode = None
                pending_count = 0
        else:
            pending_mode = None
            pending_count = 0

        # 月度调仓
        if is_rebalance_day(date) and i >= MA_LOOKBACK + 10:
            tv = total_value(date)
            mode = confirmed_mode

            if mode == "BEAR":
                for code in list(holdings.keys()):
                    if code != CASH_ETF:
                        sell_all(code, date)
                if cash > 1000:
                    buy_value(CASH_ETF, cash, date)

                if i < 500 or i % 500 == 0 or (2019 <= date.year <= 2020):
                    pos_str = ", ".join([f"{c}:{s}" for c, s in sorted(holdings.items())])
                    print(f"{date.strftime('%Y-%m-%d')} [BEAR] 总={tv:.0f} 现金={cash:.0f} {pos_str}")

            elif mode == "BULL":
                top5 = generate_momentum_top5(date)

                # 卖出所有非目标持仓
                for code in list(holdings.keys()):
                    if code not in top5 and code != CORE_ETF:
                        sell_all(code, date)

                # 30%核心510300
                core_value = tv * CORE_RATIO
                if CORE_ETF in holdings:
                    p = get_price(CORE_ETF, date)
                    if p and p > 0:
                        cur_val = holdings[CORE_ETF] * p
                        diff = cur_val - core_value
                        if abs(diff) / core_value > 0.15:
                            if diff > 0:
                                sell_shares = int(diff / p / 100) * 100
                                if sell_shares > 0:
                                    cash += sell_shares * p * (1 - TXN_COST)
                                    holdings[CORE_ETF] -= sell_shares
                                    if holdings[CORE_ETF] <= 0: del holdings[CORE_ETF]
                            else:
                                buy_value(CORE_ETF, -diff, date)
                else:
                    buy_value(CORE_ETF, core_value, date)

                # 70%卫星动量Top5
                sat_value = tv * (1 - CORE_RATIO)
                if top5 and cash > 1000:
                    per_etf = sat_value / len(top5)
                    for code in top5:
                        buy_value(code, per_etf, date)

                if i < 500 or i % 500 == 0 or (2019 <= date.year <= 2020):
                    pos_str = ", ".join([f"{c}:{s}" for c, s in sorted(holdings.items())])
                    print(f"{date.strftime('%Y-%m-%d')} [BULL] 总={tv:.0f} 现金={cash:.0f} Top5={top5} {pos_str}")

        nav_history.append({'date': date, 'total_value': total_value(date), 'mode': confirmed_mode})

    return nav_history

# ==================== 报告 ====================
def generate_report(nav_history):
    results = pd.DataFrame(nav_history)
    nav = results['total_value'].values
    initial = nav[0]; final = nav[-1]
    years = (results['date'].iloc[-1] - results['date'].iloc[0]).days / 365.25
    total_ret = final / initial - 1
    ann_ret = (final / initial) ** (1 / years) - 1
    daily_r = pd.Series(nav).pct_change().dropna().values
    ann_vol = daily_r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    cummax = pd.Series(nav).cummax()
    dd = (nav - cummax) / cummax
    max_dd = dd.min()
    win_rate = (daily_r > 0).mean()

    hs300 = [get_price('510300', d) for d in results['date']]
    hs = np.array([x for x in hs300 if x is not None])
    hs_total = hs[-1] / hs[0] - 1
    hs_ann = (hs[-1] / hs[0]) ** (1 / years) - 1
    hs_vol = np.diff(hs) / hs[:-1]
    hs_annvol = hs_vol.std() * np.sqrt(252)
    hs_sharpe = hs_ann / hs_annvol if hs_annvol > 0 else 0
    hs_cummax = np.maximum.accumulate(hs)
    hs_maxdd = ((hs - hs_cummax) / hs_cummax).min()

    mode_counts = results['mode'].value_counts()
    bull_days = mode_counts.get('BULL', 0)
    bear_days = mode_counts.get('BEAR', 0)

    print(f"\n{'='*100}")
    print(f"{' 双动量v2回测报告 (MA200+20天确认+30%核心) ':=^100}")
    print(f"{'='*100}")
    print(f"{'指标':<20} {'双动量v2':>15} {'v13(无择时)':>15} {'v10c':>12} {'沪深300':>12}")
    print("-" * 100)
    print(f"{'总收益':<20} {total_ret*100:>14.2f}% {'69.23%':>15} {'28.98%':>12} {hs_total*100:>11.2f}%")
    print(f"{'年化收益':<20} {ann_ret*100:>14.2f}% {'6.44%':>15} {'3.06%':>12} {hs_ann*100:>11.2f}%")
    print(f"{'年化波动':<20} {ann_vol*100:>14.2f}% {'27.10%':>15} {'9.42%':>12} {hs_annvol*100:>11.2f}%")
    print(f"{'夏普比率':<20} {sharpe:>15.3f} {'0.238':>15} {'0.325':>12} {hs_sharpe:>12.3f}")
    print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {'-48.06%':>15} {'-19.67%':>12} {hs_maxdd*100:>11.2f}%")
    print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>15.3f} {'1.441':>15} {'1.473':>12} {abs(hs_total/hs_maxdd):>12.3f}")
    print(f"{'牛/熊天数':<20} {bull_days}/{bear_days} ({bull_days/len(results)*100:.0f}%/{bear_days/len(results)*100:.0f}%)")
    print("=" * 100)

    results['year'] = results['date'].dt.year
    results['daily_ret'] = results['total_value'].pct_change()

    print(f"\n{'年份':<8} {'收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8} {'牛市天':>8} {'熊市天':>8}")
    print("-" * 90)
    for year in sorted(results['year'].unique()):
        sub = results[results['year'] == year].reset_index(drop=True)
        if len(sub) < 10: continue
        ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
        vol = sub['daily_ret'].std() * np.sqrt(252)
        s = ret / vol if vol > 0 else 0
        cm = sub['total_value'].cummax()
        mdd = ((sub['total_value'] - cm) / cm).min()
        wr = (sub['daily_ret'] > 0).mean()
        bull = (sub['mode'] == 'BULL').sum()
        bear = (sub['mode'] == 'BEAR').sum()
        m = " *" if ret > 0 else ""
        print(f"{year:<8} {ret*100:>9.2f}% {vol*100:>9.2f}% {s:>8.3f} {mdd*100:>9.2f}% {wr*100:>7.2f}% {bull:>8} {bear:>8}{m}")

    print(f"\n牛熊切换时间线 (MA{MA_LOOKBACK}, {CONFIRM_DAYS}天确认):")
    prev = None
    for _, row in results.iterrows():
        mode = row['mode']
        if mode != prev:
            print(f"  {row['date'].strftime('%Y-%m-%d')} -> {mode}")
            prev = mode

if __name__ == '__main__':
    nav_history = run_backtest()
    generate_report(nav_history)
    pd.DataFrame(nav_history).to_csv(os.path.join(CACHE_DIR, '..', 'backtest_dual_momentum_v2.csv'), index=False)
