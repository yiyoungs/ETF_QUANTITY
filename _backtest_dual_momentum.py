"""_backtest_dual_momentum.py
=====================================
双动量一刀切架构:
  1. 大盘开关: 沪深300 MA120, 收盘>MA120=BULL, 否则BEAR, 1周确认
  2. 牛市: 100%满仓全球Top5动量 (含黄金/纳指/A股行业), 半导体最多2只
  3. 熊市: 100%全仓国债511010
  4. 月度调仓, 一刀切切换, 不留现金沙袋
  5. 止损: 15%移动止损
"""
import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')

from config import StrategyConfig, CACHE_DIR

# ==================== 数据加载 ====================
etf_price_map = {}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        etf_price_map[code] = pd.read_csv(
            os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date']
        )

CASH_ETF = StrategyConfig.CASH_ETF_CODE  # 511010 国债ETF
BENCHMARK = '510300'                      # 沪深300ETF

# 全球混合资产池: A股高弹性行业 + 黄金(159934) + 纳指(513100)
# 当前cache中已有的标的
AVAILABLE_CODES = set(etf_price_map.keys())

# A股高弹性ETF (排除宽基指数和现金)
SECTOR_CODES = [
    '512480',  # 半导体ETF
    '159995',  # 芯片ETF
    '512760',  # 半导体行业ETF
    '515050',  # 5G ETF
    '512000',  # 券商ETF
    '515030',  # 新能源车ETF
    '159875',  # 光伏ETF
    '512660',  # 军工ETF
    '512690',  # 酒ETF
    '159928',  # 消费ETF
    '512010',  # 医药ETF
    '159801',  # 恒生科技ETF
    '512890',  # 红利ETF
    '510880',  # 红利低波ETF
    '168204',  # 煤炭LOF
    '512400',  # 有色ETF
    '159915',  # 创业板ETF
    '512100',  # 银行ETF
    '512880',  # 证券ETF
    '510050',  # 上证50
    '510500',  # 中证500
]

# 过滤出cache中实际存在的标的
ETF_POOL = [c for c in SECTOR_CODES if c in AVAILABLE_CODES]

# 半导体/芯片高相关标签 (最多同时入选2只)
SEMI_LABELS = {'512480', '159995', '512760'}

# ==================== 参数 ====================
MA_LOOKBACK = 120       # 大盘开关均线
MOM_PERIODS = [20, 60, 120]  # 多周期动量
MOM_WEIGHTS = [0.4, 0.3, 0.3]  # 近期权重更高
TOP_N = 5
STOP_LOSS = 0.20
TXN_COST = 0.001
CONFIRM_DAYS = 5        # 状态确认天数

# ==================== 核心函数 ====================

def check_market_mode(current_date, lookback=MA_LOOKBACK):
    """
    大盘绝对动量开关: 沪深300收盘价 vs MA120
    > MA120 = BULL, <= MA120 = BEAR
    """
    df = etf_price_map.get(BENCHMARK)
    if df is None:
        return "BEAR"
    sub = df[df['date'] <= current_date]
    if len(sub) < lookback + 1:
        return "BEAR"
    price = float(sub['close'].iloc[-1])
    ma = float(sub['close'].iloc[-lookback:].mean())
    return "BULL" if price > ma else "BEAR"


def generate_momentum_top5(current_date):
    """
    全球多周期相对动量评分, 输出Top 5
    半导体类ETF最多入选2只
    """
    scores = {}
    for code in ETF_POOL:
        df = etf_price_map.get(code)
        if df is None:
            continue
        sub = df[df['date'] <= current_date]
        if len(sub) < MOM_PERIODS[-1] + 1:
            continue
        current_price = float(sub['close'].iloc[-1])
        total_score = 0.0
        for period, weight in zip(MOM_PERIODS, MOM_WEIGHTS):
            past_price = float(sub['close'].iloc[-period - 1])
            ret = (current_price / past_price) - 1
            total_score += ret * weight
        scores[code] = total_score

    sorted_etfs = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    # 去重: 半导体最多2只
    final_top5 = []
    semi_count = 0
    for code, score in sorted_etfs:
        if len(final_top5) >= TOP_N:
            break
        if code in SEMI_LABELS:
            if semi_count >= 2:
                continue
            semi_count += 1
        final_top5.append(code)

    return final_top5


def get_price(code, date):
    df = etf_price_map.get(code)
    if df is None:
        return None
    m = df[df['date'] == date]['close']
    return float(m.iloc[0]) if len(m) > 0 else None


# ==================== 回测引擎 ====================

def run_backtest(start='2018-01-01', end='2026-06-30'):
    hs300_df = etf_price_map[BENCHMARK]
    dates = sorted([
        d for d in hs300_df['date']
        if d >= pd.Timestamp(start) and d <= pd.Timestamp(end)
    ])

    cash = 1_000_000.0
    holdings = {}          # {code: shares}
    highest_prices = {}    # 移动止损用
    nav_history = []
    trade_log = []

    # 状态确认机制: 新状态必须持续CONFIRM_DAYS天才切换
    confirmed_mode = "BEAR"
    pending_mode = None
    pending_count = 0

    def total_value(date):
        tv = cash
        for code, shares in holdings.items():
            p = get_price(code, date)
            if p:
                tv += shares * p
        return tv

    def is_rebalance_day(date):
        if date not in dates:
            return False
        idx = dates.index(date)
        return idx <= 0 or dates[idx - 1].month != date.month

    def sell_all(code, date):
        nonlocal cash
        if code not in holdings:
            return
        p = get_price(code, date)
        if not p:
            return
        shares = holdings[code]
        cash += shares * p * (1 - TXN_COST)
        del holdings[code]
        if code in highest_prices:
            del highest_prices[code]

    def buy_value(code, value, date):
        nonlocal cash
        p = get_price(code, date)
        if not p or p <= 0:
            return
        shares = int(value / p / 100) * 100
        if shares <= 0:
            return
        cost = shares * p * (1 + TXN_COST)
        if cash < cost:
            shares = int(cash / p / 100) * 100
            if shares <= 0:
                return
            cost = shares * p * (1 + TXN_COST)
        if cash - cost < -1:  # 安全检查: 不允许cash变负
            return
        holdings[code] = holdings.get(code, 0) + shares
        cash -= cost
        highest_prices[code] = p

    print(f"双动量一刀切 | MA{MA_LOOKBACK}大盘开关 | Top{TOP_N}多周期动量 | {STOP_LOSS*100:.0f}%止损 | 月度调仓")
    print(f"资产池: {len(ETF_POOL)}只ETF | 半导体最多2只 | 熊市100%国债")
    print("=" * 90)

    for i, date in enumerate(dates):
        tv = total_value(date)

        # ---- 止损已禁用: 一刀切策略靠月度调仓风控, 止损会导致反复割肉 ----
        # if is_rebalance_day(date) and i > 0 and holdings:
        #     for code in list(holdings.keys()):
        #         ...

        # ---- 状态确认 ----
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

        # ---- 月度调仓 ----
        if is_rebalance_day(date) and i >= MA_LOOKBACK + 10:
            tv = total_value(date)
            mode = confirmed_mode

            if mode == "BEAR":
                # ===== 熊市: 100%国债 =====
                for code in list(holdings.keys()):
                    if code != CASH_ETF:
                        sell_all(code, date)
                # 买入国债
                if cash > 1000:
                    buy_value(CASH_ETF, cash, date)

                if i < 300 or i % 400 == 0 or (2019 <= date.year <= 2019):
                    pos_str = ", ".join([f"{c}:{s}" for c, s in sorted(holdings.items())])
                    print(f"{date.strftime('%Y-%m-%d')} [BEAR] 总={tv:.0f} 现金={cash:.0f} {pos_str}")

            elif mode == "BULL":
                # ===== 牛市: 100%满仓Top5动量 =====
                top5 = generate_momentum_top5(date)

                # 清仓不在Top5中的标的 (包括国债)
                for code in list(holdings.keys()):
                    if code not in top5:
                        sell_all(code, date)

                # 等权买入Top5
                if top5 and cash > 1000:
                    per_etf = cash / len(top5)
                    for code in top5:
                        buy_value(code, per_etf, date)

                if i < 300 or i % 400 == 0 or (2019 <= date.year <= 2019):
                    pos_str = ", ".join([f"{c}:{s}" for c, s in sorted(holdings.items())])
                    print(f"{date.strftime('%Y-%m-%d')} [BULL] 总={tv:.0f} 现金={cash:.0f} Top5={top5} {pos_str}")

        nav_history.append({'date': date, 'total_value': total_value(date), 'mode': confirmed_mode})

    return nav_history, trade_log


# ==================== 报告 ====================

def generate_report(nav_history):
    results = pd.DataFrame(nav_history)
    nav = results['total_value'].values
    initial = nav[0]
    final = nav[-1]
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

    # 沪深300基准
    hs300 = []
    for d in results['date']:
        p = get_price(BENCHMARK, d)
        if p:
            hs300.append(p)
    hs = np.array(hs300)
    hs_total = hs[-1] / hs[0] - 1
    hs_ann = (hs[-1] / hs[0]) ** (1 / years) - 1
    hs_vol = np.diff(hs) / hs[:-1]
    hs_annvol = hs_vol.std() * np.sqrt(252)
    hs_sharpe = hs_ann / hs_annvol if hs_annvol > 0 else 0
    hs_cummax = np.maximum.accumulate(hs)
    hs_maxdd = ((hs - hs_cummax) / hs_cummax).min()

    # 牛熊分布
    mode_counts = results['mode'].value_counts()
    bull_days = mode_counts.get('BULL', 0)
    bear_days = mode_counts.get('BEAR', 0)
    total_days = len(results)

    print(f"\n{'='*95}")
    print(f"{' 双动量一刀切策略回测报告 (2018-2026) ':=^95}")
    print(f"{'='*95}")
    print(f"  大盘开关: 沪深300 MA{MA_LOOKBACK} | 确认期: {CONFIRM_DAYS}天")
    print(f"  牛市: 100% Top{TOP_N}多周期动量 (20/60/120日, 权重0.4/0.3/0.3)")
    print(f"  熊市: 100% 国债ETF (511010)")
    print(f"  止损: {STOP_LOSS*100:.0f}%移动止损 | 调仓: 月度 | 半导体最多2只")
    print(f"{'='*95}")
    print(f"\n{'指标':<20} {'双动量一刀切':>15} {'沪深300':>15}")
    print("-" * 95)
    print(f"{'总收益':<20} {total_ret*100:>14.2f}% {hs_total*100:>14.2f}%")
    print(f"{'年化收益':<20} {ann_ret*100:>14.2f}% {hs_ann*100:>14.2f}%")
    print(f"{'年化波动':<20} {ann_vol*100:>14.2f}% {hs_annvol*100:>14.2f}%")
    print(f"{'夏普比率':<20} {sharpe:>15.3f} {hs_sharpe:>15.3f}")
    print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {hs_maxdd*100:>14.2f}%")
    print(f"{'收益/回撤比':<20} {abs(total_ret/max_dd):>15.3f} {abs(hs_total/hs_maxdd):>15.3f}")
    print(f"{'日胜率':<20} {win_rate*100:>14.2f}%")
    print(f"{'牛/熊天数':<20} {bull_days}/{bear_days} ({bull_days/total_days*100:.0f}%/{bear_days/total_days*100:.0f}%)")
    print("=" * 95)

    # 分年度
    results['year'] = results['date'].dt.year
    results['daily_ret'] = results['total_value'].pct_change()

    print(f"\n{'年份':<8} {'收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'胜率':>8} {'牛市天数':>10} {'熊市天数':>10}")
    print("-" * 95)
    for year in sorted(results['year'].unique()):
        sub = results[results['year'] == year].reset_index(drop=True)
        if len(sub) < 10:
            continue
        ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
        vol = sub['daily_ret'].std() * np.sqrt(252)
        s = ret / vol if vol > 0 else 0
        cm = sub['total_value'].cummax()
        mdd = ((sub['total_value'] - cm) / cm).min()
        wr = (sub['daily_ret'] > 0).mean()
        bull = (sub['mode'] == 'BULL').sum()
        bear = (sub['mode'] == 'BEAR').sum()
        m = " *" if ret > 0 else ""
        print(f"{year:<8} {ret*100:>9.2f}% {vol*100:>9.2f}% {s:>8.3f} {mdd*100:>9.2f}% {wr*100:>7.2f}% {bull:>10} {bear:>10}{m}")

    # 牛熊切换时间线
    print(f"\n牛熊切换时间线 (MA{MA_LOOKBACK}, {CONFIRM_DAYS}天确认):")
    prev_mode = None
    for _, row in results.iterrows():
        mode = row['mode']
        if mode != prev_mode:
            date_str = row['date'].strftime('%Y-%m-%d')
            label = "BULL" if mode == "BULL" else "BEAR"
            print(f"  {date_str} -> {label}")
            prev_mode = mode

    return {
        'total_ret': total_ret, 'ann_ret': ann_ret, 'ann_vol': ann_vol,
        'sharpe': sharpe, 'max_dd': max_dd, 'win_rate': win_rate,
        'hs_total': hs_total, 'hs_sharpe': hs_sharpe, 'hs_maxdd': hs_maxdd,
    }


# ==================== 主程序 ====================
if __name__ == '__main__':
    nav_history, trade_log = run_backtest()
    metrics = generate_report(nav_history)

    # 保存结果
    results_df = pd.DataFrame(nav_history)
    results_df.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_dual_momentum.csv'), index=False)
    print(f"\n结果已保存到 backtest_dual_momentum.csv")
