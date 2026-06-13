"""v6: 核心-卫星策略 (Core-Satellite)
核心 (Core): 510300 (沪深300 ETF) - 40% 固定配置
卫星 (Satellite): top 3 动量行业ETF - 各 20% (合计 60%)
趋势过滤: 单ETF 必须 价格 > MA60 且 60日动量 > 0
月度调仓 + 15% 单ETF止损
无合格标的时: 持有国债ETF
"""
import pandas as pd
import numpy as np
import os

from config import StrategyConfig, CACHE_DIR

etf_price_map = {}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        etf_price_map[code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])

cash_etf = StrategyConfig.CASH_ETF_CODE
core_etf = '510300'  # 沪深300 - 核心
# 卫星ETF: 行业/主题 ETF (排除宽基510050/510300/510500和现金)
satellite_codes = [c for c in etf_price_map.keys() if c not in [cash_etf, core_etf, '510050', '510500', '510880']]

# 实际上: 510050=上证50, 510880=红利, 510500=中证500 - 这些也可以作为卫星
# 让我把510050, 510880, 510500 也加入卫星池
satellite_codes = [c for c in etf_price_map.keys() if c not in [cash_etf, core_etf]]

hs300_df = etf_price_map[core_etf]
dates = sorted([d for d in hs300_df['date'] if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

MA_TREND = 60
MOM_WINDOW = 60
TOP_N = 3
STOP_LOSS = 0.15
TXN_COST = 0.001
CORE_RATIO = 0.40  # 40% 沪深300核心
SATELLITE_RATIO = 1.0 - CORE_RATIO  # 60% 卫星

cash = float(StrategyConfig.INITIAL_CAPITAL)
positions = {}
highest_prices = {}
nav_history = []

def get_current_price(code, date):
    df = etf_price_map.get(code)
    if df is None:
        return None
    p = df[df['date'] == date]['close']
    if len(p) == 0:
        return None
    return float(p.iloc[0])

def get_total(date):
    total = cash
    for code, shares in positions.items():
        price = get_current_price(code, date)
        if price is not None:
            total += shares * price
    return total

def is_first_trading(date):
    if date not in dates:
        return False
    idx = dates.index(date)
    if idx <= 0:
        return True
    return dates[idx - 1].month != date.month

print("v6: Core-Satellite (40% 沪深300 + 60% Top3 动量行业)")
print(f"    MA{MA_TREND}趋势过滤 + {STOP_LOSS*100:.0f}%止损 + 月度调仓")
print("=" * 80)

for i, date in enumerate(dates):
    current_total = get_total(date)

    # 止损检查（只对卫星ETF）
    if i > 0 and positions:
        stop_codes = []
        for code in list(positions.keys()):
            if code == cash_etf or code == core_etf:
                continue
            price = get_current_price(code, date)
            if price is None:
                continue
            if code not in highest_prices or price > highest_prices[code]:
                highest_prices[code] = price
            dd = (price - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
            if dd <= -STOP_LOSS:
                stop_codes.append(code)

        for code in stop_codes:
            price = get_current_price(code, date)
            if price is None:
                continue
            shares = positions[code]
            cash += shares * price * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= MOM_WINDOW + 20:
        # 计算卫星候选
        candidates = []
        for code in satellite_codes:
            df = etf_price_map.get(code)
            if df is None:
                continue
            sub = df[df['date'] <= date]
            if len(sub) < MOM_WINDOW + 1:
                continue
            t1_price = float(sub['close'].iloc[-1])
            mom_price = float(sub['close'].iloc[-MOM_WINDOW - 1])
            ma60 = float(sub['close'].iloc[-MA_TREND:].mean())
            momentum = t1_price / mom_price - 1
            if t1_price > ma60 and momentum > 0:
                candidates.append((code, momentum))

        candidates.sort(key=lambda x: x[1], reverse=True)
        satellite_targets = [c[0] for c in candidates[:TOP_N]]

        total_value = get_total(date)

        # 卖出所有现有持仓（除了核心）
        for code in list(positions.keys()):
            if code == cash_etf:
                continue
            price = get_current_price(code, date)
            if price is None:
                continue
            shares = positions[code]
            cash += shares * price * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 分配
        # 核心: 40% -> 510300 (若510300自己也通过简单趋势过滤)
        core_df = etf_price_map[core_etf]
        core_sub = core_df[core_df['date'] <= date]
        buy_core = True
        if len(core_sub) >= 200:
            # 核心也做简单过滤: 510300 是否在 MA200 之上
            ma200 = float(core_sub['close'].iloc[-200:].mean())
            core_price = float(core_sub['close'].iloc[-1])
            if core_price < ma200:
                buy_core = False  # 熊市时降低核心暴露

        if buy_core:
            core_budget = total_value * CORE_RATIO
            price = get_current_price(core_etf, date)
            if price is not None and price > 0:
                shares = int(core_budget / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[core_etf] = shares
                        cash -= cost
                        highest_prices[core_etf] = price

        # 卫星: 等权分配到 top N
        if satellite_targets:
            sat_total = total_value * SATELLITE_RATIO
            per_budget = sat_total / len(satellite_targets)
            for code in satellite_targets:
                price = get_current_price(code, date)
                if price is None or price <= 0:
                    continue
                shares = int(per_budget / price / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * price * (1 + TXN_COST)
                if cash < cost:
                    shares = int(cash / price / 100) * 100
                    if shares <= 0:
                        break
                    cost = shares * price * (1 + TXN_COST)
                positions[code] = shares
                cash -= cost
                highest_prices[code] = price

        # 剩余现金 -> 国债ETF
        if cash > 1000:
            price = get_current_price(cash_etf, date)
            if price is not None and price > 0:
                shares = int(cash * 0.98 / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = shares
                        cash -= cost

        if i < 300 or i % 500 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"{date.strftime('%Y-%m-%d')}: 总={current_total:.0f}, 现金={cash:.0f}, {pos_detail}")

    nav_history.append({'date': date, 'total_value': current_total})

# 计算指标
results = pd.DataFrame(nav_history)
nav = results['total_value'].values
initial = nav[0]
final = nav[-1]
years = (dates[-1] - dates[0]).days / 365.25
total_return = final / initial - 1

daily_returns = pd.Series(nav).pct_change().dropna().values
annualized_return = (final / initial) ** (1 / years) - 1
annualized_vol = daily_returns.std() * np.sqrt(252)
sharpe = annualized_return / annualized_vol if annualized_vol > 0 else 0

cummax = pd.Series(nav).cummax()
drawdown = (nav - cummax) / cummax
max_dd = drawdown.min()
win_rate = (daily_returns > 0).mean()

# 沪深300基准
hs300_prices = [float(hs300_df[hs300_df['date'] == d]['close'].iloc[0]) if len(hs300_df[hs300_df['date'] == d]) > 0 else None for d in dates]
hs300_nav_clean = [p for p in hs300_prices if p is not None]
hs300_total = hs300_nav_clean[-1] / hs300_nav_clean[0] - 1
hs300_daily = pd.Series(hs300_nav_clean).pct_change().dropna().values
hs300_years = (dates[-1] - dates[0]).days / 365.25
hs300_ann = (hs300_nav_clean[-1] / hs300_nav_clean[0]) ** (1 / hs300_years) - 1
hs300_vol = hs300_daily.std() * np.sqrt(252)
hs300_sharpe = hs300_ann / hs300_vol if hs300_vol > 0 else 0
hs300_cummax = pd.Series(hs300_nav_clean).cummax()
hs300_dd = (hs300_nav_clean - hs300_cummax) / hs300_cummax
hs300_maxdd = hs300_dd.min()

print(f"\n\n{'指标':<20} {'v6 Core-Satellite':>20} {'沪深300':>15}")
print("=" * 70)
print(f"{'总资产 (最终)':<20} {final:>20,.0f} {hs300_nav_clean[-1]*initial/hs300_nav_clean[0]:>15,.0f}")
print(f"{'总收益':<20} {total_return*100:>19.2f}% {hs300_total*100:>14.2f}%")
print(f"{'年化收益':<20} {annualized_return*100:>19.2f}% {hs300_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {annualized_vol*100:>19.2f}% {hs300_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>20.3f} {hs300_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>19.2f}% {hs300_maxdd*100:>14.2f}%")
print(f"{'日胜率':<20} {win_rate*100:>19.2f}%")
print("=" * 70)

results['year'] = results['date'].dt.year
results['daily_ret'] = results['total_value'].pct_change()

print(f"\n{'年份':<8} {'交易日':>7} {'年度收益':>10} {'年化波动':>10} {'夏普':>8} {'最大回撤':>10} {'日胜率':>8}")
print("-" * 85)
for year in sorted(results['year'].unique()):
    sub = results[results['year'] == year].reset_index(drop=True)
    if len(sub) < 10:
        continue
    ret = sub['total_value'].iloc[-1] / sub['total_value'].iloc[0] - 1
    vol = sub['daily_ret'].std() * np.sqrt(252)
    sharpe_y = ret / vol if vol > 0 else 0
    cummax = sub['total_value'].cummax()
    dd = ((sub['total_value'] - cummax) / cummax).min()
    wr = (sub['daily_ret'] > 0).mean()
    print(f"{year:<8} {len(sub):>7} {ret*100:>9.2f}% {vol*100:>9.2f}% {sharpe_y:>8.3f} {dd*100:>9.2f}% {wr*100:>7.2f}%")

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v6.csv'), index=False)
