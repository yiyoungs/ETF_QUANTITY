"""v7: 平衡动量策略
- 宏观择时: 510300 > MA200 → 权益目标 70%；否则 20%
- 动量: 120日动量 > 0
- 趋势过滤: 价格 > MA60
- 持仓: Top 5 动量ETF 等权
- 月度调仓 + 15% 单ETF止损
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
etf_codes = [c for c in etf_price_map.keys() if c != cash_etf]

hs300_df = etf_price_map['510300']
dates = sorted([d for d in hs300_df['date'] if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

MOM_WINDOW = 120
MA_TREND = 60
TOP_N = 5
STOP_LOSS = 0.15
TXN_COST = 0.001
BULL_RATIO = 0.70  # 牛市 70% 权益
BEAR_RATIO = 0.20  # 熊市 20% 权益

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

print("v7: 宏观择时(MA200) + ETF动量(120日) + 趋势过滤(MA60) + TOP5")
print(f"    权益目标: 牛市 {BULL_RATIO*100:.0f}% / 熊市 {BEAR_RATIO*100:.0f}%")
print("=" * 80)

for i, date in enumerate(dates):
    current_total = get_total(date)

    # 止损检查
    if i > 0 and positions:
        stop_codes = []
        for code in list(positions.keys()):
            if code == cash_etf:
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
        # 1. 宏观择时: 510300 vs MA200
        core_sub = hs300_df[hs300_df['date'] <= date]
        if len(core_sub) >= 200:
            ma200 = float(core_sub['close'].iloc[-200:].mean())
            core_price = float(core_sub['close'].iloc[-1])
            equity_ratio = BULL_RATIO if core_price > ma200 else BEAR_RATIO
        else:
            equity_ratio = BEAR_RATIO

        # 2. 计算动量候选
        candidates = []
        for code in etf_codes:
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
        targets = [c[0] for c in candidates[:TOP_N]]

        total_value = get_total(date)

        # 3. 全仓卖出所有风险资产
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

        # 4. 卖出多余国债ETF
        target_equity_value = total_value * equity_ratio
        target_cash_etf_value = total_value * (1 - equity_ratio)

        if cash_etf in positions:
            price = get_current_price(cash_etf, date)
            if price is not None and price > 0:
                current_value = positions[cash_etf] * price
                if current_value > target_cash_etf_value * 1.05:
                    excess_shares = int((current_value - target_cash_etf_value) / price / 100) * 100
                    if excess_shares > 0:
                        cash += excess_shares * price * (1 - TXN_COST)
                        positions[cash_etf] -= excess_shares
                        if positions[cash_etf] <= 0:
                            del positions[cash_etf]

        # 5. 买入目标ETF
        if targets and cash > 1000:
            per_budget = target_equity_value / len(targets)
            for code in targets:
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

        # 6. 剩余现金 -> 国债ETF
        if cash > 1000:
            price = get_current_price(cash_etf, date)
            if price is not None and price > 0:
                shares = int(cash * 0.98 / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = positions.get(cash_etf, 0) + shares
                        cash -= cost

        if i < 500 or i % 500 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"{date.strftime('%Y-%m-%d')} 权益{equity_ratio*100:.0f}%: 总={current_total:.0f}, 现金={cash:.0f}, {pos_detail}")

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

print(f"\n\n{'指标':<20} {'v7 平衡动量':>15} {'沪深300':>15}")
print("=" * 70)
print(f"{'总资产 (最终)':<20} {final:>15,.0f} {hs300_nav_clean[-1]*initial/hs300_nav_clean[0]:>15,.0f}")
print(f"{'总收益':<20} {total_return*100:>14.2f}% {hs300_total*100:>14.2f}%")
print(f"{'年化收益':<20} {annualized_return*100:>14.2f}% {hs300_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {annualized_vol*100:>14.2f}% {hs300_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {hs300_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {hs300_maxdd*100:>14.2f}%")
print(f"{'日胜率':<20} {win_rate*100:>14.2f}%")
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

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v7.csv'), index=False)
