"""v5: 100% 风险资产策略 - 与沪深300同风险等级
- 动量：120日动量 - 过滤负动量
- 趋势：价格 > MA60
- 配置：全仓等权 top5 动量ETF，无任何现金缓冲
- 无合格标的时：100% 国债
- 月度调仓 + 15% 单ETF止损
"""
import pandas as pd
import numpy as np
import os

from config import StrategyConfig, CACHE_DIR

# 加载数据
etf_price_map = {}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        etf_price_map[code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])

cash_etf = StrategyConfig.CASH_ETF_CODE
etf_codes = [c for c in etf_price_map.keys() if c != cash_etf]

# 交易日
hs300_df = etf_price_map['510300']
dates = sorted([d for d in hs300_df['date'] if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

LOOKBACK_LONG = 120
MA_WINDOW = 60
TOP_N = 5
STOP_LOSS = 0.15
TXN_COST = 0.001

# 回测状态
cash = float(StrategyConfig.INITIAL_CAPITAL)
positions = {}
highest_prices = {}
nav_history = []
rebalance_count = 0

def get_price_at(code, date, n):
    df = etf_price_map.get(code)
    if df is None:
        return None
    sub = df[df['date'] <= date]
    if len(sub) < n + 1:
        return None
    return float(sub['close'].iloc[-n - 1])  # -1 = T-1

def get_ma(code, date, window):
    df = etf_price_map.get(code)
    if df is None:
        return None
    sub = df[df['date'] <= date]
    if len(sub) < window:
        return None
    return float(sub['close'].iloc[-window - 1:-1].mean())

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

print("v5: 100% 风险资产 - 120日动量 + MA60趋势过滤 + TOP5 + 15%止损")
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
    if is_first_trading(date) and i >= LOOKBACK_LONG + 10:
        # 计算候选ETF
        candidates = []
        for code in etf_codes:
            current_price = get_price_at(code, date, 0)  # T-1 close (iloc[-0-1]=-1)
            # 等等，我们需要 T-1 收盘价 - get_price_at(code, date, 0) 用 iloc[-1] 正好是 T-1
            # 这里"当前"用 iloc[-1] 就是 T-1（sub 是 <= date，取最后一个就是 date 自身）
            # 让我改一下：
            pass

        # 重新计算：用 T-1 收盘
        candidates = []
        for code in etf_codes:
            df = etf_price_map.get(code)
            if df is None:
                continue
            sub = df[df['date'] <= date]
            if len(sub) < LOOKBACK_LONG + 1:
                continue
            # T-1 收盘
            t1_price = float(sub['close'].iloc[-1])
            # 120日前收盘
            long_price = float(sub['close'].iloc[-LOOKBACK_LONG - 1])
            # MA60
            if len(sub) < MA_WINDOW:
                continue
            ma60 = float(sub['close'].iloc[-MA_WINDOW:].mean())
            momentum = t1_price / long_price - 1
            if t1_price > ma60 and momentum > 0:
                candidates.append((code, momentum))

        candidates.sort(key=lambda x: x[1], reverse=True)
        target = [c[0] for c in candidates[:TOP_N]]

        total_value = get_total(date)

        # 全仓卖出所有风险资产
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

        # 全仓卖出国债ETF
        if cash_etf in positions:
            price = get_current_price(cash_etf, date)
            if price is not None:
                shares = positions[cash_etf]
                cash += shares * price * (1 - TXN_COST)
                del positions[cash_etf]

        # 全仓买入目标ETF
        if target and cash > 1000:
            per_budget = cash * 0.98 / len(target)
            for code in target:
                price = get_current_price(code, date)
                if price is None or price <= 0:
                    continue
                shares = int(per_budget / price / 100) * 100
                if shares <= 0:
                    continue
                cost = shares * price * (1 + TXN_COST)
                if cost > cash:
                    shares = int(cash / price / 100) * 100
                    if shares <= 0:
                        break
                    cost = shares * price * (1 + TXN_COST)
                positions[code] = shares
                cash -= cost
                highest_prices[code] = price

        # 没目标时全仓国债
        if not target and cash > 1000:
            price = get_current_price(cash_etf, date)
            if price is not None and price > 0:
                shares = int(cash * 0.98 / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = shares
                        cash -= cost

        # 剩余现金兜底买国债
        if cash > 1000:
            price = get_current_price(cash_etf, date)
            if price is not None and price > 0:
                shares = int(cash * 0.98 / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = positions.get(cash_etf, 0) + shares
                        cash -= cost

        rebalance_count += 1
        if rebalance_count <= 3 or rebalance_count % 12 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"调仓#{rebalance_count} {date.strftime('%Y-%m-%d')}: 总={current_total:.0f}, 现金={cash:.0f}, {pos_detail}")

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

print(f"\n\n{'指标':<20} {'v5 全仓动量':>15} {'沪深300':>15}")
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

results.to_csv(os.path.join(CACHE_DIR, '..', 'backtest_v5.csv'), index=False)
