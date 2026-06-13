"""策略 v4: 简化版 - 经典动量策略
- 动量窗: 120-20日动量
- 只买价格 > MA60 的ETF
- 月度调仓
- 单只ETF 独立止损
- 全仓投资（无现金缓冲，30%国债ETF作为现金替代）
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
        df = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        etf_price_map[code] = df

# ETF清单
cash_etf = StrategyConfig.CASH_ETF_CODE
etf_codes = [c for c in etf_price_map.keys() if c != cash_etf]

# 沪深300 交易日
hs300_df = etf_price_map.get('510300')
dates = sorted([d for d in hs300_df['date'] if d >= pd.Timestamp('2018-01-01') and d <= pd.Timestamp('2026-06-30')])

LOOKBACK_LONG = 120
LOOKBACK_SHORT = 20
MA_WINDOW = 60
TOP_N = 5
STOP_LOSS = 0.15
TXN_COST = 0.001

# 为每只ETF 建立价格序列
def get_price_at(code, date, n):
    df = etf_price_map.get(code)
    if df is None:
        return None
    sub = df[df['date'] <= date]
    if len(sub) < n + 1:
        return None
    return float(sub['close'].iloc[-n])

def get_ma(code, date, window):
    df = etf_price_map.get(code)
    if df is None:
        return None
    sub = df[df['date'] <= date]
    if len(sub) < window:
        return None
    return float(sub['close'].iloc[-window:].mean())

# 回测
cash = float(StrategyConfig.INITIAL_CAPITAL)
positions = {}
highest_prices = {}
nav_history = []
rebalance_count = 0

def get_total(date):
    total = cash
    for code, shares in positions.items():
        df = etf_price_map.get(code)
        if df is None:
            continue
        p = df[df['date'] == date]['close']
        if len(p) > 0:
            total += shares * float(p.iloc[0])
    return total

def is_first_trading(date):
    if date not in dates:
        return False
    idx = dates.index(date)
    if idx <= 0:
        return True
    return dates[idx - 1].month != date.month

print(f"v4: 120-20日动量 + MA60趋势过滤 + TOP5 + 15%止损")
print("=" * 80)

for i, date in enumerate(dates):
    current_total = get_total(date)

    # 止损
    if i > 0 and positions:
        stop_codes = []
        for code in list(positions.keys()):
            if code == cash_etf:
                continue
            df = etf_price_map.get(code)
            if df is None:
                continue
            p = df[df['date'] == date]['close']
            if len(p) == 0:
                continue
            price = float(p.iloc[0])
            if code not in highest_prices or price > highest_prices[code]:
                highest_prices[code] = price
            dd = (price - highest_prices[code]) / highest_prices[code] if highest_prices[code] > 0 else 0
            if dd <= -STOP_LOSS:
                stop_codes.append(code)

        for code in stop_codes:
            df = etf_price_map.get(code)
            p = df[df['date'] == date]['close']
            if len(p) == 0:
                continue
            price = float(p.iloc[0])
            shares = positions[code]
            cash += shares * price * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

    # 月度调仓
    if is_first_trading(date) and i >= 130:
        candidates = []
        for code in etf_codes:
            # 当前价格（使用T-1避免lookahead）
            current_price = get_price_at(code, date, 1)  # T-1
            if current_price is None:
                continue
            price_20 = get_price_at(code, date, LOOKBACK_SHORT + 1)
            price_120 = get_price_at(code, date, LOOKBACK_LONG + 1)
            if price_20 is None or price_120 is None:
                continue
            # MA60
            ma60 = get_ma(code, date, MA_WINDOW)
            if ma60 is None:
                continue
            # 动量
            mom_20 = current_price / price_20 - 1
            mom_120 = current_price / price_120 - 1
            score = 0.3 * mom_20 + 0.7 * mom_120
            if current_price > ma60 and score > 0:
                candidates.append((code, score))

        candidates.sort(key=lambda x: x[1], reverse=True)
        target = [c[0] for c in candidates[:TOP_N]]

        total_value = get_total(date)

        # 全仓卖出非目标持仓
        for code in list(positions.keys()):
            if code == cash_etf or code in target:
                continue
            df = etf_price_map.get(code)
            if df is None:
                continue
            p = df[df['date'] == date]['close']
            if len(p) == 0:
                continue
            price = float(p.iloc[0])
            shares = positions[code]
            cash += shares * price * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 确定买入预算
        # 剩余目标持仓需要调整权重 = 全仓卖出再统一重新买入
        # 先检查是否还有持仓（如果是target的一部分，需保持）
        # 简化：先全仓卖出所有，再重新买
        for code in list(positions.keys()):
            if code == cash_etf:
                continue
            df = etf_price_map.get(code)
            p = df[df['date'] == date]['close']
            if len(p) == 0:
                continue
            price = float(p.iloc[0])
            shares = positions[code]
            cash += shares * price * (1 - TXN_COST)
            del positions[code]
            if code in highest_prices:
                del highest_prices[code]

        # 全仓买入目标ETF
        if target and cash > 1000:
            # 等权配置
            per_budget = cash * 0.98 / len(target)
            for code in target:
                df = etf_price_map.get(code)
                p = df[df['date'] == date]['close']
                if len(p) == 0:
                    continue
                price = float(p.iloc[0])
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

        # 剩余现金买入国债ETF
        remaining = cash * 0.98
        if remaining > 1000:
            df = etf_price_map.get(cash_etf)
            p = df[df['date'] == date]['close']
            if len(p) > 0:
                price = float(p.iloc[0])
                shares = int(remaining / price / 100) * 100
                if shares > 0:
                    cost = shares * price * (1 + TXN_COST)
                    if cash >= cost:
                        positions[cash_etf] = positions.get(cash_etf, 0) + shares
                        cash -= cost

        rebalance_count += 1
        if rebalance_count <= 3 or rebalance_count % 12 == 0:
            pos_detail = ", ".join([f"{c}:{s}" for c, s in sorted(positions.items())])
            print(f"调仓#{rebalance_count} {date.strftime('%Y-%m-%d')}: 总={current_total:.0f}, 现金={cash:.0f}, {pos_detail}")

    nav_history.append({
        'date': date,
        'total_value': current_total,
    })

print(f"\n回测完成: {rebalance_count} 次调仓")

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

print("\n" + "=" * 70)
print(f"{'指标':<20} {'v4 简化动量':>15} {'沪深300':>15}")
print("=" * 70)
print(f"{'总资产 (最终)':<20} {final:>15,.0f} {hs300_nav_clean[-1]*initial/hs300_nav_clean[0]:>15,.0f}")
print(f"{'总收益':<20} {total_return*100:>14.2f}% {hs300_total*100:>14.2f}%")
print(f"{'年化收益':<20} {annualized_return*100:>14.2f}% {hs300_ann*100:>14.2f}%")
print(f"{'年化波动':<20} {annualized_vol*100:>14.2f}% {hs300_vol*100:>14.2f}%")
print(f"{'夏普比率':<20} {sharpe:>15.3f} {hs300_sharpe:>15.3f}")
print(f"{'最大回撤':<20} {max_dd*100:>14.2f}% {hs300_maxdd*100:>14.2f}%")
print(f"{'日胜率':<20} {win_rate*100:>14.2f}%")
print("=" * 70)

# 分年度
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
