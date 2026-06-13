import pandas as pd
import os
from config import CACHE_DIR, StrategyConfig
from portfolio_manager import PortfolioManager

# 加载数据
all_data = {'qfq': {}, 'hfq': {}, 'dividend': {}}
for fname in sorted(os.listdir(CACHE_DIR)):
    if fname.endswith('_hfq.csv'):
        code = fname.split('_')[0]
        all_data['hfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_hfq.csv'), parse_dates=['date'])
        all_data['qfq'][code] = pd.read_csv(os.path.join(CACHE_DIR, code + '_qfq.csv'), parse_dates=['date'])

etf_pool = pd.DataFrame({'code': list(all_data['hfq'].keys())})
pm = PortfolioManager()

test_dates = ['2019-06-21', '2020-07-03', '2025-06-06']

for d in test_dates:
    print(f"\n=== [{d}] 调仓前 cash = {pm.cash:.0f} ===")
    hs300 = all_data['qfq']['510300']
    idx = hs300[hs300.date <= pd.to_datetime(d)]
    if len(idx) >= 202:
        close_prev = float(idx['close'].iloc[-2])
        ma200 = float(idx['close'].rolling(200).mean().iloc[-2])
        deviation = (close_prev - ma200) / ma200
        mode = 'bull' if close_prev > ma200 else 'bear'
        if deviation < 0.02:
            eq = 0.5
        elif deviation < 0.05:
            eq = 0.8
        else:
            eq = 1.0 if deviation >= 0 else 0.0
        print(f"  mode={mode}, equity_ratio={eq}, 510300 close={close_prev:.2f}, MA200={ma200:.2f}")

    pm.rebalance(all_data, etf_pool, d, market_mode=mode, equity_ratio=eq)
    print(f"  调仓后 cash = {pm.cash:.0f}")
    print(f"  持仓: {pm.positions}")
    total_val = pm.get_total_value(all_data, d)
    print(f"  总资产 = {total_val:.0f}")

    # 拆分资产
    risk_val = 0
    bond_val = 0
    for code, shares in pm.positions.items():
        if code == StrategyConfig.CASH_ETF_CODE:
            price = all_data['hfq'][code][all_data['hfq'][code]['date'] <= pd.to_datetime(d)]['close'].iloc[-1]
            bond_val = shares * price
        else:
            if code in all_data['hfq']:
                df = all_data['hfq'][code]
                filt = df[df['date'] <= pd.to_datetime(d)]
                if len(filt) > 0:
                    price = float(filt['close'].iloc[-1])
                    risk_val += shares * price

    print(f"  风险资产市值={risk_val:.0f}, 国债={bond_val:.0f}, 现金={pm.cash:.0f}")
    print(f"  风险占比={risk_val / total_val * 100:.1f}%, 国债占比={bond_val/total_val*100:.1f}%")
