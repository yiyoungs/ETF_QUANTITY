import pandas as pd
import numpy as np
from analyzer import Analyzer

dates = pd.date_range('2020-01-01', periods=200, freq='B')
nav = np.ones(200)
nav[0:50] = np.linspace(1.0, 1.2, 50)
nav[50:80] = np.linspace(1.2, 0.9, 30)
nav[80:130] = np.linspace(0.9, 1.25, 50)
nav[130:150] = np.linspace(1.25, 1.1, 20)
nav[150:200] = np.linspace(1.1, 1.4, 50)

df = pd.DataFrame({
    'date': dates,
    'nav': nav,
    'daily_return': np.diff(nav, prepend=nav[0]) / nav
})

a = Analyzer(df)
m = a.calculate_metrics()
print(f'MaxDD={m["max_drawdown"]*100:.2f}% MaxRecovery={m["max_recovery_days"]}d AvgRecovery={m["avg_recovery_days"]:.1f}d')
