# ETF Quantity

ETF 行业/主题轮动研究项目。历史回测只用于研究，不构成投资建议或收益承诺。

## 当前研究版本

`run_adaptive_backtest.py` 是新的自适应轮动实验：

- 使用 20/60/120 日截面动量选择最多 4 只 ETF；
- 用 100 日趋势过滤和沪深 300 + 市场广度决定 0%、25%、50%、75%、100% 风险仓位；
- 对入选标的按逆波动率配置，并限制单标的权重；
- 周五收盘生成信号，下一交易日才变更仓位，避免同一收盘价选股并成交的前视偏差；
- 在本地 `data/cache` 行情上离线运行，不会下载或修改行情。

以仓库现有缓存运行时，回测有效起点为 2021-08-17（需要全部候选 ETF
都有可用价格）。截至 2026-06-09，策略总收益为 5.24%、年化收益 1.11%、
最大回撤 -23.72%，同期沪深 300 总收益 -1.59%、国债 ETF 总收益 13.05%。
策略未跑赢现金防守基准，因此当前版本仅保留为研究基线，不应采纳。这些结果包含 0.10% 单边
交易成本，仍不包含滑点、冲击成本、税费和真实可交易性约束，不能据此实盘决策。

## 运行

```powershell
python -m pip install -r requirements.txt
python run_adaptive_backtest.py
python -m pytest -q test_adaptive_strategy.py
```

若未安装 pytest，可直接执行测试函数：

```powershell
python -c "from test_adaptive_strategy import test_strategy_has_no_same_day_execution, test_metrics_are_finite; test_strategy_has_no_same_day_execution(); test_metrics_are_finite()"
```

输出保存在 `reports/adaptive_backtest_results.csv` 和
`reports/adaptive_rebalance_log.csv`。`reports/adaptive_walk_forward.csv` 给出
预先定义的逐年 walk-forward 检验，不会根据任何检验期结果调整参数。默认样本为 2021-01-01 至本地缓存的
2026-06-09；报告同时打印 2021--2023 与 2024--2026 两段表现，后者用于避免
只依据同一段数据评价策略。
