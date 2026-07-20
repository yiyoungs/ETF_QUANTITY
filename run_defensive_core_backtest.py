from pathlib import Path

from adaptive_strategy import load_close_prices, performance_metrics
from config import get_all_high_elastic_etfs
from defensive_core_strategy import report, run_defensive_core_backtest


def main():
    root = Path(__file__).resolve().parent
    defensive_and_balanced = {"511010", "510880", "512890", "510300", "510500"}
    offensive = [code for code in get_all_high_elastic_etfs() if code not in defensive_and_balanced]
    codes = offensive + ["511010", "510880", "512890", "510300", "510500"]
    prices = load_close_prices(str(root / "data" / "cache"), codes).loc["2021-01-01":"2026-06-09"]
    results, decisions = run_defensive_core_backtest(prices, offensive)
    reports = root / "reports"
    results.to_csv(reports / "defensive_core_results.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(reports / "defensive_core_rebalance_log.csv", index=False, encoding="utf-8-sig")
    walk_forward = report(results)
    walk_forward.to_csv(reports / "defensive_core_walk_forward.csv", index=False, encoding="utf-8-sig")
    metrics = performance_metrics(results)
    print("Defensive core experiment (pre-registered rules)")
    for name, value in metrics.items():
        print(f"{name}: {value:.2%}" if any(key in name for key in ("return", "volatility", "drawdown", "weight")) else f"{name}: {value:.2f}")
    print(f"benchmark_total_return: {results['benchmark_nav'].iloc[-1] - 1:.2%}")
    print(f"cash_total_return: {results['cash_nav'].iloc[-1] - 1:.2%}")
    print(walk_forward[["period", "total_return", "max_drawdown", "benchmark_total_return", "cash_total_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
