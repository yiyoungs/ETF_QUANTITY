"""Run the adaptive strategy on the repository's local cached ETF data."""

from pathlib import Path

from adaptive_strategy import AdaptiveConfig, load_close_prices, performance_metrics, run_adaptive_backtest, walk_forward_report
from config import get_all_high_elastic_etfs


def main() -> None:
    root = Path(__file__).resolve().parent
    candidates = get_all_high_elastic_etfs()
    prices = load_close_prices(str(root / "data" / "cache"), candidates + ["510300", "511010"])
    # The smallest candidate history begins in 2020.  Start after all lookback
    # windows can be formed, keeping the sample free of synthetic fills.
    prices = prices.loc["2021-01-01":"2026-06-09"]
    results, decisions = run_adaptive_backtest(prices, candidates, config=AdaptiveConfig())
    reports = root / "reports"
    reports.mkdir(exist_ok=True)
    results.to_csv(reports / "adaptive_backtest_results.csv", index=False, encoding="utf-8-sig")
    decisions.to_csv(reports / "adaptive_rebalance_log.csv", index=False, encoding="utf-8-sig")
    windows = [
        ("2022", "2022-01-01", "2022-12-31"),
        ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"),
        ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", "2026-06-09"),
    ]
    walk_forward = walk_forward_report(results, windows)
    walk_forward.to_csv(reports / "adaptive_walk_forward.csv", index=False, encoding="utf-8-sig")

    metrics = performance_metrics(results)
    print("Adaptive ETF rotation (next-session execution)")
    for name, value in metrics.items():
        print(f"{name}: {value:.2%}" if "return" in name or "volatility" in name or "drawdown" in name or "weight" in name else f"{name}: {value:.2f}")
    print(f"benchmark_total_return: {results['benchmark_nav'].iloc[-1] - 1:.2%}")
    print(f"cash_total_return: {results['cash_nav'].iloc[-1] - 1:.2%}")
    for label, subset in (("in_sample_2021_2023", results.loc[results["date"] < "2024-01-01"]),
                          ("out_of_sample_2024_2026", results.loc[results["date"] >= "2024-01-01"])):
        rebased = subset.copy()
        rebased["nav"] = rebased["nav"] / rebased["nav"].iloc[0]
        rebased["drawdown"] = rebased["nav"] / rebased["nav"].cummax() - 1
        print(f"{label}_total_return: {performance_metrics(rebased)['total_return']:.2%}")
    print("walk_forward periods")
    print(walk_forward[["period", "total_return", "max_drawdown", "benchmark_total_return", "cash_total_return"]].to_string(index=False))


if __name__ == "__main__":
    main()
