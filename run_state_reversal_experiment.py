from pathlib import Path

from adaptive_strategy import performance_metrics
from state_reversal_experiment import run_from_cache, split_metrics


def main():
    root = Path(__file__).resolve().parent
    results, trades = run_from_cache(root)
    reports = root / "reports"
    results.to_csv(reports / "state_reversal_results.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(reports / "state_reversal_trades.csv", index=False, encoding="utf-8-sig")
    metrics = performance_metrics(results)
    print("Exploratory state-conditioned reversal experiment")
    for key, value in metrics.items():
        print(f"{key}: {value:.2%}" if any(name in key for name in ("return", "volatility", "drawdown", "weight")) else f"{key}: {value:.2f}")
    print(f"trades: {len(trades)}")
    print(split_metrics(results).to_string(index=False))


if __name__ == "__main__":
    main()
