import numpy as np
import pandas as pd

from adaptive_strategy import AdaptiveConfig, performance_metrics, run_adaptive_backtest, walk_forward_report


def synthetic_prices(days=320):
    dates = pd.date_range("2022-01-03", periods=days, freq="B")
    rng = np.random.default_rng(7)
    data = {}
    for code, drift in {"A": 0.001, "B": 0.0004, "C": -0.0002, "510300": 0.0003, "511010": 0.00002}.items():
        data[code] = 100 * np.cumprod(1 + rng.normal(drift, 0.01, days))
    return pd.DataFrame(data, index=dates)


def test_strategy_has_no_same_day_execution():
    prices = synthetic_prices()
    config = AdaptiveConfig(momentum_windows=((20, 1.0),), trend_window=30, market_window=50, volatility_window=20)
    results, decisions = run_adaptive_backtest(prices, ["A", "B", "C"], config=config)
    first_signal = decisions.iloc[0]["signal_date"]
    signal_index = results.index[results["date"] == first_signal][0]
    # The Friday signal leaves Friday's recorded allocation unchanged; the next
    # session is the earliest point at which it may take effect.
    assert results.loc[signal_index, "equity_weight"] == 0.0
    assert results.loc[signal_index + 1, "equity_weight"] >= 0.0


def test_metrics_are_finite():
    results, _ = run_adaptive_backtest(synthetic_prices(), ["A", "B", "C"], config=AdaptiveConfig(market_window=50, trend_window=30, volatility_window=20))
    assert all(np.isfinite(value) for value in performance_metrics(results).values())


def test_walk_forward_uses_requested_periods():
    results, _ = run_adaptive_backtest(synthetic_prices(), ["A", "B", "C"], config=AdaptiveConfig(market_window=50, trend_window=30, volatility_window=20))
    report = walk_forward_report(results, [("first", "2022-01-01", "2022-12-31"), ("second", "2023-01-01", "2023-12-31")])
    assert report["period"].tolist() == ["first", "second"]
    assert (report["benchmark_total_return"] > -1).all()
    assert (report["cash_total_return"] > -1).all()
