import numpy as np
import pandas as pd

from defensive_core_strategy import run_defensive_core_backtest


def test_defensive_core_handles_overlapping_universes_after_deduplication():
    dates = pd.date_range("2020-01-01", periods=330, freq="B")
    codes = ["511010", "510880", "512890", "510300", "510500", "512480", "159995"]
    rng = np.random.default_rng(13)
    prices = pd.DataFrame({code: 100 * np.cumprod(1 + rng.normal(0.0002, 0.01, len(dates))) for code in codes}, index=dates)
    results, decisions = run_defensive_core_backtest(prices, ["512480", "159995"])
    assert len(results) == len(prices)
    assert not decisions.empty
    assert np.isfinite(results[["nav", "daily_return", "equity_weight"]].to_numpy()).all()
    assert results["nav"].iloc[-1] > 0
