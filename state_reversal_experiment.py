"""Exploratory replication of a state-conditioned short-term reversal signal.

This module is deliberately labelled exploratory: the state and factor were
selected after inspecting the same history.  Its final period is therefore a
sanity check, not a clean out-of-sample validation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_strategy import load_close_prices, performance_metrics
from factor_diagnostics import classify_market


CANDIDATES = ["512480", "159995", "512760", "515050", "512000", "515030", "159875", "512660", "512690", "159928", "512010", "159801", "512890", "510880", "168204", "512400"]


def run_experiment(prices: pd.DataFrame, transaction_cost: float = 0.001, hold_days: int = 20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Buy the four largest 5-day losers only in bear_high_vol, then hold 20 days."""
    required = CANDIDATES + ["510300", "511010"]
    prices = prices[required].dropna().copy()
    returns = prices.pct_change().fillna(0.0)
    states = classify_market(prices)
    weights = pd.Series(0.0, index=required)
    weights["511010"] = 1.0
    pending = None
    days_remaining = 0
    nav = 1.0
    rows, trades = [], []

    for index, (date, daily_return) in enumerate(returns.iterrows()):
        prior_nav = nav
        nav *= 1 + float((weights * daily_return).sum())
        turnover = 0.0
        if pending is not None:
            turnover = float((pending - weights).abs().sum())
            nav *= 1 - turnover * transaction_cost
            weights = pending
            pending = None

        rows.append({
            "date": date, "nav": nav, "daily_return": nav / prior_nav - 1 if rows else 0.0,
            "benchmark_nav": prices["510300"].iloc[index] / prices["510300"].iloc[0],
            "cash_nav": prices["511010"].iloc[index] / prices["511010"].iloc[0],
            "equity_weight": float(weights[CANDIDATES].sum()), "turnover": turnover,
            "market_state": states.iloc[index],
        })

        if days_remaining > 0:
            days_remaining -= 1
            if days_remaining == 0:
                target = pd.Series(0.0, index=required)
                target["511010"] = 1.0
                pending = target
                trades.append({"signal_date": date, "action": "exit_to_treasury", "state": states.iloc[index], "selected": ""})
            continue

        # Signals use Friday close and are executed on the following trading day.
        if date.weekday() == 4 and index >= 252 and states.iloc[index] == "bear_high_vol":
            recent_losses = -prices[CANDIDATES].iloc[index].div(prices[CANDIDATES].iloc[index - 5]).sub(1)
            selected = recent_losses.nlargest(4).index.tolist()
            target = pd.Series(0.0, index=required)
            target[selected] = 1.0 / len(selected)
            pending = target
            days_remaining = hold_days
            trades.append({"signal_date": date, "action": "buy_reversal_basket", "state": states.iloc[index], "selected": ",".join(selected)})

    results = pd.DataFrame(rows)
    results["drawdown"] = results["nav"] / results["nav"].cummax() - 1
    return results, pd.DataFrame(trades)


def split_metrics(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label, start, end in (("discovery_2022_2024", "2022-09-02", "2024-12-31"), ("later_2025_2026", "2025-01-01", "2026-05-12")):
        sample = results[(results["date"] >= start) & (results["date"] <= end)].copy()
        if len(sample) < 2:
            continue
        sample["nav"] /= sample["nav"].iloc[0]
        sample["daily_return"] = sample["nav"].pct_change().fillna(0)
        sample["drawdown"] = sample["nav"] / sample["nav"].cummax() - 1
        metric = performance_metrics(sample)
        metric.update({"period": label, "benchmark_return": sample["benchmark_nav"].iloc[-1] / sample["benchmark_nav"].iloc[0] - 1, "cash_return": sample["cash_nav"].iloc[-1] / sample["cash_nav"].iloc[0] - 1})
        rows.append(metric)
    return pd.DataFrame(rows)


def run_from_cache(project_root: Path):
    prices = load_close_prices(str(project_root / "data" / "cache"), CANDIDATES + ["510300", "511010"])
    return run_experiment(prices.loc["2021-01-01":"2026-05-12"])
