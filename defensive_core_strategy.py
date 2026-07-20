"""Pre-registered defensive-core ETF allocation experiment.

Rules are copied from alternative-strategy-research.md before the backtest is
run.  They are intentionally not changed in response to the results below.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from adaptive_strategy import _capped_inverse_volatility, performance_metrics, walk_forward_report


def _group_weights(returns: pd.DataFrame, members: list[str], budget: float, cap: float = 0.35) -> pd.Series:
    result = pd.Series(0.0, index=returns.columns)
    available = [code for code in members if code in returns.columns]
    if not available or budget <= 0:
        return result
    weights = _capped_inverse_volatility(returns[available].tail(60), cap)
    result.loc[weights.index] = weights * budget
    return result


def run_defensive_core_backtest(prices: pd.DataFrame, offensive: list[str], transaction_cost: float = 0.001):
    """Run weekly, next-session execution with fixed core/satellite budgets."""
    defensive = ["511010", "510880", "512890"]
    balanced = ["510300", "510500"]
    required = list(dict.fromkeys(defensive + balanced + offensive))
    prices = prices[required].dropna().copy()
    returns = prices.pct_change().fillna(0.0)
    weights = pd.Series(0.0, index=required)
    weights["511010"] = 1.0
    pending = None
    nav = 1.0
    rows, decisions = [], []

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
            "equity_weight": float(weights[balanced + offensive].sum()), "turnover": turnover,
        })

        if date.weekday() != 4 or index < 200:
            continue
        history = prices.iloc[: index + 1]
        broad_trend = history["510300"].iloc[-1] > history["510300"].rolling(200).mean().iloc[-1]
        broad_eligible = [code for code in balanced if history[code].iloc[-1] > history[code].rolling(100).mean().iloc[-1]]
        offensive_eligible = [code for code in offensive if history[code].iloc[-1] > history[code].rolling(100).mean().iloc[-1]]

        target = pd.Series(0.0, index=required)
        if broad_trend:
            # 25% defensive core; 35% diversified broad equity; up to 40% theme satellite.
            target += _group_weights(returns.iloc[: index + 1], defensive, 0.25)
            target += _group_weights(returns.iloc[: index + 1], broad_eligible, 0.35)
            if offensive_eligible:
                momentum = history[offensive_eligible].pct_change(60).iloc[-1].nlargest(4).index.tolist()
                target += _group_weights(returns.iloc[: index + 1], momentum, 0.40)
            else:
                target += _group_weights(returns.iloc[: index + 1], defensive, 0.40)
        else:
            # Bear regime: at least 60% defensive; retain broad equities only when
            # their own 100-day trend remains positive.
            target += _group_weights(returns.iloc[: index + 1], defensive, 0.60)
            target += _group_weights(returns.iloc[: index + 1], broad_eligible, 0.40)
        # Any unavailable sleeve remains in the cash-like Treasury ETF rather
        # than being silently left as uninvested capital.
        target.loc["511010"] += 1 - target.sum()

        pending = target
        decisions.append({
            "signal_date": date, "broad_trend": bool(broad_trend),
            "defensive_weight": float(target[defensive].sum()),
            "balanced_weight": float(target[balanced].sum()),
            "offensive_weight": float(target[offensive].sum()),
            "offensive_members": ",".join([code for code in offensive if target[code] > 0]),
        })

    results = pd.DataFrame(rows)
    results["drawdown"] = results["nav"] / results["nav"].cummax() - 1
    return results, pd.DataFrame(decisions)


def report(results: pd.DataFrame) -> pd.DataFrame:
    periods = [
        ("2022", "2022-01-01", "2022-12-31"), ("2023", "2023-01-01", "2023-12-31"),
        ("2024", "2024-01-01", "2024-12-31"), ("2025", "2025-01-01", "2025-12-31"),
        ("2026_ytd", "2026-01-01", "2026-06-09"),
    ]
    return walk_forward_report(results, periods)
