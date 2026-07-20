"""A bias-aware, risk-managed ETF rotation backtest.

Signals are calculated after a close and can only change the portfolio on the
following trading day.  This deliberately differs from the legacy engine,
which trades on the same close used to calculate its signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptiveConfig:
    momentum_windows: tuple[tuple[int, float], ...] = ((20, 0.45), (60, 0.35), (120, 0.20))
    trend_window: int = 100
    market_window: int = 200
    volatility_window: int = 40
    holdings: int = 4
    max_weight: float = 0.35
    transaction_cost: float = 0.001
    rebalance_weekday: int = 4  # Friday close signal, Monday/next session execution


def load_close_prices(cache_dir: str, codes: Iterable[str]) -> pd.DataFrame:
    """Load cached QFQ close prices and align them by date."""
    frames = []
    for code in dict.fromkeys(codes):
        path = f"{cache_dir}/{code}_qfq.csv"
        frame = pd.read_csv(path, usecols=["date", "close"], parse_dates=["date"])
        frames.append(frame.rename(columns={"close": code}).set_index("date"))
    return pd.concat(frames, axis=1).sort_index()


def _capped_inverse_volatility(returns: pd.DataFrame, maximum: float) -> pd.Series:
    volatility = returns.std().replace(0, np.nan)
    inverse = (1 / volatility).replace([np.inf, -np.inf], np.nan).dropna()
    if inverse.empty:
        return pd.Series(dtype=float)
    weights = inverse / inverse.sum()
    # Iterative cap keeps weights summing to one when feasible.
    for _ in range(len(weights)):
        capped = weights.clip(upper=maximum)
        remainder = 1 - capped.sum()
        eligible = weights.index[capped < maximum - 1e-12]
        if remainder <= 1e-12 or len(eligible) == 0:
            return capped / capped.sum()
        capped.loc[eligible] += remainder * (weights.loc[eligible] / weights.loc[eligible].sum())
        weights = capped
    return weights / weights.sum()


def _risk_budget(prices: pd.DataFrame, benchmark: str, candidates: list[str], config: AdaptiveConfig) -> float:
    """Map market breadth and benchmark trend to a 0--100% equity budget."""
    if len(prices) < config.market_window:
        return 0.0
    trend = prices[candidates].iloc[-1] / prices[candidates].rolling(config.trend_window).mean().iloc[-1] - 1
    breadth = trend.gt(0).mean()
    benchmark_trend = prices[benchmark].iloc[-1] > prices[benchmark].rolling(config.market_window).mean().iloc[-1]
    if not benchmark_trend:
        return 0.0 if breadth < 0.45 else 0.25
    if breadth < 0.35:
        return 0.25
    if breadth < 0.55:
        return 0.50
    if breadth < 0.70:
        return 0.75
    return 1.00


def run_adaptive_backtest(
    prices: pd.DataFrame,
    candidates: list[str],
    benchmark: str = "510300",
    cash_asset: str = "511010",
    config: AdaptiveConfig = AdaptiveConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return daily NAV and rebalance decisions using close-to-close returns."""
    required = list(dict.fromkeys(candidates + [benchmark, cash_asset]))
    prices = prices[required].dropna().copy()
    returns = prices.pct_change().fillna(0.0)
    weights = pd.Series(0.0, index=required)
    weights[cash_asset] = 1.0
    nav = 1.0
    rows: list[dict] = []
    decisions: list[dict] = []
    pending: pd.Series | None = None

    minimum_history = max(config.market_window, max(window for window, _ in config.momentum_windows))
    for index, (date, daily_return) in enumerate(returns.iterrows()):
        previous_nav = nav
        gross_return = float((weights * daily_return).sum())
        nav *= 1 + gross_return

        # Execute the prior close's decision at this close, after earning today's
        # return on the prior portfolio.  This is the earliest close-to-close
        # representation available without intraday/open data.
        turnover = 0.0
        if pending is not None:
            turnover = float((pending - weights).abs().sum())
            nav *= 1 - turnover * config.transaction_cost
            weights = pending
            pending = None

        rows.append({
            "date": date,
            "nav": nav,
            "daily_return": nav / previous_nav - 1 if rows else 0.0,
            "benchmark_nav": float(prices[benchmark].iloc[index] / prices[benchmark].iloc[0]),
            "cash_nav": float(prices[cash_asset].iloc[index] / prices[cash_asset].iloc[0]),
            "equity_weight": float(weights[candidates].sum()),
            "turnover": turnover,
        })

        if date.weekday() != config.rebalance_weekday or index < minimum_history:
            continue
        history = prices.iloc[: index + 1]
        budget = _risk_budget(history, benchmark, candidates, config)
        momentum = pd.Series(0.0, index=candidates)
        for window, weight in config.momentum_windows:
            momentum += weight * history[candidates].pct_change(window).iloc[-1].rank(pct=True)
        eligible = momentum[(history[candidates].iloc[-1] > history[candidates].rolling(config.trend_window).mean().iloc[-1])].dropna()
        selected = eligible.nlargest(config.holdings).index.tolist()
        target = pd.Series(0.0, index=required)
        if selected and budget > 0:
            asset_weights = _capped_inverse_volatility(returns[selected].iloc[max(0, index - config.volatility_window + 1): index + 1], config.max_weight)
            target.loc[asset_weights.index] = asset_weights * budget
        target[cash_asset] = 1 - target.sum()
        pending = target
        decisions.append({"signal_date": date, "risk_budget": budget, "breadth": float((history[candidates].iloc[-1] > history[candidates].rolling(config.trend_window).mean().iloc[-1]).mean()), "selected": ",".join(selected)})

    result = pd.DataFrame(rows)
    result["drawdown"] = result["nav"] / result["nav"].cummax() - 1
    return result, pd.DataFrame(decisions)


def performance_metrics(results: pd.DataFrame) -> dict[str, float]:
    daily = results["daily_return"]
    years = len(results) / 252
    annual_return = results["nav"].iloc[-1] ** (1 / years) - 1
    volatility = daily.std() * np.sqrt(252)
    return {
        "total_return": results["nav"].iloc[-1] - 1,
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe": daily.mean() / daily.std() * np.sqrt(252) if daily.std() else 0.0,
        "max_drawdown": results["drawdown"].min(),
        "average_equity_weight": results["equity_weight"].mean(),
        "total_turnover": results["turnover"].sum(),
    }


def walk_forward_report(results: pd.DataFrame, periods: Iterable[tuple[str, str, str]]) -> pd.DataFrame:
    """Evaluate fixed, pre-declared calendar periods without refitting parameters."""
    rows = []
    for label, start, end in periods:
        sample = results.loc[(results["date"] >= start) & (results["date"] <= end)].copy()
        if len(sample) < 2:
            continue
        sample["nav"] /= sample["nav"].iloc[0]
        sample["daily_return"] = sample["nav"].pct_change().fillna(0.0)
        sample["drawdown"] = sample["nav"] / sample["nav"].cummax() - 1
        metrics = performance_metrics(sample)
        metrics.update({
            "period": label,
            "start": sample["date"].iloc[0].strftime("%Y-%m-%d"),
            "end": sample["date"].iloc[-1].strftime("%Y-%m-%d"),
            "benchmark_total_return": sample["benchmark_nav"].iloc[-1] / sample["benchmark_nav"].iloc[0] - 1,
            "cash_total_return": sample["cash_nav"].iloc[-1] / sample["cash_nav"].iloc[0] - 1,
        })
        rows.append(metrics)
    return pd.DataFrame(rows)
