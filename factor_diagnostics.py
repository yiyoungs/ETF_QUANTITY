"""Cross-sectional ETF factor diagnostics with no same-day forward labels."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from adaptive_strategy import load_close_prices


FACTOR_COLUMNS = ("reversal_5", "momentum_20", "momentum_60", "momentum_120", "low_volatility_20", "drawdown_20")
HORIZONS = (5, 20)


def classify_market(prices: pd.DataFrame, benchmark: str = "510300") -> pd.Series:
    """Classify only from information available at each close."""
    close = prices[benchmark]
    ma200 = close.rolling(200).mean()
    return_60 = close.pct_change(60)
    annualized_vol = close.pct_change().rolling(20).std() * np.sqrt(252)
    high_vol = annualized_vol > annualized_vol.rolling(252, min_periods=100).median()
    state = pd.Series("range", index=prices.index, dtype="object")
    state[(close > ma200) & (return_60 > 0)] = "bull"
    state[(close < ma200) & (return_60 < 0)] = "bear"
    state[high_vol] = state[high_vol] + "_high_vol"
    return state


def build_factor_panel(prices: pd.DataFrame, candidates: list[str], benchmark: str = "510300") -> pd.DataFrame:
    """Create Friday signal snapshots and future-return labels for each ETF."""
    prices = prices[list(dict.fromkeys(candidates + [benchmark]))].dropna().copy()
    market_state = classify_market(prices, benchmark)
    rows = []
    for index, date in enumerate(prices.index):
        if date.weekday() != 4 or index < 252 or index + max(HORIZONS) >= len(prices):
            continue
        history = prices.iloc[: index + 1]
        window20 = history[candidates].tail(20)
        for code in candidates:
            close = history[code]
            row = {
                "date": date,
                "code": code,
                "market_state": market_state.iloc[index],
                # Positive reversal score means a larger recent drop.
                "reversal_5": -close.pct_change(5).iloc[-1],
                "momentum_20": close.pct_change(20).iloc[-1],
                "momentum_60": close.pct_change(60).iloc[-1],
                "momentum_120": close.pct_change(120).iloc[-1],
                "low_volatility_20": -window20[code].pct_change().std(),
                "drawdown_20": close.iloc[-1] / close.tail(20).max() - 1,
            }
            for horizon in HORIZONS:
                row[f"forward_return_{horizon}"] = prices[code].iloc[index + horizon] / prices[code].iloc[index] - 1
            rows.append(row)
    return pd.DataFrame(rows)


def _summarize_group(panel: pd.DataFrame, factor: str, horizon: int, label: str) -> dict:
    target = f"forward_return_{horizon}"
    grouped = panel.groupby("date", sort=False)
    ic_values, spreads = [], []
    for _, sample in grouped:
        sample = sample[[factor, target]].dropna()
        if len(sample) < 8:
            continue
        ic_values.append(sample[factor].rank().corr(sample[target].rank()))
        n = max(1, len(sample) // 4)
        ordered = sample.sort_values(factor)
        spreads.append(ordered.tail(n)[target].mean() - ordered.head(n)[target].mean())
    observations = len(ic_values)
    mean_ic = float(np.nanmean(ic_values)) if observations else np.nan
    return {
        "factor": factor,
        "horizon_days": horizon,
        "market_state": label,
        "weeks": observations,
        "mean_rank_ic": mean_ic,
        "ic_t_stat": mean_ic / (np.nanstd(ic_values, ddof=1) / np.sqrt(observations)) if observations > 1 and np.nanstd(ic_values, ddof=1) else np.nan,
        "top_minus_bottom_return": float(np.nanmean(spreads)) if spreads else np.nan,
        "top_minus_bottom_hit_rate": float(np.mean(np.array(spreads) > 0)) if spreads else np.nan,
    }


def summarize_factors(panel: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for factor in FACTOR_COLUMNS:
        for horizon in HORIZONS:
            summaries.append(_summarize_group(panel, factor, horizon, "all"))
            for state in sorted(panel["market_state"].dropna().unique()):
                summaries.append(_summarize_group(panel[panel["market_state"] == state], factor, horizon, state))
    return pd.DataFrame(summaries)


def render_report(summary: pd.DataFrame) -> str:
    overall = summary[summary["market_state"] == "all"].copy()
    overall["abs_ic"] = overall["mean_rank_ic"].abs()
    strongest = overall.sort_values("abs_ic", ascending=False).head(6)
    lines = [
        "# ETF Factor Diagnostics", "", "Signals are calculated at Friday close; labels begin after the signal close.", "",
        "## Highest-information factors in the full sample", "",
        "| factor | forward window | weeks | rank IC | IC t-stat | top-bottom mean return | hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in strongest.itertuples():
        lines.append(f"| {row.factor} | {row.horizon_days}d | {row.weeks} | {row.mean_rank_ic:.3f} | {row.ic_t_stat:.2f} | {row.top_minus_bottom_return:.2%} | {row.top_minus_bottom_hit_rate:.1%} |")
    lines.extend([
        "", "## Interpretation", "",
        "- Rank IC is the Spearman correlation between the factor and future-return ranks.",
        "- Top-bottom is the weekly top-quartile minus bottom-quartile average future return before costs.",
        "- A factor needs stable results across market states and a viable cost profile before any strategy use.",
        "", "See reports/factor_summary.csv for every factor, horizon, and market state.",
    ])
    return "\n".join(lines) + "\n"

def run(project_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = ["512480", "159995", "512760", "515050", "512000", "515030", "159875", "512660", "512690", "159928", "512010", "159801", "512890", "510880", "168204", "512400"]
    prices = load_close_prices(str(project_root / "data" / "cache"), candidates + ["510300"])
    prices = prices.loc["2021-01-01":"2026-05-12"]
    panel = build_factor_panel(prices, candidates)
    return panel, summarize_factors(panel)

