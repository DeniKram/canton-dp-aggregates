"""Synthetic trade data shaped like a small Canton market (e.g. one tokenized bond).

Default market (all numbers are illustrative, not measured on Canton):
  dealers D            = 3
  days T               = 22   (one trading month)
  trades/dealer/day    ~ Poisson(20)
  trade size (USD)     ~ LogNormal(median 250k, sigma 0.8), capped at 5M
  -> one dealer-day    ~ 6M,  one market-day (D=3) ~ 18M
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MarketConfig:
    dealers: int = 3
    days: int = 22
    trades_per_day: float = 20.0
    median_trade_usd: float = 250_000.0
    size_sigma: float = 0.8
    max_trade_usd: float = 5_000_000.0
    participation: float = 1.0  # probability a dealer trades at all on a given day


@dataclass
class Market:
    """trades[t][d] is a 1-D array of trade sizes for dealer d on day t."""

    trades: list[list[np.ndarray]]
    config: MarketConfig

    @property
    def dealer_day(self) -> np.ndarray:
        """Daily volume per dealer. shape (T, D)"""
        return np.array([[day[d].sum() for d in range(self.config.dealers)] for day in self.trades])

    @property
    def daily_total(self) -> np.ndarray:
        """Market volume per day. shape (T,)"""
        return self.dealer_day.sum(axis=1)

    @property
    def max_trade(self) -> float:
        return max((x.max() for day in self.trades for x in day if x.size), default=0.0)


def generate(cfg: MarketConfig, rng: np.random.Generator) -> Market:
    mu = np.log(cfg.median_trade_usd)
    trades: list[list[np.ndarray]] = []
    for _ in range(cfg.days):
        day = []
        for _ in range(cfg.dealers):
            if rng.random() > cfg.participation:
                day.append(np.zeros(0))
                continue
            n = rng.poisson(cfg.trades_per_day)
            sizes = np.minimum(rng.lognormal(mu, cfg.size_sigma, size=n), cfg.max_trade_usd)
            day.append(sizes)
        trades.append(day)
    return Market(trades=trades, config=cfg)
