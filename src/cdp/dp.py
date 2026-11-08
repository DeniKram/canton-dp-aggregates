"""Differential privacy primitives: clipping, the Laplace mechanism, a budget accountant.

Privacy unit decides what is protected (and therefore the sensitivity Δ):
  "trade"      - hide any single trade.            Δ = trade clip C_t  (e.g. 1M)
  "dealer_day" - hide a whole dealer's daily volume. Δ = dealer-day clip C_d (e.g. 10M)

Laplace mechanism: release f(x) + Lap(b), b = Δ / ε.  Noise std = √2 · b.
  Example: Δ = 1M, ε = 1  ->  b = 1M, std ≈ 1.41M.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .data import Market


def laplace_scale(sensitivity: float, epsilon: float) -> float:
    if epsilon <= 0:
        raise ValueError("epsilon must be > 0")
    return sensitivity / epsilon


def laplace_noise(sensitivity: float, epsilon: float, rng: np.random.Generator, size=None):
    return rng.laplace(0.0, laplace_scale(sensitivity, epsilon), size=size)


def clipped_dealer_day(market: Market, unit: str, clip: float) -> np.ndarray:
    """Per-dealer daily contribution after bounding, so sensitivity really is `clip`. shape (T, D)

    unit="trade":      every trade is capped at `clip` before summing.
    unit="dealer_day": every dealer's daily volume is capped at `clip`.
    """
    if unit == "trade":
        return np.array([[np.minimum(x, clip).sum() for x in day] for day in market.trades])
    if unit == "dealer_day":
        return np.minimum(market.dealer_day, clip)
    raise ValueError(f"unknown privacy unit {unit!r}")


def clipped_daily_total(market: Market, unit: str, clip: float) -> np.ndarray:
    """Sum over dealers of the clipped contributions. shape (T,)"""
    return clipped_dealer_day(market, unit, clip).sum(axis=1)


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class PrivacyAccountant:
    """Basic sequential composition: total ε spent on the same data must stay ≤ budget.

    This is the rule the Daml `PrivacyBudget` contract will enforce on-ledger.
      budget 10, five releases at ε=2 -> ok; the sixth raises BudgetExceeded.
    """

    budget: float
    spent: float = 0.0
    log: list[float] = field(default_factory=list)

    def charge(self, epsilon: float) -> None:
        if self.spent + epsilon > self.budget + 1e-12:
            raise BudgetExceeded(f"spent {self.spent:.2f} + {epsilon:.2f} > budget {self.budget:.2f}")
        self.spent += epsilon
        self.log.append(epsilon)

    @property
    def remaining(self) -> float:
        return self.budget - self.spent


def release(
    value: float,
    sensitivity: float,
    epsilon: float,
    accountant: PrivacyAccountant,
    rng: np.random.Generator,
) -> float:
    """One DP release; charges the accountant first, so an over-budget query never leaks."""
    accountant.charge(epsilon)
    return float(value + laplace_noise(sensitivity, epsilon, rng))
