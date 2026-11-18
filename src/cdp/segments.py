"""Segment-level statistics for private companies: differential privacy against a count threshold.

Question: can a publisher that holds many cap tables release *breakdowns* (one cell per sector x
stage) instead of a single aggregate, without exposing any one company?

Per cell, two statistics are released:
  count            number of companies in the cell                        (sensitivity 1)
  typical value    geometric mean valuation of the companies in the cell

The privacy unit is one company. Each company sits in exactly one cell, so cells compose in
parallel: every cell can use the full release budget epsilon. Within a cell the budget is split
evenly between the noisy count and the noisy sum of log-valuations.

Valuations are log-normal (median 20M USD). Logs are taken relative to a public reference value
and clipped to a public range [lo, hi]; the clipping range sets the sensitivity of the log-sum.

Attack: a coalition that knows every other company in the cell (the DP threat model) subtracts
their log-valuations from the released log-sum. Under an exact release with a count threshold this
recovers the target exactly whenever the cell is published.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ValuationConfig:
    median: float = 20e6
    sigma: float = 1.2      # log-normal shape; ~90% of companies between 3M and 140M
    ref: float = 20e6       # public reference value for the log transform
    lo: float = 1e6         # public clip range
    hi: float = 1e9

    @property
    def log_sensitivity(self) -> float:
        """Most one company can move the sum of clipped log(v / ref)."""
        return max(abs(np.log(self.lo / self.ref)), abs(np.log(self.hi / self.ref)))


def sample_valuations(cfg: ValuationConfig, n: int, rng: np.random.Generator) -> np.ndarray:
    """shape (n,) valuations in USD."""
    return cfg.median * np.exp(cfg.sigma * rng.standard_normal(n))


def clipped_logs(cfg: ValuationConfig, v: np.ndarray) -> np.ndarray:
    """shape (n,) log(v / ref) clipped to the public range."""
    return np.clip(np.log(v / cfg.ref), np.log(cfg.lo / cfg.ref), np.log(cfg.hi / cfg.ref))


def geometric_mean(v: np.ndarray) -> float:
    return float(np.exp(np.mean(np.log(v))))


@dataclass(frozen=True)
class CellRelease:
    count: float          # noisy count
    log_sum: float        # noisy sum of clipped logs
    typical: float        # released geometric mean, USD


def dp_release(cfg: ValuationConfig, v: np.ndarray, epsilon: float, rng: np.random.Generator) -> CellRelease:
    """Laplace release of (count, log-sum) for one cell, epsilon split evenly."""
    half = epsilon / 2
    count = len(v) + rng.laplace(0.0, 1.0 / half)
    log_sum = clipped_logs(cfg, v).sum() + rng.laplace(0.0, cfg.log_sensitivity / half)
    typical = cfg.ref * float(np.exp(log_sum / max(count, 1.0)))
    return CellRelease(count, log_sum, typical)


def coalition_estimate(cfg: ValuationConfig, release_log_sum: float, others: np.ndarray) -> float:
    """Target's valuation as seen by a coalition that knows every other company in the cell."""
    return cfg.ref * float(np.exp(release_log_sum - clipped_logs(cfg, others).sum()))


def cell_sizes(n_companies: int, n_cells: int, concentration: float, rng: np.random.Generator) -> np.ndarray:
    """shape (n_cells,) companies per cell; Dirichlet shares give a few large and many small cells."""
    shares = rng.dirichlet(np.full(n_cells, concentration))
    return rng.multinomial(n_companies, shares)
