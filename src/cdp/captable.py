"""Daily snapshots of a private company's cap table, and continual-release DP for them.

Model (illustrative): a company with 10M shares outstanding. Over T days new grants arrive
(Poisson, 0.15/day); grant size log-normal with median 100k shares, capped at 2M.
A feed publishes total shares outstanding once a day (a running sum), like a daily snapshot.

Attack: the day-over-day difference of published totals is the size of that day's grant.

Defences compared, all for the privacy unit "one grant" (clipped at C shares):
  naive-per-day  each daily total + Lap(C/ε)       -> every snapshot spends ε: T days cost T·ε in total
  naive-split    each daily total + Lap(C·T/ε)     -> honest total ε, but noise grows with T
  increments     each day's NEW shares + Lap(C/ε), then a running sum
                                                   -> honest total ε, noise grows with √T
  tree           binary-tree continual release     -> honest total ε, noise grows with log T
                 (Dwork, Naor, Pitassi, Rothblum 2010; Chan, Shi, Song 2011)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CapTableConfig:
    days: int = 64
    initial_shares: float = 10_000_000.0
    grants_per_day: float = 0.15
    median_grant: float = 100_000.0
    grant_sigma: float = 1.0
    max_grant: float = 2_000_000.0


def generate_grants(cfg: CapTableConfig, rng: np.random.Generator) -> np.ndarray:
    """Total shares granted on each day. shape (T,). Most days are 0."""
    counts = rng.poisson(cfg.grants_per_day, size=cfg.days)
    mu = math.log(cfg.median_grant)
    return np.array([np.minimum(rng.lognormal(mu, cfg.grant_sigma, size=n), cfg.max_grant).sum() for n in counts])


def single_grant_days(cfg: CapTableConfig, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Like generate_grants but also returns a mask of days with exactly one grant. shapes (T,), (T,)"""
    counts = rng.poisson(cfg.grants_per_day, size=cfg.days)
    mu = math.log(cfg.median_grant)
    daily = np.array([np.minimum(rng.lognormal(mu, cfg.grant_sigma, size=n), cfg.max_grant).sum() for n in counts])
    return daily, counts == 1


def tree_levels(days: int) -> int:
    """Number of dyadic levels covering `days` steps: ceil(log2 T) + 1.  T=64 -> 7"""
    return math.ceil(math.log2(days)) + 1


def tree_prefix_sums(x: np.ndarray, sensitivity: float, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Binary-tree mechanism: noisy running sums of x with total privacy loss ε.

    Every dyadic block [k·2^j, (k+1)·2^j) gets its own noisy sum. One value x[t] lies in exactly
    one block per level, i.e. in L = tree_levels(T) blocks, so each block gets Lap(Δ·L/ε) and the
    whole stream costs ε. The running sum at t adds at most L blocks.
        x shape (T,) -> returns shape (T,)
    """
    T = x.shape[0]
    L = tree_levels(T)
    b = sensitivity * L / epsilon
    noisy_block: dict[tuple[int, int], float] = {}
    for j in range(L):
        size = 2**j
        for k in range(math.ceil(T / size)):
            noisy_block[(j, k)] = float(x[k * size:(k + 1) * size].sum() + rng.laplace(0.0, b))
    out = np.empty(T)
    for t in range(T):
        n, acc, start = t + 1, 0.0, 0
        for j in reversed(range(L)):  # greedy dyadic decomposition of [0, n)
            size = 2**j
            if n - start >= size:
                acc += noisy_block[(j, start // size)]
                start += size
        out[t] = acc
    return out


def increment_prefix_sums(x: np.ndarray, sensitivity: float, epsilon: float, rng: np.random.Generator) -> np.ndarray:
    """Input perturbation: noise every daily increment once, then take the running sum.

    One grant changes exactly one increment x[t] by at most Δ, so Lap(Δ/ε) per increment makes the
    whole stream cost ε. The running sum at day t carries t independent noises: std = √(2t)·Δ/ε.
    Better than the tree for short streams, worse for long ones (√T against log^1.5 T).
        x shape (T,) -> returns shape (T,)
    """
    return np.cumsum(x + rng.laplace(0.0, sensitivity / epsilon, size=x.shape[0]))
