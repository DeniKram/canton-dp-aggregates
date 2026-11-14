"""One aggregate across many cap-table registers: threshold suppression against DP.

Setting, after the Canton Asset Data Feed proposal (PR #890): a transfer agent publishes ONE daily
figure, shares outstanding summed over all R registers it operates, never a company-level item, and
applies a suppression rule such as `min-count-5` to daily changes.

Model (illustrative): R registers, each with Poisson(0.15) grants a day, grant size log-normal with
median 100k shares (as in captable.py). A grant is the triple (day, register, size).

Release rules compared. Each produces a list of `Release`s; a release is one published change of
the level, together with the grants that went into it (kept for scoring the attacks):
  exact       the level moves every day. Suppressing only the "change" item while still publishing
              the level daily is the same thing: the change is the difference of two levels.
  threshold   the level moves only when at least k grants are pending, then by their exact sum.
  dp          the level moves every day by sum(min(grant, C)) + Lap(C/ε): increment noise, ε per grant.

Attackers, all guessing that what they cannot explain in a published change is ONE grant:
  outsider    knows nothing: guesses the largest grant in a release equals the whole change.
  insider     one register: subtracts its own grants, then guesses the same about the remainder.
  coalition   knows every grant but the target (the DP threat model): subtracts them all.
A grant counts as recovered when the guess is within ±10% of its true size.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RegistersConfig:
    registers: int = 20
    days: int = 64
    grants_per_day: float = 0.15   # per register
    median_grant: float = 100_000.0
    grant_sigma: float = 1.0
    max_grant: float = 2_000_000.0


@dataclass
class Grants:
    """Parallel arrays, one entry per grant, sorted by day. shapes (N,)"""

    day: np.ndarray
    register: np.ndarray
    size: np.ndarray


@dataclass
class Release:
    day: int            # day the level moved
    change: float       # published change of the level
    idx: np.ndarray     # indices (into Grants) of the grants inside this change
    contrib: np.ndarray  # what each of them added to `change` (size, or clipped size under DP)


def generate(cfg: RegistersConfig, rng: np.random.Generator) -> Grants:
    counts = rng.poisson(cfg.grants_per_day, size=(cfg.days, cfg.registers))      # (T, R)
    day, reg = np.nonzero(counts)
    day, reg = np.repeat(day, counts[day, reg]), np.repeat(reg, counts[day, reg])  # (N,), (N,)
    size = np.minimum(rng.lognormal(math.log(cfg.median_grant), cfg.grant_sigma, size=day.size), cfg.max_grant)
    return Grants(day=day, register=reg, size=size)


def true_level(g: Grants, days: int) -> np.ndarray:
    """Cumulative shares issued by the end of each day. shape (T,)"""
    return np.cumsum(np.bincount(g.day, weights=g.size, minlength=days))


def release_exact(g: Grants, days: int) -> list[Release]:
    return [Release(t, float(g.size[i].sum()), i, g.size[i]) for t in range(days)
            if (i := np.nonzero(g.day == t)[0]).size]


def release_threshold(g: Grants, days: int, k: int = 5) -> list[Release]:
    out, pending = [], np.zeros(0, dtype=int)
    for t in range(days):
        pending = np.concatenate([pending, np.nonzero(g.day == t)[0]])
        if pending.size >= k:
            out.append(Release(t, float(g.size[pending].sum()), pending, g.size[pending]))
            pending = np.zeros(0, dtype=int)
    return out


def release_dp(g: Grants, days: int, clip: float, epsilon: float, rng: np.random.Generator) -> list[Release]:
    out = []
    for t in range(days):  # every day is published, also days without grants
        i = np.nonzero(g.day == t)[0]
        c = np.minimum(g.size[i], clip)
        out.append(Release(t, float(c.sum() + rng.laplace(0.0, clip / epsilon)), i, c))
    return out


def published_level(releases: list[Release], days: int) -> np.ndarray:
    """The level a reader sees at the end of each day (stale between releases). shape (T,)"""
    step = np.zeros(days)
    for r in releases:
        step[r.day] += r.change
    return np.cumsum(step)


def recovered(releases: list[Release], g: Grants, attacker: str, tol: float = 0.10) -> tuple[int, int]:
    """(grants recovered, grants the attacker is after). The insider is register 0."""
    hits = 0
    for r in releases:
        if r.idx.size == 0:
            continue
        size, contrib, reg = g.size[r.idx], r.contrib, g.register[r.idx]
        if attacker == "coalition":       # each grant in turn is the target; all the others are known
            guess = r.change - (contrib.sum() - contrib)
            hits += int(np.sum(np.abs(guess - size) <= tol * size))
            continue
        unknown = np.ones(r.idx.size, dtype=bool) if attacker == "outsider" else reg != 0
        if not unknown.any():
            continue
        guess = r.change - contrib[~unknown].sum()
        target = size[unknown].max()      # the best single grant this guess could be about
        hits += int(abs(guess - target) <= tol * target)
    total = g.size.size if attacker != "insider" else int(np.sum(g.register != 0))
    return hits, total
