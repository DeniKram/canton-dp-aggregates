"""Threshold release for cap-table metrics: one shared counter vs one counter per metric.

Rule under test: hold back metric updates until k qualifying
changes have accumulated, then publish. The intent is that every published delta mixes at least
k changes, so no single change can be read off it.

The question is *what* is counted.

  shared      one counter over all change types; when it reaches k, every metric is published.
              A metric touched by only one of those k changes still exposes that change exactly.
  per-metric  one counter per metric (and per share class); a metric is published only when
              its own counter reaches k. Exposure is zero by construction; the cost is delay.

Company model (illustrative, not measured on real data). Change types and daily rates:

  issuance       common shares granted/exercised   shares_issued, outstanding_common, holders_count*
  transfer       secondary sale of common          holders_count*
  conversion     preferred -> common               outstanding_common, outstanding_preferred
  preferred_round new preferred shares             shares_issued, outstanding_preferred, holders_count*

  * holders_count changes only if the change adds or removes a holder (probability `new_holder_p`).

A valuation metric is left out: it is usually gated by a count of companies, not of changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

METRICS = ("shares_issued", "outstanding_common", "outstanding_preferred", "holders_count")

# change type -> metrics it always touches; holders_count is added separately (probabilistic)
TOUCHES: dict[str, tuple[str, ...]] = {
    "issuance": ("shares_issued", "outstanding_common"),
    "transfer": (),
    "conversion": ("outstanding_common", "outstanding_preferred"),
    "preferred_round": ("shares_issued", "outstanding_preferred"),
}
MAY_CHANGE_HOLDERS = {"issuance", "transfer", "preferred_round"}


@dataclass(frozen=True)
class ThresholdConfig:
    days: int = 365
    rates: dict[str, float] = field(default_factory=lambda: {
        "issuance": 0.15,
        "transfer": 0.20,
        "conversion": 0.05,
        "preferred_round": 0.01,
    })
    new_holder_p: float = 0.5


@dataclass(frozen=True)
class Events:
    """One row per change. day shape (N,), touches shape (N, M) bool, M = len(METRICS)."""
    day: np.ndarray
    touches: np.ndarray


def generate_events(cfg: ThresholdConfig, rng: np.random.Generator) -> Events:
    """Changes over `cfg.days`, sorted by day (ties in random order)."""
    days, rows = [], []
    for kind, rate in cfg.rates.items():
        n = rng.poisson(rate * cfg.days)
        d = rng.integers(0, cfg.days, size=n)
        for di in d:
            row = np.zeros(len(METRICS), dtype=bool)
            for m in TOUCHES[kind]:
                row[METRICS.index(m)] = True
            if kind in MAY_CHANGE_HOLDERS and rng.random() < cfg.new_holder_p:
                row[METRICS.index("holders_count")] = True
            days.append(int(di))
            rows.append(row)
    if not rows:
        return Events(np.zeros(0, dtype=int), np.zeros((0, len(METRICS)), dtype=bool))
    order = np.lexsort((rng.random(len(days)), np.array(days)))
    return Events(np.array(days)[order], np.array(rows)[order])


@dataclass(frozen=True)
class ReleaseResult:
    """
    releases          number of (metric, release) pairs where the metric's value changed
    exposed_single    of those, how many mix exactly 1 change  -> that change is read off exactly
    below_k           of those, how many mix fewer than k changes (includes exposed_single)
    lag_days          per published change, days from the change to its publication, shape (P,)
    unpublished       changes (per metric) still held back at the end of the horizon
    total_changes     (change, metric) pairs in the period
    """
    releases: int
    exposed_single: int
    below_k: int
    lag_days: np.ndarray
    unpublished: int
    total_changes: int


def release_shared(ev: Events, k: int) -> ReleaseResult:
    """One counter over all changes; every k-th change triggers publication of all metrics."""
    M = ev.touches.shape[1]
    pending: list[list[int]] = [[] for _ in range(M)]  # day of each held change, per metric
    count = 0
    releases = exposed = below = 0
    lags: list[int] = []
    for d, row in zip(ev.day, ev.touches):
        for m in np.flatnonzero(row):
            pending[m].append(int(d))
        count += 1
        if count >= k:
            for m in range(M):
                n = len(pending[m])
                if n == 0:
                    continue
                releases += 1
                exposed += n == 1
                below += n < k
                lags.extend(int(d) - p for p in pending[m])
                pending[m] = []
            count = 0
    return ReleaseResult(releases, exposed, below, np.array(lags), sum(map(len, pending)),
                         int(ev.touches.sum()))


def release_per_metric(ev: Events, k: int) -> ReleaseResult:
    """One counter per metric; a metric is published when its own counter reaches k."""
    M = ev.touches.shape[1]
    pending: list[list[int]] = [[] for _ in range(M)]
    releases = exposed = below = 0
    lags: list[int] = []
    for d, row in zip(ev.day, ev.touches):
        for m in np.flatnonzero(row):
            pending[m].append(int(d))
            if len(pending[m]) >= k:
                releases += 1
                exposed += len(pending[m]) == 1  # only possible when k == 1
                below += len(pending[m]) < k
                lags.extend(int(d) - p for p in pending[m])
                pending[m] = []
    return ReleaseResult(releases, exposed, below, np.array(lags), sum(map(len, pending)),
                         int(ev.touches.sum()))


def per_metric_exposure(ev: Events, k: int) -> np.ndarray:
    """Shared counter: share of each metric's releases that expose a single change. shape (M,)"""
    M = ev.touches.shape[1]
    pending = np.zeros(M, dtype=int)
    rel, exp_ = np.zeros(M), np.zeros(M)
    count = 0
    for row in ev.touches:
        pending += row
        count += 1
        if count >= k:
            hit = pending > 0
            rel += hit
            exp_ += pending == 1
            pending[:] = 0
            count = 0
    return np.divide(exp_, rel, out=np.full(M, np.nan), where=rel > 0)
