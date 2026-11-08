"""Attacks on published aggregates.

1. Differencing (coalition) attack - the standard DP threat model:
   all dealers except the victim collude. They know their own volumes exactly, so
       victim_estimate = published_total - sum(own volumes)
   Exact publication -> recovered perfectly. With DP the error is exactly the noise.

2. Single-insider attack with sparse participation (more realistic, D=3):
   on days when the third dealer did not trade, an insider recovers the other one exactly.

3. Averaging attack: if the same quantity can be re-queried with fresh noise,
   averaging k answers shrinks noise std by √k. A privacy budget stops this.
"""

from __future__ import annotations

import numpy as np

from .data import Market


def coalition_attack(published: np.ndarray, contributions: np.ndarray, victim: int = 0) -> np.ndarray:
    """Estimate the victim's daily contribution.

    published:     (T,)   released totals
    contributions: (T, D) what each dealer actually contributed to the total (after any clipping;
                          colluders know theirs, and the clipping rule is public)
    returns:       (T,)   victim estimate
    """
    others = np.delete(contributions, victim, axis=1).sum(axis=1)
    return published - others


def recovery_rate(estimate: np.ndarray, truth: np.ndarray, tol: float = 0.10) -> float:
    """Share of days where the estimate is within ±tol (relative) of the truth; only days victim traded."""
    mask = truth > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs(estimate[mask] - truth[mask]) <= tol * truth[mask]))


def insider_attack_sparse(published: np.ndarray, market: Market, insider: int = 0) -> dict:
    """Insider sees published totals plus own volume. On days exactly one other dealer traded,
    that dealer's volume is fully revealed. Returns how many dealer-days leaked."""
    dd = market.dealer_day
    leaked, exact = 0, 0
    for t in range(dd.shape[0]):
        others = [d for d in range(dd.shape[1]) if d != insider and dd[t, d] > 0]
        if len(others) == 1:
            # The insider does not know who traded, only that the residual belongs to one dealer;
            # in thin markets this is often obvious (e.g. from chat/RFQ context). Count it as leaked.
            leaked += 1
            est = published[t] - dd[t, insider]
            exact += int(abs(est - dd[t, others[0]]) <= 0.10 * dd[t, others[0]])
    return {"days": int(dd.shape[0]), "leaked_days": leaked, "recovered_within_10pct": exact}


def snapshot_diff_attack(published: np.ndarray, start: float) -> np.ndarray:
    """Day-over-day differences of published running totals = estimated daily inflow. (T,) -> (T,)"""
    return np.diff(np.concatenate([[start], published]))


def snapshot_jump_attack(published: np.ndarray, others_running: np.ndarray, t: int) -> float:
    """Strong attacker who knows every other grant: subtract them, then compare the mean level
    after day t with the mean level before it. Averages over many noisy snapshots, so it beats
    independent per-day noise whose total budget was never accounted for.
        published, others_running: (T,)  ->  estimated grant on day t
    """
    r = published - others_running
    return float(r[t:].mean() - (r[:t].mean() if t > 0 else 0.0))


def averaging_attack(true_value: float, noisy_answers: np.ndarray) -> float:
    """Mean of k noisy answers to the same query."""
    return float(np.mean(noisy_answers))
