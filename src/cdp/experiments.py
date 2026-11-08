"""Experiments for the article. Each returns a pandas DataFrame; `run_all` also saves CSV + PNG.

    python -m cdp.experiments --out results --seeds 200
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import attacks, dp
from .data import MarketConfig, generate

TRADE_CLIP = 1_000_000.0       # privacy unit "trade": hide any single trade up to 1M
DEALER_DAY_CLIP = 12_000_000.0  # privacy unit "dealer_day": hide a dealer's whole day up to 12M
EPSILONS = [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0]


def _clip_for(unit: str) -> float:
    return TRADE_CLIP if unit == "trade" else DEALER_DAY_CLIP


# --------------------------------------------------------------- E1 baseline
def e1_exact_publication(seeds: int) -> pd.DataFrame:
    """No DP. (a) coalition recovery, D=3; (b) single insider with 60% participation."""
    rows = []
    for s in range(seeds):
        rng = np.random.default_rng(s)
        m = generate(MarketConfig(), rng)
        est = attacks.coalition_attack(m.daily_total, m.dealer_day, victim=0)
        rows.append({"attack": "coalition (2 of 3 collude)", "recovered": attacks.recovery_rate(est, m.dealer_day[:, 0])})
        m2 = generate(MarketConfig(participation=0.6), rng)
        r = attacks.insider_attack_sparse(m2.daily_total, m2, insider=0)
        rows.append({"attack": "single insider, 60% participation", "recovered": r["recovered_within_10pct"] / r["days"]})
    return pd.DataFrame(rows).groupby("attack", as_index=False)["recovered"].mean()


# --------------------------------------------------------------- E2 eps sweep
def e2_epsilon_sweep(seeds: int, dealers: int = 3) -> pd.DataFrame:
    """Daily releases. utility = median |error| / true total; privacy = coalition recovery rate."""
    rows = []
    for unit in ("trade", "dealer_day"):
        clip = _clip_for(unit)
        for eps in EPSILONS:
            errs, recs = [], []
            for s in range(seeds):
                rng = np.random.default_rng(10_000 + s)
                m = generate(replace(MarketConfig(), dealers=dealers), rng)
                contrib = dp.clipped_dealer_day(m, unit, clip)                     # (T, D)
                truth = contrib.sum(axis=1)                                        # (T,)
                pub = truth + dp.laplace_noise(clip, eps, rng, size=truth.shape)  # (T,)
                errs.append(np.median(np.abs(pub - truth) / truth))
                est = attacks.coalition_attack(pub, contrib, victim=0)             # (T,)
                recs.append(attacks.recovery_rate(est, contrib[:, 0]))
            rows.append({"unit": unit, "epsilon": eps, "median_rel_error": np.mean(errs), "victim_recovered": np.mean(recs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E3 granularity
def e3_granularity(seeds: int, epsilon: float = 1.0) -> pd.DataFrame:
    """Same ε, daily vs monthly release. Monthly sums are ~22x larger, noise is the same."""
    rows = []
    for unit in ("trade", "dealer_day"):
        clip = _clip_for(unit)
        d_err, m_err = [], []
        for s in range(seeds):
            rng = np.random.default_rng(20_000 + s)
            m = generate(MarketConfig(), rng)
            daily = dp.clipped_daily_total(m, unit, clip)
            d_err.append(np.median(np.abs(dp.laplace_noise(clip, epsilon, rng, size=daily.shape)) / daily))
            # monthly: one dealer contributes up to T*clip in dealer-month terms; for "trade" Δ stays clip
            month_sens = clip if unit == "trade" else clip * m.config.days
            m_err.append(abs(dp.laplace_noise(month_sens, epsilon, rng)) / daily.sum())
        rows.append({"unit": unit, "release": "daily", "median_rel_error": np.mean(d_err)})
        rows.append({"unit": unit, "release": "monthly", "median_rel_error": np.mean(m_err)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E4 market size
def e4_dealers(seeds: int, epsilon: float = 1.0) -> pd.DataFrame:
    """Dealer-day protection gets cheaper as the market gets bigger."""
    rows = []
    for n in (3, 5, 10, 20, 50):
        errs = []
        for s in range(seeds):
            rng = np.random.default_rng(30_000 + s)
            m = generate(replace(MarketConfig(), dealers=n), rng)
            truth = dp.clipped_daily_total(m, "dealer_day", DEALER_DAY_CLIP)
            errs.append(np.median(np.abs(dp.laplace_noise(DEALER_DAY_CLIP, epsilon, rng, size=truth.shape)) / truth))
        rows.append({"dealers": n, "median_rel_error": np.mean(errs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E5 averaging
def e5_averaging(seeds: int, epsilon: float = 1.0, budget: float = 5.0) -> pd.DataFrame:
    """Re-query the same daily total k times. Without a budget the noise averages away."""
    rows = []
    for k in (1, 2, 5, 10, 25, 50, 100):
        errs, blocked = [], 0
        for s in range(seeds):
            rng = np.random.default_rng(40_000 + s)
            m = generate(MarketConfig(), rng)
            truth = float(dp.clipped_daily_total(m, "trade", TRADE_CLIP)[0])
            answers = truth + dp.laplace_noise(TRADE_CLIP, epsilon, rng, size=k)
            errs.append(abs(attacks.averaging_attack(truth, answers) - truth))
            acc = dp.PrivacyAccountant(budget=budget)
            try:
                for _ in range(k):
                    dp.release(truth, TRADE_CLIP, epsilon, acc, rng)
            except dp.BudgetExceeded:
                blocked += 1
        rows.append({"queries": k, "mean_abs_error_usd": np.mean(errs), "blocked_by_budget": blocked / seeds})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- plots
def _plot_eps(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for unit, g in df.groupby("unit"):
        ax[0].plot(g.epsilon, g.median_rel_error * 100, marker="o", label=unit)
        ax[1].plot(g.epsilon, g.victim_recovered * 100, marker="o", label=unit)
    ax[0].set(xscale="log", xlabel="ε", ylabel="median error of published total, %", title="Utility (lower = better)")
    ax[1].set(xscale="log", xlabel="ε", ylabel="victim days recovered within ±10%, %", title="Privacy (lower = better)")
    ax[1].axhline(100, ls="--", c="grey", lw=1)
    ax[1].text(0.11, 95, "exact publication = 100%", fontsize=8, color="grey", va="top")
    for a in ax:
        a.legend(title="privacy unit")
        a.grid(alpha=0.3)
    fig.suptitle("3 dealers, daily releases, Laplace mechanism")
    fig.tight_layout()
    fig.savefig(out / "fig1_epsilon_tradeoff.png", dpi=150)
    plt.close(fig)


def _plot_dealers(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(df.dealers, df.median_rel_error * 100, marker="o")
    ax.set(xscale="log", xlabel="dealers in the market", ylabel="median error, %",
           title="Hiding a whole dealer-day (ε=1, daily)")
    ax.set_xticks(df.dealers, [str(d) for d in df.dealers])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_market_size.png", dpi=150)
    plt.close(fig)


def _plot_averaging(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(df.queries, df.mean_abs_error_usd / 1e3, marker="o", label="attacker's error")
    ax.axvline(5, ls="--", c="red", lw=1)
    ax.text(5.5, ax.get_ylim()[1] * 0.9, "budget Σε=5\nstops here", color="red", fontsize=8, va="top")
    ax.set(xscale="log", xlabel="repeated queries of the same total (ε=1 each)", ylabel="error, thousand USD",
           title="Averaging attack vs privacy budget")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_averaging_attack.png", dpi=150)
    plt.close(fig)


def run_all(out: Path, seeds: int) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "e1_exact_publication": e1_exact_publication(seeds),
        "e2_epsilon_sweep": e2_epsilon_sweep(seeds),
        "e3_granularity": e3_granularity(seeds),
        "e4_dealers": e4_dealers(seeds),
        "e5_averaging": e5_averaging(seeds),
    }
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
    _plot_eps(res["e2_epsilon_sweep"], out)
    _plot_dealers(res["e4_dealers"], out)
    _plot_averaging(res["e5_averaging"], out)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    pd.set_option("display.width", 120)
    for name, df in run_all(Path(args.out), args.seeds).items():
        print(f"\n== {name}\n{df.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
