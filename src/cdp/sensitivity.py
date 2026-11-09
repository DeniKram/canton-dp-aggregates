"""E8: do the conclusions survive other assumptions about the market?

Each scenario changes one assumption of the base case. Clip bounds follow a fixed public rule
instead of being tuned per scenario (tuning on the data would itself leak):
  trade clip       = 4 x median trade size
  dealer-day clip  = 2 x expected dealer-day volume
  grant clip       = 2 x median grant

Conclusions checked (each must hold in every scenario):
  C1  trade-level ε=1 still exposes a dealer's day often      (recovered > 25%)
  C2  dealer-day ε=1 exposes it rarely                         (recovered < 10%)
  C3  dealer-day protection: costly with 3 dealers, cheaper with 20 (error at 3 > 2.5 x error at 20)
  C4  tree beats an evenly split budget for daily snapshots    (error ratio at ε=5 > 2)
  C5  unaccounted per-day noise leaks under averaging          (strong attack, claimed ε=5, > 15%)
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pandas as pd

from . import attacks, captable, dp
from .data import MarketConfig, generate

MARKET_SCENARIOS = {
    "base": {},
    "few trades (5/day)": {"trades_per_day": 5.0},
    "many trades (50/day)": {"trades_per_day": 50.0},
    "heavy tails (sigma 1.5)": {"size_sigma": 1.5},
    "small trades (median 50k)": {"median_trade_usd": 50_000.0},
    "large trades (median 1M)": {"median_trade_usd": 1_000_000.0},
    "sparse (60% participation)": {"participation": 0.6},
}

CAP_SCENARIOS = {
    "base": {},
    "rare grants (0.05/day)": {"grants_per_day": 0.05},
    "frequent grants (0.4/day)": {"grants_per_day": 0.4},
    "uniform grants (sigma 0.5)": {"grant_sigma": 0.5},
    "heavy-tailed grants (sigma 1.5)": {"grant_sigma": 1.5},
    "short horizon (32 days)": {"days": 32},
    "long horizon (128 days)": {"days": 128},
}


def _market_cfg(changes: dict, dealers: int) -> MarketConfig:
    cfg = replace(MarketConfig(), dealers=dealers, **changes)
    return replace(cfg, max_trade_usd=20 * cfg.median_trade_usd)


def _clips(cfg: MarketConfig) -> tuple[float, float]:
    mean_trade = cfg.median_trade_usd * math.exp(cfg.size_sigma**2 / 2)
    return 4 * cfg.median_trade_usd, 2 * cfg.trades_per_day * mean_trade


def _market_metrics(changes: dict, seeds: int) -> dict:
    out = {}
    for unit in ("trade", "dealer_day"):
        errs, recs = [], []
        for s in range(seeds):
            rng = np.random.default_rng(80_000 + s)
            cfg = _market_cfg(changes, 3)
            clip = _clips(cfg)[0 if unit == "trade" else 1]
            m = generate(cfg, rng)
            contrib = dp.clipped_dealer_day(m, unit, clip)                  # (T, D)
            pub = contrib.sum(axis=1) + dp.laplace_noise(clip, 1.0, rng, size=m.config.days)
            errs.append(np.median(np.abs(pub - m.daily_total) / np.maximum(m.daily_total, 1)))
            r = attacks.recovery_rate(attacks.coalition_attack(pub, contrib), m.dealer_day[:, 0])
            if not np.isnan(r):
                recs.append(r)
        out[f"{unit}_err"], out[f"{unit}_leak"] = float(np.mean(errs)), float(np.mean(recs))
    errs20 = []
    for s in range(seeds):
        rng = np.random.default_rng(90_000 + s)
        cfg = _market_cfg(changes, 20)
        clip = _clips(cfg)[1]
        m = generate(cfg, rng)
        pub = dp.clipped_daily_total(m, "dealer_day", clip) + dp.laplace_noise(clip, 1.0, rng, size=cfg.days)
        errs20.append(np.median(np.abs(pub - m.daily_total) / np.maximum(m.daily_total, 1)))
    out["dealer_day_err_20"] = float(np.mean(errs20))
    return out


def _cap_metrics(changes: dict, seeds: int) -> dict:
    cfg = replace(captable.CapTableConfig(), **changes)
    C = 2 * cfg.median_grant
    res = {"tree": [], "split": []}
    strong = []
    for s in range(seeds):
        rng = np.random.default_rng(95_000 + s)
        daily, one = captable.single_grant_days(cfg, rng)
        x = np.minimum(daily, C)
        truth = cfg.initial_shares + np.cumsum(daily)
        tree = cfg.initial_shares + captable.tree_prefix_sums(x, C, 5.0, rng)
        split = cfg.initial_shares + np.cumsum(x) + dp.laplace_noise(C * cfg.days, 5.0, rng, size=cfg.days)
        perday = cfg.initial_shares + np.cumsum(x) + dp.laplace_noise(C, 5.0, rng, size=cfg.days)
        res["tree"].append(np.median(np.abs(tree - truth) / truth))
        res["split"].append(np.median(np.abs(split - truth) / truth))
        for t in np.where(one & (daily > 0))[0]:
            others = cfg.initial_shares + np.cumsum(np.where(np.arange(cfg.days) == t, 0.0, x))
            strong.append(abs(attacks.snapshot_jump_attack(perday, others, int(t)) - daily[t]) <= 0.1 * daily[t])
    tree_e, split_e = float(np.mean(res["tree"])), float(np.mean(res["split"]))
    return {"tree_err": tree_e, "split_err": split_e, "split_over_tree": split_e / tree_e,
            "perday_strong_leak": float(np.mean(strong)) if strong else float("nan")}


def e8_sensitivity(seeds: int = 100) -> tuple[pd.DataFrame, pd.DataFrame]:
    mrows = []
    for name, ch in MARKET_SCENARIOS.items():
        r = _market_metrics(ch, seeds)
        r["scenario"] = name
        r["C1"] = r["trade_leak"] > 0.25
        r["C2"] = r["dealer_day_leak"] < 0.10
        r["C3"] = r["dealer_day_err"] > 2.5 * r["dealer_day_err_20"]
        mrows.append(r)
    crows = []
    for name, ch in CAP_SCENARIOS.items():
        r = _cap_metrics(ch, seeds)
        r["scenario"] = name
        r["C4"] = r["split_over_tree"] > 2
        r["C5"] = r["perday_strong_leak"] > 0.15
        crows.append(r)
    cols_m = ["scenario", "trade_err", "trade_leak", "dealer_day_err", "dealer_day_leak", "dealer_day_err_20", "C1", "C2", "C3"]
    cols_c = ["scenario", "tree_err", "split_err", "split_over_tree", "perday_strong_leak", "C4", "C5"]
    return pd.DataFrame(mrows)[cols_m], pd.DataFrame(crows)[cols_c]


if __name__ == "__main__":
    pd.set_option("display.width", 160)
    m, c = e8_sensitivity()
    print(m.round(3).to_string(index=False))
    print()
    print(c.round(3).to_string(index=False))
