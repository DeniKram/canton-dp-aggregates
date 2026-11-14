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

from . import attacks, captable, dp, registers, sensitivity
from .data import MarketConfig, generate

TRADE_CLIP = 1_000_000.0       # privacy unit "trade": hide any single trade up to 1M
MONTHLY_TRADE_CLIP = 2_000_000.0  # a monthly sum tolerates a larger bound: less clipping bias
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
            errs, p90s, recs = [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(10_000 + s)
                m = generate(replace(MarketConfig(), dealers=dealers), rng)
                contrib = dp.clipped_dealer_day(m, unit, clip)                     # (T, D)
                clipped = contrib.sum(axis=1)                                      # (T,)
                pub = clipped + dp.laplace_noise(clip, eps, rng, size=clipped.shape)  # (T,)
                truth = m.daily_total                                              # (T,) unclipped
                rel = np.abs(pub - truth) / truth                                  # (T,)
                errs.append(np.median(rel))
                p90s.append(np.quantile(rel, 0.9))
                est = attacks.coalition_attack(pub, contrib, victim=0)             # (T,)
                recs.append(attacks.recovery_rate(est, m.dealer_day[:, 0]))       # vs TRUE volume
            rows.append({"unit": unit, "epsilon": eps, "median_rel_error": np.mean(errs),
                         "p90_rel_error": np.mean(p90s), "victim_recovered": np.mean(recs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E3 granularity
def e3_granularity(seeds: int, epsilon: float = 1.0) -> pd.DataFrame:
    """Same ε, daily vs monthly release. Monthly sums are ~22x larger; for the trade unit the
    monthly release uses a larger clip bound (2M), since a month accumulates clipping bias.

    Both rows report a MEDIAN: daily = median over the 22 days (averaged over markets), monthly =
    median over markets of the one monthly figure. `mean_rel_error` is the mean of the same errors;
    for Laplace noise the mean is ~1.44x the median, so never compare a mean with a median."""
    rows = []
    for unit in ("trade", "dealer_day"):
        clip = _clip_for(unit)
        d_err, d_mean, m_err = [], [], []
        for s in range(seeds):
            rng = np.random.default_rng(20_000 + s)
            m = generate(MarketConfig(), rng)
            clipped, truth = dp.clipped_daily_total(m, unit, clip), m.daily_total   # (T,), (T,)
            d_pub = clipped + dp.laplace_noise(clip, epsilon, rng, size=clipped.shape)
            d_err.append(np.median(np.abs(d_pub - truth) / truth))
            d_mean.append(np.mean(np.abs(d_pub - truth) / truth))
            # monthly: one dealer contributes up to T*clip in dealer-month terms; for "trade" Δ stays clip
            if unit == "trade":
                m_clip = MONTHLY_TRADE_CLIP
                m_sum, month_sens = dp.clipped_daily_total(m, unit, m_clip).sum(), m_clip
            else:
                m_sum, month_sens = clipped.sum(), clip * m.config.days
            m_pub = m_sum + dp.laplace_noise(month_sens, epsilon, rng)
            m_err.append(abs(m_pub - truth.sum()) / truth.sum())
        rows.append({"unit": unit, "release": "daily", "median_rel_error": np.mean(d_err), "mean_rel_error": np.mean(d_mean)})
        rows.append({"unit": unit, "release": "monthly", "median_rel_error": np.median(m_err), "mean_rel_error": np.mean(m_err)})
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
            clipped, truth = dp.clipped_daily_total(m, "dealer_day", DEALER_DAY_CLIP), m.daily_total
            pub = clipped + dp.laplace_noise(DEALER_DAY_CLIP, epsilon, rng, size=clipped.shape)
            errs.append(np.median(np.abs(pub - truth) / truth))
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


# --------------------------------------------------------------- E6 cap table
GRANT_CLIP = 200_000.0  # privacy unit "one grant": a day's new shares clipped at 200k


def _publish(mode: str, x: np.ndarray, start: float, eps: float, days: int, rng: np.random.Generator,
             raw: np.ndarray | None = None) -> np.ndarray:
    if mode == "exact":           # the real, unclipped running total
        return start + np.cumsum(raw if raw is not None else x)
    truth = start + np.cumsum(x)
    if mode == "naive-per-day":   # ε per snapshot, never accounted: real cost days·ε
        return truth + dp.laplace_noise(GRANT_CLIP, eps, rng, size=truth.shape)
    if mode == "naive-split":     # honest: ε split over all snapshots
        return truth + dp.laplace_noise(GRANT_CLIP * days, eps, rng, size=truth.shape)
    if mode == "tree":            # honest: binary-tree continual release
        return start + captable.tree_prefix_sums(x, GRANT_CLIP, eps, rng)
    if mode == "increments":      # honest: noise each day's new shares once, then running sum
        return start + captable.increment_prefix_sums(x, GRANT_CLIP, eps, rng)
    raise ValueError(mode)


def e6_cap_table(seeds: int) -> pd.DataFrame:
    """Daily cap-table totals (64 days). Error of the published total vs share of single-grant days
    whose grant size an attacker recovers within ±10%, with a simple (day-over-day) and a strong
    (knows all other grants, averages) attack.

    Error is reported three ways, because the 10M starting total is public and inflates the base:
      median_rel_error       |published - true| / true total
      median_abs_error       the same error in shares
      error_vs_issuance      that error in shares / all shares issued during the 64 days"""
    cfg = captable.CapTableConfig()
    rows = []
    for mode in ("exact", "naive-per-day", "naive-split", "tree", "increments"):
        for eps in ([1.0] if mode == "exact" else [1.0, 2.0, 5.0, 10.0]):
            errs, abs_errs, issued, simple, strong = [], [], [], [], []
            for s in range(seeds):
                rng = np.random.default_rng(50_000 + s)
                daily, one = captable.single_grant_days(cfg, rng)
                x = np.minimum(daily, GRANT_CLIP)                                   # (T,) clipped
                truth = cfg.initial_shares + np.cumsum(daily)                       # (T,) unclipped
                pub = _publish(mode, x, cfg.initial_shares, eps, cfg.days, rng, raw=daily)  # (T,)
                errs.append(np.median(np.abs(pub - truth) / truth))
                abs_errs.append(np.median(np.abs(pub - truth)))
                issued.append(daily.sum())
                diff = attacks.snapshot_diff_attack(pub, cfg.initial_shares)
                contrib = daily if mode == "exact" else x   # what each grant adds to what was published
                for t in np.where(one & (daily > 0))[0]:
                    others = cfg.initial_shares + np.cumsum(np.where(np.arange(cfg.days) == t, 0.0, contrib))
                    g = daily[t]                             # the TRUE grant the attacker wants
                    simple.append(abs(diff[t] - g) <= 0.1 * g)
                    strong.append(abs(attacks.snapshot_jump_attack(pub, others, int(t)) - g) <= 0.1 * g)
            rows.append({
                "mechanism": mode,
                "eps_claimed": float("inf") if mode == "exact" else eps,
                "eps_actual": float("inf") if mode == "exact" else (eps * cfg.days if mode == "naive-per-day" else eps),
                "median_rel_error": np.mean(errs),
                "median_abs_error": np.mean(abs_errs),
                "error_vs_issuance": np.mean(abs_errs) / np.mean(issued),
                "recovered_simple": np.mean(simple),
                "recovered_strong": np.mean(strong),
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E7 clipping
def e7_clipping(seeds: int) -> pd.DataFrame:
    """Clip bound C trades bias (large contributions cut) against noise (Lap scale grows with C).
    Error is measured against the TRUE (unclipped) total."""
    rows = []
    cfg = captable.CapTableConfig()
    for C in (50e3, 100e3, 200e3, 500e3, 1e6):
        errs, bias = [], []
        for s in range(seeds):
            rng = np.random.default_rng(50_000 + s)  # same markets as E6
            daily, _ = captable.single_grant_days(cfg, rng)
            x = np.minimum(daily, C)
            truth = cfg.initial_shares + np.cumsum(daily)
            pub = cfg.initial_shares + captable.tree_prefix_sums(x, C, 5.0, rng)
            errs.append(np.median(np.abs(pub - truth) / truth))
            bias.append(np.median((truth - cfg.initial_shares - np.cumsum(x)) / truth))
        rows.append({"case": "cap table, tree, eps=5", "clip": C, "median_rel_error": np.mean(errs), "clip_bias": np.mean(bias)})
    for C in (0.25e6, 0.5e6, 1e6, 2e6, 5e6):
        errs, bias = [], []
        for s in range(seeds):
            rng = np.random.default_rng(70_000 + s)
            m = generate(MarketConfig(), rng)
            clipped, truth = dp.clipped_daily_total(m, "trade", C), m.daily_total
            pub = clipped + dp.laplace_noise(C, 1.0, rng, size=clipped.shape)
            errs.append(np.median(np.abs(pub - truth) / truth))
            bias.append(np.median((truth - clipped) / truth))
        rows.append({"case": "3 dealers, trade unit, eps=1", "clip": C, "median_rel_error": np.mean(errs), "clip_bias": np.mean(bias)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E9 horizon
def e9_horizon(seeds: int, epsilon: float = 5.0) -> pd.DataFrame:
    """Which honest mechanism for which stream length? Same total ε, same clip bound, T = 32..2048.
    Also tracks what clipping does to a running total: the cut shares never come back, so the
    published total drifts further below the true one the longer the feed runs."""
    rows = []
    for T in (32, 64, 128, 256, 512, 1024, 2048):
        cfg = captable.CapTableConfig(days=T)
        acc = {k: {"rel": [], "abs": []} for k in ("naive-split", "tree", "increments")}
        issued, cut = [], []
        for s in range(seeds):
            rng = np.random.default_rng(60_000 + s)
            daily, _ = captable.single_grant_days(cfg, rng)
            x = np.minimum(daily, GRANT_CLIP)
            truth = cfg.initial_shares + np.cumsum(daily)
            issued.append(daily.sum())
            cut.append(daily.sum() - x.sum())
            for mode in acc:
                pub = _publish(mode, x, cfg.initial_shares, epsilon, T, rng)
                acc[mode]["rel"].append(np.median(np.abs(pub - truth) / truth))
                acc[mode]["abs"].append(np.median(np.abs(pub - truth)))
        row = {"days": T, "issued_shares": np.mean(issued), "clipped_away_shares": np.mean(cut),
               "clipped_away_share": np.sum(cut) / np.sum(issued)}
        for mode, d in acc.items():
            row[f"{mode}_rel_error"], row[f"{mode}_abs_error"] = np.mean(d["rel"]), np.mean(d["abs"])
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------- E10 threshold vs DP
def e10_threshold_vs_dp(seeds: int) -> pd.DataFrame:
    """One aggregate over R registers (the CADF equity design): exact daily level, min-count-5
    suppression, and DP increment noise. Utility: median error of the published level, in shares and
    against shares issued so far. Privacy: grants recovered within ±10% by three attackers."""
    rules = [
        ("exact daily level", None),
        ("threshold min-count-5", None),
        ("DP eps=5, clip 200k", (200_000.0, 5.0)),
        ("DP eps=5, clip 1M", (1_000_000.0, 5.0)),
        ("DP eps=1, clip 1M", (1_000_000.0, 1.0)),
    ]
    rows = []
    for R in (2, 5, 20, 50):
        cfg = registers.RegistersConfig(registers=R)
        for name, par in rules:
            abs_err, rel_err, lag = [], [], []
            hits = {a: [0, 0] for a in ("outsider", "insider", "coalition")}
            for s in range(seeds):
                rng = np.random.default_rng(110_000 + s)
                g = registers.generate(cfg, rng)
                if g.size.size == 0:
                    continue
                truth = registers.true_level(g, cfg.days)                    # (T,)
                if name.startswith("exact"):
                    rel = registers.release_exact(g, cfg.days)
                elif name.startswith("threshold"):
                    rel = registers.release_threshold(g, cfg.days, k=5)
                else:
                    rel = registers.release_dp(g, cfg.days, par[0], par[1], rng)
                pub = registers.published_level(rel, cfg.days)               # (T,)
                err = np.abs(pub - truth)
                abs_err.append(np.median(err))
                rel_err.append(np.median(err[truth > 0] / truth[truth > 0]))
                lag.append(np.mean(~np.isclose(pub, truth, rtol=1e-9, atol=1e-3)) if par is None else np.nan)
                for a in hits:
                    h, n = registers.recovered(rel, g, a)
                    hits[a][0] += h
                    hits[a][1] += n
            rows.append({
                "registers": R, "rule": name,
                "median_abs_error": np.mean(abs_err), "error_vs_issued": np.mean(rel_err),
                "days_level_is_stale": np.mean(lag),
                **{f"recovered_{a}": (h / n if n else float("nan")) for a, (h, n) in hits.items()},
            })
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


def _plot_cap_table(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, style in (("naive-split", "o--"), ("tree", "o-"), ("increments", "s-")):
        g = df[df.mechanism == mode]
        ax.plot(g.eps_claimed, g.median_rel_error * 100, style, label=mode)
    ax.set(xscale="log", yscale="log", xlabel="total privacy budget ε (honestly accounted)",
           ylabel="median error of published total, %", title="Daily cap-table totals, 64 days")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig4_cap_table.png", dpi=150)
    plt.close(fig)


def _plot_horizon(df: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for mode, style in (("naive-split", "o--"), ("tree", "o-"), ("increments", "s-")):
        ax.plot(df.days, df[f"{mode}_abs_error"] / 1e3, style, label=mode)
    ax.plot(df.days, df.clipped_away_shares / 1e3, "k:", label="shares cut by clipping (bias)")
    ax.set(xscale="log", yscale="log", xlabel="days in the stream", ylabel="thousand shares",
           title="Cap table, ε = 5: median error of the running total")
    ax.set_xticks(df.days, [str(d) for d in df.days])
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig5_horizon.png", dpi=150)
    plt.close(fig)


def run_all(out: Path, seeds: int) -> dict[str, pd.DataFrame]:
    out.mkdir(parents=True, exist_ok=True)
    res = {
        "e1_exact_publication": e1_exact_publication(seeds),
        "e2_epsilon_sweep": e2_epsilon_sweep(seeds),
        "e3_granularity": e3_granularity(seeds),
        "e4_dealers": e4_dealers(seeds),
        "e5_averaging": e5_averaging(seeds),
        "e6_cap_table": e6_cap_table(seeds),
        "e7_clipping": e7_clipping(seeds),
        "e9_horizon": e9_horizon(min(seeds, 100)),
        "e10_threshold_vs_dp": e10_threshold_vs_dp(seeds),
    }
    res["e8_sensitivity_market"], res["e8_sensitivity_cap_table"] = sensitivity.e8_sensitivity(min(seeds, 100))
    for name, df in res.items():
        df.to_csv(out / f"{name}.csv", index=False)
    _plot_eps(res["e2_epsilon_sweep"], out)
    _plot_dealers(res["e4_dealers"], out)
    _plot_averaging(res["e5_averaging"], out)
    _plot_cap_table(res["e6_cap_table"], out)
    _plot_horizon(res["e9_horizon"], out)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=200)
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    for name, df in run_all(Path(args.out), args.seeds).items():
        print(f"\n== {name}\n{df.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
