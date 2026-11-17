import numpy as np
import pytest

from cdp import attacks, dp
from cdp.data import MarketConfig, generate


@pytest.fixture
def market():
    return generate(MarketConfig(dealers=3, days=22), np.random.default_rng(0))


def test_shapes(market):
    assert market.dealer_day.shape == (22, 3)
    assert market.daily_total.shape == (22,)
    assert dp.clipped_dealer_day(market, "trade", 1e6).shape == (22, 3)


def test_clipping_bounds_contributions(market):
    trade = dp.clipped_dealer_day(market, "trade", 1e6)
    assert all(np.minimum(x, 1e6).max(initial=0) <= 1e6 for day in market.trades for x in day)
    assert (trade <= market.dealer_day + 1e-6).all()
    assert (dp.clipped_dealer_day(market, "dealer_day", 5e6) <= 5e6).all()


def test_exact_publication_is_fully_recovered(market):
    est = attacks.coalition_attack(market.daily_total, market.dealer_day, victim=0)
    assert np.allclose(est, market.dealer_day[:, 0])
    assert attacks.recovery_rate(est, market.dealer_day[:, 0]) == 1.0


def test_laplace_scale_and_empirical_std():
    assert dp.laplace_scale(1e6, 1.0) == 1e6
    assert dp.laplace_scale(1e6, 0.5) == 2e6
    noise = dp.laplace_noise(1e6, 1.0, np.random.default_rng(1), size=200_000)
    assert noise.std() == pytest.approx(np.sqrt(2) * 1e6, rel=0.02)  # std = √2·b
    with pytest.raises(ValueError):
        dp.laplace_scale(1.0, 0.0)


def test_accountant_blocks_over_budget():
    acc = dp.PrivacyAccountant(budget=10)
    for _ in range(5):
        acc.charge(2.0)
    assert acc.remaining == pytest.approx(0)
    with pytest.raises(dp.BudgetExceeded):
        acc.charge(0.1)


def test_release_charges_before_noise():
    acc = dp.PrivacyAccountant(budget=1)
    rng = np.random.default_rng(2)
    dp.release(100.0, 1.0, 1.0, acc, rng)
    with pytest.raises(dp.BudgetExceeded):
        dp.release(100.0, 1.0, 1.0, acc, rng)
    assert acc.log == [1.0]


def test_averaging_shrinks_error():
    rng = np.random.default_rng(3)
    one = [abs(dp.laplace_noise(1e6, 1.0, rng)) for _ in range(2000)]
    hundred = [abs(np.mean(dp.laplace_noise(1e6, 1.0, rng, size=100))) for _ in range(2000)]
    assert np.mean(hundred) < np.mean(one) / 5  # ≈ /10 in theory (√100)


def test_sparse_insider_leaks_when_dealers_absent():
    m = generate(MarketConfig(participation=0.5), np.random.default_rng(4))
    r = attacks.insider_attack_sparse(m.daily_total, m, insider=0)
    assert r["leaked_days"] > 0 and r["recovered_within_10pct"] == r["leaked_days"]


def test_tree_levels():
    from cdp.captable import tree_levels
    assert tree_levels(64) == 7 and tree_levels(1) == 1 and tree_levels(65) == 8


def test_tree_prefix_sums_exact_without_noise():
    from cdp.captable import tree_prefix_sums
    x = np.arange(1.0, 38.0)  # odd length, not a power of two
    out = tree_prefix_sums(x, sensitivity=1.0, epsilon=1e12, rng=np.random.default_rng(5))
    assert np.allclose(out, np.cumsum(x), atol=1e-6)


def test_tree_noise_grows_like_log_not_linear():
    from cdp.captable import tree_prefix_sums
    rng = np.random.default_rng(6)
    x = np.zeros(64)
    err = np.mean([np.abs(tree_prefix_sums(x, 1.0, 1.0, rng)).mean() for _ in range(200)])
    naive_split = 64 * 1.0  # mean |Lap(64)|
    assert err < naive_split / 2


def test_snapshot_attacks_exact():
    x = np.array([0.0, 5.0, 0.0, 7.0])
    pub = 100 + np.cumsum(x)
    assert np.allclose(attacks.snapshot_diff_attack(pub, 100.0), x)
    others = 100 + np.cumsum(np.array([0.0, 5.0, 0.0, 0.0]))
    assert attacks.snapshot_jump_attack(pub, others, 3) == pytest.approx(7.0)


def test_sensitivity_runs_and_scale_invariant():
    from cdp import sensitivity
    m, c = sensitivity.e8_sensitivity(seeds=5)
    assert len(m) == len(sensitivity.MARKET_SCENARIOS) and len(c) == len(sensitivity.CAP_SCENARIOS)
    base = m[m.scenario == "base"].iloc[0]
    big = m[m.scenario == "large trades (median 1M)"].iloc[0]
    assert abs(base.trade_err - big.trade_err) < 1e-9  # same seeds, everything scales together


def test_increment_prefix_sums_exact_without_noise():
    from cdp.captable import increment_prefix_sums
    x = np.arange(1.0, 38.0)
    out = increment_prefix_sums(x, sensitivity=1.0, epsilon=1e12, rng=np.random.default_rng(7))
    assert np.allclose(out, np.cumsum(x), atol=1e-6)


def test_increment_noise_grows_like_sqrt_t():
    from cdp.captable import increment_prefix_sums
    rng = np.random.default_rng(8)
    runs = np.array([increment_prefix_sums(np.zeros(64), 1.0, 1.0, rng) for _ in range(4000)])  # (4000, 64)
    assert runs[:, 0].std() == pytest.approx(np.sqrt(2 * 1), rel=0.1)    # day 1:  √2·b
    assert runs[:, 63].std() == pytest.approx(np.sqrt(2 * 64), rel=0.1)  # day 64: √(2·64)·b


def test_short_stream_increments_beat_tree_long_stream_tree_wins():
    from cdp.captable import increment_prefix_sums, tree_prefix_sums
    rng = np.random.default_rng(9)
    def mean_abs(fn, T, n):
        return np.mean([np.abs(fn(np.zeros(T), 1.0, 1.0, rng)).mean() for _ in range(n)])
    assert mean_abs(increment_prefix_sums, 64, 300) < mean_abs(tree_prefix_sums, 64, 60)
    assert mean_abs(increment_prefix_sums, 4096, 100) > mean_abs(tree_prefix_sums, 4096, 15)


def test_e3_reports_one_statistic_for_both_windows():
    from cdp.experiments import e3_granularity
    df = e3_granularity(seeds=200).set_index(["unit", "release"])
    daily, monthly = df.loc[("dealer_day", "daily")], df.loc[("dealer_day", "monthly")]
    # protecting a dealer's month scales Δ and the total by the same 22x: relative error is unchanged
    assert monthly.median_rel_error == pytest.approx(daily.median_rel_error, rel=0.2)


def _toy_grants():
    from cdp.registers import Grants
    # day 0: two grants (registers 0, 1); day 1: one grant (register 1); day 3: three grants
    return Grants(day=np.array([0, 0, 1, 3, 3, 3]), register=np.array([0, 1, 1, 0, 1, 2]),
                  size=np.array([10.0, 30.0, 50.0, 5.0, 5.0, 90.0]))


def test_threshold_release_waits_for_k_grants_and_keeps_the_total():
    from cdp import registers as rg
    g = _toy_grants()
    rel = rg.release_threshold(g, days=5, k=5)
    assert [r.day for r in rel] == [3] and rel[0].change == pytest.approx(190.0)   # 2+1+3 = 6 grants by day 3
    pub, truth = rg.published_level(rel, 5), rg.true_level(g, 5)
    assert list(pub) == [0, 0, 0, 190, 190] and list(truth) == [40, 90, 90, 190, 190]


def test_exact_and_threshold_fall_to_the_coalition():
    from cdp import registers as rg
    g = _toy_grants()
    for rel in (rg.release_exact(g, 5), rg.release_threshold(g, 5, k=5)):
        assert rg.recovered(rel, g, "coalition") == (6, 6)   # total minus everything else is exact


def test_outsider_and_insider_on_exact_daily_levels():
    from cdp import registers as rg
    g = _toy_grants()
    rel = rg.release_exact(g, 5)
    # outsider: day 1 is a lone grant (hit); day 3 change 100 vs largest 90 is outside ±10% (miss)
    assert rg.recovered(rel, g, "outsider") == (1, 6)
    # register 0 subtracts its own: day 0 -> 30 (hit), day 1 -> 50 (hit), day 3 -> 95 vs 90 (hit)
    assert rg.recovered(rel, g, "insider") == (3, 4)


def test_dp_release_without_noise_equals_clipped_sums():
    from cdp import registers as rg
    g = _toy_grants()
    rel = rg.release_dp(g, 5, clip=40.0, epsilon=1e12, rng=np.random.default_rng(10))
    assert len(rel) == 5 and rel[3].change == pytest.approx(5 + 5 + 40, abs=1e-6)


# ---------------------------------------------------------------- E11 threshold counters
from cdp import threshold  # noqa: E402


def _events(rows, days=None):
    t = np.array(rows, dtype=bool)
    d = np.arange(len(rows)) if days is None else np.array(days)
    return threshold.Events(d, t)


def test_shared_counter_exposes_lone_change():
    # metrics: shares_issued, outstanding_common, outstanding_preferred, holders_count
    conversion = [False, True, True, False]
    issuance = [True, True, False, False]
    ev = _events([conversion] * 4 + [issuance])  # k=5 reached; shares_issued saw only 1 change
    r = threshold.release_shared(ev, k=5)
    assert r.releases == 3                        # issued, common, preferred changed
    assert r.exposed_single == 1                  # shares_issued delta = that one issuance
    assert r.below_k == 2                         # issued (1) and preferred (4) both < 5


def test_per_metric_counter_never_exposes_below_k():
    rng = np.random.default_rng(1)
    ev = threshold.generate_events(threshold.ThresholdConfig(days=200), rng)
    r = threshold.release_per_metric(ev, k=4)
    assert r.exposed_single == 0 and r.below_k == 0
    assert r.lag_days.min() >= 0
    assert len(r.lag_days) + r.unpublished == r.total_changes
