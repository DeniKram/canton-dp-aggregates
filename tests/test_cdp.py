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
