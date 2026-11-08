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
