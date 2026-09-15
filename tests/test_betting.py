import pytest

import betting


def test_kelly_criterion_positive_edge():
    # 60% win probability at even money (decimal odds 2.0) has positive
    # expected value, so full Kelly should recommend betting a positive
    # fraction of the bankroll: f = (bp - q) / b = (1*0.6 - 0.4)/1 = 0.2
    f = betting.kelly_criterion(win_prob=0.6, odds=2.0, fraction=1.0)
    assert f == pytest.approx(0.2)


def test_kelly_criterion_negative_edge_clamped_to_zero():
    # A losing proposition (40% win prob at even money) should never
    # recommend a negative stake.
    f = betting.kelly_criterion(win_prob=0.4, odds=2.0, fraction=1.0)
    assert f == 0


def test_kelly_criterion_fractional():
    full = betting.kelly_criterion(win_prob=0.6, odds=2.0, fraction=1.0)
    quarter = betting.kelly_criterion(win_prob=0.6, odds=2.0, fraction=0.25)
    assert quarter == pytest.approx(full * 0.25)


def test_kelly_criterion_never_exceeds_full_bankroll():
    # Even an extreme edge should be capped at 100% of bankroll.
    f = betting.kelly_criterion(win_prob=0.99, odds=50.0, fraction=1.0)
    assert f <= 1.0


def test_calculate_expected_value_positive_edge():
    # 60% win probability at even money (odds=2.0) has positive EV.
    ev = betting.calculate_expected_value(model_prob=0.6, market_prob=0.5, stake=10, odds=2.0)
    assert ev == pytest.approx(0.6 * 10 - 0.4 * 10)


def test_calculate_expected_value_fair_coin_even_money_is_zero():
    ev = betting.calculate_expected_value(model_prob=0.5, market_prob=0.5, stake=10, odds=2.0)
    assert ev == pytest.approx(0.0)
