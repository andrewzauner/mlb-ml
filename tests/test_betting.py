import numpy as np
import pandas as pd
import pytest

import betting


def test_american_odds_to_decimal_favorite():
    # -150 favorite: 100/150 + 1 = 1.6667
    assert betting.american_odds_to_decimal(-150) == pytest.approx(100 / 150 + 1)


def test_american_odds_to_decimal_underdog():
    # +130 underdog: 130/100 + 1 = 2.3
    assert betting.american_odds_to_decimal(130) == pytest.approx(2.3)


def test_american_odds_to_decimal_even_money():
    assert betting.american_odds_to_decimal(100) == pytest.approx(2.0)


class _StubModel:
    """Minimal stand-in for a fitted sklearn Pipeline's predict_proba."""
    def __init__(self, home_win_probs):
        self._probs = np.asarray(home_win_probs)

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


def test_evaluate_betting_performance_uses_real_odds_for_stake_and_payout():
    # One game: model says home win prob 0.65, vegas (from features) says
    # 0.50, real home moneyline is -100 (decimal 2.0).
    # edge = 0.15 > min_edge 0.03 -> bet home.
    # kelly: b=1.0, p=0.65, q=0.35 -> raw kelly=(1*0.65-0.35)/1=0.30;
    # *0.25 fraction = 0.075; stake = min(0.075*1000, 0.05*1000) = 50
    # (capped by max_bet_pct). Home wins -> profit = 50*(2.0-1) = 50.
    model = _StubModel([0.65])
    X_test = pd.DataFrame({
        'home_implied_prob_normalized': [0.5],
        'away_implied_prob_normalized': [0.5],
    })
    y_test = pd.Series([1])
    odds_df = pd.DataFrame({'home_moneyline': [-100], 'away_moneyline': [-100]}, index=X_test.index)

    result = betting.evaluate_betting_performance(
        model, X_test, y_test, odds_df=odds_df,
        kelly_fraction=0.25, min_edge=0.03, max_bet_pct=0.05, initial_bankroll=1000
    )

    assert result['used_real_odds'] is True
    assert result['final_bankroll'] == pytest.approx(1050.0)
    assert result['total_bets'] == 1


def test_evaluate_betting_performance_falls_back_to_fair_odds_without_odds_df():
    model = _StubModel([0.65])
    X_test = pd.DataFrame({
        'home_implied_prob_normalized': [0.5],
        'away_implied_prob_normalized': [0.5],
    })
    y_test = pd.Series([1])

    result = betting.evaluate_betting_performance(
        model, X_test, y_test, odds_df=None,
        kelly_fraction=0.25, min_edge=0.03, max_bet_pct=0.05, initial_bankroll=1000
    )

    assert result['used_real_odds'] is False
    # Same math as the real-odds test above, since fair odds = 2.0 decimal
    # matches the -100 moneyline used there.
    assert result['final_bankroll'] == pytest.approx(1050.0)


def test_evaluate_betting_performance_picks_the_larger_edge_side():
    # Home has a small edge, away has a much larger edge -> should bet away.
    model = _StubModel([0.52])  # home_pred=0.52, away_pred=0.48
    X_test = pd.DataFrame({
        'home_implied_prob_normalized': [0.50],   # home edge = 0.02 (below min_edge)
        'away_implied_prob_normalized': [0.30],   # away edge = 0.18
    })
    y_test = pd.Series([0])  # away wins
    odds_df = pd.DataFrame({'home_moneyline': [-110], 'away_moneyline': [150]}, index=X_test.index)

    result = betting.evaluate_betting_performance(
        model, X_test, y_test, odds_df=odds_df, min_edge=0.03, initial_bankroll=1000
    )

    assert result['total_bets'] == 1
    assert result['winning_bets'] == 1  # away won, and we bet away


def test_evaluate_betting_performance_includes_flat_comparison():
    model = _StubModel([0.65, 0.70])
    X_test = pd.DataFrame({
        'home_implied_prob_normalized': [0.5, 0.5],
        'away_implied_prob_normalized': [0.5, 0.5],
    })
    y_test = pd.Series([1, 0])
    odds_df = pd.DataFrame({'home_moneyline': [-100, -100], 'away_moneyline': [-100, -100]},
                           index=X_test.index)

    result = betting.evaluate_betting_performance(model, X_test, y_test, odds_df=odds_df)

    assert 'flat_final_bankroll' in result
    assert 'flat_roi' in result
    assert 'flat_total_staked' in result


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
