import pandas as pd
import pytest

import features


def test_calculate_implied_probability_favorite():
    # -150 favorite: 150 / (150 + 100) = 0.6
    assert features.calculate_implied_probability(-150) == pytest.approx(0.6)


def test_calculate_implied_probability_underdog():
    # +130 underdog: 100 / (130 + 100) = 0.4347...
    assert features.calculate_implied_probability(130) == pytest.approx(100 / 230)


def _make_merged_games(rows):
    """Build a minimal merged game+odds DataFrame like merge_game_and_odds_data
    would produce, with just the columns engineer_features needs."""
    df = pd.DataFrame(rows)
    df['date_str'] = df['date']
    df['home_moneyline'] = -120
    df['away_moneyline'] = 110
    return df


def test_engineer_features_drops_games_with_no_prior_history():
    # Two teams' very first meeting has no history for either side, so it
    # should be dropped (matches the original algorithm's behavior).
    rows = [
        {'date': '20180401', 'home_team': 'NYA', 'visiting_team': 'BOS',
         'home_score': 5, 'visiting_score': 3, 'day_night': 'N', 'season': 2018},
        {'date': '20180402', 'home_team': 'BOS', 'visiting_team': 'NYA',
         'home_score': 2, 'visiting_score': 6, 'day_night': 'N', 'season': 2018},
    ]
    merged = _make_merged_games(rows)

    feature_df = features.engineer_features(merged, window_sizes=[5])

    # First game has no prior history for either team -> dropped.
    # Second game: both teams now have exactly one prior game -> kept.
    assert len(feature_df) == 1
    assert feature_df.iloc[0]['home_team'] == 'BOS'


def test_engineer_features_home_win_target():
    rows = [
        {'date': '20180401', 'home_team': 'NYA', 'visiting_team': 'BOS',
         'home_score': 5, 'visiting_score': 3, 'day_night': 'N', 'season': 2018},
        {'date': '20180402', 'home_team': 'BOS', 'visiting_team': 'NYA',
         'home_score': 2, 'visiting_score': 6, 'day_night': 'N', 'season': 2018},
        {'date': '20180403', 'home_team': 'NYA', 'visiting_team': 'BOS',
         'home_score': 1, 'visiting_score': 9, 'day_night': 'N', 'season': 2018},
    ]
    merged = _make_merged_games(rows)

    feature_df = features.engineer_features(merged, window_sizes=[5])

    # Game 2 (BOS home, lost 2-6) and game 3 (NYA home, lost 1-9) both have
    # history and should be kept, with home_win reflecting who actually won.
    assert len(feature_df) == 2
    by_date = feature_df.set_index('date_str')
    assert by_date.loc['20180402', 'home_win'] == 0  # BOS (home) lost
    assert by_date.loc['20180403', 'home_win'] == 0  # NYA (home) lost


def test_rolling_stats_no_leakage_simple_average():
    # Hand-verifiable check of the vectorized rolling-stats implementation
    # using simple (non-EWMA) averages: team A plays 3 games scoring
    # 2, 4, 6 runs. The 3rd game's "prior" rolling average should be the
    # mean of games 1-2 only (3.0), never including the game itself.
    rows = [
        {'date': '20180401', 'home_team': 'A', 'visiting_team': 'B',
         'home_score': 2, 'visiting_score': 1, 'day_night': 'N', 'season': 2018},
        {'date': '20180402', 'home_team': 'B', 'visiting_team': 'A',
         'home_score': 1, 'visiting_score': 4, 'day_night': 'N', 'season': 2018},
        {'date': '20180403', 'home_team': 'A', 'visiting_team': 'B',
         'home_score': 6, 'visiting_score': 1, 'day_night': 'N', 'season': 2018},
    ]
    merged = _make_merged_games(rows)
    merged['game_date'] = pd.to_datetime(merged['date_str'], format='%Y%m%d')
    merged = merged.sort_values('game_date')

    long_stats = features._build_team_long_stats(merged, [5], use_ewma=False, ewma_span=10)
    result = features._merge_rolling_stats(merged, long_stats, [5])

    game3 = result[result['date_str'] == '20180403'].iloc[0]
    assert game3['home_rolling_5_runs_scored'] == pytest.approx((2 + 4) / 2)
