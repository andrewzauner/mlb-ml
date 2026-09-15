import numpy as np
import pandas as pd
import pytest

import modeling


REQUIRED_COLUMNS = [
    'home_implied_prob_normalized', 'away_implied_prob_normalized',
    'home_rolling_5_runs_scored', 'home_rolling_5_runs_allowed', 'home_rolling_5_win_pct',
    'visiting_rolling_5_runs_scored', 'visiting_rolling_5_runs_allowed', 'visiting_rolling_5_win_pct',
    'home_rolling_10_win_pct', 'visiting_rolling_10_win_pct',
    'home_rolling_20_win_pct', 'visiting_rolling_20_win_pct',
    'home_rolling_10_pythag_win_pct', 'visiting_rolling_10_pythag_win_pct', 'pythag_win_pct_diff_10',
    'runs_scored_diff_5', 'runs_allowed_diff_5', 'win_pct_diff_5',
    'runs_scored_diff_10', 'runs_allowed_diff_10', 'win_pct_diff_10',
    'home_rolling_5_run_diff', 'visiting_rolling_5_run_diff',
    'days_rest_advantage', 'is_night_game',
]
DAY_COLUMNS = [f'day_{i}' for i in range(7)]
MONTH_COLUMNS = [f'month_{m}' for m in range(4, 11)]


def make_synthetic_feature_df(n=120, seed=0):
    rng = np.random.RandomState(seed)
    data = {col: rng.rand(n) for col in REQUIRED_COLUMNS}
    for col in DAY_COLUMNS + MONTH_COLUMNS:
        data[col] = rng.randint(0, 2, size=n)
    data['season'] = 2018
    data['home_win'] = rng.randint(0, 2, size=n)
    return pd.DataFrame(data)


def test_select_features_filters_by_min_year():
    df = make_synthetic_feature_df(n=20)
    df.loc[:9, 'season'] = 2005  # below min_year, should be excluded
    df.loc[10:, 'season'] = 2015

    X, y = modeling._select_features(df, min_year=2010)

    assert len(X) == 10
    assert len(y) == 10


def test_select_features_warns_on_missing_columns(capsys):
    df = make_synthetic_feature_df(n=20)
    df = df.drop(columns=['home_rolling_5_win_pct'])

    X, y = modeling._select_features(df, min_year=2010)

    assert 'home_rolling_5_win_pct' not in X.columns
    captured = capsys.readouterr()
    assert 'not found' in captured.out


def test_prepare_model_data_uses_chronological_split_not_shuffled():
    n = 100
    df = make_synthetic_feature_df(n=n)
    # Make row order meaningful (e.g. an increasing "day count" feature)
    # so we can confirm the split doesn't shuffle it.
    df['home_rolling_5_runs_scored'] = np.arange(n, dtype=float)

    X_train, X_test, y_train, y_test = modeling.prepare_model_data(df, test_size=0.2, min_year=2010)

    assert len(X_train) == 80
    assert len(X_test) == 20
    # Chronological split: all training values precede all test values.
    assert X_train['home_rolling_5_runs_scored'].max() < X_test['home_rolling_5_runs_scored'].min()


def test_walk_forward_validation_returns_per_fold_and_summary_metrics():
    df = make_synthetic_feature_df(n=150, seed=1)

    results = modeling.walk_forward_validation(df, n_splits=3, min_year=2010, grid_search=False)

    assert len(results['folds']) == 3
    for fold in results['folds']:
        assert 0.0 <= fold['accuracy'] <= 1.0
        assert 0.0 <= fold['roc_auc'] <= 1.0
    for metric in ('accuracy', 'roc_auc', 'brier_score', 'log_loss'):
        assert metric in results['mean']
        assert metric in results['std']

    # Expanding window: each fold should train on strictly more games than
    # the previous one (walk-forward, not a fixed-size sliding window).
    train_sizes = [f['train_size'] for f in results['folds']]
    assert train_sizes == sorted(train_sizes)
    assert train_sizes[0] < train_sizes[-1]
