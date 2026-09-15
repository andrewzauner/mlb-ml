import numpy as np
import pandas as pd
import pytest

import modeling


class _StubModel:
    """Minimal stand-in for a fitted sklearn Pipeline's predict_proba."""
    def __init__(self, home_win_probs):
        self._probs = np.asarray(home_win_probs)

    def predict_proba(self, X):
        return np.column_stack([1 - self._probs, self._probs])


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


def test_calibration_report_perfectly_calibrated_model_has_zero_ece():
    # 10 games per predicted-probability bucket where the actual win rate
    # exactly matches the predicted probability -> ECE should be ~0.
    probs = np.concatenate([np.full(10, 0.2), np.full(10, 0.8)])
    outcomes = np.concatenate([
        np.array([1] * 2 + [0] * 8),  # 20% win rate, matches predicted 0.2
        np.array([1] * 8 + [0] * 2),  # 80% win rate, matches predicted 0.8
    ])
    model = _StubModel(probs)
    X_test = pd.DataFrame(index=range(len(probs)))
    y_test = pd.Series(outcomes)

    report = modeling.calibration_report(model, X_test, y_test, n_bins=10)

    assert report['ece'] == pytest.approx(0.0, abs=1e-9)
    assert set(['predicted_prob', 'actual_win_rate', 'gap', 'count']) <= set(report['bins'].columns)


def test_calibration_report_handles_realistic_random_probabilities():
    # Regression test: calibration_report used to compute its own bin
    # counts with np.digitize while calibration_curve (sklearn) bins
    # internally with np.searchsorted - the two disagree on values that
    # land exactly on a bin edge, desyncing the two bin counts and
    # crashing with a shape mismatch. Random probabilities exercise many
    # more edge/bin configurations than a small handcrafted case would.
    rng = np.random.RandomState(0)
    n = 500
    probs = rng.rand(n)
    outcomes = (rng.rand(n) < probs).astype(int)
    model = _StubModel(probs)
    X_test = pd.DataFrame(index=range(n))
    y_test = pd.Series(outcomes)

    report = modeling.calibration_report(model, X_test, y_test, n_bins=10)

    assert 0.0 <= report['ece'] <= 1.0
    assert report['bins']['count'].sum() == n


def test_calibration_report_detects_overconfident_model():
    # Model always predicts 0.9 but is only right half the time -> should
    # show a large calibration gap.
    probs = np.full(20, 0.9)
    outcomes = np.array([1] * 10 + [0] * 10)
    model = _StubModel(probs)
    X_test = pd.DataFrame(index=range(len(probs)))
    y_test = pd.Series(outcomes)

    report = modeling.calibration_report(model, X_test, y_test, n_bins=10)

    assert report['ece'] > 0.3


def test_shap_feature_importance_basic():
    pytest.importorskip('shap')
    rng = np.random.RandomState(0)
    n = 60
    X_train = pd.DataFrame(rng.rand(n, 4), columns=['a', 'b', 'c', 'd'])
    y_train = pd.Series(rng.randint(0, 2, size=n))

    model = modeling.train_model(X_train, y_train, grid_search=False, calibrate=False)
    result = modeling.shap_feature_importance(model, X_train, max_samples=30)

    assert result is not None
    assert set(['feature', 'mean_abs_shap', 'mean_shap']) <= set(result.columns)
    assert set(result['feature']) == {'a', 'b', 'c', 'd'}
    assert (result['mean_abs_shap'] >= 0).all()


def test_shap_feature_importance_averages_ensemble_members():
    pytest.importorskip('shap')
    pytest.importorskip('xgboost')
    rng = np.random.RandomState(0)
    n = 60
    X_train = pd.DataFrame(rng.rand(n, 4), columns=['a', 'b', 'c', 'd'])
    y_train = pd.Series(rng.randint(0, 2, size=n))

    model = modeling.train_model(X_train, y_train, grid_search=False,
                                 calibrate=False, model_type='ensemble')
    result = modeling.shap_feature_importance(model, X_train, max_samples=30)

    assert result is not None
    assert set(result['feature']) == {'a', 'b', 'c', 'd'}


def test_build_classifier_gbm():
    from sklearn.ensemble import GradientBoostingClassifier
    classifier, param_grid = modeling._build_classifier('gbm', random_state=42, tuned=False)
    assert isinstance(classifier, GradientBoostingClassifier)
    assert 'classifier__n_estimators' in param_grid


def test_build_classifier_xgboost():
    xgboost = pytest.importorskip('xgboost')
    classifier, param_grid = modeling._build_classifier('xgboost', random_state=42, tuned=True)
    assert isinstance(classifier, xgboost.XGBClassifier)


def test_build_classifier_unknown_type_raises():
    with pytest.raises(ValueError):
        modeling._build_classifier('not_a_real_model', random_state=42, tuned=False)


def test_train_model_applies_sample_weight_without_crashing():
    # Regression test: use_class_weight used to compute a class_weight
    # dict that was printed but never actually passed to fit(). This just
    # exercises the fast path end-to-end (grid_search=False,
    # calibrate=False) and checks the result is a usable fitted model.
    rng = np.random.RandomState(0)
    n = 60
    X_train = pd.DataFrame(rng.rand(n, 4), columns=['a', 'b', 'c', 'd'])
    y_train = pd.Series(rng.randint(0, 2, size=n))

    model = modeling.train_model(X_train, y_train, grid_search=False,
                                 use_class_weight=True, calibrate=False)

    preds = model.predict_proba(X_train)
    assert preds.shape == (n, 2)


def test_averaging_ensemble_classifier_averages_member_probabilities():
    class _Stub:
        def __init__(self, probs):
            self._probs = np.asarray(probs)
            self.feature_names_in_ = np.array(['a', 'b'])

        def predict_proba(self, X):
            return np.column_stack([1 - self._probs, self._probs])

    ensemble = modeling.AveragingEnsembleClassifier([_Stub([0.2, 0.8]), _Stub([0.6, 0.6])])

    proba = ensemble.predict_proba(None)

    assert proba[:, 1] == pytest.approx([0.4, 0.7])
    assert list(ensemble.predict(None)) == [0, 1]  # 0.4 < 0.5, 0.7 >= 0.5
    assert list(ensemble.feature_names_in_) == ['a', 'b']  # delegates to first member


def test_averaging_ensemble_classifier_requires_at_least_one_model():
    with pytest.raises(ValueError):
        modeling.AveragingEnsembleClassifier([])


def test_train_model_ensemble_returns_working_averaging_classifier():
    pytest.importorskip('xgboost')
    rng = np.random.RandomState(0)
    n = 80
    X_train = pd.DataFrame(rng.rand(n, 4), columns=['a', 'b', 'c', 'd'])
    y_train = pd.Series(rng.randint(0, 2, size=n))

    model = modeling.train_model(X_train, y_train, grid_search=False,
                                 calibrate=False, model_type='ensemble')

    assert isinstance(model, modeling.AveragingEnsembleClassifier)
    assert len(model.models) == 2
    preds = model.predict_proba(X_train)
    assert preds.shape == (n, 2)
    # feature_importance() and shap_feature_importance() must handle the
    # ensemble case (averaging across members) without crashing.
    importance_df = modeling.feature_importance(model)
    assert importance_df is not None
    assert set(importance_df['feature']) == {'a', 'b', 'c', 'd'}


def test_train_model_unknown_model_type_raises():
    rng = np.random.RandomState(0)
    n = 30
    X_train = pd.DataFrame(rng.rand(n, 3), columns=['a', 'b', 'c'])
    y_train = pd.Series(rng.randint(0, 2, size=n))

    with pytest.raises(ValueError):
        modeling.train_model(X_train, y_train, grid_search=False,
                             calibrate=False, model_type='not_a_real_model')
