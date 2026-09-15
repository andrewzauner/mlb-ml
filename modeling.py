"""
Modeling module for MLB prediction.

This module handles model training, evaluation, and feature importance analysis.

TODO: Experiment with alternative models (LightGBM, neural networks)
TODO: Implement proper hyperparameter optimization (Bayesian optimization)
TODO: Implement ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                            f1_score, roc_auc_score, brier_score_loss,
                            log_loss)
from typing import Tuple, Dict, Optional
import warnings


def prepare_model_data(feature_df: pd.DataFrame,
                       test_size: float = 0.2,
                       min_year: int = 2010) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                       pd.Series, pd.Series]:
    """
    Prepare data for model training by selecting features and splitting.
    
    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame with engineered features
    test_size : float, optional
        Proportion of data to use for testing (default: 0.2)
    min_year : int, optional
        Minimum year to include (default: 2010 for recent trends)
    
    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
        
    Notes
    -----
    Uses chronological split (not random) to prevent data leakage.
    This is crucial for time series data.
    
    OVERSIMPLIFICATION WARNING:
    - Currently using a simple 80/20 chronological split
    - No separate validation set for hyperparameter tuning
    - Not accounting for seasonal effects (playoff vs regular season)
    
    TODO: Create separate validation set for hyperparameter tuning

    For a more robust, multi-fold evaluation of the same chronological-
    ordering constraint, see walk_forward_validation() below.
    """
    X, y = _select_features(feature_df, min_year)

    # CRITICAL: Chronological split for time series data
    # We use the first 80% for training and last 20% for testing
    # This simulates real-world usage where we predict future games
    train_cutoff = int(len(X) * (1 - test_size))
    X_train, X_test = X.iloc[:train_cutoff], X.iloc[train_cutoff:]
    y_train, y_test = y.iloc[:train_cutoff], y.iloc[train_cutoff:]

    print(f"Training data shape: {X_train.shape}")
    print(f"Testing data shape: {X_test.shape}")
    print(f"Class balance in training: {y_train.mean():.3f} (home win rate)")
    print(f"Class balance in testing: {y_test.mean():.3f} (home win rate)")

    # TODO: Check for class imbalance and consider SMOTE or class weights
    # Baseball typically has ~54% home win rate, so slight imbalance exists

    return X_train, X_test, y_train, y_test


def _select_features(feature_df: pd.DataFrame, min_year: int) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Select the model's feature columns and target from engineered features,
    restricted to seasons >= min_year. Shared by prepare_model_data (single
    chronological split) and walk_forward_validation (multiple expanding
    chronological splits) so both use the same feature set.
    """
    # Feature selection
    # These are the core features that have proven predictive
    features = [
        # Vegas odds features (typically the strongest signal)
        'home_implied_prob_normalized', 
        'away_implied_prob_normalized',
        
        # Short-term team performance (5 games)
        'home_rolling_5_runs_scored', 
        'home_rolling_5_runs_allowed', 
        'home_rolling_5_win_pct',
        'visiting_rolling_5_runs_scored', 
        'visiting_rolling_5_runs_allowed', 
        'visiting_rolling_5_win_pct',
        
        # Medium-term performance (10 games)
        'home_rolling_10_win_pct', 
        'visiting_rolling_10_win_pct',
        
        # Longer-term performance (20 games)
        'home_rolling_20_win_pct', 
        'visiting_rolling_20_win_pct',
        
        # NEW: Pythagenpat expected win% (run differential based)
        'home_rolling_10_pythag_win_pct',
        'visiting_rolling_10_pythag_win_pct',
        'pythag_win_pct_diff_10',
        
        # Comparison features
        'runs_scored_diff_5', 
        'runs_allowed_diff_5', 
        'win_pct_diff_5',
        'runs_scored_diff_10', 
        'runs_allowed_diff_10', 
        'win_pct_diff_10',
        'home_rolling_5_run_diff', 
        'visiting_rolling_5_run_diff',
        
        # Rest and game conditions
        'days_rest_advantage', 
        'is_night_game',
        
        # Day of week (some teams perform better on certain days)
        'day_0', 'day_1', 'day_2', 'day_3', 'day_4', 'day_5', 'day_6',
        
        # Month (captures seasonal trends, weather)
        'month_4', 'month_5', 'month_6', 'month_7', 
        'month_8', 'month_9', 'month_10'
    ]
    
    # Try to add home/away split features if they exist
    optional_features = [
        'home_rolling_10_home_win_pct',
        'visiting_rolling_10_away_win_pct',
    ]
    
    for feat in optional_features:
        if feat in feature_df.columns:
            features.append(feat)
    
    # Filter to available features
    available_features = [f for f in features if f in feature_df.columns]
    
    if len(available_features) < len(features):
        missing = set(features) - set(available_features)
        print(f"Warning: {len(missing)} features not found: {missing}")
    
    # Filter to recent years only (older data may not be relevant)
    # TODO: Experiment with different time windows
    # TODO: Consider weighting recent years more heavily
    recent_data = feature_df[feature_df['season'] >= min_year].copy()
    
    print(f"Using {len(recent_data)} games from {min_year} onwards")
    
    X = recent_data[available_features]
    y = recent_data['home_win']

    return X, y


class AveragingEnsembleClassifier:
    """
    Combines several already-trained classifiers by averaging their
    predicted probabilities - the "multiple model ensemble" idea in its
    simplest form (equal-weight averaging, no stacking/meta-learner).

    Each member can be a plain sklearn Pipeline or a calibrated model
    (CalibratedClassifierCV); this only relies on predict_proba(), so it
    works with anything train_model() returns.

    Implements enough of the sklearn estimator interface (predict_proba,
    predict, feature_names_in_) to be a drop-in replacement for a single
    model everywhere else in this project uses one - evaluate_model(),
    calibration_report(), and betting.evaluate_betting_performance() all
    only call predict()/predict_proba() and work unmodified; predict.py's
    predict_game() additionally needs feature_names_in_, delegated here
    to the first member (all members are trained on the same X_train, so
    they expect the same feature columns in the same order).

    TODO: Weighted averaging (e.g. by each member's validation log loss)
    instead of equal weights
    TODO: Stacking with a meta-learner instead of simple averaging
    """
    def __init__(self, models: list):
        if not models:
            raise ValueError("AveragingEnsembleClassifier needs at least one model")
        self.models = models
        self.classes_ = np.array([0, 1])

    @property
    def feature_names_in_(self):
        return self.models[0].feature_names_in_

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X) for m in self.models], axis=0)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _build_classifier(model_type: str, random_state: int, tuned: bool):
    """
    Construct the (untrained) classifier step for the given model_type.

    Parameters
    ----------
    model_type : str
        'gbm' (sklearn GradientBoostingClassifier) or 'xgboost'
    random_state : int
        Random seed
    tuned : bool
        If False, use fixed "reasonable default" hyperparameters directly
        on the classifier (the grid_search=False fast path). If True, use
        bare defaults - hyperparameters are set via GridSearchCV instead.

    Returns
    -------
    (classifier, param_grid)
        param_grid is the GridSearchCV grid to use when tuned=True; it's
        unused (and can be ignored) when tuned=False.
    """
    if model_type == 'xgboost':
        from xgboost import XGBClassifier
        if tuned:
            classifier = XGBClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric='logloss',
                random_state=random_state,
            )
        else:
            classifier = XGBClassifier(eval_metric='logloss', random_state=random_state)
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 4, 5],
            'classifier__subsample': [0.8, 1.0],
            'classifier__colsample_bytree': [0.8, 1.0],
        }
    elif model_type == 'gbm':
        if tuned:
            classifier = GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,
                max_depth=4,
                min_samples_split=5,
                subsample=0.8,
                random_state=random_state,
            )
        else:
            classifier = GradientBoostingClassifier(random_state=random_state)
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],
            'classifier__learning_rate': [0.01, 0.05, 0.1],
            'classifier__max_depth': [3, 4, 5],
            'classifier__min_samples_split': [2, 5],
            'classifier__subsample': [0.8, 1.0],
        }
    else:
        raise ValueError(f"Unknown model_type '{model_type}'. Use 'gbm' or 'xgboost'.")

    return classifier, param_grid


def train_model(X_train: pd.DataFrame,
                y_train: pd.Series,
                grid_search: bool = True,
                random_state: int = 42,
                use_class_weight: bool = True,
                calibrate: bool = True,
                model_type: str = 'gbm') -> Pipeline:
    """
    Train a machine learning model for game prediction.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series
        Training target
    grid_search : bool, optional
        Whether to perform grid search for hyperparameter tuning
    random_state : int, optional
        Random seed for reproducibility
    use_class_weight : bool, optional
        Whether to weight training samples inversely to class frequency
        (recommended for the ~54% home win rate)
    calibrate : bool, optional
        Whether to calibrate probabilities using isotonic regression (default: True)
        IMPORTANT: Calibration improves probability estimates for betting
    model_type : str, optional
        'gbm' (default) for sklearn's GradientBoostingClassifier,
        'xgboost' for XGBClassifier (requires the optional `xgboost`
        package - falls back to 'gbm' with a warning if it isn't
        installed), or 'ensemble' to train both and average their
        predicted probabilities (see AveragingEnsembleClassifier)

    Returns
    -------
    sklearn.pipeline.Pipeline or AveragingEnsembleClassifier
        Trained model

    Notes
    -----
    CURRENT MODEL CHOICES:
    - GradientBoostingClassifier (default) or XGBClassifier: both handle
      structured/tabular data with mixed feature types well
    - StandardScaler: Normalizes features (important for some models)

    IMPROVEMENTS IMPLEMENTED:
    ✅ Added random_state for reproducibility
    ✅ Included StandardScaler in pipeline
    ✅ Grid search over key hyperparameters (with TimeSeriesSplit CV)
    ✅ Per-sample weights to handle home field advantage imbalance
    ✅ Optional XGBoost backend
    ✅ Probability calibration (CalibratedClassifierCV)

    TODO: Model variants to try:
    1. LightGBM
    2. Neural networks (MLPClassifier or deep learning)
    3. Ensemble methods:
       - Stack multiple models (GBM + XGBoost + Logistic)
       - Voting classifier
       - Weighted averaging based on recent performance

    TODO: Hyperparameter optimization approaches:
    - Bayesian optimization (scikit-optimize, Optuna)
    - Randomized search for faster initial exploration
    - Nested cross-validation for unbiased performance estimates
    """
    print("Training model...")

    if model_type == 'ensemble':
        member_types = ['gbm', 'xgboost']
        try:
            import xgboost  # noqa: F401
        except ImportError:
            print("xgboost is not installed (pip install xgboost) - "
                  "ensemble will only use 'gbm' (no averaging benefit).")
            member_types = ['gbm']

        print(f"Training ensemble members: {member_types}")
        members = [
            train_model(X_train, y_train, grid_search=grid_search, random_state=random_state,
                       use_class_weight=use_class_weight, calibrate=calibrate, model_type=mt)
            for mt in member_types
        ]
        model = AveragingEnsembleClassifier(members)
        print("Ensemble training complete.")
        return model

    if model_type == 'xgboost':
        try:
            import xgboost  # noqa: F401
        except ImportError:
            print("xgboost is not installed (pip install xgboost) - falling back to model_type='gbm'.")
            model_type = 'gbm'

    # Per-sample weights for class imbalance. Baseball has a ~54% home win
    # rate, so a slight imbalance exists.
    # BUG FIX: this used to compute a class_weight *dict* and only print
    # it - GradientBoostingClassifier doesn't accept class_weight, and the
    # dict was never turned into sample_weight or passed to fit(), so
    # use_class_weight=True (the default) silently did nothing.
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        print(f"Using class weights: {class_weight_dict}")
        sample_weight = compute_sample_weight('balanced', y_train)
    else:
        sample_weight = None

    if grid_search:
        classifier, param_grid = _build_classifier(model_type, random_state, tuned=False)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])

        # TODO: Use RandomizedSearchCV for faster search with more parameters
        # TODO: Implement Bayesian optimization for more efficient search

        # Perform grid search with cross-validation.
        # X_train/y_train are already in chronological order (prepare_model_data
        # never shuffles), so TimeSeriesSplit gives each fold's validation set
        # strictly after its training set - a plain KFold would let the model
        # tune hyperparameters using "future" games to predict "past" ones.
        n_splits = min(5, max(2, len(X_train) // 50))
        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=TimeSeriesSplit(n_splits=n_splits),
            scoring='neg_log_loss',  # Better for probability calibration than accuracy
            n_jobs=-1,
            verbose=1
        )

        fit_kwargs = {'classifier__sample_weight': sample_weight} if sample_weight is not None else {}
        grid.fit(X_train, y_train, **fit_kwargs)
        model = grid.best_estimator_

        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV score: {-grid.best_score_:.4f} (log loss)")

    else:
        # Use fixed hyperparameters (faster, good for initial testing)
        classifier, _ = _build_classifier(model_type, random_state, tuned=True)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', classifier)
        ])

        fit_kwargs = {'classifier__sample_weight': sample_weight} if sample_weight is not None else {}
        model = pipeline.fit(X_train, y_train, **fit_kwargs)

    # IMPROVEMENT: Add probability calibration
    # This is CRITICAL for betting applications where probability accuracy matters
    if calibrate:
        print("Calibrating probabilities with isotonic regression...")
        # Use 'isotonic' for non-parametric calibration (works well for tree ensembles)
        # Split training data (and its sample weights, kept aligned) for
        # calibration to avoid overfitting.
        from sklearn.model_selection import train_test_split as split
        if sample_weight is not None:
            X_train_sub, X_cal, y_train_sub, y_cal, w_train_sub, w_cal = split(
                X_train, y_train, sample_weight, test_size=0.2,
                random_state=random_state, stratify=y_train
            )
        else:
            X_train_sub, X_cal, y_train_sub, y_cal = split(
                X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
            )
            w_train_sub = None

        sub_fit_kwargs = {'classifier__sample_weight': w_train_sub} if w_train_sub is not None else {}

        # Refit base model on subset
        if grid_search:
            grid.fit(X_train_sub, y_train_sub, **sub_fit_kwargs)
            base_model = grid.best_estimator_
        else:
            base_model = pipeline.fit(X_train_sub, y_train_sub, **sub_fit_kwargs)

        # Now calibrate on held-out calibration set
        # scikit-learn >=1.6 removed cv='prefit' in favor of wrapping the
        # already-fitted estimator in FrozenEstimator; fall back to the old
        # API for older installations.
        try:
            from sklearn.frozen import FrozenEstimator
            calibrated_model = CalibratedClassifierCV(
                FrozenEstimator(base_model),
                method='isotonic',  # Isotonic regression for tree models
            )
        except ImportError:
            calibrated_model = CalibratedClassifierCV(
                base_model,
                method='isotonic',
                cv='prefit'  # Model already trained
            )
        # Deliberately unweighted: calibration should map predicted
        # probabilities to the *true* observed outcome frequency, not a
        # class-rebalanced one.
        calibrated_model.fit(X_cal, y_cal)
        model = calibrated_model
        print("✅ Calibration complete - probabilities should be more accurate")
    
    print("Model training complete.")
    return model


def evaluate_model(model: Pipeline,
                   X_test: pd.DataFrame, 
                   y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate the model on test data.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test target
    
    Returns
    -------
    dict
        Dictionary of evaluation metrics
        
    Notes
    -----
    IMPROVEMENT: Added additional metrics beyond the original:
    - Brier score: Measures quality of probabilistic predictions
    - Log loss: Penalizes confident wrong predictions
    
    TODO: Add calibration analysis
    - Plot calibration curves
    - Compute Expected Calibration Error (ECE)
    
    TODO: Analyze performance by subgroups:
    - Home favorites vs underdogs
    - High vs low scoring games
    - Different months/weather conditions
    - Team strength tiers
    """
    if model is None:
        print("Model not trained yet.")
        return None
    
    # Generate predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'brier_score': brier_score_loss(y_test, y_pred_proba),  # NEW: probability calibration
        'log_loss': log_loss(y_test, y_pred_proba),  # NEW: better for betting
    }
    
    print("\nModel Evaluation Metrics:")
    print("=" * 50)
    for metric, value in metrics.items():
        print(f"{metric:.<30} {value:.4f}")
    print("=" * 50)
    
    # Baseline comparison
    # In baseball, home teams win ~54% of games
    baseline_accuracy = y_test.mean() if y_test.mean() > 0.5 else 1 - y_test.mean()
    print(f"\nBaseline (always predict home/away): {baseline_accuracy:.4f}")
    print(f"Model improvement: {(metrics['accuracy'] - baseline_accuracy):.4f}")
    
    # TODO: Add confusion matrix analysis
    # TODO: Plot ROC curve and precision-recall curve
    # TODO: Analyze predictions by confidence level

    return metrics


def walk_forward_validation(feature_df: pd.DataFrame,
                            n_splits: int = 5,
                            min_year: int = 2010,
                            grid_search: bool = False,
                            random_state: int = 42) -> Dict:
    """
    Evaluate the model with expanding-window walk-forward validation.

    prepare_model_data() gives a single chronological 80/20 split: one
    train window, one test window. That's a reasonable quick check, but
    it's one sample - performance on that particular test window could be
    unusually good or bad by chance. Walk-forward validation instead
    carves the season-ordered data into n_splits chronological folds
    (via TimeSeriesSplit): fold i trains on everything before it and
    tests on the fold right after, so every fold's test set is still
    strictly in the future relative to its training set (no leakage),
    and the reported metrics are an average over multiple such
    train/test boundaries instead of one.

    Parameters
    ----------
    feature_df : pd.DataFrame
        DataFrame with engineered features (output of engineer_features)
    n_splits : int, optional
        Number of expanding-window folds (default: 5)
    min_year : int, optional
        Minimum season to include (default: 2010)
    grid_search : bool, optional
        Whether to grid search hyperparameters within each fold (default:
        False - grid searching n_splits times over is expensive; the
        single-split prepare_model_data()/train_model(grid_search=True)
        path is the place to tune hyperparameters)
    random_state : int, optional
        Random seed passed through to train_model

    Returns
    -------
    dict
        'folds': list of per-fold metric dicts (each augmented with
            'train_size' and 'test_size')
        'mean': metrics averaged across folds
        'std': per-metric standard deviation across folds

    TODO: Support calibration inside each fold (currently disabled for
    speed - see train_model(calibrate=...))
    """
    X, y = _select_features(feature_df, min_year)

    splitter = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    print(f"\nRunning walk-forward validation ({n_splits} folds)...")
    print("=" * 70)

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        print(f"\nFold {fold_idx}/{n_splits}: train={len(X_train)} games, "
              f"test={len(X_test)} games")

        model = train_model(X_train, y_train, grid_search=grid_search,
                            random_state=random_state, calibrate=False)

        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'brier_score': brier_score_loss(y_test, y_pred_proba),
            'log_loss': log_loss(y_test, y_pred_proba),
            'train_size': len(X_train),
            'test_size': len(X_test),
        }
        fold_metrics.append(metrics)
        print(f"  accuracy={metrics['accuracy']:.4f}  roc_auc={metrics['roc_auc']:.4f}  "
              f"brier={metrics['brier_score']:.4f}  log_loss={metrics['log_loss']:.4f}")

    metric_names = ['accuracy', 'roc_auc', 'brier_score', 'log_loss']
    mean_metrics = {m: float(np.mean([f[m] for f in fold_metrics])) for m in metric_names}
    std_metrics = {m: float(np.std([f[m] for f in fold_metrics])) for m in metric_names}

    print("\n" + "=" * 70)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("=" * 70)
    for m in metric_names:
        print(f"{m:.<20} {mean_metrics[m]:.4f} (+/- {std_metrics[m]:.4f})")
    print("=" * 70)

    return {'folds': fold_metrics, 'mean': mean_metrics, 'std': std_metrics}


def calibration_report(model: Pipeline,
                       X_test: pd.DataFrame,
                       y_test: pd.Series,
                       n_bins: int = 10) -> Dict:
    """
    Assess how well-calibrated the model's predicted probabilities are.

    A model can have good accuracy/AUC while still being poorly
    calibrated (e.g. systematically overconfident) - which matters a lot
    for betting, since bet sizing (Kelly criterion) depends on the
    predicted probability being an honest estimate, not just on which
    side of 50% it falls on.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained (optionally calibrated) prediction model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test outcomes
    n_bins : int, optional
        Number of probability bins for the reliability diagram (default: 10)

    Returns
    -------
    dict
        'bins': DataFrame with one row per bin (predicted avg probability,
            actual win rate, bin count)
        'ece': Expected Calibration Error - the bin-count-weighted average
            gap between predicted probability and actual outcome frequency
            (0 = perfectly calibrated, higher = worse)
        'max_calibration_error': the single worst bin's gap

    Notes
    -----
    TODO: Plot an actual reliability diagram (this reports the same
    underlying data as text/a DataFrame, since the project has no
    plotting dependency yet)
    """
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    prob_true, prob_pred = calibration_curve(y_test, y_pred_proba, n_bins=n_bins, strategy='uniform')

    # calibration_curve silently drops empty bins, so recompute counts per
    # bin ourselves to weight the ECE correctly and report them. Must use
    # the exact same bin assignment sklearn uses internally (searchsorted
    # against the interior edges), not np.digitize - the two disagree on
    # values landing exactly on a bin edge, which then desyncs the
    # nonzero-bin counts from calibration_curve's (prob_true, prob_pred).
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.searchsorted(bin_edges[1:-1], y_pred_proba)
    counts = np.bincount(bin_idx, minlength=n_bins)[:n_bins]
    non_empty = counts > 0
    bin_counts = counts[non_empty]

    gaps = np.abs(prob_true - prob_pred)
    ece = float(np.sum(bin_counts * gaps) / bin_counts.sum())
    max_calibration_error = float(gaps.max()) if len(gaps) else 0.0

    bins_df = pd.DataFrame({
        'predicted_prob': prob_pred,
        'actual_win_rate': prob_true,
        'gap': gaps,
        'count': bin_counts,
    })

    print("\nCalibration / Reliability Report:")
    print("=" * 60)
    print(f"{'Predicted':>12} {'Actual':>12} {'Gap':>10} {'Count':>8}")
    for _, row in bins_df.iterrows():
        print(f"{row['predicted_prob']:>12.3f} {row['actual_win_rate']:>12.3f} "
              f"{row['gap']:>10.3f} {int(row['count']):>8}")
    print("-" * 60)
    print(f"Expected Calibration Error (ECE): {ece:.4f}")
    print(f"Max Calibration Error:            {max_calibration_error:.4f}")
    print("=" * 60)

    return {'bins': bins_df, 'ece': ece, 'max_calibration_error': max_calibration_error}


def _unwrap_pipeline(model) -> Pipeline:
    """
    Get the underlying (scaler, classifier) Pipeline from a trained model.

    When probability calibration is enabled, `model` is a
    CalibratedClassifierCV wrapping the underlying Pipeline(s) rather than
    a Pipeline itself, so `named_steps` isn't available directly on it.
    This unwraps to the fitted pipeline used by the first calibrator - the
    same one for every calibrator when calibrate=True used a single
    FrozenEstimator/prefit base model rather than cv-fitting several.
    """
    if hasattr(model, 'calibrated_classifiers_'):
        base_estimator = model.calibrated_classifiers_[0].estimator
        return getattr(base_estimator, 'estimator', base_estimator)
    return model


def feature_importance(model: Pipeline) -> pd.DataFrame:
    """
    Extract and display feature importances from the trained model.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model pipeline

    Returns
    -------
    pd.DataFrame
        DataFrame with feature names and importance scores

    Notes
    -----
    Feature importance helps identify which factors are most predictive.
    This uses each model's built-in (impurity-based) importances, which
    are fast but biased toward high-cardinality features and don't show
    the *direction* of a feature's effect. See shap_feature_importance()
    for a more rigorous alternative.

    Expected top features (based on domain knowledge):
    1. Vegas implied probabilities (strongest signal)
    2. Recent win percentage
    3. Run differential
    4. Rest days advantage

    TODO: Analyze feature interactions
    - Which features work together?
    - Are there redundant features?
    """
    if model is None:
        print("Model not trained yet.")
        return None

    if isinstance(model, AveragingEnsembleClassifier):
        # Average each member's importances (aligned by feature name - all
        # members were trained on the same X_train, so they share the same
        # feature columns).
        per_member = [_raw_feature_importance(m) for m in model.models]
        per_member = [r for r in per_member if r is not None]
        if not per_member:
            print("Error extracting feature importances: no ensemble member returned importances.")
            return None
        feature_names = per_member[0][0]
        importances = np.mean([imp for _, imp in per_member], axis=0)
    else:
        raw = _raw_feature_importance(model)
        if raw is None:
            return None
        feature_names, importances = raw

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    print("\nTop 15 Most Important Features:")
    print("=" * 60)
    for idx, row in importance_df.head(15).iterrows():
        bar = '█' * int(row['importance'] * 200)  # Visual bar
        print(f"{row['feature']:.<40} {row['importance']:.4f} {bar}")
    print("=" * 60)

    return importance_df


def _raw_feature_importance(model) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract (feature_names, importances) from a single (non-ensemble)
    trained model, without any printing. Shared by feature_importance()
    and its ensemble-averaging path above.
    """
    try:
        pipeline = _unwrap_pipeline(model)
        # Feature names come from the *pipeline* (which delegates to its
        # first step, the scaler), not the classifier step - inside a
        # Pipeline, the classifier is fit on the scaler's plain ndarray
        # output, so it never sees column names and has no
        # feature_names_in_ of its own. feature_importances_ is still
        # read from the classifier itself, in the same column order the
        # scaler passed it.
        feature_names = pipeline.named_steps['scaler'].feature_names_in_
        importances = pipeline.named_steps['classifier'].feature_importances_
        return feature_names, importances
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        return None


def shap_feature_importance(model: Pipeline,
                            X: pd.DataFrame,
                            max_samples: int = 500,
                            random_state: int = 42) -> Optional[pd.DataFrame]:
    """
    Compute SHAP-based feature importance and effect direction.

    Unlike feature_importance()'s impurity-based scores, SHAP values are
    computed per-prediction and can be averaged to show not just *how
    much* a feature matters but *which direction* it tends to push
    predictions (e.g. "more home rest days pushes toward a home win, on
    average") - useful both for sanity-checking the model against
    baseball intuition and for explaining individual predictions.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained (optionally calibrated) prediction model
    X : pd.DataFrame
        Feature data to explain (e.g. X_test) - real games, not synthetic
        rows, so the resulting importances reflect actual data patterns
    max_samples : int, optional
        SHAP's TreeExplainer is fast, but computing + printing importances
        for very large X is unnecessary; randomly sample at most this many
        rows (default: 500)
    random_state : int, optional
        Random seed for the row sample

    Returns
    -------
    pd.DataFrame or None
        Columns: 'feature', 'mean_abs_shap' (overall importance, use this
        for ranking), 'mean_shap' (signed average - direction of effect).
        None if the optional `shap` package isn't installed.

    Notes
    -----
    TODO: Analyze feature interactions (SHAP interaction values)
    TODO: Per-prediction explanations for predict_game()
    """
    if model is None:
        print("Model not trained yet.")
        return None

    try:
        import shap
    except ImportError:
        print("shap is not installed (pip install shap) - skipping SHAP analysis.")
        return None

    X_sample = X.sample(n=min(max_samples, len(X)), random_state=random_state) if len(X) > max_samples else X

    members = model.models if isinstance(model, AveragingEnsembleClassifier) else [model]
    per_member = [_raw_shap_importance(m, X_sample, shap) for m in members]
    per_member = [r for r in per_member if r is not None]
    if not per_member:
        return None

    mean_abs_shap = np.mean([r[0] for r in per_member], axis=0)
    mean_shap = np.mean([r[1] for r in per_member], axis=0)

    importance_df = pd.DataFrame({
        'feature': X_sample.columns,
        'mean_abs_shap': mean_abs_shap,
        'mean_shap': mean_shap,
    }).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

    print(f"\nSHAP Feature Importance (n={len(X_sample)} games"
          f"{', averaged over ensemble members' if len(members) > 1 else ''}):")
    print("=" * 70)
    print(f"{'Feature':<40} {'|SHAP|':>10} {'Direction':>15}")
    for _, row in importance_df.head(15).iterrows():
        direction = '+ home win' if row['mean_shap'] > 0 else '+ away win'
        print(f"{row['feature']:<40} {row['mean_abs_shap']:>10.4f} {direction:>15}")
    print("=" * 70)

    return importance_df


def _raw_shap_importance(model, X_sample: pd.DataFrame, shap_module) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute (mean_abs_shap, mean_shap) for a single (non-ensemble) trained
    model against X_sample, scaled with that model's own fitted scaler.
    Shared by shap_feature_importance() and its ensemble-averaging path.
    """
    try:
        pipeline = _unwrap_pipeline(model)
        scaler = pipeline.named_steps['scaler']
        classifier = pipeline.named_steps['classifier']

        X_scaled = pd.DataFrame(scaler.transform(X_sample), columns=X_sample.columns, index=X_sample.index)

        explainer = shap_module.TreeExplainer(classifier)
        shap_values = explainer.shap_values(X_scaled)

        # Different shap/model versions return shap_values in different
        # shapes for binary classification: a plain (n, features) array
        # for the positive class, a list of two such arrays
        # [class0, class1], or a (n, features, 2) array. Normalize to the
        # positive-class array.
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            shap_values = shap_values[:, :, 1]

        return np.abs(shap_values).mean(axis=0), shap_values.mean(axis=0)
    except Exception as e:
        print(f"Error computing SHAP values: {e}")
        return None
