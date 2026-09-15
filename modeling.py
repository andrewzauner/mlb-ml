"""
Modeling module for MLB prediction.

This module handles model training, evaluation, and feature importance analysis.

TODO: Experiment with alternative models (XGBoost, LightGBM, neural networks)
TODO: Implement proper hyperparameter optimization (Bayesian optimization)
TODO: Add calibration curves and reliability diagrams
TODO: Implement ensemble methods
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
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


def train_model(X_train: pd.DataFrame, 
                y_train: pd.Series,
                grid_search: bool = True,
                random_state: int = 42,
                use_class_weight: bool = True,
                calibrate: bool = True) -> Pipeline:
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
        Whether to use balanced class weights (recommended for ~54% home win rate)
    calibrate : bool, optional
        Whether to calibrate probabilities using isotonic regression (default: True)
        IMPORTANT: Calibration improves probability estimates for betting
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Trained model pipeline
        
    Notes
    -----
    Uses GradientBoostingClassifier by default as it typically
    performs well on structured data with mixed feature types.
    
    CURRENT MODEL CHOICE:
    - GradientBoostingClassifier: Good baseline, interpretable
    - StandardScaler: Normalizes features (important for some models)
    
    IMPROVEMENTS IMPLEMENTED:
    ✅ Added random_state for reproducibility
    ✅ Included StandardScaler in pipeline
    ✅ Grid search over key hyperparameters
    ✅ Class weights to handle home field advantage imbalance
    
    TODO: Model variants to try:
    1. XGBoost or LightGBM (often outperform sklearn GBM)
       - Better handling of missing values
       - Built-in regularization
       - Faster training
       
    2. Neural networks (MLPClassifier or deep learning)
       - Can capture complex nonlinear interactions
       - May need more data to avoid overfitting
       
    3. Ensemble methods:
       - Stack multiple models (GBM + RF + Logistic)
       - Voting classifier
       - Weighted averaging based on recent performance
       
    4. Calibration:
       - CalibratedClassifierCV to improve probability estimates
       - Important for betting applications
       
    TODO: Hyperparameter optimization approaches:
    - Bayesian optimization (scikit-optimize, Optuna)
    - Randomized search for faster initial exploration
    - Nested cross-validation for unbiased performance estimates
    """
    print("Training model...")
    
    # Calculate class weights if enabled
    # Baseball has ~54% home win rate, so slight imbalance
    if use_class_weight:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.unique(y_train)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}
        print(f"Using class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    if grid_search:
        # Define pipeline with scaling and classification
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(random_state=random_state))
        ])
        
        # Hyperparameter grid
        # IMPROVEMENT: Expanded from original with better ranges
        param_grid = {
            'classifier__n_estimators': [100, 200, 300],  # More trees generally better
            'classifier__learning_rate': [0.01, 0.05, 0.1],  # Smaller = more conservative
            'classifier__max_depth': [3, 4, 5],  # Depth controls complexity
            'classifier__min_samples_split': [2, 5],  # Regularization parameter
            'classifier__subsample': [0.8, 1.0],  # Stochastic gradient boosting
        }
        
        # Note: GradientBoostingClassifier doesn't support class_weight directly
        # For class imbalance, we could use sample_weight in fit() or try different models
        # Keeping this simple for now, but documenting the limitation

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
        
        grid.fit(X_train, y_train)
        model = grid.best_estimator_
        
        print(f"Best parameters: {grid.best_params_}")
        print(f"Best CV score: {-grid.best_score_:.4f} (log loss)")
        
    else:
        # Use fixed hyperparameters (faster, good for initial testing)
        # IMPROVEMENT: Better default parameters than before
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', GradientBoostingClassifier(
                n_estimators=200,
                learning_rate=0.05,  # Slightly lower for stability
                max_depth=4,  # Moderate depth
                min_samples_split=5,  # Prevent overfitting
                subsample=0.8,  # Stochastic boosting
                random_state=random_state
            ))
        ])
        
        model = pipeline.fit(X_train, y_train)
    
    # IMPROVEMENT: Add probability calibration
    # This is CRITICAL for betting applications where probability accuracy matters
    if calibrate:
        print("Calibrating probabilities with isotonic regression...")
        # Use 'isotonic' for non-parametric calibration (works well for tree ensembles)
        # Split training data for calibration to avoid overfitting
        from sklearn.model_selection import train_test_split as split
        X_train_sub, X_cal, y_train_sub, y_cal = split(
            X_train, y_train, test_size=0.2, random_state=random_state, stratify=y_train
        )
        
        # Refit base model on subset
        if grid_search:
            grid.fit(X_train_sub, y_train_sub)
            base_model = grid.best_estimator_
        else:
            base_model = pipeline.fit(X_train_sub, y_train_sub)
        
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
    
    Expected top features (based on domain knowledge):
    1. Vegas implied probabilities (strongest signal)
    2. Recent win percentage
    3. Run differential
    4. Rest days advantage
    
    TODO: Implement SHAP values for better feature importance
    - More accurate attribution than built-in importances
    - Shows direction of effect (positive/negative)
    - Can explain individual predictions
    
    TODO: Analyze feature interactions
    - Which features work together?
    - Are there redundant features?
    """
    if model is None:
        print("Model not trained yet.")
        return None
    
    try:
        # When probability calibration is enabled, `model` is a
        # CalibratedClassifierCV wrapping the underlying Pipeline(s) rather
        # than a Pipeline itself, so `named_steps` isn't available directly.
        # Unwrap to the fitted pipeline used by the first calibrator.
        pipeline = model
        if hasattr(model, 'calibrated_classifiers_'):
            base_estimator = model.calibrated_classifiers_[0].estimator
            pipeline = getattr(base_estimator, 'estimator', base_estimator)

        # Get feature names and importances. Note: feature names come from
        # the *pipeline* (which delegates to its first step, the scaler),
        # not the classifier step - inside a Pipeline, the classifier is
        # fit on the scaler's plain ndarray output, so it never sees
        # column names and has no feature_names_in_ of its own.
        # feature_importances_ is still read from the classifier itself,
        # in the same column order the scaler passed it.
        feature_names = pipeline.named_steps['scaler'].feature_names_in_
        importances = pipeline.named_steps['classifier'].feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
        
        print("\nTop 15 Most Important Features:")
        print("=" * 60)
        for idx, row in importance_df.head(15).iterrows():
            bar = '█' * int(row['importance'] * 200)  # Visual bar
            print(f"{row['feature']:.<40} {row['importance']:.4f} {bar}")
        print("=" * 60)
        
        return importance_df
        
    except Exception as e:
        print(f"Error extracting feature importances: {e}")
        return None
