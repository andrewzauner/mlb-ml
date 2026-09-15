"""
Pipeline orchestration for MLB prediction model.

This module coordinates the complete workflow from data loading
through model training and evaluation.

TODO: Add checkpointing to save/resume pipeline progress
TODO: Implement parallel processing for faster feature engineering
TODO: Add logging with different verbosity levels
"""

import os
from typing import List, Optional, Tuple, Dict
import pandas as pd
from sklearn.pipeline import Pipeline

import data_loader
import features
import modeling
import betting


def run_complete_pipeline(years: Optional[List[int]] = None,
                          data_dir: str = './data',
                          test_size: float = 0.2,
                          grid_search: bool = True,
                          evaluate_betting: bool = True,
                          model_path: Optional[str] = None,
                          model_type: str = 'gbm') -> Dict:
    """
    Run the complete model pipeline from data download to evaluation.
    
    This function orchestrates all steps:
    1. Download Retrosheet data
    2. Load and parse game logs
    3. Generate/download odds data
    4. Merge game and odds data
    5. Engineer features
    6. Prepare training/test splits
    7. Train model
    8. Evaluate model
    9. Display feature importance
    10. Evaluate betting performance (optional)
    
    Parameters
    ----------
    years : list of int, optional
        Years to include in analysis (default: 2015-2022)
    data_dir : str, optional
        Directory for data storage (default: './data')
    test_size : float, optional
        Proportion of data for testing (default: 0.2)
    grid_search : bool, optional
        Whether to perform hyperparameter grid search (default: True)
    evaluate_betting : bool, optional
        Whether to evaluate betting performance (default: True)
    model_path : str, optional
        If given, save the trained model pipeline to this path (via
        joblib) after training completes.
    model_type : str, optional
        'gbm' (default) or 'xgboost' - see modeling.train_model

    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Trained model pipeline
        - 'game_data': Historical game data
        - 'metrics': Model evaluation metrics
        - 'feature_importance': Feature importance DataFrame
        - 'calibration': Calibration/reliability report (see modeling.calibration_report)
        - 'betting_results': Betting performance metrics (if evaluate_betting=True)
        
    Examples
    --------
    >>> results = run_complete_pipeline(years=[2018, 2019, 2020])
    >>> print(f"Model accuracy: {results['metrics']['accuracy']:.3f}")
    >>> print(f"ROI: {results['betting_results']['roi']:.1%}")
    
    Notes
    -----
    The pipeline uses chronological splits to prevent data leakage.
    This means the model is trained on earlier games and tested on later games,
    simulating real-world usage.
    
    TODO: Implement incremental updates (add new season data without retraining)
    TODO: Add data validation steps
    TODO: Implement automated hyperparameter tuning schedules
    """
    print("\n" + "=" * 70)
    print("MLB PREDICTION MODEL - COMPLETE PIPELINE")
    print("=" * 70 + "\n")
    
    # Set default years if not provided
    if years is None:
        years = list(range(2015, 2023))
    
    print(f"Running pipeline for years: {min(years)} - {max(years)}")
    print(f"Data directory: {data_dir}")
    print(f"Test set size: {test_size:.0%}")
    print(f"Grid search: {'Enabled' if grid_search else 'Disabled'}\n")
    
    # Step 1: Download Retrosheet data
    print("STEP 1/10: Downloading Retrosheet data...")
    print("-" * 70)
    data_loader.download_retrosheet_data(years, data_dir)
    print()
    
    # Step 2: Load game data
    print("STEP 2/10: Loading game data...")
    print("-" * 70)
    game_data = data_loader.load_retrosheet_data(years, data_dir)
    
    if game_data is None or len(game_data) == 0:
        raise ValueError("Failed to load game data. Pipeline aborted.")
    print()
    
    # Step 3: Get odds data
    print("STEP 3/10: Generating odds data...")
    print("-" * 70)
    odds_data = data_loader.download_odds_data(years, game_data=game_data)
    print()
    
    # Step 4: Merge datasets
    print("STEP 4/10: Merging game and odds data...")
    print("-" * 70)
    merged_data = data_loader.merge_game_and_odds_data(game_data, odds_data)
    
    if merged_data is None or len(merged_data) == 0:
        raise ValueError("Failed to merge data. Pipeline aborted.")
    print()
    
    # Step 5: Engineer features
    print("STEP 5/10: Engineering features...")
    print("-" * 70)
    feature_df = features.engineer_features(merged_data)
    print()
    
    # Step 6: Prepare model data
    print("STEP 6/10: Preparing training and test sets...")
    print("-" * 70)
    X_train, X_test, y_train, y_test = modeling.prepare_model_data(
        feature_df, 
        test_size=test_size
    )
    print()
    
    # Step 7: Train model
    print("STEP 7/10: Training model...")
    print("-" * 70)
    model = modeling.train_model(X_train, y_train, grid_search=grid_search, model_type=model_type)
    print()
    
    # Step 8: Evaluate model
    print("STEP 8/10: Evaluating model...")
    print("-" * 70)
    metrics = modeling.evaluate_model(model, X_test, y_test)
    print()
    
    # Step 9: Feature importance
    print("STEP 9/10: Analyzing feature importance...")
    print("-" * 70)
    importance_df = modeling.feature_importance(model)
    print()

    calibration = modeling.calibration_report(model, X_test, y_test)
    print()
    
    # Step 10: Betting performance (optional)
    betting_results = None
    if evaluate_betting:
        print("STEP 10/10: Evaluating betting performance...")
        print("-" * 70)
        odds_cols = ['home_moneyline', 'away_moneyline']
        test_odds = (feature_df.loc[X_test.index, odds_cols]
                     if all(c in feature_df.columns for c in odds_cols) else None)
        betting_results = betting.evaluate_betting_performance(
            model, X_test, y_test, odds_df=test_odds
        )
        print()
    else:
        print("STEP 10/10: Skipping betting evaluation (disabled)")
        print("-" * 70)
        print()

    if model_path:
        save_model(model, model_path)
        print()

    # Pipeline complete
    print("=" * 70)
    print("PIPELINE COMPLETE")
    print("=" * 70)
    
    # Return results
    return {
        'model': model,
        'game_data': game_data,
        'metrics': metrics,
        'feature_importance': importance_df,
        'calibration': calibration,
        'betting_results': betting_results,
        'test_data': (X_test, y_test)
    }


def save_model(model: Pipeline, filepath: str = './model.pkl') -> None:
    """
    Save trained model to disk.
    
    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained model to save
    filepath : str, optional
        Path to save model (default: './model.pkl')
        
    Notes
    -----
    TODO: Save model metadata (training date, features used, performance)
    TODO: Version control for models
    """
    import joblib
    
    print(f"Saving model to {filepath}...")
    joblib.dump(model, filepath)
    print(f"Model saved successfully.")


def load_model(filepath: str = './model.pkl') -> Pipeline:
    """
    Load trained model from disk.
    
    Parameters
    ----------
    filepath : str, optional
        Path to load model from (default: './model.pkl')
    
    Returns
    -------
    sklearn.pipeline.Pipeline
        Loaded model
        
    Notes
    -----
    TODO: Validate model compatibility with current code version
    TODO: Load and display model metadata
    """
    import joblib
    
    print(f"Loading model from {filepath}...")
    model = joblib.load(filepath)
    print(f"Model loaded successfully.")
    return model
