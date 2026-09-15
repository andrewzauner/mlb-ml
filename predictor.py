"""
Main predictor class for MLB game prediction.

This module provides a unified interface to the prediction system,
maintaining the same external API as the original implementation
while delegating to the refactored modules.
"""

from typing import List, Optional, Dict
from sklearn.pipeline import Pipeline
import pandas as pd

import data_loader
import features
import modeling
import betting
import predict as predict_module
import pipeline


class MLBPredictor:
    """
    Main class for MLB game prediction and betting analysis.
    
    This class provides a unified interface to:
    - Download and load historical game data
    - Train prediction models
    - Evaluate model performance
    - Make predictions on individual games
    - Simulate betting strategies
    
    Attributes
    ----------
    data_dir : str
        Directory for storing downloaded data
    game_data : pd.DataFrame or None
        Loaded historical game data
    odds_data : pd.DataFrame or None
        Loaded odds data
    model : sklearn.pipeline.Pipeline or None
        Trained prediction model
    
    Examples
    --------
    >>> predictor = MLBPredictor(data_dir='./data')
    >>> results = predictor.run_complete_pipeline([2018, 2019, 2020])
    >>> prediction = predictor.predict_game(
    ...     home_team="NYA",
    ...     visiting_team="BOS",
    ...     game_date="20240601"
    ... )
    """
    
    def __init__(self, data_dir: str = './data'):
        """
        Initialize the MLB Predictor.
        
        Parameters
        ----------
        data_dir : str, optional
            Directory to store downloaded data (default: './data')
        """
        self.data_dir = data_dir
        self.game_data = None
        self.odds_data = None
        self.model = None
        
        # Create data directory if it doesn't exist
        import os
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
    
    def download_retrosheet_data(self, years: List[int]) -> None:
        """
        Download game data from Retrosheet for specified years.
        
        Parameters
        ----------
        years : list of int
            Years to download (e.g., [2018, 2019, 2020])
        """
        data_loader.download_retrosheet_data(years, self.data_dir)
    
    def load_retrosheet_data(self, years: List[int]) -> pd.DataFrame:
        """
        Load and parse Retrosheet game logs.
        
        Parameters
        ----------
        years : list of int
            Years to load
        
        Returns
        -------
        pd.DataFrame
            Loaded game data
        """
        self.game_data = data_loader.load_retrosheet_data(years, self.data_dir)
        return self.game_data
    
    def download_odds_data(self, years: List[int]) -> pd.DataFrame:
        """
        Generate placeholder odds data.
        
        Parameters
        ----------
        years : list of int
            Years to generate odds for
        
        Returns
        -------
        pd.DataFrame
            Odds data
            
        Notes
        -----
        This is currently a placeholder. Replace with actual odds API.
        If game data has already been loaded (via load_retrosheet_data),
        odds are generated for the real schedule so they actually line up
        with game_data during the merge step.
        """
        self.odds_data = data_loader.download_odds_data(years, game_data=self.game_data)
        return self.odds_data
    
    def merge_game_and_odds_data(self) -> pd.DataFrame:
        """
        Merge game data with odds data.
        
        Returns
        -------
        pd.DataFrame
            Merged dataset
        """
        return data_loader.merge_game_and_odds_data(self.game_data, self.odds_data)
    
    def engineer_features(self, data: pd.DataFrame, 
                         window_sizes: List[int] = [5, 10, 20]) -> pd.DataFrame:
        """
        Engineer features for prediction.
        
        Parameters
        ----------
        data : pd.DataFrame
            Merged game and odds data
        window_sizes : list of int, optional
            Window sizes for rolling statistics
        
        Returns
        -------
        pd.DataFrame
            Feature DataFrame
        """
        return features.engineer_features(data, window_sizes)
    
    def prepare_model_data(self, feature_df: pd.DataFrame,
                          test_size: float = 0.2) -> tuple:
        """
        Prepare data for modeling.
        
        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature DataFrame
        test_size : float, optional
            Test set proportion
        
        Returns
        -------
        tuple
            (X_train, X_test, y_train, y_test)
        """
        return modeling.prepare_model_data(feature_df, test_size)
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                   grid_search: bool = True, model_type: str = 'gbm') -> Pipeline:
        """
        Train the prediction model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training labels
        grid_search : bool, optional
            Whether to use grid search for hyperparameters
        model_type : str, optional
            'gbm' (default) or 'xgboost' - see modeling.train_model

        Returns
        -------
        Pipeline
            Trained model
        """
        self.model = modeling.train_model(X_train, y_train, grid_search=grid_search, model_type=model_type)
        return self.model
    
    def evaluate_model(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate model performance.
        
        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        
        Returns
        -------
        dict
            Evaluation metrics
        """
        return modeling.evaluate_model(self.model, X_test, y_test)
    
    def feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Returns
        -------
        pd.DataFrame
            Feature importance scores
        """
        return modeling.feature_importance(self.model)

    def shap_feature_importance(self, X: pd.DataFrame, max_samples: int = 500) -> Optional[pd.DataFrame]:
        """
        Compute SHAP-based feature importance (overall importance plus
        direction of effect). Requires the optional `shap` package.

        Parameters
        ----------
        X : pd.DataFrame
            Feature data to explain (e.g. X_test)
        max_samples : int, optional
            Randomly sample at most this many rows for speed

        Returns
        -------
        pd.DataFrame or None
            None if `shap` isn't installed
        """
        return modeling.shap_feature_importance(self.model, X, max_samples=max_samples)

    def walk_forward_validation(self, feature_df: pd.DataFrame,
                               n_splits: int = 5,
                               grid_search: bool = False) -> Dict:
        """
        Evaluate the model with expanding-window walk-forward validation
        instead of a single train/test split.

        Parameters
        ----------
        feature_df : pd.DataFrame
            Feature DataFrame (output of engineer_features)
        n_splits : int, optional
            Number of expanding-window folds
        grid_search : bool, optional
            Whether to grid search hyperparameters within each fold

        Returns
        -------
        dict
            Per-fold metrics plus mean/std across folds
        """
        return modeling.walk_forward_validation(
            feature_df, n_splits=n_splits, grid_search=grid_search
        )

    def calibration_report(self, X_test: pd.DataFrame,
                          y_test: pd.Series,
                          n_bins: int = 10) -> Dict:
        """
        Assess how well-calibrated the model's predicted probabilities are.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        n_bins : int, optional
            Number of probability bins for the reliability diagram

        Returns
        -------
        dict
            Per-bin reliability stats plus Expected Calibration Error (ECE)
        """
        return modeling.calibration_report(self.model, X_test, y_test, n_bins=n_bins)

    def evaluate_betting_performance(self, X_test: pd.DataFrame,
                                    y_test: pd.Series,
                                    odds_df: Optional[pd.DataFrame] = None,
                                    kelly_fraction: float = 0.25) -> Dict:
        """
        Evaluate betting strategy performance.

        Parameters
        ----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test labels
        odds_df : pd.DataFrame, optional
            DataFrame with 'home_moneyline'/'away_moneyline' columns,
            indexed the same as X_test/y_test, for realistic bet sizing
            and payouts. Falls back to assumed fair odds if omitted.
        kelly_fraction : float, optional
            Kelly criterion fraction for bet sizing

        Returns
        -------
        dict
            Betting performance metrics
        """
        return betting.evaluate_betting_performance(
            self.model, X_test, y_test, odds_df=odds_df, kelly_fraction=kelly_fraction
        )
    
    def predict_game(self, home_team: str, visiting_team: str,
                    game_date: Optional[str] = None,
                    odds: Optional[Dict] = None) -> Dict:
        """
        Predict outcome of a specific game.
        
        Parameters
        ----------
        home_team : str
            Home team abbreviation
        visiting_team : str
            Visiting team abbreviation
        game_date : str, optional
            Game date in YYYYMMDD format
        odds : dict, optional
            Betting odds {'home_moneyline': -150, 'away_moneyline': +130}
        
        Returns
        -------
        dict
            Prediction details
        """
        return predict_module.predict_game(
            self.model, self.game_data, home_team, visiting_team, 
            game_date, odds
        )
    
    def run_complete_pipeline(self, years: Optional[List[int]] = None,
                             test_size: float = 0.2,
                             grid_search: bool = True,
                             evaluate_betting: bool = True,
                             model_type: str = 'gbm',
                             compute_shap: bool = False) -> Dict:
        """
        Run the complete pipeline from data download to evaluation.

        Parameters
        ----------
        years : list of int, optional
            Years to include (default: 2015-2022)
        test_size : float, optional
            Test set proportion
        grid_search : bool, optional
            Whether to use grid search
        evaluate_betting : bool, optional
            Whether to evaluate betting performance
        model_type : str, optional
            'gbm' (default), 'xgboost', or 'ensemble' - see modeling.train_model
        compute_shap : bool, optional
            Whether to also compute SHAP-based feature importance

        Returns
        -------
        dict
            Results including model, metrics, and betting performance
        """
        results = pipeline.run_complete_pipeline(
            years=years,
            data_dir=self.data_dir,
            test_size=test_size,
            grid_search=grid_search,
            evaluate_betting=evaluate_betting,
            model_type=model_type,
            compute_shap=compute_shap
        )
        
        # Store model and data for later use
        self.model = results['model']
        self.game_data = results['game_data']
        
        return results
