"""
MLB Game Prediction Model - Main Entry Point

This script demonstrates the complete workflow:
1. Train model on historical data (or load a previously saved one)
2. Evaluate model performance
3. Demonstrate single game prediction

Run with: python main.py
Run with: python main.py --help    for available options
"""
import argparse

from predictor import MLBPredictor
from predict import format_prediction_output
import pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train and evaluate the MLB game prediction model."
    )
    parser.add_argument(
        '--years', type=int, nargs=2, metavar=('START', 'END'), default=(2018, 2022),
        help="Inclusive range of seasons to train on (default: 2018 2022)"
    )
    parser.add_argument(
        '--data-dir', type=str, default='./data',
        help="Directory for downloaded/cached Retrosheet data (default: ./data)"
    )
    parser.add_argument(
        '--no-grid-search', action='store_true',
        help="Skip hyperparameter grid search and use fixed defaults (much faster)"
    )
    parser.add_argument(
        '--no-betting', action='store_true',
        help="Skip the betting simulation step"
    )
    parser.add_argument(
        '--save-model', type=str, default=None, metavar='PATH',
        help="Save the trained model pipeline to PATH (via joblib) after training"
    )
    parser.add_argument(
        '--load-model', type=str, default=None, metavar='PATH',
        help="Load a previously saved model from PATH instead of training a new one. "
             "Game data for the given --years is still loaded, since it's needed to "
             "compute features for the demo prediction."
    )
    return parser.parse_args()


def main():
    """
    Run the complete MLB prediction pipeline.
    """
    args = parse_args()

    # Initialize predictor
    print("Initializing MLB Predictor...")
    mlb_predictor = MLBPredictor(data_dir=args.data_dir)

    # Define years to train on
    years = list(range(args.years[0], args.years[1] + 1))

    if args.load_model:
        # Reuse a previously trained model instead of retraining. We still
        # need game_data loaded so predict_game() can compute features for
        # the demo prediction below.
        print(f"\nLoading model from {args.load_model}...")
        mlb_predictor.model = pipeline.load_model(args.load_model)
        mlb_predictor.load_retrosheet_data(years)
        results = None
    else:
        # Run complete pipeline
        print(f"\nRunning pipeline for years {years[0]}-{years[-1]}...")
        results = mlb_predictor.run_complete_pipeline(
            years=years,
            test_size=0.2,
            grid_search=not args.no_grid_search,
            evaluate_betting=not args.no_betting
        )
        if args.save_model:
            pipeline.save_model(mlb_predictor.model, args.save_model)
    
    # Display summary of results (skipped when a saved model was loaded,
    # since no fresh evaluation/betting run was performed)
    if results is not None:
        print("\n" + "=" * 70)
        print("PIPELINE SUMMARY")
        print("=" * 70)
        print(f"Model accuracy:       {results['metrics']['accuracy']:.3f}")
        print(f"ROC AUC:              {results['metrics']['roc_auc']:.3f}")
        print(f"Brier score:          {results['metrics']['brier_score']:.4f}")

        if results['betting_results']:
            print(f"\nBetting Performance:")
            print(f"Total bets:           {results['betting_results']['total_bets']}")
            print(f"Win rate:             {results['betting_results']['win_rate']:.1%}")
            print(f"ROI:                  {results['betting_results']['roi']:.1%}")
            print(f"Final bankroll:       ${results['betting_results']['final_bankroll']:.2f}")

        print("=" * 70)
    
    # Demonstrate game prediction
    print("\n\nDEMONSTRATING SINGLE GAME PREDICTION")
    print("=" * 70)
    
    # Example prediction: Yankees vs Red Sox
    prediction = mlb_predictor.predict_game(
        home_team="NYA",    # New York Yankees
        visiting_team="BOS", # Boston Red Sox
        game_date="20230601", # June 1, 2023
        odds={
            'home_moneyline': -150,  # Yankees favored
            'away_moneyline': +130   # Red Sox underdogs
        }
    )
    
    # Display formatted prediction
    print(format_prediction_output(prediction))
    
    print("\n✅ Pipeline completed successfully!")



if __name__ == "__main__":
    main()
