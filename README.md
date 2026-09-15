# MLB Game Prediction Model

A machine learning system for predicting MLB game outcomes and evaluating betting strategies.

## Project Structure

```
mlb_predictor/
├── __init__.py           # Package initialization
├── predictor.py          # Main MLBPredictor class
├── data_loader.py        # Data acquisition and loading
├── features.py           # Feature engineering
├── modeling.py           # Model training and evaluation
├── betting.py            # Betting strategy simulation
├── predict.py            # Single game prediction
└── pipeline.py           # End-to-end workflow orchestration

main.py                   # Entry point for running the pipeline (CLI)
tests/                    # pytest unit tests
data/                     # Downloaded Retrosheet data (created automatically)
```

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

or directly:

```bash
pip install pandas numpy scikit-learn requests joblib pytest
```

### Optional (for enhancements)

```bash
pip install xgboost lightgbm shap optuna
```

## Quick Start

Run the complete pipeline:

```bash
python main.py
```

This will:
1. Download historical game data from Retrosheet (or reuse files already in `data/`)
2. Generate placeholder odds data
3. Engineer features
4. Train a gradient boosting model
5. Evaluate model performance
6. Simulate betting strategies
7. Demonstrate a single game prediction

### CLI options

```bash
python main.py --years 2018 2022              # season range (default: 2018-2022)
python main.py --no-grid-search                # skip hyperparameter search (much faster)
python main.py --no-betting                     # skip the betting simulation
python main.py --model-type xgboost             # use XGBoost instead of sklearn's GBM
python main.py --model-type ensemble            # average GBM + XGBoost predictions
python main.py --shap                           # also compute SHAP-based feature importance
python main.py --save-model ./model.pkl         # save the trained model after training
python main.py --load-model ./model.pkl         # reuse a saved model instead of retraining
python main.py --data-dir ./data                # where Retrosheet files live/are cached
```

### Running tests

```bash
python -m pytest tests/ -v
```

## Usage Examples

### Basic Usage

```python
from mlb_predictor import MLBPredictor

# Initialize
predictor = MLBPredictor(data_dir='./data')

# Run complete pipeline
results = predictor.run_complete_pipeline([2018, 2019, 2020, 2021, 2022])

# Make a prediction
prediction = predictor.predict_game(
    home_team="NYA",
    visiting_team="BOS",
    game_date="20240601",
    odds={'home_moneyline': -150, 'away_moneyline': +130}
)

print(f"Home win probability: {prediction['home_win_probability']:.1%}")
print(f"Recommendation: {prediction['recommendation']}")
```

### Step-by-Step Usage

```python
from mlb_predictor import MLBPredictor

predictor = MLBPredictor()

# 1. Download and load data
predictor.download_retrosheet_data([2020, 2021, 2022])
game_data = predictor.load_retrosheet_data([2020, 2021, 2022])

# 2. Get odds and merge
odds_data = predictor.download_odds_data([2020, 2021, 2022])
merged = predictor.merge_game_and_odds_data()

# 3. Engineer features
features = predictor.engineer_features(merged)

# 4. Prepare data and train
X_train, X_test, y_train, y_test = predictor.prepare_model_data(features)
model = predictor.train_model(X_train, y_train)

# 5. Evaluate
metrics = predictor.evaluate_model(X_test, y_test)
importance = predictor.feature_importance()
```

## Key Features

### Data Loading (`data_loader.py`)
- Downloads game logs from Retrosheet
- Properly maps batting statistics columns (fixes KeyError bug)
- Merges game data with odds data
- **BUG FIX**: Now correctly extracts `home_H`, `home_AB`, etc. from Retrosheet files (the column
  indices were verified against a real gamelog row - the previous mapping silently zeroed out
  `home_AB` for every game and pulled the wrong stat entirely for the others)
- **BUG FIX**: Loader now finds `gl{year}.txt` as well as `GL{year}.TXT` (case-insensitive lookup)
- **BUG FIX**: Placeholder odds are now generated per real scheduled game (keyed off actual
  `game_data`) instead of an independently-random schedule that almost never lined up with real
  games - this raised the odds/game match rate from well under 1% to 100%
- **BUG FIX**: Games are merged on a unique `game_id` instead of `(date, home_team,
  visiting_team)`, which isn't a unique key (doubleheaders share it) and used to fan out into
  duplicate/misaligned rows

### Feature Engineering (`features.py`)
- Rolling team statistics (5, 10, 20 game windows)
- Implied probabilities from betting odds
- Temporal features (day of week, month, rest days)
- Batting statistics and run differentials
- **DEFENSIVE**: Checks for missing columns before use
- **PERFORMANCE**: Rolling stats are computed with a vectorized `merge_asof`-based lookup instead
  of a per-game `iterrows()` scan of each team's full history - same "no data from the current or
  a future game" guarantee, but roughly two orders of magnitude faster (a full 2018-2022 run went
  from minutes to a few seconds)

### Modeling (`modeling.py`)
- Gradient Boosting Classifier with grid search
- Chronological train/test split (prevents data leakage)
- Grid search now cross-validates with `TimeSeriesSplit` instead of plain K-fold, so
  hyperparameter tuning never validates on games that happened before the training fold
- Walk-forward (expanding-window) validation via `modeling.walk_forward_validation()` /
  `MLBPredictor.walk_forward_validation()`, for a more robust multi-fold alternative to the single
  80/20 split
- Model persistence: `pipeline.save_model()` / `load_model()` are wired into the pipeline and
  exposed via `main.py --save-model` / `--load-model`
- Optional XGBoost backend alongside the default GradientBoostingClassifier - pass
  `model_type='xgboost'` (or `main.py --model-type xgboost`); falls back to the sklearn model with
  a warning if `xgboost` isn't installed
- Simple ensemble (`model_type='ensemble'`) that trains both the GBM and XGBoost backends and
  averages their predicted probabilities (`modeling.AveragingEnsembleClassifier`) - equal-weight
  averaging, not stacking
- Calibration/reliability diagnostics via `modeling.calibration_report()` - per-bin predicted vs.
  actual win rate, plus Expected Calibration Error (ECE)
- Multiple evaluation metrics (accuracy, AUC, Brier score, log loss)
- Feature importance analysis: the fast, built-in (impurity-based) importances via
  `feature_importance()`, or `shap_feature_importance()` for SHAP values, which also show
  *direction* of effect (e.g. "higher rest-day advantage pushes toward a home win") - both work
  transparently with the ensemble, averaging across its members
- **BUG FIX**: probability calibration and feature-importance extraction now work with current
  scikit-learn (`CalibratedClassifierCV(cv='prefit')` was removed upstream; feature importance
  was reading feature names off the wrong pipeline step)
- **BUG FIX**: `use_class_weight=True` (the default) computed a class-weight dict, printed it, and
  then never actually used it anywhere - `GradientBoostingClassifier` doesn't take `class_weight`,
  and nothing turned it into `sample_weight` for `.fit()`. Sample weights are now computed and
  passed through training (and the calibration re-fit) for real.

### Betting Simulation (`betting.py`)
- Kelly Criterion bet sizing, now using each bet's actual American odds (converted to decimal) for
  both stake sizing and payout, on whichever side (home or away) has the larger edge
- Minimum edge thresholds
- Bankroll management
- Performance by confidence level
- A parallel flat-betting simulation (fixed stake per bet) run on the same bets, for comparison
  against Kelly sizing
- **BUG FIX**: bet sizing and payouts used to always assume fair, no-vig 2.0 decimal odds
  regardless of the actual moneyline, and only ever considered betting the home side. Real
  `home_moneyline`/`away_moneyline` are now threaded through from the pipeline (`odds_df` param);
  omitting them still works, falling back to the old fair-odds assumption.

### Prediction (`predict.py`)
- Single game outcome prediction
- Edge calculation vs. market odds
- Betting recommendations


## Areas for Improvement

The code includes extensive TODO comments marking oversimplifications. Key areas:

### High Priority
1. **Replace placeholder odds** with real historical data
   - Current: Synthetic odds generated per real scheduled game (fixed a bug where they
     were generated for an unrelated random schedule and barely overlapped with real games -
     coverage is now 100%, but the *lines themselves* are still random, not real market data)
   - Needed: Real sportsbook data via API or scraping

2. **Add pitcher statistics**
   - Current: Team-level stats only
   - Needed: Starting pitcher ERA, WHIP, K/9, recent performance

3. **Implement park factors**
   - Current: All ballparks treated equally
   - Needed: Park-adjusted offensive/pitching metrics

### Medium Priority
4. **Better hyperparameter tuning**
   - ~~Try XGBoost~~ - done, `model_type='xgboost'` (optional dependency)
   - Still needed: LightGBM, Bayesian optimization (Optuna)

5. **Improve bet sizing**
   - ~~Kelly with actual (vig-included) American odds~~ - done, see `betting.evaluate_betting_performance(odds_df=...)`
   - Still needed: the odds themselves are still synthetic (see #1) - the *math* is now
     odds-realistic, but there's no real vig/market efficiency to bet against yet, so ROI numbers
     from the current placeholder data are not meaningful

6. **Add recency weighting**
   - ~~Simple moving averages~~ - done, EWMA is the default (see `features.engineer_features(use_ewma=True)`)

### Lower Priority (But Still Important)
7. ~~Walk-forward validation~~ - done, see `modeling.walk_forward_validation()`
8. ~~Calibration curves~~ - done, see `modeling.calibration_report()` (ECE + per-bin reliability table)
9. ~~SHAP values for feature importance~~ - done, see `modeling.shap_feature_importance()` (optional `shap` dependency)
10. Injury/roster data
11. Weather features
12. ~~Multiple model ensemble~~ - done (simple probability averaging), see `model_type='ensemble'` /
    `modeling.AveragingEnsembleClassifier`. Still a TODO: weighted averaging or stacking with a
    meta-learner instead of equal weights

## Extending the Model

Each module has clear extension points marked with TODO comments:

### Adding a New Feature
Edit `mlb_predictor/features.py`:

```python
def engineer_features(data, window_sizes=[5, 10, 20]):
    # ... existing code ...
    
    # Add your new feature here
    df['new_feature'] = calculate_new_feature(df)
    
    return feature_df
```

Then add it to the feature list in `mlb_predictor/modeling.py`:

```python
features = [
    'home_implied_prob_normalized',
    # ... existing features ...
    'new_feature',  # Add here
]
```

### Trying a Different Model
Edit `mlb_predictor/modeling.py`:

```python
def train_model(X_train, y_train, grid_search=True):
    # Replace GradientBoostingClassifier with your model
    from xgboost import XGBClassifier
    
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier())
    ])
    # ... rest of function
```

### Adding Real Odds Data
Edit `mlb_predictor/data_loader.py`:

```python
def download_odds_data(years):
    # Replace placeholder with API call
    import requests
    
    odds_data = []
    for year in years:
        response = requests.get(f"https://api.oddsapi.com/v1/{year}")
        # Parse response...
        odds_data.append(parsed_data)
    
    return pd.concat(odds_data)
```

## Performance Expectations

### Model Accuracy
- Baseline (always predict home): ~54%
- Expected model accuracy: 55-57%
- **Note**: Even small improvements over baseline are valuable for betting

### Betting Performance
- With perfect predictions at 57% accuracy and 3% edge requirement
- Expected ROI: 5-15% (but highly variable)
- **WARNING**: Current results use placeholder odds and may not reflect real performance

### Important Caveats
1. Past performance doesn't guarantee future results
2. Sportsbooks adjust lines based on sharp money
3. Vig reduces theoretical edge by 4-5%
4. Line shopping and timing matter significantly
5. Always bet responsibly with money you can afford to lose

## Contributing

When adding features or making changes:

1. **Document oversimplifications** - Add TODO comments explaining what could be better
2. **Maintain modularity** - Keep functions focused and modules independent
3. **Add docstrings** - Explain parameters, returns, and any gotchas
4. **Preserve the external API** - MLBPredictor class should remain stable
5. **Test chronological splits** - Never shuffle time series data

## License

This is an educational project. Use at your own risk. Sports betting carries risk of loss.

## Acknowledgments

- Game data from [Retrosheet](https://www.retrosheet.org/)
- Inspired by the sports analytics community
- Built with scikit-learn, pandas, and numpy
