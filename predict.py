"""
Prediction module for single MLB games.

This module handles predicting outcomes of individual games
using the trained model and historical data.

TODO: Add confidence intervals for predictions
TODO: Implement game simulation (Monte Carlo)
TODO: Add expected score predictions (not just win probability)
"""

import pandas as pd
from datetime import datetime
from sklearn.pipeline import Pipeline
from typing import Dict, Optional
from features import calculate_game_features


def predict_game(model: Pipeline,
                game_data: pd.DataFrame,
                home_team: str,
                visiting_team: str,
                game_date: Optional[str] = None,
                odds: Optional[Dict] = None,
                home_starting_pitcher_id: Optional[str] = None,
                visiting_starting_pitcher_id: Optional[str] = None,
                park_id: Optional[str] = None) -> Dict:
    """
    Predict the outcome of a specific game.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained prediction model
    game_data : pd.DataFrame
        Historical game data for feature calculation
    home_team : str
        Home team abbreviation (Retrosheet format, e.g., 'NYA', 'BOS')
    visiting_team : str
        Visiting team abbreviation
    game_date : str, optional
        Game date in YYYYMMDD format (defaults to today)
    odds : dict, optional
        Dictionary with 'home_moneyline' and 'away_moneyline' keys
        Example: {'home_moneyline': -150, 'away_moneyline': +130}
    home_starting_pitcher_id, visiting_starting_pitcher_id : str, optional
        Retrosheet pitcher IDs (e.g. "grayj003") for each starter. Used to
        compute a rolling earned-runs-allowed proxy for pitcher quality;
        omitting them falls back to a neutral league-average value for
        that feature rather than failing.
    park_id : str, optional
        Retrosheet park ID (e.g. "PHO01") for the game's venue. Used to
        compute a rolling park-factor proxy; omitting it falls back to a
        neutral league-average value.

    Returns
    -------
    dict
        Prediction details including:
        - game_date: Date of game
        - home_team: Home team
        - visiting_team: Visiting team
        - home_win_probability: Predicted probability of home win
        - visiting_win_probability: Predicted probability of away win
        - prediction: 'Home Win' or 'Visiting Win'
        - confidence: Max probability (0 to 1)
        - edge: Difference from market odds (if provided)
        - recommendation: Betting recommendation (if odds provided)
        
    Examples
    --------
    >>> prediction = predict_game(
    ...     model=trained_model,
    ...     game_data=historical_data,
    ...     home_team="NYA",
    ...     visiting_team="BOS",
    ...     game_date="20240601",
    ...     odds={'home_moneyline': -150, 'away_moneyline': +130}
    ... )
    >>> print(f"Home win probability: {prediction['home_win_probability']:.1%}")
    
    Notes
    -----
    TODO: Add prediction intervals (not just point estimates)
    TODO: Add game simulation for score predictions
    TODO: Include uncertainty quantification
    """
    if model is None:
        raise ValueError("Model not trained yet. Train model before predicting.")
    
    if game_data is None or len(game_data) == 0:
        raise ValueError("Game data not loaded. Cannot calculate features.")
    
    # Default to today if no date provided
    if game_date is None:
        game_date = datetime.now().strftime('%Y%m%d')
    
    # Calculate features for this specific game
    features = calculate_game_features(
        home_team=home_team,
        visiting_team=visiting_team,
        game_date=game_date,
        game_data=game_data,
        odds=odds,
        home_starting_pitcher_id=home_starting_pitcher_id,
        visiting_starting_pitcher_id=visiting_starting_pitcher_id,
        park_id=park_id
    )
    
    # Convert to DataFrame
    feature_df = pd.DataFrame([features])
    
    # Get model's expected features
    try:
        model_features = model.feature_names_in_
    except AttributeError:
        # For pipeline, get from classifier step
        model_features = model.named_steps['classifier'].feature_names_in_
    
    # Ensure all model features are present
    for feat in model_features:
        if feat not in feature_df.columns:
            feature_df[feat] = 0
    
    # Select features in correct order
    X = feature_df[model_features]
    
    # Make prediction
    home_win_prob = model.predict_proba(X)[0, 1]
    visiting_win_prob = 1 - home_win_prob
    
    # Determine prediction
    prediction_label = 'Home Win' if home_win_prob > 0.5 else 'Visiting Win'
    confidence = max(home_win_prob, visiting_win_prob)
    
    # Build result dictionary
    result = {
        'game_date': game_date,
        'home_team': home_team,
        'visiting_team': visiting_team,
        'home_win_probability': home_win_prob,
        'visiting_win_probability': visiting_win_prob,
        'prediction': prediction_label,
        'confidence': confidence
    }
    
    # Add betting analysis if odds provided
    if odds:
        from features import calculate_implied_probability
        
        home_ml = odds.get('home_moneyline', 0)
        away_ml = odds.get('away_moneyline', 0)
        
        # Calculate market probabilities
        home_market_prob = calculate_implied_probability(home_ml)
        away_market_prob = calculate_implied_probability(away_ml)
        
        # Normalize market probabilities
        total_market_prob = home_market_prob + away_market_prob
        home_market_prob_norm = home_market_prob / total_market_prob
        away_market_prob_norm = away_market_prob / total_market_prob
        
        # Calculate edge
        home_edge = home_win_prob - home_market_prob_norm
        away_edge = visiting_win_prob - away_market_prob_norm
        
        # Determine best bet
        if abs(home_edge) > abs(away_edge):
            best_bet = 'Home' if home_edge > 0 else None
            edge = home_edge
        else:
            best_bet = 'Away' if away_edge > 0 else None
            edge = away_edge
        
        # Generate recommendation
        if edge is not None and abs(edge) > 0.03:  # 3% edge threshold
            if best_bet == 'Home':
                recommendation = f"BET HOME ({home_team}) - Edge: {edge:.1%}"
            else:
                recommendation = f"BET AWAY ({visiting_team}) - Edge: {edge:.1%}"
        else:
            recommendation = "NO BET - Insufficient edge"
        
        # Add to result
        result.update({
            'home_moneyline': home_ml,
            'away_moneyline': away_ml,
            'market_home_prob': home_market_prob_norm,
            'market_away_prob': away_market_prob_norm,
            'home_edge': home_edge,
            'away_edge': away_edge,
            'recommendation': recommendation
        })
    
    return result


def format_prediction_output(prediction: Dict) -> str:
    """
    Format prediction dictionary as readable text.
    
    Parameters
    ----------
    prediction : dict
        Prediction result from predict_game()
    
    Returns
    -------
    str
        Formatted prediction text
    """
    output = []
    output.append("=" * 70)
    output.append("MLB GAME PREDICTION")
    output.append("=" * 70)
    output.append(f"Date:           {prediction['game_date']}")
    output.append(f"Matchup:        {prediction['visiting_team']} @ {prediction['home_team']}")
    output.append("")
    output.append("PROBABILITIES:")
    output.append(f"  Home Win:     {prediction['home_win_probability']:.1%}")
    output.append(f"  Away Win:     {prediction['visiting_win_probability']:.1%}")
    output.append("")
    output.append(f"PREDICTION:     {prediction['prediction']}")
    output.append(f"CONFIDENCE:     {prediction['confidence']:.1%}")
    
    # Add betting information if available
    if 'recommendation' in prediction:
        output.append("")
        output.append("BETTING ANALYSIS:")
        output.append(f"  Home Odds:    {prediction['home_moneyline']:+d}")
        output.append(f"  Away Odds:    {prediction['away_moneyline']:+d}")
        output.append(f"  Market Home:  {prediction['market_home_prob']:.1%}")
        output.append(f"  Market Away:  {prediction['market_away_prob']:.1%}")
        output.append(f"  Home Edge:    {prediction['home_edge']:+.1%}")
        output.append(f"  Away Edge:    {prediction['away_edge']:+.1%}")
        output.append("")
        output.append(f"RECOMMENDATION: {prediction['recommendation']}")
    
    output.append("=" * 70)
    
    return "\n".join(output)
