"""
Feature engineering module for MLB prediction model.

This module transforms raw game data into predictive features including:
- Rolling team performance statistics
- Odds-based implied probabilities
- Temporal features (day of week, month, rest days)
- Batting statistics and derived metrics

TODO: Add pitcher-specific features (ERA, WHIP, recent starts)
TODO: Implement park factors (hitting/pitching park adjustments)
TODO: Add weather features (temperature, wind, precipitation)
TODO: Include injury/roster information
TODO: Add strength of schedule adjustments
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import warnings


def calculate_implied_probability(moneyline: float) -> float:
    """
    Convert American odds moneyline to implied probability.
    
    Parameters
    ----------
    moneyline : float
        American odds (e.g., -150 for favorite, +130 for underdog)
    
    Returns
    -------
    float
        Implied probability (between 0 and 1)
        
    Notes
    -----
    American odds conversion:
    - Negative (favorite): probability = |odds| / (|odds| + 100)
    - Positive (underdog): probability = 100 / (odds + 100)
    
    The sum of both teams' implied probabilities > 1.0 due to the
    bookmaker's margin (vig/juice). This should be normalized.
    """
    if moneyline > 0:
        return 100 / (moneyline + 100)
    else:
        return abs(moneyline) / (abs(moneyline) + 100)


def engineer_features(data: pd.DataFrame, 
                      window_sizes: List[int] = [5, 10, 20],
                      use_ewma: bool = True,
                      ewma_span: int = 10) -> pd.DataFrame:
    """
    Engineer features for the prediction model.
    
    Parameters
    ----------
    data : pd.DataFrame
        Merged game and odds data
    window_sizes : list of int, optional
        Window sizes for rolling statistics (default: [5, 10, 20])
    use_ewma : bool, optional
        Whether to use exponentially weighted moving average (default: True)
        EWMA gives more weight to recent games, which is more realistic
    ewma_span : int, optional
        Span parameter for EWMA (default: 10)
        Roughly equivalent to a 10-game average but with recency bias
    
    Returns
    -------
    pd.DataFrame
        DataFrame with engineered features
        
    Notes
    -----
    This function is now defensive against missing columns and will:
    - Check for required columns before using them
    - Fill missing values appropriately
    - Log warnings for missing expected columns
    
    IMPROVEMENTS IMPLEMENTED:
    1. ✅ Exponentially weighted moving averages (EWMA) for recency weighting
       - Recent games weighted more heavily than older games
       - More realistic reflection of current team strength
       
    REMAINING OVERSIMPLIFICATIONS:
    2. No park factors - all ballparks treated equally
       TODO: Add park-adjusted stats (wRC+, FIP-, etc.)
       
    3. No pitcher information - using only team-level stats
       TODO: Add starting pitcher features (ERA, K/9, recent performance)
       
    4. No platoon splits - ignoring L/R matchups
       TODO: Include team performance vs LHP/RHP
       
    5. Simple batting average instead of advanced metrics
       TODO: Use wOBA, wRC+, ISO, BABIP for better offensive measurement
       
    6. No bullpen quality metrics
       TODO: Add bullpen ERA, high-leverage performance
    """
    print("Engineering features...")
    
    # Make a copy to avoid modifying original data
    df = data.copy()
    
    # Convert date to datetime
    df['game_date'] = pd.to_datetime(df['date_str'], format='%Y%m%d', errors='coerce')
    
    # Create target variable: 1 if home team wins, 0 otherwise
    df['home_win'] = (df['home_score'] > df['visiting_score']).astype(int)
    
    # === Basic batting statistics (with defensive checks) ===
    # This section fixes the KeyError by checking column existence first
    
    if all(col in df.columns for col in ['home_H', 'home_AB', 'visiting_H', 'visiting_AB']):
        # Calculate batting averages
        df['home_batting_avg'] = df['home_H'] / df['home_AB']
        df['visiting_batting_avg'] = df['visiting_H'] / df['visiting_AB']
        
        # Handle division by zero
        df['home_batting_avg'] = df['home_batting_avg'].fillna(0)
        df['visiting_batting_avg'] = df['visiting_batting_avg'].fillna(0)
        
        # IMPROVEMENT: Add slugging percentage (SLG) and ISO
        # SLG = (1B + 2*2B + 3*3B + 4*HR) / AB
        # ISO = SLG - AVG (isolated power - measures extra-base hit ability)
        if all(col in df.columns for col in ['home_2B', 'home_3B', 'home_HR',
                                              'visiting_2B', 'visiting_3B', 'visiting_HR']):
            # Calculate singles (H - 2B - 3B - HR)
            df['home_1B'] = df['home_H'] - df['home_2B'] - df['home_3B'] - df['home_HR']
            df['visiting_1B'] = df['visiting_H'] - df['visiting_2B'] - df['visiting_3B'] - df['visiting_HR']
            
            # Slugging percentage
            df['home_slugging'] = (
                (df['home_1B'] + 2*df['home_2B'] + 3*df['home_3B'] + 4*df['home_HR']) / df['home_AB']
            ).fillna(0)
            df['visiting_slugging'] = (
                (df['visiting_1B'] + 2*df['visiting_2B'] + 3*df['visiting_3B'] + 4*df['visiting_HR']) / df['visiting_AB']
            ).fillna(0)
            
            # Isolated power (measures extra-base hit ability)
            df['home_iso'] = df['home_slugging'] - df['home_batting_avg']
            df['visiting_iso'] = df['visiting_slugging'] - df['visiting_batting_avg']
        
        # TODO: Add on-base percentage (OBP)
        # OBP = (H + BB + HBP) / (AB + BB + HBP + SF)
        # Requires: BB (walks), HBP (hit by pitch), SF (sacrifice flies)
        
        # TODO: Add wOBA (weighted on-base average)
        # More accurate than OPS at measuring offensive value
        
    else:
        print("Warning: Batting stat columns not found. Skipping batting averages.")
        df['home_batting_avg'] = 0
        df['visiting_batting_avg'] = 0
        df['home_slugging'] = 0
        df['visiting_slugging'] = 0
        df['home_iso'] = 0
        df['visiting_iso'] = 0
    
    # === Odds-based features ===
    # Calculate implied probabilities from moneylines
    if 'home_moneyline' in df.columns and 'away_moneyline' in df.columns:
        df['home_implied_prob'] = df['home_moneyline'].apply(calculate_implied_probability)
        df['away_implied_prob'] = df['away_moneyline'].apply(calculate_implied_probability)
        
        # Normalize to remove bookmaker's vig
        # OVERSIMPLIFICATION: This assumes all books have same vig
        # TODO: Track vig/margin separately as a feature (market efficiency indicator)
        df['total_implied_prob'] = df['home_implied_prob'] + df['away_implied_prob']
        df['home_implied_prob_normalized'] = df['home_implied_prob'] / df['total_implied_prob']
        df['away_implied_prob_normalized'] = df['away_implied_prob'] / df['total_implied_prob']
        
        # TODO: Track closing line value (CLV) - difference between opening and closing odds
        # This is a strong indicator of betting skill
        
    else:
        print("Warning: Moneyline columns not found. Using neutral probabilities.")
        df['home_implied_prob_normalized'] = 0.5
        df['away_implied_prob_normalized'] = 0.5
    
    # === Temporal features ===
    # Day/night game indicator
    if 'day_night' in df.columns:
        df['is_night_game'] = (df['day_night'] == 'N').astype(int)
        # TODO: Add time of day for day games (1pm vs 4pm can matter for hitting)
    else:
        df['is_night_game'] = 1  # Default to night
    
    # === Rolling team performance statistics ===
    # Sort by date for time-series features
    df = df.sort_values(by=['game_date'])

    # Build a long-format (one row per team-game) history with rolling
    # performance stats, then attach each team's "as of just before this
    # game" stats back onto the game-level DataFrame.
    # IMPROVEMENT: Using EWMA for recency weighting when enabled
    long_stats = _build_team_long_stats(df, window_sizes, use_ewma, ewma_span)

    # TODO: Add rolling stats for:
    # - Pythagenpat expected win% (run differential based)
    # - vs. specific divisions
    # - Recent streak (last 3-5 games)

    # Merge rolling stats back into game data (drops games where either
    # team has no prior history yet, matching the original behavior)
    feature_df = _merge_rolling_stats(df, long_stats, window_sizes)

    # === Derived comparison features ===
    for window in window_sizes:
        # Run differential features
        feature_df[f'home_rolling_{window}_run_diff'] = (
            feature_df[f'home_rolling_{window}_runs_scored'] - 
            feature_df[f'home_rolling_{window}_runs_allowed']
        )
        feature_df[f'visiting_rolling_{window}_run_diff'] = (
            feature_df[f'visiting_rolling_{window}_runs_scored'] - 
            feature_df[f'visiting_rolling_{window}_runs_allowed']
        )
        
        # IMPROVEMENT: Pythagenpat expected winning percentage
        # More accurate than raw win% - based on run differential
        # Formula: exp_win% = (runs_scored^2) / (runs_scored^2 + runs_allowed^2)
        # Using exponent of 1.83 (Pythagenpat for baseball)
        exponent = 1.83
        
        home_rs = feature_df[f'home_rolling_{window}_runs_scored']
        home_ra = feature_df[f'home_rolling_{window}_runs_allowed']
        feature_df[f'home_rolling_{window}_pythag_win_pct'] = (
            (home_rs ** exponent) / ((home_rs ** exponent) + (home_ra ** exponent))
        ).fillna(0.5)  # Default to .500 if division by zero
        
        visiting_rs = feature_df[f'visiting_rolling_{window}_runs_scored']
        visiting_ra = feature_df[f'visiting_rolling_{window}_runs_allowed']
        feature_df[f'visiting_rolling_{window}_pythag_win_pct'] = (
            (visiting_rs ** exponent) / ((visiting_rs ** exponent) + (visiting_ra ** exponent))
        ).fillna(0.5)
        
        # Head-to-head comparison features
        feature_df[f'runs_scored_diff_{window}'] = (
            feature_df[f'home_rolling_{window}_runs_scored'] - 
            feature_df[f'visiting_rolling_{window}_runs_scored']
        )
        feature_df[f'runs_allowed_diff_{window}'] = (
            feature_df[f'home_rolling_{window}_runs_allowed'] - 
            feature_df[f'visiting_rolling_{window}_runs_allowed']
        )
        feature_df[f'win_pct_diff_{window}'] = (
            feature_df[f'home_rolling_{window}_win_pct'] - 
            feature_df[f'visiting_rolling_{window}_win_pct']
        )
        feature_df[f'pythag_win_pct_diff_{window}'] = (
            feature_df[f'home_rolling_{window}_pythag_win_pct'] - 
            feature_df[f'visiting_rolling_{window}_pythag_win_pct']
        )
    
    # === Calendar features (one-hot encoded) ===
    # Day of week
    feature_df['day_of_week_num'] = pd.to_datetime(feature_df['date_str']).dt.dayofweek
    day_of_week_dummies = pd.get_dummies(feature_df['day_of_week_num'], prefix='day')
    feature_df = pd.concat([feature_df, day_of_week_dummies], axis=1)
    
    # Month (baseball season only)
    feature_df['month'] = pd.to_datetime(feature_df['date_str']).dt.month
    month_dummies = pd.get_dummies(feature_df['month'], prefix='month')
    feature_df = pd.concat([feature_df, month_dummies], axis=1)
    
    # TODO: Add "days into season" or "games played" to capture team development
    # Early season stats are less predictive than late season
    
    # === Rest days ===
    # Calculate days between games for each team
    # OVERSIMPLIFICATION: This is a simplified calculation
    # TODO: Get actual off-days from schedule (including travel days, time zones)
    feature_df = feature_df.sort_values(['home_team', 'game_date'])
    feature_df['home_days_rest'] = (
        feature_df.groupby('home_team')['game_date']
        .diff()
        .dt.days
    )
    
    feature_df = feature_df.sort_values(['visiting_team', 'game_date'])
    feature_df['visiting_days_rest'] = (
        feature_df.groupby('visiting_team')['game_date']
        .diff()
        .dt.days
    )
    
    # Fill NaN with median (first game of season)
    feature_df['home_days_rest'] = feature_df['home_days_rest'].fillna(
        feature_df['home_days_rest'].median()
    )
    feature_df['visiting_days_rest'] = feature_df['visiting_days_rest'].fillna(
        feature_df['visiting_days_rest'].median()
    )
    
    # Days rest advantage
    feature_df['days_rest_advantage'] = (
        feature_df['home_days_rest'] - feature_df['visiting_days_rest']
    )
    
    # Clean up any remaining NaN values
    feature_df = feature_df.fillna(0)
    
    print(f"Feature engineering complete. Dataset has {len(feature_df)} rows "
          f"and {len(feature_df.columns)} columns.")
    
    return feature_df


def _build_team_long_stats(df: pd.DataFrame,
                           window_sizes: List[int],
                           use_ewma: bool,
                           ewma_span: int) -> pd.DataFrame:
    """
    Build a long-format (one row per team-game) history with rolling
    performance stats, used to attach each team's "as of just before this
    game" stats back onto the game-level DataFrame via merge_asof.

    Replaces a previous per-team dict + per-row iterrows() lookup, which
    was O(n^2) (every game re-filtered each team's full history) and took
    minutes on a full multi-season dataset. This computes the same rolling
    values with vectorized groupby/rolling/ewm calls.
    """
    home = pd.DataFrame({
        'game_date': df['game_date'].values,
        'team': df['home_team'].values,
        'team_score': df['home_score'].values,
        'opponent_score': df['visiting_score'].values,
        'is_home': 1,
    })
    away = pd.DataFrame({
        'game_date': df['game_date'].values,
        'team': df['visiting_team'].values,
        'team_score': df['visiting_score'].values,
        'opponent_score': df['home_score'].values,
        'is_home': 0,
    })
    long_df = pd.concat([home, away], ignore_index=True)
    # Sort each team's games chronologically so rolling/ewm windows are
    # computed in the right order.
    long_df = long_df.sort_values(['team', 'game_date'], kind='mergesort').reset_index(drop=True)

    won = long_df['team_score'] > long_df['opponent_score']
    long_df['won'] = won.astype(int)
    long_df['home_win'] = (won & (long_df['is_home'] == 1)).astype(int)
    long_df['away_win'] = (won & (long_df['is_home'] == 0)).astype(int)

    grp = long_df.groupby('team', sort=False)
    for window in window_sizes:
        if use_ewma:
            # Exponentially weighted moving average - recent games matter more
            # span parameter controls the decay rate (higher = slower decay)
            span = min(window, ewma_span)
            long_df[f'rolling_{window}_runs_scored'] = grp['team_score'].transform(
                lambda s: s.ewm(span=span, min_periods=1).mean())
            long_df[f'rolling_{window}_runs_allowed'] = grp['opponent_score'].transform(
                lambda s: s.ewm(span=span, min_periods=1).mean())
            long_df[f'rolling_{window}_win_pct'] = grp['won'].transform(
                lambda s: s.ewm(span=span, min_periods=1).mean())
        else:
            # Simple moving average (original implementation)
            long_df[f'rolling_{window}_runs_scored'] = grp['team_score'].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean())
            long_df[f'rolling_{window}_runs_allowed'] = grp['opponent_score'].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean())
            long_df[f'rolling_{window}_win_pct'] = grp['won'].transform(
                lambda s: s.rolling(window=window, min_periods=1).mean())

        # IMPROVEMENT: Add home/away splits
        # Teams often perform differently at home vs on the road
        long_df[f'rolling_{window}_home_win_pct'] = grp['home_win'].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean())
        long_df[f'rolling_{window}_away_win_pct'] = grp['away_win'].transform(
            lambda s: s.rolling(window=window, min_periods=1).mean())

    return long_df


def _merge_rolling_stats(df: pd.DataFrame,
                         long_stats: pd.DataFrame,
                         window_sizes: List[int]) -> pd.DataFrame:
    """
    Attach each team's rolling performance stats "as of just before this
    game" onto the game-level DataFrame.

    Uses merge_asof (direction='backward', allow_exact_matches=False) to
    find, for each team, its most recent game strictly before the current
    one - the same "prevent data leakage" semantics as the original
    per-row filter (`team_history['game_date'] < game_date`), but as a
    single vectorized sorted-merge per side instead of one DataFrame
    filter per game.
    """
    stat_cols = []
    for window in window_sizes:
        stat_cols += [
            f'rolling_{window}_runs_scored',
            f'rolling_{window}_runs_allowed',
            f'rolling_{window}_win_pct',
            f'rolling_{window}_home_win_pct',
            f'rolling_{window}_away_win_pct',
        ]
    stat_cols = [c for c in stat_cols if c in long_stats.columns]

    # merge_asof requires both frames sorted ascending by the 'on' column.
    long_sorted = long_stats.sort_values('game_date', kind='mergesort').reset_index(drop=True)
    working = df.sort_values('game_date', kind='mergesort').reset_index(drop=True)
    lookup = long_sorted[['game_date', 'team'] + stat_cols]

    home_matched = pd.merge_asof(
        working[['game_date', 'home_team']],
        lookup,
        on='game_date', left_by='home_team', right_by='team',
        direction='backward', allow_exact_matches=False,
    )
    visiting_matched = pd.merge_asof(
        working[['game_date', 'visiting_team']],
        lookup,
        on='game_date', left_by='visiting_team', right_by='team',
        direction='backward', allow_exact_matches=False,
    )

    # Only include games where both teams already have prior history,
    # matching the original behavior.
    have_history = home_matched['team'].notna() & visiting_matched['team'].notna()

    feature_df = working.loc[have_history].reset_index(drop=True).copy()
    home_matched = home_matched.loc[have_history].reset_index(drop=True)
    visiting_matched = visiting_matched.loc[have_history].reset_index(drop=True)

    for window in window_sizes:
        feature_df[f'home_rolling_{window}_runs_scored'] = home_matched[f'rolling_{window}_runs_scored']
        feature_df[f'home_rolling_{window}_runs_allowed'] = home_matched[f'rolling_{window}_runs_allowed']
        feature_df[f'home_rolling_{window}_win_pct'] = home_matched[f'rolling_{window}_win_pct']

        feature_df[f'visiting_rolling_{window}_runs_scored'] = visiting_matched[f'rolling_{window}_runs_scored']
        feature_df[f'visiting_rolling_{window}_runs_allowed'] = visiting_matched[f'rolling_{window}_runs_allowed']
        feature_df[f'visiting_rolling_{window}_win_pct'] = visiting_matched[f'rolling_{window}_win_pct']

        # IMPROVEMENT: Add home/away splits (asymmetric by design: the
        # home team's rolling *home* win% and the visiting team's rolling
        # *away* win%, mirroring the optional features modeling.py looks for)
        home_split_col = f'rolling_{window}_home_win_pct'
        if home_split_col in home_matched.columns:
            feature_df[f'home_rolling_{window}_home_win_pct'] = home_matched[home_split_col]

        away_split_col = f'rolling_{window}_away_win_pct'
        if away_split_col in visiting_matched.columns:
            feature_df[f'visiting_rolling_{window}_away_win_pct'] = visiting_matched[away_split_col]

    return feature_df


def calculate_game_features(home_team: str,
                            visiting_team: str,
                            game_date: str,
                            game_data: pd.DataFrame,
                            odds: Dict = None,
                            window_sizes: List[int] = [5, 10, 20]) -> Dict:
    """
    Calculate features for a single game prediction.
    
    Parameters
    ----------
    home_team : str
        Home team abbreviation
    visiting_team : str
        Visiting team abbreviation
    game_date : str
        Game date in YYYYMMDD format
    game_data : pd.DataFrame
        Historical game data
    odds : dict, optional
        Dictionary with 'home_moneyline' and 'away_moneyline' keys
    window_sizes : list of int, optional
        Window sizes for rolling statistics
    
    Returns
    -------
    dict
        Dictionary of features for the game
        
    Notes
    -----
    This function is used for making predictions on new games.
    It calculates the same features as engineer_features but for a single matchup.
    """
    features = {}
    
    # Get historical data for both teams
    home_team_data = game_data[
        (game_data['home_team'] == home_team) | 
        (game_data['visiting_team'] == home_team)
    ].sort_values('date')
    
    visiting_team_data = game_data[
        (game_data['home_team'] == visiting_team) | 
        (game_data['visiting_team'] == visiting_team)
    ].sort_values('date')
    
    # Calculate rolling stats for both teams
    for window in window_sizes:
        # Home team stats
        home_recent = home_team_data.tail(window)
        home_games = home_recent[home_recent['home_team'] == home_team]
        away_games = home_recent[home_recent['visiting_team'] == home_team]
        
        runs_scored = (
            home_games['home_score'].sum() + 
            away_games['visiting_score'].sum()
        ) / max(1, len(home_recent))
        
        runs_allowed = (
            home_games['visiting_score'].sum() + 
            away_games['home_score'].sum()
        ) / max(1, len(home_recent))
        
        wins = (
            (home_games['home_score'] > home_games['visiting_score']).sum() +
            (away_games['visiting_score'] > away_games['home_score']).sum()
        )
        win_pct = wins / max(1, len(home_recent))
        
        features[f'home_rolling_{window}_runs_scored'] = runs_scored
        features[f'home_rolling_{window}_runs_allowed'] = runs_allowed
        features[f'home_rolling_{window}_win_pct'] = win_pct
        features[f'home_rolling_{window}_run_diff'] = runs_scored - runs_allowed
        
        # Visiting team stats
        visiting_recent = visiting_team_data.tail(window)
        home_games = visiting_recent[visiting_recent['home_team'] == visiting_team]
        away_games = visiting_recent[visiting_recent['visiting_team'] == visiting_team]
        
        runs_scored = (
            home_games['home_score'].sum() + 
            away_games['visiting_score'].sum()
        ) / max(1, len(visiting_recent))
        
        runs_allowed = (
            home_games['visiting_score'].sum() + 
            away_games['home_score'].sum()
        ) / max(1, len(visiting_recent))
        
        wins = (
            (home_games['home_score'] > home_games['visiting_score']).sum() +
            (away_games['visiting_score'] > away_games['home_score']).sum()
        )
        win_pct = wins / max(1, len(visiting_recent))
        
        features[f'visiting_rolling_{window}_runs_scored'] = runs_scored
        features[f'visiting_rolling_{window}_runs_allowed'] = runs_allowed
        features[f'visiting_rolling_{window}_win_pct'] = win_pct
        features[f'visiting_rolling_{window}_run_diff'] = runs_scored - runs_allowed
    
    # Comparison features
    for window in window_sizes:
        features[f'runs_scored_diff_{window}'] = (
            features[f'home_rolling_{window}_runs_scored'] - 
            features[f'visiting_rolling_{window}_runs_scored']
        )
        features[f'runs_allowed_diff_{window}'] = (
            features[f'home_rolling_{window}_runs_allowed'] - 
            features[f'visiting_rolling_{window}_runs_allowed']
        )
        features[f'win_pct_diff_{window}'] = (
            features[f'home_rolling_{window}_win_pct'] - 
            features[f'visiting_rolling_{window}_win_pct']
        )
    
    # Rest days (simplified)
    home_last_game = home_team_data.iloc[-1]['date'] if not home_team_data.empty else None
    visiting_last_game = visiting_team_data.iloc[-1]['date'] if not visiting_team_data.empty else None
    
    if home_last_game and game_date:
        home_days_rest = (pd.to_datetime(game_date) - pd.to_datetime(home_last_game)).days
    else:
        home_days_rest = 1
    
    if visiting_last_game and game_date:
        visiting_days_rest = (pd.to_datetime(game_date) - pd.to_datetime(visiting_last_game)).days
    else:
        visiting_days_rest = 1
    
    features['days_rest_advantage'] = home_days_rest - visiting_days_rest
    
    # Temporal features
    game_datetime = pd.to_datetime(game_date)
    day_of_week = game_datetime.dayofweek
    month = game_datetime.month
    
    for i in range(7):
        features[f'day_{i}'] = 1 if i == day_of_week else 0
    
    for i in range(4, 11):
        features[f'month_{i}'] = 1 if i == month else 0
    
    # Odds features
    if odds:
        home_ml = odds.get('home_moneyline', 0)
        away_ml = odds.get('away_moneyline', 0)
        
        home_prob = calculate_implied_probability(home_ml)
        away_prob = calculate_implied_probability(away_ml)
        
        total_prob = home_prob + away_prob
        features['home_implied_prob_normalized'] = home_prob / total_prob
        features['away_implied_prob_normalized'] = away_prob / total_prob
    else:
        features['home_implied_prob_normalized'] = 0.5
        features['away_implied_prob_normalized'] = 0.5
    
    # Default to night game
    features['is_night_game'] = 1
    
    return features
