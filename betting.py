"""
Betting performance module for MLB prediction model.

This module simulates betting strategies and evaluates profitability.

TODO: Add closing line value (CLV) tracking
TODO: Add bankroll management and drawdown-based risk-of-ruin analysis
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from typing import Dict, Optional


def american_odds_to_decimal(moneyline: float) -> float:
    """
    Convert American odds to decimal odds.

    Parameters
    ----------
    moneyline : float
        American odds (e.g., -150 for favorite, +130 for underdog)

    Returns
    -------
    float
        Decimal odds (e.g., 1.667 for -150, 2.30 for +130). A $1 stake
        that wins returns `decimal_odds` total (`decimal_odds - 1` profit).
    """
    if moneyline > 0:
        return moneyline / 100 + 1
    else:
        return 100 / abs(moneyline) + 1


def evaluate_betting_performance(model: Pipeline,
                                 X_test: pd.DataFrame,
                                 y_test: pd.Series,
                                 odds_df: Optional[pd.DataFrame] = None,
                                 kelly_fraction: float = 0.25,
                                 min_edge: float = 0.03,
                                 max_bet_pct: float = 0.05,
                                 initial_bankroll: float = 1000.0,
                                 flat_bet_pct: float = 0.02) -> Dict:
    """
    Evaluate betting performance using the model's predictions.

    Parameters
    ----------
    model : sklearn.pipeline.Pipeline
        Trained prediction model
    X_test : pd.DataFrame
        Test features
    y_test : pd.Series
        Test outcomes (1 = home win, 0 = away win)
    odds_df : pd.DataFrame, optional
        DataFrame with 'home_moneyline'/'away_moneyline' columns, indexed
        the same as X_test/y_test (e.g. feature_df.loc[X_test.index, ...]).
        When given, bet sizing and payouts use the real American odds for
        whichever side (home or away) has the edge. When omitted, falls
        back to assuming fair 2.0 decimal odds on both sides (the
        previous behavior) - useful for quick checks without real odds.
    kelly_fraction : float, optional
        Fraction of Kelly Criterion to use (default: 0.25 for quarter-Kelly)
    min_edge : float, optional
        Minimum edge required to place bet (default: 0.03 = 3%)
    max_bet_pct : float, optional
        Maximum bet as fraction of bankroll (default: 0.05 = 5%)
    initial_bankroll : float, optional
        Starting bankroll (default: 1000.0)
    flat_bet_pct : float, optional
        Stake size for the flat-betting comparison, as a fraction of the
        *initial* bankroll, bet on every game Kelly would also bet on
        (default: 0.02 = 2%)

    Returns
    -------
    dict
        Dictionary with betting performance metrics for both the Kelly
        strategy and a flat-betting comparison

    Notes
    -----
    CURRENT BETTING STRATEGY:
    - Considers both sides of each game: bet when model probability beats
      vegas probability by at least min_edge, on whichever side (home or
      away) has the larger edge
    - Use fractional Kelly for bet sizing, with real American odds when
      odds_df is provided (see kelly_criterion())
    - Cap bets at max_bet_pct of bankroll for risk management
    - Also simulate flat betting (fixed stake) on the same bets, for
      comparison

    REMAINING OVERSIMPLIFICATIONS:

    1. ODDS STRUCTURE:
       - odds_df still comes from the placeholder synthetic odds
         (data_loader.download_odds_data), not a real sportsbook feed -
         the math here is now odds-realistic, the *odds* themselves
         aren't yet
       - No modeling of vig/line shopping across multiple books

    2. BANKROLL MANAGEMENT:
       - No stop-loss or take-profit rules
       - No adjustment for drawdowns
       - TODO: Implement risk of ruin calculations
       - TODO: Add dynamic Kelly fraction based on confidence

    3. MARKET EFFICIENCY:
       - Assumes we can always bet at listed odds
       - Reality: Lines move, limits exist, odds can be pulled
       - TODO: Model closing line value (CLV) to validate edge

    4. CORRELATION:
       - Treats each bet independently
       - Reality: Some games are correlated (division rivals, weather)
       - TODO: Consider portfolio effects
    """
    print("\n" + "=" * 70)
    print("EVALUATING BETTING PERFORMANCE")
    print("=" * 70)

    # Get predicted probabilities (of a home win)
    home_pred_prob = model.predict_proba(X_test)[:, 1]
    n = len(X_test)

    home_vegas_prob = (X_test['home_implied_prob_normalized'].values
                        if 'home_implied_prob_normalized' in X_test.columns
                        else np.full(n, 0.5))
    away_vegas_prob = (X_test['away_implied_prob_normalized'].values
                        if 'away_implied_prob_normalized' in X_test.columns
                        else 1 - home_vegas_prob)

    has_real_odds = (odds_df is not None
                      and 'home_moneyline' in odds_df.columns
                      and 'away_moneyline' in odds_df.columns)
    if has_real_odds:
        aligned_odds = odds_df.loc[X_test.index]
        home_decimal_odds = aligned_odds['home_moneyline'].apply(american_odds_to_decimal).values
        away_decimal_odds = aligned_odds['away_moneyline'].apply(american_odds_to_decimal).values
    else:
        # Fallback: assume fair, no-vig 2.0 decimal odds (+100 American)
        # on both sides, matching the original simplified behavior.
        home_decimal_odds = np.full(n, 2.0)
        away_decimal_odds = np.full(n, 2.0)

    betting_df = pd.DataFrame({
        'true_outcome': y_test.values,
        'home_pred_prob': home_pred_prob,
        'away_pred_prob': 1 - home_pred_prob,
        'home_vegas_prob': home_vegas_prob,
        'away_vegas_prob': away_vegas_prob,
        'home_decimal_odds': home_decimal_odds,
        'away_decimal_odds': away_decimal_odds,
    })

    # Initialize tracking (two parallel bankrolls: Kelly-sized vs. flat)
    kelly_bankroll = initial_bankroll
    flat_bankroll = initial_bankroll
    flat_stake_size = flat_bet_pct * initial_bankroll
    bets = []

    print(f"\nInitial Bankroll: ${initial_bankroll:.2f}")
    print(f"Odds source: {'real moneylines' if has_real_odds else 'assumed fair 2.0 decimal (no odds_df given)'}")
    print(f"Kelly Fraction: {kelly_fraction:.2f} (conservative)")
    print(f"Minimum Edge: {min_edge:.1%}")
    print(f"Maximum Bet: {max_bet_pct:.1%} of bankroll")
    print(f"Flat bet size: {flat_bet_pct:.1%} of initial bankroll (${flat_stake_size:.2f})\n")

    # Simulate betting on each game, choosing whichever side (home/away)
    # has the larger edge - mirrors predict.py's per-game recommendation.
    for _, row in betting_df.iterrows():
        home_edge = row['home_pred_prob'] - row['home_vegas_prob']
        away_edge = row['away_pred_prob'] - row['away_vegas_prob']

        if home_edge >= away_edge:
            side, edge = 'home', home_edge
            win_prob, decimal_odds = row['home_pred_prob'], row['home_decimal_odds']
            won = row['true_outcome'] == 1
        else:
            side, edge = 'away', away_edge
            win_prob, decimal_odds = row['away_pred_prob'], row['away_decimal_odds']
            won = row['true_outcome'] == 0

        # Only bet when we have sufficient edge
        if edge <= min_edge:
            continue

        # Proper Kelly Criterion using the side's actual decimal odds
        # (b = decimal_odds - 1), instead of assuming fair 2.0 odds.
        kelly_frac = kelly_criterion(win_prob, decimal_odds, fraction=kelly_fraction)
        kelly_stake = min(kelly_frac * kelly_bankroll, max_bet_pct * kelly_bankroll)

        if kelly_stake <= 0:
            continue

        # Payout uses the real odds too: profit on a win is
        # stake * (decimal_odds - 1), not a flat 1:1 assumption.
        kelly_profit = kelly_stake * (decimal_odds - 1) if won else -kelly_stake
        kelly_bankroll += kelly_profit

        # Flat-betting comparison: same bet selection, fixed stake size.
        flat_stake = min(flat_stake_size, max_bet_pct * flat_bankroll)
        flat_profit = flat_stake * (decimal_odds - 1) if won else -flat_stake
        flat_bankroll += flat_profit

        bets.append({
            'side': side,
            'prediction': win_prob,
            'vegas_prob': row['home_vegas_prob'] if side == 'home' else row['away_vegas_prob'],
            'decimal_odds': decimal_odds,
            'edge': edge,
            'stake': kelly_stake,
            'won': won,
            'profit': kelly_profit,
            'bankroll': kelly_bankroll,
            'flat_stake': flat_stake,
            'flat_profit': flat_profit,
            'flat_bankroll': flat_bankroll,
        })

    # Analyze results
    if not bets:
        print("⚠️  No bets were placed based on criteria.")
        print(f"   Try lowering min_edge (currently {min_edge:.1%})")
        return {
            'total_bets': 0,
            'final_bankroll': initial_bankroll,
            'roi': 0.0,
            'flat_final_bankroll': initial_bankroll,
            'flat_roi': 0.0,
        }

    bets_df = pd.DataFrame(bets)

    # Calculate key metrics
    total_bets = len(bets_df)
    winning_bets = bets_df['won'].sum()
    win_rate = winning_bets / total_bets
    total_staked = bets_df['stake'].sum()
    total_profit = bets_df['profit'].sum()
    roi = (kelly_bankroll - initial_bankroll) / initial_bankroll
    avg_profit_per_bet = bets_df['profit'].mean()

    flat_total_staked = bets_df['flat_stake'].sum()
    flat_total_profit = bets_df['flat_profit'].sum()
    flat_roi = (flat_bankroll - initial_bankroll) / initial_bankroll

    # Maximum drawdown (Kelly strategy)
    bets_df['cumulative_profit'] = bets_df['profit'].cumsum()
    bets_df['running_max'] = bets_df['cumulative_profit'].cummax()
    bets_df['drawdown'] = bets_df['running_max'] - bets_df['cumulative_profit']
    max_drawdown = bets_df['drawdown'].max()
    max_drawdown_pct = max_drawdown / initial_bankroll

    # Print summary
    print("=" * 70)
    print("BETTING RESULTS - Fractional Kelly")
    print("=" * 70)
    print(f"Total bets placed:        {total_bets:>8}  "
          f"(home: {(bets_df['side'] == 'home').sum()}, away: {(bets_df['side'] == 'away').sum()})")
    print(f"Winning bets:             {winning_bets:>8} ({win_rate:.1%})")
    print(f"Losing bets:              {total_bets - winning_bets:>8}")
    print(f"\nTotal staked:             ${total_staked:>8.2f}")
    print(f"Total profit:             ${total_profit:>8.2f}")
    print(f"Average profit/bet:       ${avg_profit_per_bet:>8.2f}")
    print(f"\nFinal bankroll:           ${kelly_bankroll:>8.2f}")
    print(f"Return on Investment:     {roi:>8.1%}")
    print(f"Max drawdown:             ${max_drawdown:>8.2f} ({max_drawdown_pct:.1%})")
    print("=" * 70)

    print("\nBETTING RESULTS - Flat betting (comparison)")
    print("-" * 70)
    print(f"Flat stake per bet:       ${flat_stake_size:>8.2f}")
    print(f"Total staked:             ${flat_total_staked:>8.2f}")
    print(f"Total profit:             ${flat_total_profit:>8.2f}")
    print(f"Final bankroll:           ${flat_bankroll:>8.2f}")
    print(f"Return on Investment:     {flat_roi:>8.1%}")
    print("-" * 70)

    # Performance by confidence level
    print("\nPerformance by Prediction Confidence (Kelly):")
    print("-" * 70)
    bets_df['confidence_bucket'] = pd.cut(
        bets_df['prediction'],
        bins=[0, 0.55, 0.60, 0.65, 0.70, 1.0],
        labels=['50-55%', '55-60%', '60-65%', '65-70%', '70%+']
    )

    bucket_stats = bets_df.groupby('confidence_bucket', observed=True).agg({
        'won': ['count', 'sum', 'mean'],
        'profit': 'sum',
        'stake': 'sum'
    })
    bucket_stats.columns = ['Bets', 'Wins', 'Win%', 'Profit', 'Staked']
    bucket_stats['ROI%'] = (bucket_stats['Profit'] / bucket_stats['Staked'] * 100).round(1)

    print(bucket_stats.to_string())
    print("-" * 70)

    # TODO: Add more analysis
    # - Streak analysis (longest winning/losing streak)
    # - Monthly performance
    # - Performance by edge size
    # - Sharpe ratio (risk-adjusted returns)

    if not has_real_odds:
        print("\n💡 IMPORTANT NOTES:")
        print("   • No odds_df given - results assume fair odds without vig")
        print("   • Real sportsbooks typically charge -110 on both sides")
        print("   • Pass odds_df (home/away moneylines) for realistic payouts")
    print("   • Always shop for best lines across multiple books")
    print("   • Track closing line value (CLV) to validate your edge")

    # Return metrics
    return {
        'total_bets': total_bets,
        'winning_bets': winning_bets,
        'win_rate': win_rate,
        'total_staked': total_staked,
        'total_profit': total_profit,
        'final_bankroll': kelly_bankroll,
        'roi': roi,
        'max_drawdown': max_drawdown,
        'max_drawdown_pct': max_drawdown_pct,
        'avg_profit_per_bet': avg_profit_per_bet,
        'performance_by_confidence': bucket_stats,
        'used_real_odds': has_real_odds,
        'flat_total_staked': flat_total_staked,
        'flat_total_profit': flat_total_profit,
        'flat_final_bankroll': flat_bankroll,
        'flat_roi': flat_roi,
    }


def calculate_expected_value(model_prob: float,
                             market_prob: float,
                             stake: float = 1.0,
                             odds: float = 2.0) -> float:
    """
    Calculate expected value of a bet.
    
    Parameters
    ----------
    model_prob : float
        Our estimated probability of winning
    market_prob : float
        Market's implied probability
    stake : float, optional
        Amount to bet (default: 1.0)
    odds : float, optional
        Decimal odds (default: 2.0 = even money)
    
    Returns
    -------
    float
        Expected value of the bet
        
    Notes
    -----
    EV = (probability of winning × amount won per bet) - 
         (probability of losing × amount lost per bet)
    
    Positive EV indicates a profitable bet in the long run.
    
    TODO: Convert American odds to decimal odds properly
    TODO: Account for push scenarios (ties) in some bet types
    """
    amount_won = stake * (odds - 1)
    amount_lost = stake
    
    ev = (model_prob * amount_won) - ((1 - model_prob) * amount_lost)
    
    return ev


def kelly_criterion(win_prob: float,
                    odds: float,
                    fraction: float = 1.0) -> float:
    """
    Calculate Kelly Criterion bet size.
    
    Parameters
    ----------
    win_prob : float
        Probability of winning the bet (0 to 1)
    odds : float
        Decimal odds received on win
    fraction : float, optional
        Fraction of Kelly to bet (default: 1.0 = full Kelly)
        Use 0.25-0.5 for fractional Kelly (more conservative)
    
    Returns
    -------
    float
        Fraction of bankroll to bet (0 to 1)
        
    Notes
    -----
    Kelly formula: f = (bp - q) / b
    where:
        f = fraction of bankroll to wager
        b = odds received (decimal odds - 1)
        p = probability of winning
        q = 1 - p (probability of losing)
    
    Kelly Criterion maximizes long-term growth rate but can be
    aggressive. Fractional Kelly (0.25 to 0.5) is recommended.
    
    TODO: Implement for American odds format
    TODO: Add safety checks (negative Kelly = don't bet)
    TODO: Consider Kelly for multiple simultaneous bets
    """
    b = odds - 1  # Net odds received
    p = win_prob
    q = 1 - p
    
    # Kelly formula
    kelly = (b * p - q) / b
    
    # Apply fractional Kelly
    kelly_fractional = kelly * fraction
    
    # Never bet more than 100% of bankroll
    kelly_fractional = max(0, min(kelly_fractional, 1.0))
    
    return kelly_fractional
