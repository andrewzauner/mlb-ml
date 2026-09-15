"""
Data loading module for MLB prediction model.

This module handles downloading Retrosheet game logs, loading historical game data,
generating/downloading odds data, and merging game and odds datasets.

TODO: Replace placeholder odds generation with real historical odds API integration
TODO: Add support for additional data sources (weather, injuries)
"""

import pandas as pd
import numpy as np
import requests
import os
import zipfile
import io
from typing import List, Optional


# Retrosheet game log column mapping based on official documentation
# Source: https://www.retrosheet.org/gamelogs/glfields.txt
# Verified against raw gamelog rows: column 21 is visiting AB (a value in the
# high 20s/30s), column 49 is home AB. The previous mapping here was off by
# -4 on the visiting side and +1 on the home side, which silently zeroed out
# home_AB for every game (AB fell on a column that's always blank) and
# pulled the wrong stat entirely for the other fields.
RETROSHEET_COLUMNS = {
    0: 'date',
    3: 'visiting_team',
    6: 'home_team',
    9: 'visiting_score',
    10: 'home_score',
    12: 'day_night',
    # Box score stats for visiting team
    21: 'visiting_AB',  # At-bats
    22: 'visiting_H',   # Hits
    23: 'visiting_2B',  # Doubles
    24: 'visiting_3B',  # Triples
    25: 'visiting_HR',  # Home runs
    # Box score stats for home team
    49: 'home_AB',  # At-bats
    50: 'home_H',   # Hits
    51: 'home_2B',  # Doubles
    52: 'home_3B',  # Triples
    53: 'home_HR',  # Home runs
    # Team earned runs allowed (i.e. earned runs charged against that
    # team's pitching staff - so "visiting_team_earned_runs" describes
    # visiting PITCHING quality, not visiting offense). Verified across a
    # full season: never exceeds the opposing team's score (earned <=
    # total runs), with a small gap from occasional unearned runs - e.g.
    # mean visiting_team_earned_runs is ~0.35 runs below home_score.
    40: 'visiting_team_earned_runs',
    68: 'home_team_earned_runs',
    # Starting pitcher Retrosheet player IDs. Verified against a real game
    # (Rockies @ Diamondbacks, 2018-03-29): column 101 held "grayj003"
    # (Jon Gray, Colorado's actual starter that day) and column 103 held
    # "corbp001" (Patrick Corbin, Arizona's), consistent with the
    # winning/losing pitcher fields and both teams' batting lineups
    # (pitchers batting 9th under NL rules at the time).
    101: 'visiting_starting_pitcher_id',
    103: 'home_starting_pitcher_id',
}


def download_retrosheet_data(years: List[int], data_dir: str = './data') -> None:
    """
    Download game data from Retrosheet for specified years.
    
    Parameters
    ----------
    years : list of int
        Years to download data for (e.g., [2018, 2019, 2020])
    data_dir : str, optional
        Directory to store downloaded data
        
    Notes
    -----
    Retrosheet provides complete game logs in a standardized format.
    Files are downloaded as .zip and extracted automatically.
    """
    print(f"Downloading Retrosheet data for years: {years}")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    
    for year in years:
        url = f"https://www.retrosheet.org/gamelogs/gl{year}.zip"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                z = zipfile.ZipFile(io.BytesIO(response.content))
                z.extractall(data_dir)
                print(f"Successfully downloaded and extracted data for {year}")
            else:
                print(f"Failed to download data for {year}: Status code {response.status_code}")
        except Exception as e:
            print(f"Error downloading data for {year}: {e}")


def _find_game_log_file(year: int, data_dir: str) -> Optional[str]:
    """
    Locate the game log file for a given year, tolerating filename case.

    Retrosheet zip archives extract to uppercase names (GL2018.TXT), but
    files placed in the data directory by other means (e.g. manual
    download, this repo's bundled sample data) commonly use lowercase
    (gl2018.txt). Both are checked so loading doesn't silently fail.
    """
    candidates = [f"GL{year}.TXT", f"gl{year}.txt"]
    for candidate in candidates:
        candidate_path = os.path.join(data_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path
    return None


def load_retrosheet_data(years: List[int], data_dir: str = './data') -> Optional[pd.DataFrame]:
    """
    Load and parse Retrosheet game logs.
    
    Parameters
    ----------
    years : list of int
        Years to load data for
    data_dir : str, optional
        Directory containing downloaded data files
    
    Returns
    -------
    pd.DataFrame or None
        DataFrame containing game data with properly mapped columns,
        or None if no data could be loaded
        
    Notes
    -----
    The function maps Retrosheet columns to meaningful names based on the
    official field specification. This fixes the KeyError issue by ensuring
    batting statistics columns (home_H, home_AB, etc.) are properly created.
    
    TODO: Add validation to check for corrupt or incomplete game log files
    TODO: Consider caching parsed data to avoid re-parsing on subsequent runs
    """
    all_data = []

    for year in years:
        file_path = _find_game_log_file(year, data_dir)
        if file_path is not None:
            try:
                # Load data without predefined column names
                year_data = pd.read_csv(file_path, header=None, sep=',', quotechar='"')
                
                # Create generic column names first
                num_columns = year_data.shape[1]
                column_names = [f'col_{i}' for i in range(num_columns)]
                year_data.columns = column_names
                
                # Map known columns based on Retrosheet specification
                rename_map = {f'col_{idx}': name for idx, name in RETROSHEET_COLUMNS.items()}
                year_data.rename(columns=rename_map, inplace=True)
                
                # Add year column for tracking
                year_data['season'] = year
                
                # Ensure batting stat columns exist (defensive programming)
                stat_columns = ['home_H', 'home_AB', 'visiting_H', 'visiting_AB',
                               'home_2B', 'home_3B', 'home_HR',
                               'visiting_2B', 'visiting_3B', 'visiting_HR',
                               'home_team_earned_runs', 'visiting_team_earned_runs']

                for col in stat_columns:
                    if col not in year_data.columns:
                        print(f"Warning: {col} not found for {year}, setting to 0")
                        year_data[col] = 0

                # Starting pitcher IDs are strings (Retrosheet player codes,
                # e.g. "grayj003"), not numeric - just ensure the columns
                # exist so downstream code can check for them uniformly.
                for col in ['home_starting_pitcher_id', 'visiting_starting_pitcher_id']:
                    if col not in year_data.columns:
                        print(f"Warning: {col} not found for {year}, setting to empty")
                        year_data[col] = ''

                # Convert numeric columns to appropriate types
                numeric_cols = ['visiting_score', 'home_score', 'home_H', 'home_AB',
                               'visiting_H', 'visiting_AB', 'home_2B', 'home_3B',
                               'home_HR', 'visiting_2B', 'visiting_3B', 'visiting_HR',
                               'home_team_earned_runs', 'visiting_team_earned_runs']

                for col in numeric_cols:
                    if col in year_data.columns:
                        year_data[col] = pd.to_numeric(year_data[col], errors='coerce').fillna(0)
                
                all_data.append(year_data)
                print(f"Loaded {len(year_data)} games from {year}")
                
            except Exception as e:
                print(f"Error loading data for {year}: {e}")
        else:
            print(f"File not found for {year}: no GL{year}.TXT or gl{year}.txt in {data_dir}")

    if all_data:
        game_data = pd.concat(all_data, ignore_index=True)
        # Stable unique ID per game. (date, home_team, visiting_team) is NOT
        # a unique key - doubleheaders put two games under the same triple,
        # which previously caused merge_game_and_odds_data() to join each
        # such game against every odds row sharing that key (a many-to-many
        # fan-out that duplicated rows and could pair a game with the wrong
        # game's odds).
        game_data['game_id'] = range(len(game_data))
        print(f"Total games loaded: {len(game_data)}")
        return game_data
    else:
        print("No data was loaded.")
        return None


def download_odds_data(years: List[int],
                       game_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Generate placeholder odds data (to be replaced with real API integration).

    Parameters
    ----------
    years : list of int
        Years to generate odds data for
    game_data : pd.DataFrame, optional
        Actual loaded game data (from load_retrosheet_data). When provided,
        one synthetic odds line is generated per real game, so the odds
        "market" lines up with the real schedule instead of a randomly
        generated set of matchups that almost never land on a real game
        date. When omitted, falls back to generating a synthetic schedule
        from scratch (kept for standalone/backwards-compatible use).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: date, home_team, visiting_team,
        home_moneyline, away_moneyline

    Notes
    -----
    This is a PLACEHOLDER function that generates synthetic odds data.

    TODO: Replace with actual historical odds data from one of these sources:
        - The Odds API (https://the-odds-api.com/) - provides historical odds
        - SportsOddsHistory.com - historical odds database
        - OddsPortal scraping (check terms of service)
        - Purchase historical data from a sports data provider

    OVERSIMPLIFICATION WARNING:
        The current random odds generation:
        - Does not reflect actual market conditions
        - Ignores home field advantage patterns
        - Doesn't account for pitcher matchups
        - Has no correlation with actual team strength

    For production use, this MUST be replaced with real historical odds.

    BUG FIX (previous version): When game_data wasn't wired in, odds were
    generated for a completely independent, randomly sampled schedule
    (random dates, random team pairings). Because that schedule almost
    never coincided with the real one, merge_game_and_odds_data() kept
    well under 1% of games - starving the model of training data. Keying
    odds generation off the real schedule keeps the "placeholder" caveat
    but restores full coverage.
    """
    print("Note: Generating placeholder odds data.")
    print("TODO: Replace with actual historical odds API integration")

    def _synthetic_moneylines(n: int):
        is_home_favorite = np.random.random(n) > 0.4
        home_line = np.where(
            is_home_favorite,
            -np.random.randint(110, 220, size=n),
            np.random.randint(100, 200, size=n)
        )
        away_line = np.where(
            is_home_favorite,
            np.random.randint(100, 200, size=n),
            -np.random.randint(110, 220, size=n)
        )
        return home_line, away_line

    if game_data is not None and len(game_data) > 0:
        # Generate one odds line per actual game, so odds coverage matches
        # the real schedule instead of a disjoint random one.
        relevant = game_data[game_data['season'].isin(years)] if 'season' in game_data.columns else game_data
        home_line, away_line = _synthetic_moneylines(len(relevant))

        odds_data = pd.DataFrame({
            'date': relevant['date'].astype(str).values,
            'home_team': relevant['home_team'].values,
            'visiting_team': relevant['visiting_team'].values,
            'home_moneyline': home_line,
            'away_moneyline': away_line
        })
        # Carry the unique game_id through when available so
        # merge_game_and_odds_data can do an exact 1:1 join instead of
        # matching on (date, home_team, visiting_team), which collides on
        # doubleheaders.
        if 'game_id' in relevant.columns:
            odds_data['game_id'] = relevant['game_id'].values
    else:
        # Fallback: no real schedule available, synthesize one from scratch.
        dates = []
        home_teams = []
        away_teams = []

        # MLB team abbreviations (Retrosheet format)
        teams = ['NYA', 'BOS', 'TOR', 'BAL', 'TBA',
                 'CHA', 'CLE', 'DET', 'KCA', 'MIN',
                 'HOU', 'LAA', 'OAK', 'SEA', 'TEX',
                 'ATL', 'MIA', 'NYN', 'PHI', 'WAS',
                 'CHN', 'CIN', 'MIL', 'PIT', 'SLN',
                 'ARI', 'COL', 'LAN', 'SDN', 'SFN']

        for year in years:
            for month in range(4, 11):  # Baseball season (April-October)
                for day in range(1, 28):
                    if np.random.random() < 0.3:  # Not every day has games
                        continue

                    # Generate 8 random games for this date
                    for _ in range(8):
                        date_str = f"{year}{month:02d}{day:02d}"
                        game_teams = np.random.choice(teams, 2, replace=False)
                        dates.append(date_str)
                        home_teams.append(game_teams[0])
                        away_teams.append(game_teams[1])

        home_line, away_line = _synthetic_moneylines(len(dates))
        odds_data = pd.DataFrame({
            'date': dates,
            'home_team': home_teams,
            'visiting_team': away_teams,
            'home_moneyline': home_line,
            'away_moneyline': away_line
        })

    print(f"Created placeholder odds data with {len(odds_data)} entries")
    return odds_data


def merge_game_and_odds_data(game_data: pd.DataFrame, 
                             odds_data: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Merge game data with odds data on date and team matchup.
    
    Parameters
    ----------
    game_data : pd.DataFrame
        Game results data from Retrosheet
    odds_data : pd.DataFrame
        Betting odds data
    
    Returns
    -------
    pd.DataFrame or None
        Merged dataset, or None if inputs are invalid
        
    Notes
    -----
    Uses inner join to ensure we only keep games where both actual results
    and odds are available. This is important for training since we need
    both features (odds) and labels (results).
    
    TODO: Add logic to handle multiple odds from different sportsbooks
    TODO: Consider keeping games without odds for evaluation purposes

    BUG FIX: (date, home_team, visiting_team) is not a unique key -
    doubleheaders put two distinct games under the same triple. Merging on
    it alone silently fans out: each such game matches every odds row
    sharing that key, duplicating rows and, with a real odds feed (multiple
    books/lines per game), pairing games with the wrong game's odds. When
    both frames carry a game_id (set by load_retrosheet_data /
    download_odds_data), merge on that instead for an exact 1:1 join.
    """
    if game_data is None or odds_data is None:
        print("Cannot merge: game_data or odds_data is None")
        return None

    # Convert date formats to match
    game_data['date_str'] = game_data['date'].astype(str)

    if 'game_id' in game_data.columns and 'game_id' in odds_data.columns:
        merged_data = pd.merge(game_data, odds_data, on='game_id', how='inner',
                               suffixes=('', '_odds'))
    else:
        # Fallback for odds data without a game_id (e.g. a real odds feed
        # keyed only by date/teams). Can still fan out on doubleheaders.
        merged_data = pd.merge(
            game_data,
            odds_data,
            left_on=['date_str', 'home_team', 'visiting_team'],
            right_on=['date', 'home_team', 'visiting_team'],
            how='inner'
        )

    print(f"Merged data has {len(merged_data)} rows")
    print(f"Match rate: {len(merged_data) / len(game_data) * 100:.1f}% of games have odds")

    return merged_data
