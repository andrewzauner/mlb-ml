import os
import numpy as np
import pandas as pd

import data_loader


# A real row from a Retrosheet game log (2018-03-29 COL @ ARI), used to
# pin down the column mapping against ground truth.
SAMPLE_ROW = (
    '"20180329","0","Thu","COL","NL",1,"ARI","NL",1,2,8,51,"N","","","","PHO01",'
    '48703,216,"100001000","30000320x",'
    '33,9,0,0,2,2,1,0,0,2,0,12,0,0,2,0,7,5,8,8,0,0,24,8,0,0,0,0,'
    '36,12,2,1,0,8,0,0,0,6,0,11,2,0,0,0,10,6,2,2,1,0,27,10,0,0,2,0,'
    '"cedeg901","Gary Cederstrom","coope901","Eric Cooper","blasc901","Cory Blaser",'
    '"sches901","Stu Scheurwater","","(none)","","(none)",'
    '"blacb001","Buddy Black","lovut001","Tony Lovullo",'
    '"corbp001","Patrick Corbin","grayj003","Jon Gray","","(none)","lambj001","Jake Lamb"'
)


def write_gamelog(path, rows):
    with open(path, 'w') as f:
        f.write('\n'.join(rows) + '\n')


def test_retrosheet_columns_batting_stats_match_known_row(tmp_path):
    # Regression test for a real bug: the batting-stat column indices were
    # off (home fields +1, visiting fields -4), which zeroed out home_AB
    # for every game. Verified against the raw file: visiting AB=33,
    # H=9; home AB=36, H=12, 2B=2, 3B=1, HR=0.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_gamelog(data_dir / "GL2018.TXT", [SAMPLE_ROW])

    df = data_loader.load_retrosheet_data([2018], str(data_dir))

    assert len(df) == 1
    row = df.iloc[0]
    assert row['visiting_AB'] == 33
    assert row['visiting_H'] == 9
    assert row['visiting_2B'] == 0
    assert row['visiting_3B'] == 0
    assert row['visiting_HR'] == 2
    assert row['home_AB'] == 36
    assert row['home_H'] == 12
    assert row['home_2B'] == 2
    assert row['home_3B'] == 1
    assert row['home_HR'] == 0
    assert row['home_team'] == 'ARI'
    assert row['visiting_team'] == 'COL'
    assert row['home_score'] == 8
    assert row['visiting_score'] == 2


def test_load_retrosheet_data_is_case_insensitive_to_filename(tmp_path):
    # Regression test: files are commonly named gl2018.txt (lowercase),
    # but the loader used to only look for GL2018.TXT.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    write_gamelog(data_dir / "gl2018.txt", [SAMPLE_ROW])

    df = data_loader.load_retrosheet_data([2018], str(data_dir))

    assert df is not None
    assert len(df) == 1


def test_load_retrosheet_data_missing_file_returns_none(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    df = data_loader.load_retrosheet_data([2099], str(data_dir))

    assert df is None


def test_download_odds_data_covers_every_real_game():
    # Regression test: the placeholder odds generator used to invent its
    # own random schedule, which almost never matched real games. When
    # game_data is supplied, odds should be generated 1:1 for every game.
    game_data = pd.DataFrame({
        'date': ['20180401', '20180401', '20180402'],
        'home_team': ['NYA', 'BOS', 'NYA'],
        'visiting_team': ['BOS', 'NYA', 'BOS'],
        'season': [2018, 2018, 2018],
        'game_id': [0, 1, 2],
    })

    odds = data_loader.download_odds_data([2018], game_data=game_data)

    assert len(odds) == len(game_data)
    assert set(odds['game_id']) == set(game_data['game_id'])
    # Moneylines should be plausible American odds (not zero/NaN).
    assert (odds['home_moneyline'].abs() >= 100).all()
    assert (odds['away_moneyline'].abs() >= 100).all()


def test_merge_handles_doubleheaders_without_row_fanout():
    # Regression test: (date, home_team, visiting_team) is not a unique
    # key - a doubleheader has two distinct games sharing it. Merging on
    # game_id must produce an exact 1:1 join instead of a many-to-many
    # fan-out that duplicates rows.
    game_data = pd.DataFrame({
        'date': ['20180401', '20180401'],
        'date_str': ['20180401', '20180401'],
        'home_team': ['DET', 'DET'],
        'visiting_team': ['PIT', 'PIT'],
        'home_score': [6, 0],
        'visiting_score': [8, 1],
        'season': [2018, 2018],
        'game_id': [0, 1],
    })
    odds_data = pd.DataFrame({
        'date': ['20180401', '20180401'],
        'home_team': ['DET', 'DET'],
        'visiting_team': ['PIT', 'PIT'],
        'home_moneyline': [-120, -110],
        'away_moneyline': [110, 100],
        'game_id': [0, 1],
    })

    merged = data_loader.merge_game_and_odds_data(game_data, odds_data)

    assert len(merged) == 2
    # Each game should be paired with its own odds, not the other game's.
    row0 = merged[merged['game_id'] == 0].iloc[0]
    row1 = merged[merged['game_id'] == 1].iloc[0]
    assert row0['home_moneyline'] == -120
    assert row1['home_moneyline'] == -110
