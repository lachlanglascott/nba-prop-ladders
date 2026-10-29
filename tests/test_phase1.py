import json
from datetime import date

import polars as pl
import pytest

from nba_ladders.player_games import COLUMNS, parse_league_gamelog
from nba_ladders.sources.nba_stats import get_league_gamelog
from nba_ladders.with_without import with_without

HEADERS = list(COLUMNS) + ["WL"]


def _row(player_id, name, game_id, game_date, minutes, pts=0, reb=0, matchup="AAA vs. BBB"):
    vals = {
        "SEASON_ID": "22025", "PLAYER_ID": player_id, "PLAYER_NAME": name, "TEAM_ID": 1,
        "TEAM_ABBREVIATION": "AAA", "GAME_ID": game_id, "GAME_DATE": game_date,
        "MATCHUP": matchup, "MIN": minutes, "PTS": pts, "OREB": 0, "DREB": reb, "REB": reb,
        "AST": 0, "FGM": 0, "FGA": 0, "FG3M": 0, "FG3A": 0, "FTM": 0, "FTA": 0, "PF": 0,
        "PLUS_MINUS": 0, "WL": "W",
    }
    return [vals[h] for h in HEADERS]


def _payload(rows):
    return {"resultSets": [{"name": "LeagueGameLog", "headers": HEADERS, "rowSet": rows}]}


def test_parse_league_gamelog():
    df = parse_league_gamelog(
        _payload([_row(7, "A", "g1", "2025-10-21", 30, matchup="AAA @ BBB")])
    )
    row = df.row(0, named=True)
    assert row["game_date"] == date(2025, 10, 21)
    assert row["is_home"] is False
    assert row["opponent"] == "BBB"
    assert row["minutes"] == 30


class FakeFetch:
    def __init__(self):
        self.calls = 0

    def __call__(self, season, season_type):
        self.calls += 1
        return {"call": self.calls}


def test_cache_reuses_snapshot_and_never_overwrites(tmp_path):
    fetch = FakeFetch()
    d1, d2 = date(2026, 10, 20), date(2026, 10, 21)
    first = get_league_gamelog("2025-26", raw_dir=tmp_path, fetch=fetch, today=d1)
    again = get_league_gamelog("2025-26", raw_dir=tmp_path, fetch=fetch, today=d2)
    assert first == again == {"call": 1}

    # refresh on the same day reuses that day's file; a new day adds a new file
    get_league_gamelog("2025-26", refresh=True, raw_dir=tmp_path, fetch=fetch, today=d1)
    assert fetch.calls == 1
    newer = get_league_gamelog("2025-26", refresh=True, raw_dir=tmp_path, fetch=fetch, today=d2)
    assert newer == {"call": 2}
    files = sorted(p.name for p in tmp_path.iterdir())
    assert len(files) == 2
    assert json.loads((tmp_path / files[0]).read_text()) == {"call": 1}


@pytest.fixture
def team():
    # S = starter (misses g3), B = backup, L = leaves the team after g2 (traded).
    rows = []
    dates = ["2025-10-21", "2025-10-23", "2025-10-25", "2025-10-27", "2025-10-29"]
    for i, d in enumerate(dates, start=1):
        g = f"g{i}"
        starter_out = g == "g3"
        if not starter_out:
            rows.append(_row(1, "Starter", g, d, 32, pts=14, reb=10))
        rows.append(_row(2, "Backup", g, d, 30 if starter_out else 10, pts=16 if starter_out else 4,
                         reb=12 if starter_out else 3))
        if i <= 2:
            rows.append(_row(3, "Leaver", g, d, 20))
    return parse_league_gamelog(_payload(rows))


def test_with_without_detects_backup_uplift(team):
    ww = with_without(team, min_games_out=1)
    row = ww.filter((pl.col("player_id") == 2) & (pl.col("mate_id") == 1)).row(0, named=True)
    assert row["games"] == 4 and row["games_mate_out"] == 1
    assert row["minutes"] == 10 and row["minutes_mate_out"] == 30
    assert row["minutes_uplift"] == 20
    assert row["mate_name"] == "Starter"


def test_traded_player_not_counted_as_absent(team):
    ww = with_without(team, min_games_out=1)
    assert ww.filter(pl.col("mate_id") == 3).is_empty()


def test_player_profiles_split_by_big_minutes(team):
    from nba_ladders.screens import player_profiles

    backup = player_profiles(team).filter(pl.col("player_id") == 2).row(0, named=True)
    # Backup: 1 game at 30 min with 16p/12r (a 10/10), 4 games at 10 min with 4p/3r.
    assert backup["gp"] == 5 and backup["n_big"] == 1
    assert backup["p_min25"] == pytest.approx(0.2)
    assert backup["p_10_10"] == pytest.approx(0.2)
    assert backup["p_10_10_big"] == 1.0


def test_backup_big_games_flags_starter_out(team):
    from nba_ladders.scenarios import backup_big_games, rung_rates

    # Starter (id 1) and Backup (id 2) both rebound like centres; Leaver is a guard.
    pos = pl.DataFrame({"player_id": [1, 2, 3], "position": ["C", "C", "G"]})
    bg = backup_big_games(team, pos, min_gp=1)
    assert set(bg["player_id"]) == {2}
    assert bg.filter(pl.col("starter_out"))["game_id"].to_list() == ["g3"]
    out = rung_rates(bg, ["starter_out"]).filter(pl.col("starter_out")).row(0, named=True)
    assert out["n"] == 1 and out["p10_10"] == 1.0 and out["p12_12"] == 1.0
