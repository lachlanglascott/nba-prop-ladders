from datetime import date

import polars as pl

from nba_ladders.sheet import depth_chart, is_new_absence


def _pg(rows):
    return pl.DataFrame(rows, schema=["player_id", "player_name", "team", "game_date", "minutes",
                                      "reb"], orient="row")


def test_depth_chart_picks_centre_over_stretch_forward():
    d = date(2026, 10, 20)
    pg = _pg([
        (1, "Stretch Four", "AAA", d, 34, 5),   # F-C, rebounds like a forward
        (2, "Real Centre", "AAA", d, 28, 11),   # C
        (3, "Backup Big", "AAA", d, 14, 6),     # C
    ])
    roster = pl.DataFrame({"player_id": [1, 2, 3], "team_now": ["AAA"] * 3})
    pos = pl.DataFrame({"player_id": [1, 2, 3], "position": ["F-C", "C", "C"]})
    dc = depth_chart(pg, roster, pos)
    assert dc.filter(pl.col("is_starter"))["player_name"].to_list() == ["Real Centre"]


def test_is_new_absence():
    pg = _pg([
        (1, "Starter", "AAA", date(2026, 10, 20), 30, 10),
        (2, "Mate", "AAA", date(2026, 10, 20), 20, 4),
        (2, "Mate", "AAA", date(2026, 10, 22), 20, 4),  # starter missed the 22nd
    ])
    assert is_new_absence(pg, 1, "AAA", date(2026, 10, 22)) is True   # played last game (20th)
    assert is_new_absence(pg, 1, "AAA", date(2026, 10, 24)) is False  # already missed the 22nd


def test_long_term_injured_centre_is_not_the_starter():
    old, now = date(2026, 3, 1), date(2026, 10, 28)
    pg = _pg([
        (1, "Injured Starter", "AAA", old, 32, 11),   # hasn't played since March
        (2, "Current Starter", "AAA", now, 26, 10),
    ])
    roster = pl.DataFrame({"player_id": [1, 2], "team_now": ["AAA"] * 2})
    pos = pl.DataFrame({"player_id": [1, 2], "position": ["C", "C"]})
    dc = depth_chart(pg, roster, pos)
    assert dc.filter(pl.col("is_starter"))["player_name"].to_list() == ["Current Starter"]
