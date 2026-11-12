from datetime import date, timedelta

import polars as pl

from nba_ladders.usage import absences


def _games(season: str, start: date, n: int, team: str = "AAA") -> list[dict]:
    rows = []
    for k in range(n):
        gid, d = f"{season}-{k}", start + timedelta(days=2 * k)
        for pid, mins in ((1, 34), (2, 30), (3, 24)):
            if pid == 1 and k == n - 1:
                continue  # star sits the last game
            if pid == 2 and season == "22026":
                continue  # player 2 left in the offseason: never plays this season
            rows.append({"season_id": season, "team": team, "game_id": gid, "game_date": d, "player_id": pid,
                         "minutes": mins, "u": 20.0 + pid, "ast": 5, "reb": 6, "grp": "G"})
    return rows


def test_absence_detected_within_season_only():
    pg = pl.DataFrame(_games("22025", date(2026, 1, 1), 8) + _games("22026", date(2026, 10, 20), 8))
    ab = absences(pg)
    # star (1) missing the last game of each season is an absence, with his usual volume
    star = ab.filter(pl.col("mate_id") == 1)
    assert sorted(star["game_id"].to_list()) == ["22025-7", "22026-7"]
    assert abs(star["vol_u"][0] - 21.0) < 1e-9
    # player 2 left: not "absent" in the new season's games (window doesn't cross seasons)
    assert ab.filter(pl.col("mate_id") == 2, pl.col("game_id").str.starts_with("22026")).is_empty()
