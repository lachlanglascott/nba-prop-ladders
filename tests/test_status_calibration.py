from datetime import date

import polars as pl

from nba_ladders.status_calibration import calibrate, outcomes


def test_outcomes_match_listed_players_to_box_scores():
    d = date(2026, 3, 15)
    report = pl.DataFrame({
        "game_date": [d, d, d, d],
        "team_name": ["BostonCeltics"] * 4,
        "player": ["Queta,Neemias", "Garza,Luka", "White,Derrick", None],
        "player_key": ["neemiasqueta", "garzaluka", "derrickwhite", None],
        "status": ["Questionable", "Questionable", "Out", "Not submitted"],
        "reason": ["Injury/Illness-Ankle", "Injury/Illness-Knee", "Rest", None],
    })
    box = pl.DataFrame({"team": ["BOS", "BOS"], "player_name": ["Neemias Queta", "Jaylen Brown"],
                        "minutes": [24, 35]})
    o = outcomes(report, box, d)
    assert o.height == 3  # the not-submitted row is dropped
    played = dict(zip(o["player"], o["played"], strict=True))
    assert played == {"Queta,Neemias": True, "Garza,Luka": False, "White,Derrick": False}
    cal = calibrate(o).filter(pl.col("status") == "Questionable").row(0, named=True)
    assert cal["listed"] == 2 and cal["p_sits"] == 0.5
