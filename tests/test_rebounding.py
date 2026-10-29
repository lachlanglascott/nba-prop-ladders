from datetime import date, timedelta

import polars as pl
import pytest

from nba_ladders.rebounding import K_OPP, expected_rebounds, opponent_factors


def test_expected_rebounds_use_only_prior_games_and_shrink_to_position():
    d0 = date(2025, 1, 1)
    pg = pl.DataFrame({
        "player_id": [1, 1, 1], "game_date": [d0 + timedelta(days=i) for i in range(3)],
        "minutes": [30, 30, 30], "oreb": [0, 0, 9], "dreb": [6, 6, 6], "opponent": ["X", "Y", "Z"],
    })
    pos = pl.DataFrame({"player_id": [1], "position": ["C"]})
    e = expected_rebounds(pg, pos)
    # game 3's own 9 offensive boards must not feed its own expectation
    assert e["exp_oreb"][2] < 3
    assert e["prior_min"].to_list() == [0, 30, 60]


def test_opponent_factor_shrinks_toward_one():
    df = pl.DataFrame({"opponent": ["X"], "oreb": [0], "dreb": [120], "exp_oreb": [0.0],
                       "exp_dreb": [100.0]})
    f = opponent_factors(df, []).row(0, named=True)
    assert f["f_dreb"] == pytest.approx((120 + K_OPP) / (100 + K_OPP))
    assert 1 < f["f_dreb"] < 1.2 and f["f_oreb"] == 1.0
