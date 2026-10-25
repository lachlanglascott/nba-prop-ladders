import math

import polars as pl
import pytest

from nba_ladders.absences import MAX_SHIFT, context_effects


def _setup(eff_min=None):
    pg = pl.DataFrame({"player_id": [1], "game_id": ["g1"], "team_id": [10]})
    pos = pl.DataFrame({"player_id": [1, 2, 3], "position": ["C", "C", "G"]})
    pri = pl.DataFrame({
        "grp": ["B", "B"], "mate_grp": ["B", "G"],
        "b_min": [0.1, 0.05], "b_pts": [0.0, 0.0], "b_reb": [0.0, 0.0], "b_ast": [0.0, 0.0],
    })
    eff = pl.DataFrame(
        {"player_id": [1], "mate_id": [2], "n_out": [10], "mate_min": [30.0],
         "eff_min": [eff_min], "eff_pts": [0.0], "eff_reb": [0.0], "eff_ast": [0.0]},
        schema_overrides={"eff_min": pl.Float64},
    )
    return pg, pos, pri, eff


def test_pair_effect_used_and_prior_fallback():
    pg, pos, pri, eff = _setup(eff_min=10.0)
    absent = pl.DataFrame({"game_id": ["g1", "g1"], "team_id": [10, 10], "mate_id": [2, 3],
                           "mate_min": [30.0, 20.0]})
    out = context_effects(pg, absent, eff, pri, pos, scale=1.0).row(0, named=True)
    # pair effect 10 for the big + prior 0.05 * 20 = 1 for the guard (no pair history)
    expected = MAX_SHIFT * math.tanh(11 / MAX_SHIFT)
    assert out["shift_min"] == pytest.approx(expected)
    assert out["n_absent"] == 2 and out["freed_min"] == 50.0


def test_negative_minutes_effects_are_floored_and_scale_applies():
    pg, pos, pri, eff = _setup(eff_min=-6.0)
    absent = pl.DataFrame({"game_id": ["g1"], "team_id": [10], "mate_id": [2], "mate_min": [30.0]})
    assert context_effects(pg, absent, eff, pri, pos, scale=1.0)["shift_min"][0] == 0.0
    pg, pos, pri, eff = _setup(eff_min=10.0)
    small = context_effects(pg, absent, eff, pri, pos, scale=0.4)["shift_min"][0]
    assert small == pytest.approx(MAX_SHIFT * math.tanh(4 / MAX_SHIFT))


def test_no_absences_means_no_shift():
    pg, pos, pri, eff = _setup(eff_min=10.0)
    absent = pl.DataFrame(schema={"game_id": pl.String, "team_id": pl.Int64, "mate_id": pl.Int64,
                                  "mate_min": pl.Float64})
    out = context_effects(pg, absent, eff, pri, pos).row(0, named=True)
    assert out["shift_min"] == 0.0 and out["mult_pts"] == 1.0
