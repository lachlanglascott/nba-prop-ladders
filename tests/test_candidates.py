import numpy as np

from nba_ladders.blend import predict
from nba_ladders.candidates import _opp


def test_opp():
    assert _opp("CHA@WAS", "CHA") == "WAS" and _opp("CHA@WAS", "WAS") == "CHA"


def test_blend_moves_toward_model_but_stays_anchored_to_book():
    w = [0.0, 1.0, 0.2] + [0.0] * 13  # book weight 1, model weight 0.2, no stat/angle terms
    lo = predict([0.3], [0.2], ["pts"], [False], [False], w)[0]
    hi = predict([0.3], [0.5], ["pts"], [False], [False], w)[0]
    assert lo < hi and 0.2 < lo < 0.5 and 0.2 < hi < 0.5
    assert np.isclose(predict([0.3], [0.3], ["pts"], [False], [False], [0.0, 1.0, 0.0] + [0.0] * 13)[0], 0.3)


def test_adjust_history_backs_out_absence_minutes_and_rates():
    import polars as pl

    from nba_ladders.context_adjust import adjust_history

    hist = pl.DataFrame({"game_id": ["a", "b"], "minutes": [20, 34], "pts": [10, 20],
                         "reb": [4, 6], "ast": [2, 8]})
    adj = pl.DataFrame({"game_id": ["b"], "shift_min": [10.0], "mult_pts": [1.0], "mult_reb": [1.0],
                        "mult_ast": [2.0]})
    out = adjust_history(hist, adj).sort("game_id")
    assert out["minutes"].to_list() == [20, 24]                  # game b: 34 - 10
    assert abs(out["ast"][1] - 8 / 2 * 24 / 34) < 1e-9           # per-minute rate halved
    assert out["ast"][0] == 2                                     # untouched game


def test_every_angle_has_focus_shapes_evidence_and_label():
    from nba_ladders.candidates import ANGLE_LABEL, EVIDENCE, FOCUS, SHAPES

    assert set(ANGLE_LABEL) == set(EVIDENCE) == set(FOCUS) == set(SHAPES)


def test_pot_miss(monkeypatch):
    from datetime import date

    from nba_ladders import candidates as C

    monkeypatch.setattr(C, "_career_conv", lambda pid, stat: 0.55)
    g = {"same": True, "potential_ast": 16, "ast": 4, "opponent": "BOS", "game_date": date(2026, 10, 29)}
    assert C.pot_miss(g, 1)["short"] == 16 * 0.55 - 4
    assert C.pot_miss({**g, "same": False}, 1) is None
    assert C.pot_miss({**g, "potential_ast": 11}, 1) is None
    assert C.pot_miss({**g, "ast": 6}, 1) is None  # 8.8 expected, only 2.8 short
    assert set(C.ANGLE_LABEL) == set(C.FOCUS) == set(C.SHAPES) == set(C.EVIDENCE)
