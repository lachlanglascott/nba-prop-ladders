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
