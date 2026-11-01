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


def test_pot_volume(monkeypatch):
    from datetime import date

    from nba_ladders import candidates as C

    monkeypatch.setattr(C, "_career_conv", lambda pid, stat: 0.55)
    monkeypatch.setattr(C, "pot_baseline", lambda pid, before: 9.0)
    g = {"same": True, "potential_ast": 22, "ast": 14, "opponent": "NOP", "game_date": date(2026, 10, 29)}
    conv = C.pot_volume(g, 1)
    assert conv["short"] < 0 and "converted" in conv["detail"]  # the Acuff case
    assert C.pot_volume({**g, "ast": 6}, 1)["short"] > 0
    assert C.pot_volume({**g, "potential_ast": 17}, 1) is None
    assert C.pot_volume({**g, "same": False}, 1) is None
    monkeypatch.setattr(C, "pot_baseline", lambda pid, before: 16.0)
    assert C.pot_volume(g, 1) is None  # 22 is not a spike for a 16-a-game passer


def test_b2b_rest_rates():
    from datetime import date, timedelta

    import polars as pl

    from nba_ladders.rest import rest_rates

    # team plays day pairs (B2B) 10 times; player A sits every second night, B never does
    rows, d0 = [], date(2025, 11, 1)
    for i in range(10):
        d1 = d0 + timedelta(days=4 * i)
        for gid, d in ((f"g{i}a", d1), (f"g{i}b", d1 + timedelta(days=1))):
            for pid in (1, 2):
                if pid == 1 and gid.endswith("b"):
                    continue
                rows.append({"season_id": "22025", "player_id": pid, "player_name": f"P{pid}", "team": "AAA",
                             "game_id": gid, "game_date": d, "minutes": 30})
    r = rest_rates(pl.DataFrame(rows), date(2026, 1, 1), "22025")
    p = dict(r.select("player_id", "p_sit_b2b").iter_rows())
    assert p[1] > 0.8 and p[2] < 0.2


def test_apply_blowout_cuts_starter_tails():
    import numpy as np

    from nba_ladders.candidates import apply_blowout

    rng = np.random.default_rng(0)
    s = {"minutes": np.full(20_000, 34.0), "pts": rng.poisson(20, 20_000), "reb": rng.poisson(6, 20_000),
         "ast": rng.poisson(5, 20_000)}
    big, close = apply_blowout(s, -14.0, 34.0), apply_blowout(s, -1.0, 34.0)
    assert (big["pts"] >= 30).mean() < (close["pts"] >= 30).mean()
    assert big["minutes"].mean() < close["minutes"].mean()
    assert apply_blowout(s, None, 34.0) is s


def test_sgm_ideas_needs_live_legs():
    import numpy as np

    from nba_ladders.candidates import sgm_ideas

    rng = np.random.default_rng(1)
    s = {"pts": rng.poisson(12, 40_000), "reb": rng.poisson(7, 40_000), "ast": rng.poisson(2, 40_000)}
    prices = {"betr_au": {("pts", 10): 2.2, ("reb", 7): 2.4, ("ast", 7): 9.0}}
    ideas = sgm_ideas(s, prices)
    assert ideas and ideas[0]["combo"] == "10p/7r" and ideas[0]["book"] == "betr_au"  # 7+ ast not live
