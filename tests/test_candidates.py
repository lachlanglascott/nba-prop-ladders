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


def test_tips_margin_and_questionable(monkeypatch):
    from datetime import date

    import polars as pl

    from nba_ladders import candidates as C

    monkeypatch.setattr(C, "_frame", lambda name: pl.DataFrame(
        {"season_id": ["22026"] * 2, "player_name": ["A B", "C D"], "minutes": [28.0, 20.0], "team": ["AAA"] * 2,
         "game_date": [date(2026, 11, 1)] * 2}))
    pick = {"star": False, "why": "w", "angles": ["unlocked"]}
    sgm = {"combo": "12p/5r", "fair": 4.0, "est": 6.0, "book": "betr_au"}
    cands = {"A B": {"team": "AAA", "opp": "BBB", "reasons": {"unlocked": ""}, "pick": pick, "proj_min": 29.0,
                     "sgm_ideas": [sgm, {**sgm, "combo": "10p/3a", "est": 4.4}]},
             "C D": {"team": "AAA", "opp": "BBB", "reasons": {"unlocked": ""}, "pick": pick, "proj_min": 30.0,
                     "sgm_ideas": [sgm]}}
    report = pl.DataFrame({"player_key": [C.player_key("C D")], "status": ["Questionable"], "team": ["AAA"]})
    card = C.tips(cands, report)
    assert [(r["player"], r["bet"]) for r in card] == [("A B", "12p/5r")]  # 10p/3a fails the 1.3x margin
    assert card[0]["take_at"] == 1.3 * 4.0 and not card[0]["risk"]


def test_carry_forward():
    from datetime import date, timedelta

    import polars as pl

    from nba_ladders.availability import carry_forward

    d0, rows = date(2026, 10, 20), []
    for i in range(4):  # team AAA plays 4 games; A plays all, B misses the last one, C misses the last two
        for pid, nm in ((1, "A Aa"), (2, "B Bb"), (3, "C Cc")):
            if (nm == "B Bb" and i == 3) or (nm == "C Cc" and i >= 2):
                continue
            rows.append({"season_id": "22026", "player_name": nm, "team": "AAA", "game_date": d0 + timedelta(days=2 * i),
                         "minutes": 30})
    pg = pl.DataFrame(rows)
    day = d0 + timedelta(days=8)
    unfiled = pl.DataFrame({"team": ["AAA"], "player_key": [None], "status": ["Not submitted"]}, schema_overrides={"player_key": pl.String})
    prev = lambda d: pl.DataFrame({"player_key": ["bbb"], "status": ["Out"], "reason": ["Injury/Illness-Back; Injury Management"]})
    cf = {r["player"]: r for r in carry_forward(unfiled, pg, day, prev).iter_rows(named=True)}
    assert set(cf) == {"B Bb", "C Cc"}
    assert cf["B Bb"]["p_sit"] == 0.56 and cf["B Bb"]["status"] == "Questionable"  # Out last game, management
    assert cf["C Cc"]["p_sit"] == 0.67  # missed 2 straight, no prior report row
    filed = pl.DataFrame({"team": ["AAA"], "player_key": ["aaa"], "status": ["Available"]})
    cf2 = carry_forward(filed, pg, day)
    assert set(cf2["p_sit"].to_list()) == {0.30}


def test_depth_chart_fc_starter():
    from datetime import date

    import polars as pl

    from nba_ladders.sheet import depth_chart

    rows = []
    for i in range(10):
        d = date(2026, 10, 20 + i)
        rows += [{"player_id": 1, "player_name": "F C", "game_date": d, "minutes": 30, "reb": 8},
                 {"player_id": 2, "player_name": "Cee", "game_date": d, "minutes": 18, "reb": 6}]
    roster = pl.DataFrame({"player_id": [1, 2], "team_now": ["AAA", "AAA"]})
    pos = pl.DataFrame({"player_id": [1, 2], "position": ["F-C", "C"]})
    dc = depth_chart(pl.DataFrame(rows), roster, pos)
    assert dc.filter(pl.col("is_starter"))["player_id"].to_list() == [1]  # the 30-mpg F-C starts, not the 18-mpg C


def test_spots_market_and_grades():
    from nba_ladders import spots as S

    assert abs(S.market_ratio(1.24) - 0.934) < 1e-6 and S.market_ratio(5.62) < S.market_ratio(2.24)  # longshot bias
    assert S.grade(0.15, 0.30, {"role"}) == "A"
    assert S.grade(0.15, 0.30, {"bounce"}) == "B"  # only lineup roles make A
    assert S.grade(0.50, 0.30, {"role"}, outlier=True) == "C"
    a = {"sits": [{"kind": "bounce", "stat": "ast", "text": "ast"}, {"kind": "bounce", "stat": "reb", "text": "reb"}]}
    assert S._why(a, {"kind": "single", "bet": "8+ reb"}) == "reb"
    assert S._why(a, {"kind": "SGM", "bet": "12p/5a"}) == "ast"


def test_spots_simulate_poisson_given_minutes():
    import numpy as np

    from nba_ladders import spots as S

    a = {"proj_min": 30.0, "min_sd": 4.0, "spread": None, "rate": {"ast": 0.2, "reb": 0.2, "pts": 0.5}}
    s = S.simulate(a, n=50_000)
    assert abs(s["ast"].mean() - 6.0) < 0.15 and abs(s["minutes"].mean() - 30.0) < 0.2
    # minutes spread makes counts over-dispersed vs Poisson, but only modestly (var/mean ~1.1-1.4)
    assert 1.0 < s["ast"].var() / s["ast"].mean() < 1.5
    assert np.corrcoef(s["ast"], s["reb"])[0, 1] > 0.05  # shared minutes correlate the stats


def test_lineup_weights_extra_absences_count_less():
    import polars as pl

    from nba_ladders.spots import EXTRA_OUT_W, _lineup_weights

    s = pl.DataFrame({"absent_wide": [[1], [1, 2, 3], [1, 9]]})
    out = _lineup_weights(s, pid=5, tonight_in={2, 3}, id_name={2: "Horford", 3: "Williams"})
    assert out["lw"].to_list() == [1.0, EXTRA_OUT_W ** 2, 1.0]  # 9 isn't playing tonight: same lineup
    assert out["also_out"].to_list() == ["", "Horford, Williams", ""]
