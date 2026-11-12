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
    out = _lineup_weights(s, pid=5, tonight_in={2, 3}, id_name={2: "Horford", 3: "Williams", ("grp", 5): "W", ("grp", 2): "W", ("grp", 3): "W"})
    assert out["lw"].to_list() == [1.0, EXTRA_OUT_W ** 2, 1.0]  # 9 isn't playing tonight: same lineup
    big = _lineup_weights(pl.DataFrame({"absent_wide": [[1, 2, 3, 4, 5]]}), pid=9, tonight_in={1, 2, 3, 4},
                          id_name={("grp", 9): "B", ("grp", 1): "G", ("grp", 2): "G", ("grp", 3): "W", ("grp", 4): "W"})
    assert 0.4 < big["lw"][0] < 0.5  # four guards/wings out barely inflates a big's minutes (Achiuwa)
    assert out["also_out"].to_list() == ["", "Horford, Williams", ""]


def test_lineup_weights_drop_games_the_starting_big_missed():
    """Kalkbrenner 12 Nov: his one Miller/Schroder-out game (35 min, 15 reb) also had Diabate out, who plays
    tonight. A backup big's games without the big ahead of him are a different role: they don't count (user)."""
    import polars as pl

    from nba_ladders.spots import _lineup_weights

    ids = {("grp", 9): "B", ("grp", 1): "B", ("grp", 2): "W", ("mpg", 9): 19.0, ("mpg", 1): 30.0, ("mpg", 2): 30.0}
    out = _lineup_weights(pl.DataFrame({"absent_wide": [[5, 1], [5], [5, 2]]}), pid=9, tonight_in={1, 2}, id_name=ids)
    assert out["lw"].to_list()[0] == 0.0  # the starting centre was out then: not tonight's role
    assert out["lw"].to_list()[1] == 1.0 and 0 < out["lw"].to_list()[2] < 1
    listed_f = _lineup_weights(pl.DataFrame({"absent_wide": [[5, 2]]}), pid=9, tonight_in={2},
                               id_name={**ids, ("r36", 2): 15.0})  # listed F, rebounds like a centre (Diabate)
    assert listed_f["lw"][0] == 0.0


def test_conditional_keeps_best_bet_per_stat():
    from nba_ladders.spots import _one_per_player

    a = {"name": "Kyshawn George"}
    reb = {"kind": "single", "bet": "8+ reb", "p": 0.16, "price": 7.75, "ev": 0.24}
    reb2 = {"kind": "single", "bet": "7+ reb", "p": 0.22, "price": 5.0, "ev": 0.10}
    pts = {"kind": "single", "bet": "20+ pts", "p": 0.31, "price": 3.70, "ev": 0.147}
    got = [b["bet"] for _, b in _one_per_player([(a, reb), (a, reb2), (a, pts)])]
    assert set(got) == {"8+ reb", "20+ pts"}  # his points no longer hide behind his rebounds


def test_builder_factor_by_shape():
    from nba_ladders.spots import builder_factor

    assert builder_factor([4.30, 1.61]) > 1.0  # uneven 7r/5a: Black slip $7.50 vs legs $6.92
    assert builder_factor([2.5, 2.2]) < 1.0  # even 6r/6a: $5.25 vs $5.50
    assert builder_factor([5.2, 5.2]) < 0.9  # long legs 8r/8a: $21 vs $27


def test_ra_lift_volatile_above_steady():
    from nba_ladders.spots import ra_lift

    legs = [("reb", 7), ("ast", 7)]
    assert ra_lift({"min_sd": 7.0, "combo_ra": True}, legs) > ra_lift({"min_sd": 4.0, "combo_ra": True}, legs) > 1.0
    assert ra_lift({"min_sd": 7.0, "combo_ra": True}, [("pts", 15), ("ast", 5)]) is None
    pr = [("pts", 12), ("reb", 10)]
    assert ra_lift({"min_sd": 7.5, "big": True}, pr) > 1.3 > ra_lift({"min_sd": 4.0, "big": True}, pr)  # Clingan-type swing
    assert ra_lift({"min_sd": 7.5}, pr) is None  # not a big


def test_builder_factor_pr_charges_correlation():
    from nba_ladders.spots import builder_factor

    assert 0.7 < builder_factor([3.3, 2.65], {"pts", "reb"}) < 0.85  # Ighodaro 10p/7r $7 vs legs $8.74
    assert builder_factor([4.30, 1.61], {"reb", "ast"}) > 1.0  # uneven R/A unchanged


def test_min_price_and_leg_floor_constants():
    from nba_ladders.spots import LEG_FLOOR, NEWS_MIN

    assert 0.15 <= LEG_FLOOR <= 0.25  # lottery legs out, not "weak stats" (that rule failed on 11 combos)
    assert 3.0 <= NEWS_MIN <= 4.0  # start/bench news = a modest shift both ways


def test_own_status_minutes_cut():
    from nba_ladders.spots import own_status_cut

    assert own_status_cut("Questionable") == 1.8
    assert own_status_cut("Probable") == 0.8
    assert own_status_cut("") == own_status_cut("Out") == 0.0
    assert own_status_cut("Questionable", "B2B rest (not on report): sat 3 of 4 second nights") == 0.0


def test_lead_with_single_keeps_combos_listed():
    from nba_ladders.spots import lead_with_single

    sgm = {"kind": "SGM", "bet": "9r/4a"}
    single = {"kind": "single", "bet": "9+ reb"}
    lean = {"kind": "single", "bet": "6+ ast", "lean": True}
    assert lead_with_single([sgm, lean, single]) == [single, sgm, lean]
    assert lead_with_single([sgm, lean]) == [sgm, lean]  # no firm single: the combo still leads


def test_side_blowout_curves():
    import numpy as np

    from nba_ladders.blowout import side_base, side_shift

    big_win, big_loss = np.array([22.0]), np.array([-22.0])
    assert side_shift(big_win, 33)[0] < -3 and side_shift(big_loss, 33)[0] < -3  # starters sit either way
    assert side_shift(big_loss, 14)[0] > side_shift(big_win, 14)[0]  # the losers' bench gets the garbage time
    assert abs(side_base(14) - 1.0) < 0.3


def test_price_plays_one_per_player_positive_ev():
    from nba_ladders.spots import price_plays

    bet = lambda stat, ev, price=3.0: {"bet": f"10+ {stat}", "kind": "single", "price": price, "p": (1 + ev) / price, "ev": ev, "own_hr": 0.4}
    near = [{"name": "A", "bets": [bet("reb", 0.10), bet("reb", 0.20)]}, {"name": "B", "bets": [bet("pts", 0.02)]},
            {"name": "C", "bets": [bet("pts", 0.15)]}, {"name": "D", "bets": [bet("ast", 0.30)]}]
    out = price_plays({"near": near}, card=[({"name": "D"}, {})])
    assert [a["name"] for a, _ in out] == ["A", "C"]  # D is on the card, B is under +4%


def test_price_plays_one_per_game_and_stat():
    """0/10 over 10-11 Nov, three rebound overs a night and two from one game (Amen, Eason): cap both (user, 10 Nov)."""
    from nba_ladders.spots import price_plays

    bet = lambda b, ev: {"bet": b, "kind": "SGM" if "/" in b else "single", "price": 3.0, "p": (1 + ev) / 3.0, "ev": ev, "own_hr": 0.4}
    near = [{"name": "Amen", "team": "HOU", "opp": "GSW", "bets": [bet("9+ reb", 0.21)]},
            {"name": "Eason", "team": "HOU", "opp": "GSW", "bets": [bet("10+ pts", 0.20)]},  # same game
            {"name": "Ayton", "team": "WAS", "opp": "IND", "bets": [bet("7+ reb", 0.19)]},  # second rebound play
            {"name": "Castle", "team": "SAS", "opp": "CHI", "bets": [bet("8+ ast", 0.17)]},
            {"name": "Boozer", "team": "MEM", "opp": "DEN", "bets": [bet("9r/7a", 0.15)]}]
    out = price_plays({"near": near}, card=[])
    assert [a["name"] for a, _ in out] == ["Amen", "Castle", "Boozer"]


def test_price_plays_prefer_single():
    from nba_ladders.spots import price_plays

    sgm = {"bet": "9r/3a", "kind": "SGM", "price": 4.8, "p": 0.27, "ev": 0.29}
    single = {"bet": "9+ reb", "kind": "single", "price": 3.0, "p": 0.36, "ev": 0.08, "own_hr": 0.4}
    out = price_plays({"near": [{"name": "Amen", "bets": [sgm, single]}]}, card=[])
    assert out[0][1]["bet"] == "9+ reb"
