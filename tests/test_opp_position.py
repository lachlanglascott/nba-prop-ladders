from nba_ladders.opp_position import angle_h, archetype, bet_factor, bet_stats


def test_bet_stats():
    assert bet_stats({"kind": "single", "bet": "17+ pts"}) == ["pts"]
    assert bet_stats({"kind": "single", "bet": "20+ pts+reb"}) == ["pts", "reb"]
    assert bet_stats({"kind": "SGM", "bet": "15p/10r"}) == ["pts", "reb"]


def test_bet_factor_geometric_mean():
    assert abs(bet_factor({"pts": 1.21, "reb": 1.0}, ["pts", "reb"]) - 1.1) < 1e-9
    assert bet_factor({}, ["ast"]) == 1.0


def test_bigs_any_spot():
    pts = {"kind": "single", "bet": "17+ pts"}
    assert angle_h({"pts": 0.97}, pts, {"role"}, "B", set())[0] == "gate"  # Poeltl vs PHI
    assert angle_h({"pts": 0.91}, pts, {"bounce"}, "B", set())[0] == "gate"  # starting big, not just backups
    assert angle_h({"pts": 1.05, "reb": 1.04}, {"kind": "SGM", "bet": "15p/10r"}, {"role"}, "B", set())[0] == "boost"
    assert angle_h({"ast": 0.5}, {"kind": "single", "bet": "4+ ast"}, {"role"}, "B", set())[0] == ""  # bigs' ast: no


def test_guards_wings_assists_only():
    ast = {"kind": "single", "bet": "6+ ast"}
    assert angle_h({"ast": 0.91}, ast, {"role"}, "W", set())[0] == "down"
    assert angle_h({"ast": 1.03}, ast, {"role"}, "G", set())[0] == "boost"
    assert angle_h({"ast": 1.10}, ast, {"bounce"}, "G", {"ast"})[0] == "boost"
    assert angle_h({"ast": 1.10}, ast, {"emerging"}, "G", set())[0] == ""  # opponent alone: no
    assert angle_h({"pts": 0.90}, {"kind": "single", "bet": "22+ pts"}, {"role"}, "G", set())[0] == ""


def test_archetype():
    assert archetype("G", 7.0, 4.0, 5.0) == "lead_guard"
    assert archetype("B", 2.0, 12.0, 0.5) == "rim_big"


def test_wing_points_bad_spot():
    from nba_ladders.opp_position import line

    pts = {"kind": "single", "bet": "20+ pts"}
    assert angle_h({"pts": 0.96}, pts, {"role"}, "W", set())[0] == "down"  # George vs DET
    assert angle_h({"pts": 0.96}, pts, {"role"}, "G", set())[0] == ""  # guards' points: flat
    assert line({"pts": 0.96, "reb": 1.0, "ast": 0.913}, "DET", "W", 1.0) == "DET vs wings: pts -4%, reb +0%, ast -9%; pace +0%"
