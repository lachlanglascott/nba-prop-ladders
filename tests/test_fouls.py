from datetime import date

import polars as pl
import pytest

from nba_ladders.fouls import (
    backup_games,
    backup_split,
    flag_starters,
    opponent_effect,
    player_game_fouls,
    starting_centres,
    within_player_uplift,
)
from nba_ladders.sources.pbp import (
    FOUL_SCHEMA,
    get_pbp,
    parse_fouls,
    parse_starters,
    seconds_elapsed,
)


def _action(n, period, clock, pid, sub, desc, team="AAA"):
    return {
        "actionNumber": n, "period": period, "clock": clock, "teamId": 1, "teamTricode": team,
        "personId": pid, "playerName": f"p{pid}", "actionType": "Foul", "subType": sub,
        "description": desc,
    }


PAYLOAD = {"game": {"gameId": "g1", "actions": [
    _action(1, 1, "PT11M26.00S", 10, "Shooting", "X S.FOUL (P1.T1) (Ref)"),
    _action(2, 1, "PT02M00.00S", 10, "Personal", "X P.FOUL (P2.T2) (Ref)"),
    _action(3, 2, "PT05M00.00S", 10, "Technical", "X T.FOUL (P0.T0) (Ref)"),
    _action(6, 2, "PT04M00.00S", 10, "Offensive", "X OFF.Foul (P3) (Ref)"),
    {"actionNumber": 7, "period": 2, "clock": "PT04M00.00S", "personId": 10,
     "actionType": "Turnover", "subType": "Offensive Foul Turnover", "description": "(P3.T1)"},
    _action(4, 2, "PT05M00.00S", 0, "Defensive 3 Second", "T.FOUL (Ref)"),
    {"actionNumber": 5, "period": 1, "clock": "PT01M00.00S", "personId": 11,
     "actionType": "Made Shot", "subType": "Jump Shot", "description": "shot"},
]}}


def test_parse_fouls_excludes_technicals():
    df = parse_fouls(PAYLOAD)
    assert df.schema == pl.Schema(FOUL_SCHEMA)
    assert df["action_number"].to_list() == [1, 2, 6]
    assert df["player_pf_count"].to_list() == [1, 2, 3]
    assert df["seconds_elapsed"].to_list() == [34.0, 600.0, 720 + 480.0]


def test_seconds_elapsed_overtime():
    assert seconds_elapsed(2, "PT12M00.00S") == 720
    assert seconds_elapsed(5, "PT04M00.00S") == 2880 + 60


def test_get_pbp_caches_and_retries(tmp_path):
    calls = []

    def flaky(gid):
        calls.append(gid)
        if len(calls) == 1:
            raise TimeoutError("boom")
        return {"game": {"gameId": gid, "actions": [{"actionType": "x"}] * 301}}

    p = get_pbp("g9", raw_dir=tmp_path, fetch=flaky, sleep=lambda s: None)
    assert len(calls) == 2 and (tmp_path / "g9.json").exists()
    assert get_pbp("g9", raw_dir=tmp_path, fetch=flaky) == p
    assert len(calls) == 2  # served from cache


def _fouls(rows):
    return pl.DataFrame(
        [{"game_id": g, "player_id": p, "period": q} for g, p, q in rows],
        schema={"game_id": pl.String, "player_id": pl.Int64, "period": pl.Int64},
    )


def test_early_trouble_flag():
    f = _fouls([
        ("g1", 1, 1), ("g1", 1, 1),               # 2 in Q1 -> trouble
        ("g1", 2, 1), ("g1", 2, 2),               # 2 in H1 -> clean
        ("g1", 3, 1), ("g1", 3, 2), ("g1", 3, 2), # 3 in H1 -> trouble
        ("g1", 4, 3), ("g1", 4, 3), ("g1", 4, 4), # late fouls -> clean
    ])
    out = player_game_fouls(f).sort("player_id")
    assert out["early_trouble"].to_list() == [True, False, True, False]
    assert out["pf_q1"].to_list() == [2, 1, 1, 0]


POS = pl.DataFrame({"player_id": [1, 2, 3, 9], "position": ["C", "C-F", "F-C", "G"]})


def _pg(rows):
    return pl.DataFrame(
        [{"season_id": "22025", "player_id": p, "player_name": f"p{p}", "team": "AAA",
          "opponent": opp, "game_id": g, "game_date": d, "minutes": m, "pts": pts,
          "reb": reb, "pf": 0} for p, g, d, opp, m, pts, reb in rows]
    )


PG = _pg([
    # player 1 is the main C; in g3 he does not play, so player 2 is that game's starter.
    (1, "g1", date(2025, 11, 1), "BBB", 32, 15, 10),
    (2, "g1", date(2025, 11, 1), "BBB", 14, 6, 5),
    (3, "g1", date(2025, 11, 1), "BBB", 10, 4, 3),
    (9, "g1", date(2025, 11, 1), "BBB", 36, 30, 4),
    (1, "g2", date(2025, 11, 3), "CCC", 20, 8, 6),
    (2, "g2", date(2025, 11, 3), "CCC", 28, 12, 11),
    (3, "g2", date(2025, 11, 3), "CCC", 16, 10, 10),
    (2, "g3", date(2025, 11, 5), "BBB", 8, 10, 9),
])


STARTERS = pl.DataFrame({  # g1: C (1) + F-C (3) start; g2: 1 starts; g3: only 2 plays
    "game_id": ["g1", "g1", "g1", "g2", "g2", "g3"],
    "team": ["AAA"] * 6,
    "player_id": [1, 3, 9, 1, 9, 2],
})


def test_starting_centre_and_backup_split():
    starters = starting_centres(PG, POS, STARTERS).sort("game_id")
    assert starters["starter_id"].to_list() == [1, 1, 2]  # C beats F-C; guard never picked

    fouls = _fouls([("g2", 1, 1), ("g2", 1, 1)])  # starter in trouble in g2 only
    flagged = flag_starters(starters, player_game_fouls(fouls)).sort("game_id")
    assert flagged["early_trouble"].to_list() == [False, True, False]

    bg = backup_games(PG, POS, flagged, STARTERS)
    # starting F-C (3 in g1) is not a backup; guard never a backup; g3 has none
    assert sorted(zip(bg["game_id"], bg["player_id"])) == [("g1", 2), ("g2", 2), ("g2", 3)]
    split = backup_split(bg)
    clean = split.filter(~pl.col("early_trouble")).row(0, named=True)
    trouble = split.filter(pl.col("early_trouble")).row(0, named=True)
    assert clean["games"] == 1 and trouble["games"] == 2
    assert clean["min_avg"] == pytest.approx(14.0)
    assert trouble["min_avg"] == pytest.approx(22.0)
    assert trouble["p_10_10"] == pytest.approx(1.0)  # both 10/10 when starter in trouble
    assert clean["p_10_10"] == pytest.approx(0.0)

    wpu = within_player_uplift(bg, min_trouble_games=1).sort("player_id")
    assert wpu["player_id"].to_list() == [2]  # 3 has no clean game
    assert wpu["min_uplift"].to_list() == [14.0]


def test_opponent_effect():
    flagged = pl.DataFrame({
        "opponent": ["BBB", "BBB", "CCC", "CCC"],
        "game_date": [date(2025, 11, d) for d in (1, 9, 2, 10)],
        "early_trouble": [True, True, False, False],
    })
    opp = opponent_effect(flagged)
    bbb = opp.filter(pl.col("opponent") == "BBB").row(0, named=True)
    assert bbb["rate"] == 1.0 and bbb["vs_league"] == pytest.approx(0.5)
    assert bbb["rate_h1"] == 1.0 and bbb["rate_h2"] == 1.0


def _ev(n, team, pid, name, kind="Made Shot", desc="", name_i=None):
    return {"actionNumber": n, "period": 1, "clock": "PT10M00.00S", "teamTricode": team,
            "personId": pid, "playerName": name, "playerNameI": name_i or name,
            "actionType": kind, "subType": "", "description": desc}


def test_parse_starters_handles_subs_diacritics_and_shared_surnames():
    acts = [_ev(i, "AAA", i, f"S{i}") for i in range(1, 5)]  # 4 obvious starters
    acts += [
        _ev(5, "AAA", 5, "Williams", name_i="J. Williams"),  # 5th starter
        _ev(6, "AAA", 1, "S1", "Substitution", "SUB: Valanciunas FOR S1"),
        _ev(7, "AAA", 7, "Valančiūnas"),  # came in: diacritics differ from sub text
        _ev(8, "AAA", 5, "Williams", "Substitution", "SUB: K. Williams FOR Williams"),
        _ev(9, "AAA", 9, "Williams", name_i="K. Williams"),  # came in, same surname
        {**_ev(10, "AAA", 4, "S4"), "period": 2},
    ]
    st = parse_starters({"game": {"gameId": "g", "actions": acts}})
    assert sorted(st["player_id"].to_list()) == [1, 2, 3, 4, 5]
