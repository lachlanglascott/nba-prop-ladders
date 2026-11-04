"""Opponent pace and defence multipliers for points and assists.

Possessions per team-game = FGA + 0.44 FTA - OREB + TOV, per 48 minutes. For each team, as at a
date: pace, and points / assists allowed per possession relative to league. This season's
games are shrunk toward last season's (itself regressed halfway to league), so early-season
numbers lean on last season.

Player multiplier vs opponent X: pace = X's pace / league (his own team's pace is already in
his history), times X's allowed rate for the stat. Rebounds use rebounding.py instead (that
already includes the opponent's miss volume).

    uv run python -m nba_ladders.matchup [--date YYYY-MM-DD]   # table of tonight's factors
"""

import argparse
from datetime import date
from functools import lru_cache

import polars as pl

from nba_ladders import config

K_PACE, K_DEF = 10.0, 15.0  # pseudo-games of last season in this season's estimate
REGRESS = 0.5  # last season's factor regressed this far to league
STATS = ("pts", "ast")


def team_games(pg: pl.DataFrame) -> pl.DataFrame:
    """One row per team-game: possessions per 48 and points / assists scored."""
    t = pg.group_by("season_id", "game_id", "game_date", "team", "opponent").agg(
        (pl.col("fga").sum() + 0.44 * pl.col("fta").sum() - pl.col("oreb").sum() + pl.col("tov").sum()).alias("poss"),
        pl.col("minutes").sum().alias("team_min"), pl.col("pts").sum(), pl.col("ast").sum())
    # both teams share the game's possessions: average them, scale overtime to 48 minutes
    g = t.group_by("game_id").agg(pl.col("poss").mean().alias("game_poss"), pl.col("team_min").max().alias("gm"))
    return t.join(g, on="game_id").with_columns(
        (pl.col("game_poss") * 240 / pl.col("gm").clip(lower_bound=200)).alias("pace"),
        (pl.col("pts") / pl.col("game_poss")).alias("pts_pp"), (pl.col("ast") / pl.col("game_poss")).alias("ast_pp"))


def factors(pg: pl.DataFrame, as_of: date, season_id: str) -> pl.DataFrame:
    """Per team, as at as_of: pace and pts/ast-allowed factors (1.0 = league average)."""
    tg = team_games(pg.filter(pl.col("game_date") < as_of))
    prev_id = str(int(season_id) - 1)
    out = None
    for sid in (prev_id, season_id):
        s = tg.filter(pl.col("season_id") == sid)
        if s.is_empty():
            continue
        lg = {"pace": s["pace"].mean(), **{x: s[f"{x}_pp"].mean() for x in STATS}}
        # pace of a team = its games' pace; defence = what its opponents scored against it
        own = s.group_by("team").agg(pl.len().alias("n"), (pl.col("pace").mean() / lg["pace"]).alias("pace"))
        allowed = s.group_by("opponent").agg(*[(pl.col(f"{x}_pp").mean() / lg[x]).alias(x) for x in STATS]) \
            .rename({"opponent": "team"})
        f = own.join(allowed, on="team")
        if out is None:  # last season, regressed toward league
            out = f.select("team", *[(1 + (1 - REGRESS) * (pl.col(c) - 1)).alias(c) for c in ("pace", *STATS)])
        else:
            m = f.join(out, on="team", how="left", suffix="_prior").fill_null(1.0)
            out = m.select("team", *[
                ((pl.col(c) * pl.col("n") + pl.col(f"{c}_prior") * (K_PACE if c == "pace" else K_DEF))
                 / (pl.col("n") + (K_PACE if c == "pace" else K_DEF))).alias(c) for c in ("pace", *STATS)])
    return out if out is not None else pl.DataFrame({"team": [], "pace": [], "pts": [], "ast": []})


@lru_cache(maxsize=16)
def _table(as_of: date, season_id: str) -> pl.DataFrame:
    return factors(pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet"), as_of, season_id)


def multipliers(opponent: str, as_of: date, season_id: str = "22026") -> dict[str, float]:
    """Rate multipliers for a player facing `opponent`: {'pace', 'pts', 'ast'} (pts/ast include pace)."""
    r = _table(as_of, season_id).filter(pl.col("team") == opponent)
    if r.is_empty():
        return {"pace": 1.0, "pts": 1.0, "ast": 1.0}
    pace, pts, ast = r.row(0)[1:]
    return {"pace": pace, "pts": pace * pts, "ast": pace * ast}


def main() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat,
                    default=datetime.now(ZoneInfo("America/New_York")).date())
    args = ap.parse_args()
    t = _table(args.date, "22026").sort("pace", descending=True)
    with pl.Config(tbl_rows=40, tbl_hide_column_data_types=True, float_precision=3):
        print(t)


if __name__ == "__main__":
    main()


# ---- by position group (G / W / B) -------------------------------------------------------------
# Half-season persistence (2024-26): opp x group factors correlate 0.44-0.52 between halves (bigs'
# points 0.65, guards' assists 0.64); the residual after the team effect only 0.10-0.24. So the
# group factor is shrunk toward the team's own factor, which is shrunk toward the league.
K_GROUP = 25.0  # pseudo-games of "same as the team overall" in each group factor
GROUP_STATS = ("pts", "reb", "ast")


def _group_table(pg: pl.DataFrame, pos: pl.DataFrame, season_id: str, as_of: date) -> pl.DataFrame:
    from nba_ladders.absences import position_groups

    d = pg.filter(pl.col("season_id") == season_id, pl.col("game_date") < as_of, pl.col("minutes") >= 10)
    if d.is_empty():
        return pl.DataFrame()
    d = d.join(position_groups(pos), on="player_id", how="left").with_columns(pl.col("grp").fill_null("W"))
    rate = d.group_by("player_id").agg(*[(pl.col(s).sum() / pl.col("minutes").sum()).alias(f"r_{s}") for s in GROUP_STATS])
    d = d.join(rate, on="player_id").with_columns(*[(pl.col(f"r_{s}") * pl.col("minutes")).alias(f"e_{s}") for s in GROUP_STATS])
    team = d.group_by("opponent").agg(pl.col("game_id").n_unique().alias("n"),
                                      *[(pl.col(s).sum() / pl.col(f"e_{s}").sum()).alias(f"t_{s}") for s in GROUP_STATS])
    grp = d.group_by("opponent", "grp").agg(*[(pl.col(s).sum() / pl.col(f"e_{s}").sum()).alias(f"g_{s}") for s in GROUP_STATS])
    return grp.join(team, on="opponent")


@lru_cache(maxsize=16)
def group_factors(as_of: date, season_id: str = "22026") -> pl.DataFrame:
    """Per opponent and position group: pts / reb / ast allowed per minute vs each player's own rate."""
    from nba_ladders.scenarios import positions

    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    pos = positions()
    out = None
    for sid in (str(int(season_id) - 1), season_id):
        t = _group_table(pg, pos, sid, as_of)
        if t.is_empty():
            continue
        # shrink group toward team, team toward league (1.0)
        f = t.with_columns(*[((pl.col(f"g_{s}") * pl.col("n") + pl.col(f"t_{s}") * K_GROUP) / (pl.col("n") + K_GROUP)).alias(f"g_{s}")
                             for s in GROUP_STATS])
        if out is None:  # last season: regressed halfway to league
            out = f.select("opponent", "grp", *[(1 + (1 - REGRESS) * (pl.col(f"g_{s}") - 1)).alias(s) for s in GROUP_STATS])
        else:
            m = f.join(out, on=["opponent", "grp"], how="left", suffix="_prior").fill_null(1.0)
            out = m.select("opponent", "grp", *[((pl.col(f"g_{s}") * pl.col("n") + pl.col(s) * K_DEF) / (pl.col("n") + K_DEF)).alias(s)
                                                for s in GROUP_STATS])
    return out if out is not None else pl.DataFrame({"opponent": [], "grp": [], "pts": [], "reb": [], "ast": []})


def player_multipliers(opponent: str, grp: str, as_of: date, season_id: str = "22026") -> dict[str, float]:
    """Pace x what this opponent allows to the player's position group (pts, ast); reb factor for display."""
    pace = multipliers(opponent, as_of, season_id)["pace"]
    r = group_factors(as_of, season_id).filter(pl.col("opponent") == opponent, pl.col("grp") == grp)
    if r.is_empty():
        team = multipliers(opponent, as_of, season_id)
        return {"pace": pace, "pts": team["pts"], "ast": team["ast"], "reb_grp": 1.0}
    row = r.row(0, named=True)
    return {"pace": pace, "pts": pace * row["pts"], "ast": pace * row["ast"], "reb_grp": row["reb"]}


def weakness_line(opponent: str, grp: str, as_of: date) -> str:
    """Plain words: what this opponent allows to this position group, vs league."""
    r = group_factors(as_of).filter(pl.col("opponent") == opponent, pl.col("grp") == grp)
    if r.is_empty():
        return ""
    row = r.row(0, named=True)
    pace = multipliers(opponent, as_of)["pace"]
    name = {"G": "guards", "W": "wings", "B": "bigs"}[grp]
    bits = [f"{lab} {(row[s] - 1) * 100:+.0f}%" for s, lab in (("pts", "pts"), ("reb", "reb"), ("ast", "ast"))]
    return f"{opponent} vs {name}: " + ", ".join(bits) + f"; pace {(pace - 1) * 100:+.0f}%"
