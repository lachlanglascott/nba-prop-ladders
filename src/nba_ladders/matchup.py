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
# Since 2026-11-13 these come from opp_position.py: per-minute vs each player's own rate, as at the date, shrunk
# group -> team -> league, last season as the prior. Tested predictive and under-priced (docs/opp_position.md);
# the previous estimator here (K_GROUP 25, pts x1.5 in spots) is in git history.
GROUP_STATS = ("pts", "reb", "ast")


@lru_cache(maxsize=16)
def group_factors(as_of: date, season_id: str = "22026") -> pl.DataFrame:
    """Per opponent and position group: pts / reb / ast allowed per minute vs each player's own rate."""
    from nba_ladders.opp_position import tonight

    rows = [{"opponent": o, "grp": g, **f} for (o, g), f in tonight(as_of).items()]
    return pl.DataFrame(rows) if rows else pl.DataFrame({"opponent": [], "grp": [], "pts": [], "reb": [], "ast": []})


def player_multipliers(opponent: str, grp: str, as_of: date, season_id: str = "22026") -> dict[str, float]:
    """Pace x what this opponent allows to the player's position group (pts, ast, weighted by OPP_W); reb for display."""
    from nba_ladders.opp_position import OPP_W

    pace = multipliers(opponent, as_of, season_id)["pace"]
    r = group_factors(as_of, season_id).filter(pl.col("opponent") == opponent, pl.col("grp") == grp)
    if r.is_empty():
        team = multipliers(opponent, as_of, season_id)
        return {"pace": pace, "pts": team["pts"], "ast": team["ast"], "reb_grp": 1.0}
    row, w = r.row(0, named=True), OPP_W.get(grp, OPP_W["W"])  # per-minute vs own rate: pace is already inside
    return {"pace": pace, "pts": row["pts"] ** w["pts"], "ast": row["ast"] ** w["ast"], "reb_grp": row["reb"]}


def weakness_line(opponent: str, grp: str, as_of: date) -> str:
    """Plain words: what this opponent allows to this position group, vs league."""
    from nba_ladders.opp_position import line

    r = group_factors(as_of).filter(pl.col("opponent") == opponent, pl.col("grp") == grp)
    if r.is_empty():
        return ""
    return line(r.row(0, named=True), opponent, grp, multipliers(opponent, as_of)["pace"])
