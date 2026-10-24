"""Candidate screens: which players have the minutes-scenario profile worth modelling."""

import polars as pl

from nba_ladders.with_without import with_without

BIG_POSITIONS = ("C", "C-F", "F-C")
CENTRE_POSITIONS = ("C", "C-F")  # centre-first; F-C are mostly stretch forwards
BIG_MINUTES = 25


def parse_player_index(payload: dict) -> pl.DataFrame:
    result = payload["resultSets"][0]
    df = pl.DataFrame(result["rowSet"], schema=result["headers"], orient="row", infer_schema_length=None)
    return df.select(
        pl.col("PERSON_ID").alias("player_id"),
        (pl.col("PLAYER_FIRST_NAME") + " " + pl.col("PLAYER_LAST_NAME")).alias("player_name"),
        pl.col("TEAM_ABBREVIATION").alias("team_now"),
        pl.col("POSITION").alias("position"),
        pl.col("HEIGHT").alias("height"),
    )


def player_profiles(pg: pl.DataFrame) -> pl.DataFrame:
    """Minutes spread and how production scales with minutes, per player over pg's games."""
    big = pl.col("minutes") >= BIG_MINUTES
    dd = (pl.col("pts") >= 10) & (pl.col("reb") >= 10)
    return pg.group_by("player_id").agg(
        pl.len().alias("gp"),
        pl.col("minutes").mean().alias("min_avg"),
        pl.col("minutes").std().alias("min_sd"),
        big.mean().alias("p_min25"),
        (pl.col("minutes") <= 12).mean().alias("p_min12"),
        (pl.col("pts").sum() / pl.col("minutes").sum() * 36).alias("pts36"),
        (pl.col("reb").sum() / pl.col("minutes").sum() * 36).alias("reb36"),
        (pl.col("pf").sum() / pl.col("minutes").sum() * 36).alias("pf36"),
        dd.mean().alias("p_10_10"),
        big.sum().alias("n_big"),
        dd.filter(big).mean().alias("p_10_10_big"),
        pl.corr("pts", "reb").alias("corr_pr"),
    )


def top_blockers(
    pg: pl.DataFrame, blocker_ids: pl.Series | None = None, min_games_out: int = 5
) -> pl.DataFrame:
    """For each player-season-team, the teammate whose absence lifted his minutes most.

    blocker_ids restricts candidate blockers (e.g. to bigs, for the backup-C thesis).
    """
    ww = with_without(pg, min_games_out=min_games_out)
    if blocker_ids is not None:
        ww = ww.filter(pl.col("mate_id").is_in(blocker_ids.implode()))
    return (
        ww.sort("minutes_uplift", descending=True)
        .group_by("player_id", "season_id", "team", maintain_order=True)
        .first()
        .select(
            "player_id", "season_id", "team",
            pl.col("mate_name").alias("blocker"),
            pl.col("games_mate_out").alias("blocker_out_gp"),
            pl.col("minutes").alias("min_blocker_in"),
            pl.col("minutes_mate_out").alias("min_blocker_out"),
        )
    )


def backup_bigs(
    pg: pl.DataFrame,
    index: pl.DataFrame,
    *,
    tier_season: str,
    history_seasons: list[str],
    current_season: str,
    min_range: tuple[float, float] = (8, 24),
    min_gp: int = 20,
    shrink_games: int = 10,
) -> pl.DataFrame:
    """Current bigs who were bench-tier (avg minutes in min_range) in tier_season.

    p_10_10_big_adj shrinks the big-minutes hit rate toward the group mean as if we had
    shrink_games extra average games, so a 4/6 doesn't outrank a 30/60.
    """
    bigs = index.filter(pl.col("position").is_in(BIG_POSITIONS))
    tier_pg = pg.filter(pl.col("season_id") == tier_season)
    season_mins = tier_pg.group_by("player_id").agg(
        pl.len().alias("gp_tier"),
        pl.col("minutes").mean().alias("min_tier"),
        pl.col("team").sort_by("game_date").last().alias("team_tier"),
    )
    tier = season_mins.filter(
        pl.col("gp_tier") >= min_gp, pl.col("min_tier").is_between(*min_range)
    )
    hist = player_profiles(pg.filter(pl.col("season_id").is_in(history_seasons)))
    blockers = top_blockers(tier_pg, blocker_ids=bigs["player_id"])
    # The centre on the player's *current* roster who played most last season.
    current_starter = (
        bigs.filter(pl.col("position").is_in(CENTRE_POSITIONS))
        .join(season_mins, on="player_id")
        .select("team_now", pl.col("player_id").alias("starter_id"),
                pl.col("player_name").alias("starter_now"),
                pl.col("min_tier").alias("starter_min"))
    )
    now = (
        pg.filter(pl.col("season_id") == current_season)
        .group_by("player_id")
        .agg(pl.len().alias("gp_now"), pl.col("minutes").mean().alias("min_now"))
    )
    out = (
        bigs.join(tier, on="player_id")
        .join(hist, on="player_id")
        .join(blockers, left_on=["player_id", "team_tier"], right_on=["player_id", "team"],
              how="left")
        .drop("season_id")
        .join(now, on="player_id", how="left")
        .with_columns((pl.col("team_now") != pl.col("team_tier")).alias("moved"))
    )
    starters = (
        out.select("player_id", "team_now", "min_tier")
        .join(current_starter, on="team_now")
        .filter(pl.col("starter_id") != pl.col("player_id"), pl.col("starter_min") > pl.col("min_tier"))
        .sort("starter_min", descending=True)
        .group_by("player_id", maintain_order=True)
        .first()
        .select("player_id", "starter_now", "starter_min")
    )
    prior = (out["p_10_10_big"] * out["n_big"]).sum() / out["n_big"].sum()
    return (
        out.join(starters, on="player_id", how="left")
        .with_columns(
            ((pl.col("p_10_10_big").fill_null(0) * pl.col("n_big") + prior * shrink_games)
             / (pl.col("n_big") + shrink_games)).alias("p_10_10_big_adj")
        )
        .sort("p_10_10_big_adj", descending=True)
    )
