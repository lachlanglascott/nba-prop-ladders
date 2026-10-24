"""How a player's minutes/production change when a teammate is absent.

A teammate counts as "absent" from a team game if he has no row for it but played for that
team both before and after it in the same season (i.e. he was on the roster: injury, rest,
suspension). Games outside his tenure with the team (pre-signing, post-trade) are excluded.
"""

import polars as pl

STATS = ["minutes", "pts", "reb", "ast"]


def team_games(pg: pl.DataFrame) -> pl.DataFrame:
    return pg.select("season_id", "team_id", "game_id", "game_date").unique()


def tenures(pg: pl.DataFrame) -> pl.DataFrame:
    return pg.group_by("season_id", "team_id", "player_id").agg(
        pl.col("game_date").min().alias("first"), pl.col("game_date").max().alias("last")
    )


def teammate_status(pg: pl.DataFrame) -> pl.DataFrame:
    """One row per (team game, rostered teammate) with played = True/False."""
    rostered = (
        team_games(pg)
        .join(tenures(pg), on=["season_id", "team_id"])
        .filter(pl.col("game_date").is_between(pl.col("first"), pl.col("last")))
        .select("season_id", "team_id", "game_id", pl.col("player_id").alias("mate_id"))
    )
    played = pg.select("game_id", "team_id", pl.col("player_id").alias("mate_id")).with_columns(
        pl.lit(True).alias("mate_played")
    )
    return rostered.join(played, on=["game_id", "team_id", "mate_id"], how="left").with_columns(
        pl.col("mate_played").fill_null(False)
    )


def with_without(pg: pl.DataFrame, min_games_out: int = 3) -> pl.DataFrame:
    """Per (player, teammate, season, team): player's average stats with mate in vs out."""
    status = teammate_status(pg)
    joined = pg.join(status, on=["season_id", "team_id", "game_id"]).filter(
        pl.col("player_id") != pl.col("mate_id")
    )
    aggs = [pl.len().alias("games")] + [pl.col(s).mean() for s in STATS]
    summary = joined.group_by(
        "season_id", "team", "team_id", "player_id", "player_name", "mate_id", "mate_played"
    ).agg(aggs)
    keys = ["season_id", "team", "team_id", "player_id", "player_name", "mate_id"]
    on = summary.filter(pl.col("mate_played")).drop("mate_played")
    off = summary.filter(~pl.col("mate_played")).drop("mate_played")
    names = pg.select(pl.col("player_id").alias("mate_id"), pl.col("player_name").alias("mate_name"))
    return (
        on.join(off, on=keys, suffix="_mate_out")
        .filter(pl.col("games_mate_out") >= min_games_out)
        .join(names.unique("mate_id"), on="mate_id", how="left")
        .with_columns(
            (pl.col(f"{s}_mate_out") - pl.col(s)).alias(f"{s}_uplift") for s in STATS
        )
        .sort("minutes_uplift", descending=True)
    )
