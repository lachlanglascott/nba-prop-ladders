"""Pre-game-knowable scenarios per player-game.

Only flags you could know before tip-off belong here (e.g. the starting centre is ruled out).
Realised minutes are an outcome, never a scenario; conditioning on them is look-ahead.
"""

import polars as pl

from nba_ladders.screens import BIG_POSITIONS, parse_player_index
from nba_ladders.sources.nba_stats import get_player_index
from nba_ladders.with_without import teammate_status

INDEX_SEASONS = [
    "2019-20", "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26", "2026-27",
]


def positions() -> pl.DataFrame:
    """Latest known position per player across all cached player indexes."""
    frames = [
        parse_player_index(get_player_index(s)).with_columns(pl.lit(s).alias("idx_season"))
        for s in INDEX_SEASONS
    ]
    return (
        pl.concat(frames)
        .sort("idx_season")
        .unique("player_id", keep="last")
        .select("player_id", "position")
    )


CENTRE_REB36 = 10.0


def team_season_roles(pg: pl.DataFrame, pos: pl.DataFrame, min_gp: int = 20) -> pl.DataFrame:
    """Per (season, team, player): avg minutes, whether a big, and the team's starting centre.

    Position labels can't separate centres from stretch forwards (Plumlee and Markkanen are
    both "F-C"), so a centre is a big who also rebounds like one (reb/36 >= CENTRE_REB36).
    """
    roles = (
        pg.group_by("season_id", "team_id", "player_id")
        .agg(
            pl.len().alias("gp"),
            pl.col("minutes").mean().alias("season_min"),
            (pl.col("reb").sum() / pl.col("minutes").sum() * 36).alias("reb36"),
        )
        .join(pos, on="player_id", how="left")
        .with_columns(
            pl.col("position").is_in(BIG_POSITIONS).fill_null(False).alias("is_big"),
        )
        .with_columns((pl.col("is_big") & (pl.col("reb36") >= CENTRE_REB36)).alias("is_centre"))
    )
    starters = (
        roles.filter(pl.col("is_centre"), pl.col("gp") >= min_gp)
        .sort("season_min", descending=True)
        .group_by("season_id", "team_id", maintain_order=True)
        .first()
        .select(
            "season_id", "team_id",
            pl.col("player_id").alias("starter_id"),
            pl.col("season_min").alias("starter_min"),
        )
    )
    return roles.join(starters, on=["season_id", "team_id"], how="left")


def backup_big_games(pg: pl.DataFrame, pos: pl.DataFrame, min_gp: int = 20) -> pl.DataFrame:
    """Player-games of bench bigs, flagged with whether the starting centre was out.

    A bench big = a big (min_gp+ games for the team) who isn't the starting centre and averages
    fewer minutes than him. starter_out = starter on the roster that day but didn't play.
    """
    roles = team_season_roles(pg, pos, min_gp)
    backups = roles.filter(
        pl.col("is_big"),
        pl.col("gp") >= min_gp,
        pl.col("player_id") != pl.col("starter_id"),
        pl.col("season_min") < pl.col("starter_min"),
    ).select("season_id", "team_id", "player_id", "starter_id", "season_min")
    status = teammate_status(pg).select(
        "game_id", "team_id", pl.col("mate_id").alias("starter_id"),
        (~pl.col("mate_played")).alias("starter_out"),
    )
    starter_names = pg.select(
        pl.col("player_id").alias("starter_id"), pl.col("player_name").alias("starter_name")
    ).unique("starter_id")
    return (
        pg.join(backups, on=["season_id", "team_id", "player_id"])
        .join(status, on=["game_id", "team_id", "starter_id"], how="left")
        .filter(pl.col("starter_out").is_not_null())  # outside the starter's tenure
        .join(starter_names, on="starter_id", how="left")
    )


RUNGS = [(10, 10), (12, 12), (15, 12)]


def rung_rates(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Games, minutes and P+R rung hit rates, plus tail ratios P(higher rung | 10/10)."""
    hits = [
        ((pl.col("pts") >= p) & (pl.col("reb") >= r)).mean().alias(f"p{p}_{r}") for p, r in RUNGS
    ]
    return (
        df.group_by(by)
        .agg(
            pl.len().alias("n"),
            pl.col("minutes").mean().alias("min"),
            ((pl.col("pts") >= 10) & (pl.col("reb") >= 10)).sum().alias("n10_10"),
            *hits,
        )
        .with_columns(
            (pl.col("p12_12") / pl.col("p10_10")).alias("tail_12"),
            (pl.col("p15_12") / pl.col("p10_10")).alias("tail_15"),
        )
    )
