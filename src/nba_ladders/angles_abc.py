"""Out-of-sample checks for angles A (conversion), B (tail), C (injury trigger).

Each test asks: does a player's *previous* season predict his *next* season in starter-out
games? If not, the angle may be real on average but you can't pick players for it in advance.
"""

from itertools import pairwise

import polars as pl

from nba_ladders.scenarios import rung_rates

SEASON_ORDER = ["22019", "22020", "22021", "22022", "22023", "22024", "22025"]


def _next_season(col: str) -> pl.Expr:
    mapping = dict(pairwise(SEASON_ORDER))
    return pl.col(col).replace_strict(mapping, default=None)


def player_season_split(bg: pl.DataFrame) -> pl.DataFrame:
    """Per player-season (all teams pooled): rung rates in starter-in and starter-out games."""
    rates = rung_rates(bg, ["player_id", "player_name", "season_id", "starter_out"])
    inn = rates.filter(~pl.col("starter_out")).drop("starter_out")
    out = rates.filter(pl.col("starter_out")).drop("starter_out")
    return inn.join(out, on=["player_id", "player_name", "season_id"], suffix="_out", how="full",
                    coalesce=True).with_columns((pl.col("min_out") - pl.col("min")).alias("min_uplift"))


def season_pairs(ps: pl.DataFrame) -> pl.DataFrame:
    """Join each player-season to the same player's following season."""
    nxt = ps.with_columns(_next_season("season_id").alias("season_id")).filter(
        pl.col("season_id").is_not_null()
    )
    # nxt now holds season t's numbers keyed on t+1; join to actual t+1 numbers.
    return ps.join(nxt, on=["player_id", "player_name", "season_id"], suffix="_prev")


def weighted_corr(df: pl.DataFrame, x: str, y: str) -> float:
    d = df.drop_nulls([x, y]).filter(pl.col(x).is_finite() & pl.col(y).is_finite())
    return d.select(pl.corr(x, y)).item() if d.height > 2 else float("nan")


def tercile_table(df: pl.DataFrame, predictor: str, target: str, weight: str) -> pl.DataFrame:
    """Bucket by predictor terciles; pooled (weighted) target per bucket."""
    d = df.drop_nulls([predictor, target, weight]).filter(pl.col(predictor).is_finite())
    return (
        d.with_columns(
            pl.col(predictor).qcut(3, labels=["low", "mid", "high"]).alias("bucket")
        )
        .group_by("bucket")
        .agg(
            pl.len().alias("players"),
            pl.col(weight).sum().alias("games"),
            pl.col(predictor).mean().alias(f"{predictor}_avg"),
            ((pl.col(target) * pl.col(weight)).sum() / pl.col(weight).sum()).alias(target),
        )
        .sort("bucket")
    )
