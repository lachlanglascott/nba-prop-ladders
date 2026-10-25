"""Teammate-absence effects on a player's minutes and per-minute rates (PPM / RPM / APM).

For each (player, teammate) pair: how the player's minutes and per-minute production change
in games the teammate missed while rostered (with/without, pooled over seasons they shared).
Pairs with few games are shrunk toward a league prior by position group, scaled by how many
minutes the absent teammate normally plays. That prior is also the estimate for pairs with no
shared history (rookies, new signings).

A game's context = the set of the player's rostered teammates who sat. Effects of several
absences add for minutes (capped) and multiply for rates (capped).
"""

import polars as pl

from nba_ladders.with_without import teammate_status, with_without

GROUPS = {"G": "G", "G-F": "G", "F-G": "W", "F": "W", "F-C": "B", "C-F": "B", "C": "B"}
RATES = ["pts", "reb", "ast"]
K_MIN = 8.0  # pseudo-games of prior for minutes effects
K_RATE = 20.0  # rates are noisier
MIN_BASE_MINUTES = 8.0  # ignore pairs where the player barely plays even with the mate in
MIN_MATE_MINUTES = 12.0  # a fringe player sitting frees ~nothing and his absence is mostly noise
MAX_SHIFT = 18.0  # cap on summed minutes unlocked in one game
# Raw pair effects overshoot ~2.5x. Scale chosen on 2024-25, validated once on 2025-26
# (docs/context_model.md).
EFFECT_SCALE = 0.4
RATE_MULT_CLIP = (0.7, 1.4)


def position_groups(pos: pl.DataFrame) -> pl.DataFrame:
    return pos.select(
        "player_id", pl.col("position").replace_strict(GROUPS, default="W").alias("grp")
    )


def mate_minutes(pg: pl.DataFrame) -> pl.DataFrame:
    """Each player's average minutes per (season, team) when he plays: what his absence frees."""
    return pg.group_by("season_id", "team_id", "player_id").agg(
        pl.col("minutes").mean().alias("mate_min")
    ).rename({"player_id": "mate_id"})


def pair_effects(pg: pl.DataFrame, pos: pl.DataFrame) -> pl.DataFrame:
    """Raw (unshrunk) pair effects pooled across shared seasons, with position groups."""
    ww = with_without(pg, min_games_out=1).join(
        mate_minutes(pg), on=["season_id", "team_id", "mate_id"]
    ).filter(pl.col("mate_min") >= MIN_MATE_MINUTES)
    w = pl.col("games_mate_out")
    rate = lambda s, suf="": pl.col(f"{s}{suf}") / pl.col(f"minutes{suf}")
    pooled = (
        ww.filter(pl.col("minutes") >= MIN_BASE_MINUTES, pl.col("minutes_mate_out") > 0)
        .with_columns(
            (pl.col("minutes_mate_out") - pl.col("minutes")).alias("d_min"),
            *[(rate(s, "_mate_out") / rate(s) - 1).fill_nan(0).alias(f"rel_{s}") for s in RATES],
        )
        .with_columns(*[pl.col(f"rel_{s}").clip(-0.6, 1.0) for s in RATES])
        .group_by("player_id", "mate_id")
        .agg(
            w.sum().alias("n_out"),
            ((pl.col("d_min") * w).sum() / w.sum()).alias("d_min"),
            ((pl.col("mate_min") * w).sum() / w.sum()).alias("mate_min"),
            *[((pl.col(f"rel_{s}") * w).sum() / w.sum()).alias(f"rel_{s}") for s in RATES],
        )
    )
    groups = position_groups(pos)
    return pooled.join(groups, on="player_id", how="left").join(
        groups.rename({"player_id": "mate_id", "grp": "mate_grp"}), on="mate_id", how="left"
    ).with_columns(pl.col("grp", "mate_grp").fill_null("W"))


def priors(effects: pl.DataFrame) -> pl.DataFrame:
    """League prior per (player group, mate group): effect per minute the mate normally plays."""
    w = pl.col("n_out")
    per = lambda c: ((pl.col(c) * w).sum() / (pl.col("mate_min") * w).sum())
    return effects.group_by("grp", "mate_grp").agg(
        per("d_min").alias("b_min"), *[per(f"rel_{s}").alias(f"b_{s}") for s in RATES]
    )


def shrunk_effects(effects: pl.DataFrame, pri: pl.DataFrame) -> pl.DataFrame:
    """Pair effects shrunk toward the prior: (n * raw + K * prior) / (n + K)."""
    e = effects.join(pri, on=["grp", "mate_grp"], how="left")
    shrink = lambda raw, prior, k: (pl.col("n_out") * raw + k * prior) / (pl.col("n_out") + k)
    return e.select(
        "player_id", "mate_id", "n_out", "mate_min",
        shrink(pl.col("d_min"), pl.col("b_min") * pl.col("mate_min"), K_MIN).alias("eff_min"),
        *[shrink(pl.col(f"rel_{s}"), pl.col(f"b_{s}") * pl.col("mate_min"), K_RATE)
          .alias(f"eff_{s}") for s in RATES],
    )


def game_absences(pg: pl.DataFrame) -> pl.DataFrame:
    """Per (game, team): each rostered teammate who sat, with his usual minutes."""
    st = teammate_status(pg).filter(~pl.col("mate_played"))
    return (
        st.join(mate_minutes(pg), on=["season_id", "team_id", "mate_id"])
        .filter(pl.col("mate_min") >= MIN_MATE_MINUTES)
        .select("game_id", "team_id", "mate_id", "mate_min")
    )


def context_effects(
    player_games: pl.DataFrame,
    absent: pl.DataFrame,
    eff: pl.DataFrame,
    pri: pl.DataFrame,
    pos: pl.DataFrame,
    scale: float | None = None,
) -> pl.DataFrame:
    """For each player-game row, the combined effect of the teammates who sat.

    player_games needs player_id, game_id, team_id. absent: game_absences() rows (or a
    hand-built set for a future game). Pairs with no history fall back to the prior.
    Returns player_id, game_id, shift_min, mult_pts, mult_reb, mult_ast.

    Per-pair minutes effects are floored at 0 (negative ones were noise in the backtest);
    several absences combine with diminishing returns: MAX_SHIFT * tanh(sum / MAX_SHIFT).
    """
    a = EFFECT_SCALE if scale is None else scale
    groups = position_groups(pos)
    rows = (
        player_games.select("player_id", "game_id", "team_id")
        .join(absent, on=["game_id", "team_id"])
        .filter(pl.col("player_id") != pl.col("mate_id"))
        .join(groups, on="player_id", how="left")
        .join(groups.rename({"player_id": "mate_id", "grp": "mate_grp"}), on="mate_id", how="left")
        .with_columns(pl.col("grp", "mate_grp").fill_null("W"))
        .join(pri, on=["grp", "mate_grp"], how="left")
        .join(eff.drop("mate_min"), on=["player_id", "mate_id"], how="left")
        .with_columns(
            (a * pl.coalesce("eff_min", pl.col("b_min") * pl.col("mate_min")))
            .clip(lower_bound=0).alias("e_min"),
            *[(a * pl.coalesce(f"eff_{s}", pl.col(f"b_{s}") * pl.col("mate_min")))
              .alias(f"e_{s}") for s in RATES],
        )
    )
    out = rows.group_by("player_id", "game_id").agg(
        (MAX_SHIFT * (pl.col("e_min").sum() / MAX_SHIFT).tanh()).alias("shift_min"),
        *[(1 + pl.col(f"e_{s}")).product().clip(*RATE_MULT_CLIP).alias(f"mult_{s}") for s in RATES],
        pl.len().alias("n_absent"),
        pl.col("mate_min").sum().alias("freed_min"),
    )
    base = player_games.select("player_id", "game_id").unique()
    return base.join(out, on=["player_id", "game_id"], how="left").with_columns(
        pl.col("shift_min").fill_null(0.0),
        *[pl.col(f"mult_{s}").fill_null(1.0) for s in RATES],
        pl.col("n_absent").fill_null(0),
        pl.col("freed_min").fill_null(0.0),
    )
