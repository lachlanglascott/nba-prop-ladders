"""Angle D: early foul trouble of starting centres, backup-big uplift, and opponent effect.

Pure functions over three tables:
- fouls: personal fouls from play-by-play (see sources/pbp.parse_fouls)
- pg: player-game box table (data/processed/player_games.parquet)
- positions: player_id -> position (from the player index)

Run `uv run python -m nba_ladders.fouls` to print the report tables for 2025-26.
"""

import polars as pl

from nba_ladders.screens import BIG_POSITIONS

Q1_PF = 2  # >= this many PF in Q1 ...
H1_PF = 3  # ... or >= this many before halftime = early foul trouble


def player_game_fouls(fouls: pl.DataFrame) -> pl.DataFrame:
    """Per player-game: PF in Q1, PF in first half, total PF, and the early-trouble flag."""
    return (
        fouls.group_by("game_id", "player_id")
        .agg(
            (pl.col("period") == 1).sum().alias("pf_q1"),
            (pl.col("period") <= 2).sum().alias("pf_h1"),
            pl.len().alias("pf_pbp"),
        )
        .with_columns(
            ((pl.col("pf_q1") >= Q1_PF) | (pl.col("pf_h1") >= H1_PF)).alias("early_trouble")
        )
    )


POSITION_PRIORITY = {"C": 0, "C-F": 1, "F-C": 2}


def starting_centres(
    pg: pl.DataFrame, positions: pl.DataFrame, starters: pl.DataFrame
) -> pl.DataFrame:
    """One row per team-game: the starting centre.

    Among the game's actual starters (inferred from play-by-play), take the most centre-like
    big (C > C-F > F-C), breaking ties by season minutes for that team. Team-games that
    started no C/C-F/F-C are dropped. F-C is included because the player index lists several
    true starting centres that way (Wembanyama, Sabonis, Okongwu, Mobley).
    """
    pri = positions.filter(pl.col("position").is_in(BIG_POSITIONS)).select(
        "player_id", pl.col("position").replace_strict(POSITION_PRIORITY).alias("pos_rank"),
        pl.col("position"),
    )
    season_min = pg.group_by("season_id", "team", "player_id").agg(
        pl.col("minutes").sum().alias("season_minutes")
    )
    return (
        pg.join(starters.select("game_id", "team", "player_id"), on=["game_id", "team", "player_id"])
        .join(pri, on="player_id")
        .join(season_min, on=["season_id", "team", "player_id"])
        .sort(["pos_rank", "season_minutes"], descending=[False, True])
        .group_by("game_id", "team", maintain_order=True)
        .first()
        .select(
            "season_id", "game_id", "game_date", "team", "opponent",
            pl.col("player_id").alias("starter_id"),
            pl.col("player_name").alias("starter_name"),
            pl.col("position").alias("starter_pos"),
            pl.col("minutes").alias("starter_min"),
            pl.col("pf").alias("starter_pf"),
        )
    )


def flag_starters(starters: pl.DataFrame, pgf: pl.DataFrame) -> pl.DataFrame:
    """Attach the starter's foul timing; starters with no PBP fouls get zeros."""
    out = starters.join(
        pgf.rename({"player_id": "starter_id"}), on=["game_id", "starter_id"], how="left"
    )
    return out.with_columns(
        pl.col("pf_q1", "pf_h1", "pf_pbp").fill_null(0),
        pl.col("early_trouble").fill_null(False),
    )


def centre_rates(flagged: pl.DataFrame, min_games: int = 20) -> pl.DataFrame:
    """Early-foul-trouble rate per starting centre (team-specific stints)."""
    return (
        flagged.group_by("starter_id", "starter_name", "team")
        .agg(
            pl.len().alias("games"),
            pl.col("early_trouble").sum().alias("trouble_games"),
            pl.col("early_trouble").mean().alias("trouble_rate"),
            pl.col("starter_min").mean().alias("min_avg"),
            pl.col("starter_min").filter(pl.col("early_trouble")).mean().alias("min_trouble"),
            pl.col("starter_min").filter(~pl.col("early_trouble")).mean().alias("min_clean"),
        )
        .filter(pl.col("games") >= min_games)
        .sort("trouble_rate", descending=True)
    )


def backup_games(
    pg: pl.DataFrame, positions: pl.DataFrame, flagged: pl.DataFrame, starters: pl.DataFrame
) -> pl.DataFrame:
    """Bench bigs (C/C-F/F-C who did not start) in each team-game, with the starter's flag.

    pg is a box table, so DNPs are absent: a backup who never got in is not counted (see
    caveats in docs/angle_d_fouls.md).
    """
    bigs = positions.filter(pl.col("position").is_in(BIG_POSITIONS)).select("player_id")
    return (
        pg.join(bigs, on="player_id")
        .join(starters.select("game_id", "team", "player_id"),
              on=["game_id", "team", "player_id"], how="anti")
        .join(
            flagged.select("game_id", "team", "starter_id", "early_trouble"),
            on=["game_id", "team"],
        )
        .filter(pl.col("player_id") != pl.col("starter_id"))
        .with_columns(((pl.col("pts") >= 10) & (pl.col("reb") >= 10)).alias("dd"))
    )


def primary_backups(bg: pl.DataFrame) -> pl.DataFrame:
    """Keep only each team-season's main backup big (most season minutes among backups)."""
    top = (
        bg.group_by("season_id", "team", "player_id")
        .agg(pl.col("minutes").sum().alias("m"))
        .sort("m", descending=True)
        .group_by("season_id", "team", maintain_order=True)
        .first()
        .select("season_id", "team", "player_id")
    )
    return bg.join(top, on=["season_id", "team", "player_id"])


def backup_split(bg: pl.DataFrame) -> pl.DataFrame:
    """Backup-big outcomes when the starter had early foul trouble vs not (pooled)."""
    return (
        bg.group_by("early_trouble")
        .agg(
            pl.len().alias("games"),
            pl.col("minutes").mean().alias("min_avg"),
            pl.col("pts").mean().alias("pts_avg"),
            pl.col("reb").mean().alias("reb_avg"),
            pl.col("dd").mean().alias("p_10_10"),
            pl.col("dd").sum().alias("n_10_10"),
        )
        .sort("early_trouble")
    )


def within_player_uplift(bg: pl.DataFrame, min_trouble_games: int = 3) -> pl.DataFrame:
    """Per backup: minutes/P(10-10) with starter in trouble minus clean. Removes mix effects."""
    t, c = pl.col("early_trouble"), ~pl.col("early_trouble")
    return (
        bg.group_by("player_id", "player_name", "team")
        .agg(
            t.sum().alias("n_trouble"),
            c.sum().alias("n_clean"),
            pl.col("minutes").filter(t).mean().alias("min_trouble"),
            pl.col("minutes").filter(c).mean().alias("min_clean"),
            pl.col("dd").filter(t).mean().alias("p1010_trouble"),
            pl.col("dd").filter(c).mean().alias("p1010_clean"),
        )
        .filter(pl.col("n_trouble") >= min_trouble_games, pl.col("n_clean") >= min_trouble_games)
        .with_columns((pl.col("min_trouble") - pl.col("min_clean")).alias("min_uplift"))
        .sort("min_uplift", descending=True)
    )


def severity_split(bg: pl.DataFrame, flagged: pl.DataFrame) -> pl.DataFrame:
    """Starter and primary-backup minutes by the starter's first-half PF count (0,1,2,3,4+)."""
    bucket = pl.col("pf_h1").clip(upper_bound=4)
    st = flagged.group_by(bucket.alias("starter_pf_h1")).agg(
        pl.len().alias("team_games"), pl.col("starter_min").mean().alias("starter_min")
    )
    bk = (
        bg.join(flagged.select("game_id", "team", "pf_h1"), on=["game_id", "team"])
        .group_by(bucket.alias("starter_pf_h1"))
        .agg(
            pl.len().alias("backup_games"),
            pl.col("minutes").mean().alias("backup_min"),
            pl.col("dd").mean().alias("backup_p1010"),
        )
    )
    return st.join(bk, on="starter_pf_h1", how="left").sort("starter_pf_h1")


def within_starter_minutes(flagged: pl.DataFrame, min_trouble_games: int = 3) -> float:
    """Trouble-games-weighted mean of (starter minutes in trouble - clean), per starter."""
    t = pl.col("early_trouble")
    d = (
        flagged.group_by("starter_id", "team")
        .agg(
            t.sum().alias("n"),
            (~t).sum().alias("n_clean"),
            (pl.col("starter_min").filter(t).mean() - pl.col("starter_min").filter(~t).mean())
            .alias("delta"),
        )
        .filter(pl.col("n") >= min_trouble_games, pl.col("n_clean") >= min_trouble_games)
    )
    return (d["delta"] * d["n"]).sum() / d["n"].sum() if d.height else float("nan")


def opponent_effect(flagged: pl.DataFrame) -> pl.DataFrame:
    """Per opponent: how often the starting centre facing them got into early foul trouble.

    Also split by half of season (by date midpoint) for a persistence check, and a z-score vs
    the league rate using the binomial SE for that opponent's game count.
    """
    mid = flagged["game_date"].sort()[flagged.height // 2]
    league = flagged["early_trouble"].mean()
    t = pl.col("early_trouble")
    first = pl.col("game_date") < mid
    return (
        flagged.group_by("opponent")
        .agg(
            pl.len().alias("games"),
            t.sum().alias("trouble_games"),
            t.mean().alias("rate"),
            t.filter(first).mean().alias("rate_h1"),
            first.sum().alias("games_h1"),
            t.filter(~first).mean().alias("rate_h2"),
            (~first).sum().alias("games_h2"),
        )
        .with_columns(
            (pl.col("rate") - league).alias("vs_league"),
            ((pl.col("rate") - league) / (league * (1 - league) / pl.col("games")).sqrt())
            .alias("z"),
        )
        .sort("rate", descending=True)
    )


def persistence(by_group: pl.DataFrame) -> dict:
    """First-half vs second-half correlation of rates, plus observed vs binomial-noise spread."""
    if by_group.height < 3:
        return {"groups": by_group.height}
    r = by_group.select(pl.corr("rate_h1", "rate_h2")).item()
    p = by_group["rate"].mean()
    n = by_group["games"].mean()
    return {
        "groups": by_group.height,
        "corr_h1_h2": r,
        "sd_observed": by_group["rate"].std(),
        "sd_binomial_noise": (p * (1 - p) / n) ** 0.5,
    }


def centre_persistence(flagged: pl.DataFrame, min_games_half: int = 15) -> dict:
    """Same split-half check for the centres themselves (is foul-proneness a trait?)."""
    mid = flagged["game_date"].sort()[flagged.height // 2]
    first = pl.col("game_date") < mid
    t = pl.col("early_trouble")
    g = (
        flagged.group_by("starter_id", "team")
        .agg(
            pl.len().alias("games"),
            t.mean().alias("rate"),
            t.filter(first).mean().alias("rate_h1"),
            first.sum().alias("n1"),
            t.filter(~first).mean().alias("rate_h2"),
            (~first).sum().alias("n2"),
        )
        .filter(pl.col("n1") >= min_games_half, pl.col("n2") >= min_games_half)
    )
    return persistence(g)


def _main() -> None:  # pragma: no cover - report runner over local data
    import json

    from nba_ladders import config
    from nba_ladders.screens import parse_player_index
    from nba_ladders.sources.nba_stats import get_player_index

    pl.Config.set_tbl_rows(40)
    pl.Config.set_tbl_cols(20)
    pl.Config.set_fmt_str_lengths(30)
    season_id = "22025"
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(
        pl.col("season_id") == season_id
    )
    fouls = pl.read_parquet(config.DATA_PROCESSED / f"fouls_{season_id}.parquet")
    pg = pg.filter(pl.col("game_id").is_in(fouls["game_id"].unique().implode()))
    # Season-appropriate positions, falling back to the newest index for anyone missing.
    idx = pl.concat([
        parse_player_index(get_player_index("2025-26")).with_columns(pl.lit(0).alias("o")),
        parse_player_index(get_player_index("2026-27")).with_columns(pl.lit(1).alias("o")),
    ]).sort("o").unique("player_id", keep="first")

    pgf = player_game_fouls(fouls)
    chk = pg.join(pgf, on=["game_id", "player_id"], how="left").with_columns(
        pl.col("pf_pbp").fill_null(0)
    )
    print("PBP PF == box PF:", (chk["pf_pbp"] == chk["pf"]).mean(), "rows", chk.height)

    starters = pl.read_parquet(config.DATA_PROCESSED / f"starters_{season_id}.parquet")
    flagged = flag_starters(starting_centres(pg, idx, starters), pgf)
    n_tg = pg.select("game_id", "team").unique().height
    print("team-games", n_tg, "with a big starter", flagged.height)
    print(flagged["starter_pos"].value_counts())
    print("team-games", flagged.height, "league early-trouble rate",
          round(flagged["early_trouble"].mean(), 4))
    print(flagged.group_by("early_trouble").agg(pl.len(), pl.col("starter_min").mean()))
    print(centre_rates(flagged))

    bg = backup_games(pg, idx, flagged, starters)
    print("ALL backup bigs"); print(backup_split(bg))
    pb = primary_backups(bg)
    print("PRIMARY backup"); print(backup_split(pb))
    wpu = within_player_uplift(pb)
    print(wpu)
    print("within-player mean uplift (weighted by trouble n):",
          (wpu["min_uplift"] * wpu["n_trouble"]).sum() / wpu["n_trouble"].sum())

    print("within-starter minutes delta (trouble - clean):", within_starter_minutes(flagged))
    print("severity split (primary backup)"); print(severity_split(pb, flagged))

    opp = opponent_effect(flagged)
    print(opp)
    print("opponent persistence", json.dumps(persistence(opp), indent=1))
    print("centre persistence", json.dumps(centre_persistence(flagged), indent=1))

    # Does the opponent's FT-drawing (box FTA per game) explain the opponent rate?
    fta = pg.group_by("team").agg((pl.col("fta").sum() / pl.col("game_id").n_unique())
                                  .alias("fta_pg"))
    j = opp.join(fta, left_on="opponent", right_on="team")
    print("corr(opponent FTA/g, induced trouble rate):",
          j.select(pl.corr("fta_pg", "rate")).item())
    # Out-of-sample: FTA/g from first-half games vs induced trouble rate in second half.
    mid = flagged["game_date"].sort()[flagged.height // 2]
    fta1 = pg.filter(pl.col("game_date") < mid).group_by("team").agg(
        (pl.col("fta").sum() / pl.col("game_id").n_unique()).alias("fta_h1"))
    j2 = opp.join(fta1, left_on="opponent", right_on="team")
    print("split date", mid, "corr(FTA/g H1, rate H2):",
          j2.select(pl.corr("fta_h1", "rate_h2")).item(),
          "corr(rate H1, rate H2):", j2.select(pl.corr("rate_h1", "rate_h2")).item())


if __name__ == "__main__":
    _main()
