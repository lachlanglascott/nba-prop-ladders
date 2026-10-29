"""Rebounding vs opponent and by position: is there a persistent opponent effect?

Expected rebounds per player-game = his offensive/defensive rebound rate from strictly earlier
games (shrunk toward his position group's league rate) x his actual minutes. For each opponent,
season and position group (G / W / B), compare actual vs expected. A real opponent effect must
persist: first half of the season should predict the second half.

Also tests the "rebound environment": does an opponent's miss volume (FGA - FGM, a mix of pace
and shooting) predict extra defensive rebounds for the players facing them?

    uv run python -m nba_ladders.rebounding
"""

import polars as pl

from nba_ladders import config
from nba_ladders.absences import position_groups
from nba_ladders.scenarios import positions

K_MIN = 150.0  # pseudo-minutes of position-group rate in each player's prior rate
K_OPP = 150.0  # pseudo-expected-rebounds of "no effect" when shrinking opponent factors
SIDES = ("oreb", "dreb")


def expected_rebounds(pg: pl.DataFrame, pos: pl.DataFrame) -> pl.DataFrame:
    """Add exp_oreb / exp_dreb: as-at per-minute rate (shrunk to position) x actual minutes."""
    df = pg.join(position_groups(pos), on="player_id", how="left").with_columns(
        pl.col("grp").fill_null("W")
    ).sort("player_id", "game_date")
    league = df.group_by("grp").agg(
        *[(pl.col(s).sum() / pl.col("minutes").sum()).alias(f"lg_{s}") for s in SIDES]
    )
    df = df.join(league, on="grp")
    prior = lambda c: pl.col(c).shift(1).cum_sum().over("player_id").fill_null(0)
    return df.with_columns(
        *[(((prior(s) + K_MIN * pl.col(f"lg_{s}")) / (prior("minutes") + K_MIN)) * pl.col("minutes"))
          .alias(f"exp_{s}") for s in SIDES],
        prior("minutes").alias("prior_min"),
    )


def season_halves(df: pl.DataFrame) -> pl.DataFrame:
    mid = df.group_by("season_id").agg(pl.col("game_date").median().alias("mid"))
    return df.join(mid, on="season_id").with_columns(
        pl.when(pl.col("game_date") < pl.col("mid")).then(pl.lit("H1")).otherwise(pl.lit("H2")).alias("half")
    )


def opponent_factors(df: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Actual / expected rebounds allowed, shrunk toward 1, per (opponent, *by)."""
    keys = ["opponent", *by]
    return df.group_by(keys).agg(
        *[pl.col(s).sum().alias(f"act_{s}") for s in SIDES],
        *[pl.col(f"exp_{s}").sum().alias(f"exp_{s}") for s in SIDES],
    ).with_columns(
        *[((pl.col(f"act_{s}") + K_OPP) / (pl.col(f"exp_{s}") + K_OPP)).alias(f"f_{s}") for s in SIDES]
    )


def persistence(df: pl.DataFrame, side: str, by: list[str]) -> dict:
    """Corr of opponent factors between season halves, and spread vs pure noise."""
    f = opponent_factors(df, ["season_id", "half", *by])
    w = f.pivot(on="half", index=["opponent", "season_id", *by], values=[f"f_{side}", f"exp_{side}"])
    w = w.drop_nulls()
    corr = w.select(pl.corr(f"f_{side}_H1", f"f_{side}_H2")).item()
    # binomial-ish noise on a ratio with ~E expected rebounds: sd ~ 1/sqrt(E)
    noise = (1 / w[f"exp_{side}_H1"]).mean() ** 0.5
    return {"groups": w.height, "corr_h1_h2": round(corr, 3),
            "sd_observed": round(w[f"f_{side}_H1"].std(), 3), "sd_noise_approx": round(noise, 3)}


def environment(pg: pl.DataFrame, df: pl.DataFrame) -> dict:
    """Do opponents' misses per game (pace x shooting) predict extra DREB vs expected?"""
    team = pg.group_by("season_id", "team", "game_id").agg(
        (pl.col("fga") - pl.col("fgm")).sum().alias("misses"))
    opp_miss = season_halves(team.join(pg.select("game_id", "game_date").unique("game_id"), on="game_id")) \
        .group_by("season_id", "team", "half").agg(pl.col("misses").mean().alias("opp_misses_pg")) \
        .rename({"team": "opponent"})
    f = opponent_factors(df, ["season_id", "half"]).join(opp_miss, on=["season_id", "opponent", "half"])
    h1 = f.filter(pl.col("half") == "H1").select("season_id", "opponent", pl.col("opp_misses_pg").alias("miss_h1"))
    h2 = f.filter(pl.col("half") == "H2").select("season_id", "opponent", "f_dreb")
    j = h1.join(h2, on=["season_id", "opponent"])
    return {
        "same_half_corr(misses, dreb factor)": round(f.select(pl.corr("opp_misses_pg", "f_dreb")).item(), 3),
        "H1 misses -> H2 dreb factor corr": round(j.select(pl.corr("miss_h1", "f_dreb")).item(), 3),
        "team-halves": j.height,
    }


def main() -> None:
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(
        pl.col("season_id") != "22026")
    df = season_halves(expected_rebounds(pg, positions())).filter(pl.col("prior_min") >= 300)
    print(f"{df.height:,} player-games with 300+ prior minutes")
    for side in SIDES:
        print(f"\n{side.upper()} opponent effect, season halves:")
        print("  all positions:", persistence(df, side, []))
        print("  by position:  ", persistence(df, side, ["grp"]))
    print("\nrebound environment:", environment(pg, df))
    cur = opponent_factors(df.filter(pl.col("season_id") == "22025"), ["grp"])
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=12):
        for g in ["B", "W", "G"]:
            t = cur.filter(pl.col("grp") == g).sort("f_dreb", descending=True)
            print(f"\n2025-26 DREB allowed to {g} vs expected (top/bottom 4):")
            print(pl.concat([t.head(4), t.tail(4)]).select("opponent", pl.col("act_dreb").round(0),
                  pl.col("exp_dreb").round(0), pl.col("f_dreb").round(3), pl.col("f_oreb").round(3)))


if __name__ == "__main__":
    main()
