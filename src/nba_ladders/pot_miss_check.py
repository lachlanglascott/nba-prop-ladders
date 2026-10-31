"""Backtest the last-game potential-assist angles against Sportsbet assist rungs.

    uv run python -m nba_ladders.pot_miss_check

- pot_miss_ast: monster potentials last game that didn't convert (12+ potentials, 4+ short).
- pot_volume_ast: monster potentials last game, converted or not (volume spike vs his average).

Lineup = 20+ mpg (10+ games) teammates absent; "same" = same absent set as the previous game.
Conversion = career-to-date assists / potentials, shrunk to league. Several variants per angle
were tried (listed below); the chosen ones are marked in candidates.py (mild multiple testing).
"""
from datetime import date

import polars as pl

from nba_ladders import config
from nba_ladders.luck import K_CONV, league_conversion, load
from nba_ladders.market_backtest import roi

SPLIT = date(2025, 8, 1)


def features() -> pl.DataFrame:
    """Per player-game (2024-25, 2025-26, 20+ min last game): last game's potentials, assists,
    shortfall vs conversion, spike vs prior average, and whether the lineup is unchanged."""
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    mpg = pg.group_by("season_id", "team_id", "player_id").agg(pl.col("minutes").mean().alias("mpg"),
                                                               pl.len().alias("gp"))
    rot = mpg.filter(pl.col("mpg") >= 20, pl.col("gp") >= 10)
    games = pg.select("season_id", "team_id", "game_id", "game_date").unique()
    played = pg.filter(pl.col("minutes") > 0).select("game_id", "team_id", "player_id")
    absent = (games.join(rot.select("season_id", "team_id", "player_id"), on=["season_id", "team_id"])
              .join(played, on=["game_id", "team_id", "player_id"], how="anti")
              .group_by("game_id", "team_id").agg(pl.col("player_id").sort().alias("absent")))

    t = load().sort("player_id", "game_date")
    lg = league_conversion(t)
    t = t.join(absent, on=["game_id", "team_id"], how="left").with_columns(pl.col("absent").fill_null([]))
    o = ["player_id", "season_id"]
    cum = lambda c: pl.col(c).cum_sum().over("player_id") - pl.col(c)  # career before this game
    t = t.with_columns(
        ((cum("ast") + K_CONV["ast"] * lg["ast"]) / (cum("potential_ast") + K_CONV["ast"])).alias("conv"),
        pl.col("potential_ast").shift(1).rolling_mean(10, min_samples=3).over(o).alias("pot_avg_prior"),
        pl.col("ast").shift(1).rolling_mean(10, min_samples=3).over(o).alias("ast_avg_prior"),
    )
    t = t.with_columns(
        pl.col("potential_ast").shift(1).over(o).alias("pot_prev"), pl.col("ast").shift(1).over(o).alias("ast_prev"),
        pl.col("conv").shift(1).over(o).alias("conv_prev"),
        pl.col("pot_avg_prior").shift(1).over(o).alias("pot_base"),  # average before the previous game
        pl.col("ast_avg_prior").shift(1).over(o).alias("ast_base"),
        pl.col("absent").shift(1).over(o).alias("absent_prev"),
        pl.col("minutes").shift(1).over(o).alias("min_prev"),
    ).with_columns(
        (pl.col("absent") == pl.col("absent_prev")).alias("same"),
        (pl.col("pot_prev") * pl.col("conv_prev") - pl.col("ast_prev")).alias("short"),
        (pl.col("pot_prev") / pl.col("pot_base")).alias("spike"),
    ).drop_nulls(["pot_prev", "pot_base", "short"])
    return t.filter(pl.col("season_id").is_in(["22024", "22025"]), pl.col("min_prev") >= 20)


same, pot, short, spike = pl.col("same"), pl.col("pot_prev"), pl.col("short"), pl.col("spike")
VARIANTS = {
    "all ast rungs": pl.lit(True),
    # pot_miss_ast
    "miss: pot12 short3 spike1.5": (spike >= 1.5) & (pot >= 12) & (short >= 3),
    "miss: pot12 short3 spike1.5 same": (spike >= 1.5) & (pot >= 12) & (short >= 3) & same,
    "miss: pot10 short3 spike1.3 same": (spike >= 1.3) & (pot >= 10) & (short >= 3) & same,
    "miss: pot12 short4 same (chosen)": (pot >= 12) & (short >= 4) & same,
    # pot_volume_ast
    "vol: pot15 same": (pot >= 15) & same,
    "vol: pot18 same": (pot >= 18) & same,
    "vol: pot15 spike1.5 same": (pot >= 15) & (spike >= 1.5) & same,
    "vol: pot15 spike1.5": (pot >= 15) & (spike >= 1.5),
    "vol: pot18 spike1.5 same": (pot >= 18) & (spike >= 1.5) & same,
    "vol: pot15 spike1.5 same, converted": (pot >= 15) & (spike >= 1.5) & same & (short <= 0),
}


def market(t: pl.DataFrame) -> pl.DataFrame:
    m = pl.read_parquet(config.DATA_PROCESSED / "market_backtest.parquet").filter(pl.col("stat") == "ast")
    return m.join(t.select("player_id", "game_date", "spike", "pot_prev", "short", "same", "ast_prev",
                           "pot_base", "ast_base"), on=["player_id", "game_date"], how="inner")


def main() -> None:
    t = features()
    print(f"player-games {t.height:,}\n")
    for nm, c in VARIANTS.items():
        d = t.filter(c)
        print(f"{nm:40s} n={d.height:5d} next pot {d['potential_ast'].mean():5.2f} (avg {d['pot_base'].mean():5.2f}) "
              f"next ast {d['ast'].mean():5.2f} (avg {d['ast_base'].mean():5.2f})")
    f = market(t)
    print(f"\nSportsbet ast rungs joined: {f.height:,}")
    for nm, c in VARIANTS.items():
        for lo, lab in ((1.0, "all"), (3.0, "$3+")):
            x = f.filter(c, pl.col("price") >= lo)
            a = roi(x, 1000)
            if not a["bets"]:
                continue
            s1, s2 = roi(x.filter(pl.col("game_date") < SPLIT), 300), roi(x.filter(pl.col("game_date") >= SPLIT), 300)
            print(f"{nm:40s} {lab:4s} bets {a['bets']:5d} hit {a['hit_rate']:.2f} avg ${a['avg_price']:.2f} "
                  f"roi {a['roi']:+.1%} ci [{a['roi_ci90'][0]:+.1%},{a['roi_ci90'][1]:+.1%}]  "
                  f"24-25 {s1.get('roi', float('nan')):+.1%} (n {s1.get('bets')})  "
                  f"25-26 {s2.get('roi', float('nan')):+.1%} (n {s2.get('bets')})")


if __name__ == "__main__":
    main()
