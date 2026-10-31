"""Backtest: monster potential assists last game that didn't convert (the pot_miss_ast angle).

    uv run python -m nba_ladders.pot_miss_check

Lineup = 20+ mpg (10+ games) teammates absent; "same" = same absent set as the previous game.
Conversion = career-to-date assists / potentials, shrunk to league. Market = Sportsbet ast rungs.
Five variants were tried; pot12 short4 same was chosen (mild multiple testing).
"""
from datetime import date

import polars as pl

from nba_ladders import config
from nba_ladders.luck import K_CONV, league_conversion, load
from nba_ladders.market_backtest import roi

pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
# lineup: absent 20+ mpg (season) teammates per team-game
mpg = pg.group_by("season_id", "team_id", "player_id").agg(pl.col("minutes").mean().alias("mpg"), pl.len().alias("gp"))
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
)
t = t.with_columns(
    pl.col("potential_ast").shift(1).over(o).alias("pot_prev"), pl.col("ast").shift(1).over(o).alias("ast_prev"),
    pl.col("conv").shift(1).over(o).alias("conv_prev"),
    pl.col("pot_avg_prior").shift(1).over(o).alias("pot_base"),  # average before the previous game
    pl.col("absent").shift(1).over(o).alias("absent_prev"),
    pl.col("minutes").shift(1).over(o).alias("min_prev"),
).with_columns(
    (pl.col("absent") == pl.col("absent_prev")).alias("same"),
    (pl.col("pot_prev") * pl.col("conv_prev") - pl.col("ast_prev")).alias("short"),
    (pl.col("pot_prev") / pl.col("pot_base")).alias("spike"),
).drop_nulls(["pot_prev", "pot_base", "short"])
t = t.filter(pl.col("season_id").is_in(["22024", "22025"]), pl.col("min_prev") >= 20)
print("player-games", t.height)
def show(name, cond):
    d = t.filter(cond)
    print(f"{name:52s} n={d.height:5d} next pot {d['potential_ast'].mean():5.2f} (base {d['pot_base'].mean():5.2f}) "
          f"next ast {d['ast'].mean():5.2f} vs exp-from-base {(d['pot_base']*d['conv']).mean():5.2f}")
for nm, c in {
    "all": pl.lit(True),
    "spike>=1.5 & pot>=12": (pl.col("spike") >= 1.5) & (pl.col("pot_prev") >= 12),
    "spike>=1.5 & pot>=12 & short>=3": (pl.col("spike") >= 1.5) & (pl.col("pot_prev") >= 12) & (pl.col("short") >= 3),
    "spike>=1.5 & pot>=12 & short>=3 & same": (pl.col("spike") >= 1.5) & (pl.col("pot_prev") >= 12) & (pl.col("short") >= 3) & pl.col("same"),
    "spike>=1.3 & pot>=10 & short>=3 & same": (pl.col("spike") >= 1.3) & (pl.col("pot_prev") >= 10) & (pl.col("short") >= 3) & pl.col("same"),
    "short>=4 (any)": pl.col("short") >= 4,
}.items():
    show(nm, c)

m = pl.read_parquet(config.DATA_PROCESSED / "market_backtest.parquet").filter(pl.col("stat") == "ast")
f = m.join(t.select("player_id", "game_date", "spike", "pot_prev", "short", "same", "ast_prev", "pot_base"),
           on=["player_id", "game_date"], how="inner")
print("\nmarket ast rungs joined", f.height)
split = date(2025, 8, 1)
for nm, c in {
    "all ast rungs": pl.lit(True),
    "spike1.5 pot12 short3": (pl.col("spike") >= 1.5) & (pl.col("pot_prev") >= 12) & (pl.col("short") >= 3),
    "spike1.5 pot12 short3 same": (pl.col("spike") >= 1.5) & (pl.col("pot_prev") >= 12) & (pl.col("short") >= 3) & pl.col("same"),
    "spike1.3 pot10 short3 same": (pl.col("spike") >= 1.3) & (pl.col("pot_prev") >= 10) & (pl.col("short") >= 3) & pl.col("same"),
    "pot12 short4 same": (pl.col("pot_prev") >= 12) & (pl.col("short") >= 4) & pl.col("same"),
}.items():
    for pr, lab in ((1.0, "all prices"), (3.0, "tails $3+")):
        x = f.filter(c, pl.col("price") >= pr)
        a = roi(x, 1000)
        if not a["bets"]:
            continue
        s1, s2 = roi(x.filter(pl.col("game_date") < split), 300), roi(x.filter(pl.col("game_date") >= split), 300)
        print(f"{nm:30s} {lab:10s} bets {a['bets']:5d} hit {a['hit_rate']:.2f} avg ${a['avg_price']:.2f} roi {a['roi']:+.1%} "
              f"ci [{a['roi_ci90'][0]:+.1%},{a['roi_ci90'][1]:+.1%}]  24-25 {s1.get('roi', float('nan')):+.1%} (n {s1.get('bets')})  25-26 {s2.get('roi', float('nan')):+.1%} (n {s2.get('bets')})")
