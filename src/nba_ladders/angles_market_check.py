"""Backtest every angle against historical Sportsbet ladder prices (blind, tails).

    uv run python -m nba_ladders.angles_market_check

For each historical Sportsbet rung (market_backtest.parquet) adds pre-game features:
- luck: season-to-date luck multiplier for the rung's stat (potential assists / reb chances)
- freed: usual minutes of rotation teammates (12+ mpg) who sat that game
- role_trend: last-5 minutes minus last-20 minutes (minutes rising)
- foul: backup bigs only; starter's fouls/36 x opponent FTA per game (both prior-only)
- starter / variance: 28+ mpg over the last 20, and sd of minutes / CV of P+R over the last 20

Then blind ROI of tail rungs (price >= $3) per angle, overall, by half of season (Oct-Jan,
Feb-Apr, pooled across seasons) and by season (2024-25, 2025-26). An angle only counts if it
holds in both seasons.
Caveat: freed minutes use each absent teammate's season-average minutes (mild look-ahead).
"""

from datetime import date

import polars as pl

from nba_ladders import config
from nba_ladders.absences import position_groups
from nba_ladders.luck import BLEND, CHANCES, K_CONV, league_conversion
from nba_ladders.luck import load as load_tracking
from nba_ladders.market_backtest import roi
from nba_ladders.scenarios import positions
from nba_ladders.with_without import teammate_status

TAIL = 3.0
SEASON_SPLIT = date(2025, 8, 1)  # before = 2024-25, after = 2025-26


def rolling_luck(window: int = 10) -> pl.DataFrame:
    """Per player-game, from strictly earlier games: last-`window` luck multiplier for AST and REB
    (blended / actual, blend weights from luck.py) and potential assists per game."""
    trk = load_tracking().sort("player_id", "game_date")
    lg = league_conversion(trk)
    cols = []
    for s, c in CHANCES.items():
        prior_cum = lambda col: pl.col(col).shift(1).cum_sum().over("player_id").fill_null(0)
        win = lambda col: pl.col(col).shift(1).rolling_sum(window, min_samples=window).over("player_id")
        conv = (prior_cum(s) + K_CONV[s] * lg[s]) / (prior_cum(c) + K_CONV[s])
        blended = BLEND[s] * win(c) * conv + (1 - BLEND[s]) * win(s)
        cols.append((blended / win(s).clip(lower_bound=1)).alias(f"luck10_{s}"))
    cols.append((pl.col("potential_ast").shift(1).rolling_mean(window, min_samples=window)
                 .over("player_id")).alias("pot_ast10"))
    return trk.with_columns(cols).select("player_id", "game_date", "luck10_ast", "luck10_reb",
                                         "pot_ast10")


def features() -> pl.DataFrame:
    d = pl.read_parquet(config.DATA_PROCESSED / "market_backtest.parquet")
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").sort("player_id", "game_date")
    prior = lambda c, n: pl.col(c).shift(1).rolling_mean(n, min_samples=min(n, 5)).over("player_id")
    pr = pl.col("pts") + pl.col("reb")
    roll = pg.with_columns(
        prior("minutes", 20).alias("min20"), prior("minutes", 5).alias("min5"),
        pl.col("minutes").shift(1).rolling_std(20, min_samples=10).over("player_id").alias("min_sd"),
        (pr.shift(1).rolling_std(20, min_samples=10) / pr.shift(1).rolling_mean(20, min_samples=10))
        .over("player_id").alias("pr_cv"),
        (pl.col("pf").shift(1).rolling_sum(20, min_samples=10)
         / pl.col("minutes").shift(1).rolling_sum(20, min_samples=10) * 36).over("player_id").alias("pf36"),
    ).select("player_id", "game_id", "game_date", "team", "opponent", "min20", "min5", "min_sd",
             "pr_cv", "pf36")

    # minutes freed by absent rotation teammates (season-average minutes; mild look-ahead)
    mate_min = pg.group_by("season_id", "team_id", "player_id").agg(pl.col("minutes").mean().alias("mate_min"))
    absent = (teammate_status(pg).filter(~pl.col("mate_played"))
              .join(mate_min.rename({"player_id": "mate_id"}), on=["season_id", "team_id", "mate_id"])
              .filter(pl.col("mate_min") >= 12)
              .group_by("game_id", "team_id").agg(pl.col("mate_min").sum().alias("freed")))
    gt = pg.select("game_id", "team_id", "team").unique()
    freed = absent.join(gt, on=["game_id", "team_id"]).select("game_id", "team", "freed")

    # opponent free-throw drawing (team FTA per game over its previous 20 games)
    team_fta = (pg.group_by("game_id", "team", "game_date").agg(pl.col("fta").sum())
                .sort("team", "game_date")
                .with_columns(pl.col("fta").shift(1).rolling_mean(20, min_samples=5).over("team").alias("opp_fta"))
                .select("game_id", pl.col("team").alias("opponent"), "opp_fta"))
    # backup bigs: their season's starting centre's prior foul rate that game
    bg = pl.read_parquet(config.DATA_PROCESSED / "backup_big_games.parquet").select(
        "player_id", "game_id", "starter_id")
    starter_pf = roll.select(pl.col("player_id").alias("starter_id"), "game_id",
                             pl.col("pf36").alias("starter_pf36"))
    out = (d.join(roll, on=["player_id", "game_date"], how="left")
           .join(freed, on=["game_id", "team"], how="left")
           .join(team_fta, on=["game_id", "opponent"], how="left")
           .join(bg, on=["player_id", "game_id"], how="left")
           .join(starter_pf, on=["starter_id", "game_id"], how="left")
           .with_columns(
               pl.col("freed").fill_null(0.0),
               (pl.col("min5") - pl.col("min20")).alias("role_trend"),
               pl.when(pl.col("stat").str.contains("ast") & ~pl.col("stat").str.contains("reb"))
               .then(pl.col("luck_ast")).otherwise(pl.col("luck_reb")).alias("luck_stat"),
               (pl.col("min20") >= 28).alias("starter"),
               pl.col("starter_id").is_not_null().alias("backup_big"),
           ))
    out = (out.join(rolling_luck(), on=["player_id", "game_date"], how="left")
           .join(position_groups(positions()), on="player_id", how="left"))
    q = out.select(pl.col("min_sd").quantile(0.667), pl.col("pr_cv").quantile(0.667),
                   (pl.col("starter_pf36") * pl.col("opp_fta")).quantile(0.667)).row(0)
    return out.with_columns(
        (pl.col("min_sd") >= q[0]).alias("hi_min_var"),
        (pl.col("pr_cv") >= q[1]).alias("hi_pr_cv"),
        (pl.col("backup_big") & (pl.col("starter_pf36") * pl.col("opp_fta") >= q[2])).alias("foul_spot"),
    )


ANGLES = {
    "all tails (baseline)": pl.lit(True),
    "luck >= 1.05": pl.col("luck_stat") >= 1.05,
    "luck >= 1.10": pl.col("luck_stat") >= 1.10,
    "freed >= 20 min": pl.col("freed") >= 20,
    "freed >= 40 min": pl.col("freed") >= 40,
    "role trend >= +4 min": pl.col("role_trend") >= 4,
    "starter out (backup big)": pl.col("starter_out"),
    "starter out x high min var": pl.col("starter_out") & pl.col("hi_min_var"),
    "foul spot (backup big)": pl.col("foul_spot"),
    "starter x high P+R CV": pl.col("starter") & pl.col("hi_pr_cv"),
    "starter x high min var": pl.col("starter") & pl.col("hi_min_var"),
    "starter x high P+R CV, reb/P+R": pl.col("starter") & pl.col("hi_pr_cv") & pl.col("stat").is_in(["reb", "pts+reb"]),
    "freed>=20 & luck>=1.05": (pl.col("freed") >= 20) & (pl.col("luck_stat") >= 1.05),
    # rolling last-10 potentials (works all season, unlike season-to-date)
    "AST rungs, luck10 >= 1.10": (pl.col("stat") == "ast") & (pl.col("luck10_ast") >= 1.10),
    "AST rungs, luck10 >= 1.20": (pl.col("stat") == "ast") & (pl.col("luck10_ast") >= 1.20),
    "AST rungs, guards, pot>=8, luck10>=1.15": (pl.col("stat") == "ast") & (pl.col("grp") == "G")
        & (pl.col("pot_ast10") >= 8) & (pl.col("luck10_ast") >= 1.15),
    "AST rungs, luck10 <= 0.90 (lucky: fade?)": (pl.col("stat") == "ast") & (pl.col("luck10_ast") <= 0.90),
    "REB rungs, luck10 >= 1.10": (pl.col("stat") == "reb") & (pl.col("luck10_reb") >= 1.10),
    "REB/P+R rungs, luck10 >= 1.10": pl.col("stat").is_in(["reb", "pts+reb"]) & (pl.col("luck10_reb") >= 1.10),
}


def table(f: pl.DataFrame, min_price: float = TAIL) -> pl.DataFrame:
    tails = f.filter(pl.col("price") >= min_price)
    rows = []
    for name, cond in ANGLES.items():
        x = tails.filter(cond)
        a = roi(x, 1000)
        if not a["bets"]:
            continue
        early = pl.col("game_date").dt.month().is_in([10, 11, 12, 1])
        h1, h2 = roi(x.filter(early), 500), roi(x.filter(~early), 500)
        s1 = roi(x.filter(pl.col("game_date") < SEASON_SPLIT), 500)
        s2 = roi(x.filter(pl.col("game_date") >= SEASON_SPLIT), 500)
        rows.append({"angle": name, "bets": a["bets"], "hit": a["hit_rate"], "avg_$": a["avg_price"],
                     "roi": a["roi"], "ci90_lo": a["roi_ci90"][0], "ci90_hi": a["roi_ci90"][1],
                     "roi_oct_jan": h1.get("roi"), "n1": h1.get("bets"),
                     "roi_feb_apr": h2.get("roi"), "n2": h2.get("bets"),
                     "roi_2425": s1.get("roi"), "n_2425": s1.get("bets"),
                     "roi_2526": s2.get("roi"), "n_2526": s2.get("bets")})
    return pl.DataFrame(rows)


def main() -> None:
    f = features()
    print(f"{f.height:,} rungs; tails (>= ${TAIL:.0f}): {f.filter(pl.col('price') >= TAIL).height:,}")
    cols = ["angle", "bets", "hit", "avg_$", "roi", "ci90_lo", "ci90_hi", "roi_oct_jan",
            "roi_feb_apr", "roi_2425", "n_2425", "roi_2526", "n_2526"]
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=40, tbl_cols=14, tbl_width_chars=240,
                   fmt_str_lengths=40):
        print(f"\nTAILS (price >= ${TAIL:.0f})")
        print(table(f).select(cols))
        print("\nALL PRICES")
        print(table(f, min_price=1.0).select(cols))


if __name__ == "__main__":
    main()
