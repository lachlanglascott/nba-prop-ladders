"""Back-to-back rest: who sits the second night, and is tonight one?

For every second game of a team back-to-back (played the previous calendar day) where the
player played the first leg: did he play? Compared with his miss rate in other games after he
played the team's previous game (injuries, illness), the excess is rest. Seasons are weighted
toward the recent ones and shrunk toward the league rate, so one skipped B2B doesn't make a
rester. Game logs only list players who appeared, so "missed" = not in the log.

    uv run python -m nba_ladders.rest [--date YYYY-MM-DD]   # league resters + tonight's B2B risks
"""

import argparse
from datetime import date, timedelta

import polars as pl

from nba_ladders import config

SEASONS_BACK = 3
SEASON_WEIGHTS = {0: 1.0, 1: 0.7, 2: 0.4}  # this season, last, the one before
K_B2B = 2.0  # pseudo back-to-backs at his usual miss rate + the league rest excess
MIN_MPG = 24.0  # only starters/rotation players get a rest rate
SIT_OUT, SIT_WATCH = 0.70, 0.50  # tonight: treat as out / flag as a watch (0.3-0.5 over-predicted)


def player_legs(pg: pl.DataFrame) -> pl.DataFrame:
    """Per player and team game after one he played: was it a B2B second leg, and did he play?"""
    tg = pg.select("season_id", "team", "game_id", "game_date").unique().sort("team", "game_date")
    tg = tg.with_columns(
        pl.col("game_id").shift(1).over("team", "season_id").alias("prev_game"),
        ((pl.col("game_date") - pl.col("game_date").shift(1).over("team", "season_id")).dt.total_days() == 1)
        .alias("b2b"))
    played = pg.filter(pl.col("minutes") > 0).select("player_id", "team", "game_id", "minutes")
    # every (player, team game) where he played the team's previous game
    legs = played.rename({"game_id": "prev_game", "minutes": "prev_min"}).join(
        tg.drop_nulls("prev_game"), on=["team", "prev_game"])
    return legs.join(played.select("player_id", "game_id", pl.lit(True).alias("played")),
                     on=["player_id", "game_id"], how="left").with_columns(pl.col("played").fill_null(False))


def rest_rates(pg: pl.DataFrame, as_of: date, season_id: str) -> pl.DataFrame:
    """Per player: P(sits a B2B second leg | played the first), excess over his usual miss rate."""
    sid = int(season_id)
    keep = [str(sid - i) for i in range(SEASONS_BACK)]
    hist = pg.filter(pl.col("game_date") < as_of, pl.col("season_id").is_in(keep))
    legs = player_legs(hist).with_columns(
        pl.col("season_id").cast(pl.Int64).map_elements(lambda s: SEASON_WEIGHTS.get(sid - s, 0.0),
                                                        return_dtype=pl.Float64).alias("w"))
    mpg = hist.group_by("player_id").agg(pl.col("minutes").mean().alias("mpg"),
                                         pl.col("player_name").last())
    lg_b2b = legs.filter(pl.col("b2b"))
    league = float((~lg_b2b["played"]).mean() - (~legs.filter(~pl.col("b2b"))["played"]).mean())
    agg = legs.group_by("player_id").agg(
        (pl.col("w") * pl.col("b2b")).sum().alias("n_b2b"),
        (pl.col("w") * (pl.col("b2b") & ~pl.col("played"))).sum().alias("sat_b2b"),
        (pl.col("w") * ~pl.col("b2b")).sum().alias("n_other"),
        (pl.col("w") * (~pl.col("b2b") & ~pl.col("played"))).sum().alias("sat_other"),
        pl.col("b2b").sum().alias("b2b_games"), (pl.col("b2b") & ~pl.col("played")).sum().alias("b2b_sat"))
    base = (pl.col("sat_other") / pl.col("n_other").clip(lower_bound=1))
    prior = (base + league).clip(0, 1)
    return (agg.join(mpg, on="player_id").filter(pl.col("mpg") >= MIN_MPG)
            .with_columns(base.alias("usual_miss"),
                          ((pl.col("sat_b2b") + K_B2B * prior) / (pl.col("n_b2b") + K_B2B)).clip(0, 1).alias("p_sit_b2b"))
            .with_columns((pl.col("p_sit_b2b") - pl.col("usual_miss")).alias("rest_excess"))
            .sort("p_sit_b2b", descending=True))


def tonight(pg: pl.DataFrame, day: date, teams: list[str], season_id: str = "22026") -> pl.DataFrame:
    """Players on teams playing the second night of a B2B on `day` who played yesterday,
    with their rest probability."""
    y = day - timedelta(days=1)
    yday = pg.filter(pl.col("game_date") == y, pl.col("team").is_in(teams), pl.col("minutes") > 0)
    if yday.is_empty():
        return pl.DataFrame()
    rates = rest_rates(pg, day, season_id)
    return (yday.select("player_id", "player_name", "team", pl.col("minutes").alias("min_yday"))
            .join(rates.drop("player_name"), on="player_id")
            .filter(pl.col("p_sit_b2b") >= SIT_WATCH).sort("p_sit_b2b", descending=True))


def main() -> None:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    ap = argparse.ArgumentParser()
    ap.add_argument("--date", type=date.fromisoformat, default=datetime.now(ZoneInfo("America/New_York")).date())
    args = ap.parse_args()
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    r = rest_rates(pg, args.date, "22026")
    cols = ["player_name", "mpg", "b2b_games", "b2b_sat", "usual_miss", "rest_excess", "p_sit_b2b"]
    with pl.Config(tbl_rows=30, tbl_hide_column_data_types=True, float_precision=2, tbl_width_chars=160):
        print("League: most likely to sit a B2B second leg (players who played the first)")
        print(r.select(cols).head(25))
        from nba_ladders.market import TEAM_ABBR
        from nba_ladders.sources.injury_report import report_for_date

        rep = report_for_date(args.date).filter(pl.col("game_date") == args.date)
        abbr = {k.replace(" ", ""): v for k, v in TEAM_ABBR.items()}
        teams = [abbr[n] for n in rep["team_name"].drop_nulls().unique() if n in abbr]
        t = tonight(pg, args.date, teams)
        print(f"\nTonight ({args.date}) second-leg risks:" if t.height else "\nNo B2B rest risks tonight.")
        if t.height:
            print(t.select("player_name", "team", "min_yday", "b2b_games", "b2b_sat", "p_sit_b2b"))


if __name__ == "__main__":
    main()
