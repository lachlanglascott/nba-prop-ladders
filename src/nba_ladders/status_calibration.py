"""Calibrate P(sits | injury-report status) from past reports vs box scores.

For each game day, take the report published at ~5 PM ET (before most tip-offs) and check
whether each listed player appeared in that day's box score.

    uv run python -m nba_ladders.status_calibration --season-id 22025
"""

import argparse
import time
from datetime import date, datetime

import polars as pl

from nba_ladders import config
from nba_ladders.sources.injury_report import ET, fetch_report, parse_report, player_key

REPORT_TIMES = [(17, 0), (16, 30), (17, 30), (16, 0), (18, 0)]  # first available wins
STATUS_ORDER = ["Out", "Doubtful", "Questionable", "Probable", "Available"]
TEAM_ABBR = None  # filled lazily (avoids importing nba_api static at module import)


def day_report(d: date) -> pl.DataFrame | None:
    for h, m in REPORT_TIMES:
        ts = datetime(d.year, d.month, d.day, h, m, tzinfo=ET)
        path = fetch_report(ts)
        time.sleep(0.3)  # static CDN, but be polite
        if path:
            return parse_report(path, ts)
    return None


def outcomes(report: pl.DataFrame, pg_day: pl.DataFrame, d: date) -> pl.DataFrame:
    """Listed players on d with whether they played (box score), matched on team + name key."""
    global TEAM_ABBR
    if TEAM_ABBR is None:
        from nba_api.stats.static import teams

        TEAM_ABBR = {t["full_name"].replace(" ", ""): t["abbreviation"] for t in teams.get_teams()}
    listed = (
        report.filter(pl.col("game_date") == d, pl.col("player").is_not_null(),
                      pl.col("status").is_in(STATUS_ORDER))
        .with_columns(pl.col("team_name").replace_strict(TEAM_ABBR, default=None).alias("team"))
        .select("team", "player", "player_key", "status", "reason")
    )
    played = pg_day.select(
        "team", pl.col("player_name").map_elements(player_key, return_dtype=pl.String).alias("player_key"),
        pl.lit(True).alias("played"), "minutes",
    )
    teams_playing = pg_day["team"].unique().implode()
    return (
        listed.filter(pl.col("team").is_in(teams_playing))  # drop games postponed/not in data
        .join(played, on=["team", "player_key"], how="left")
        .with_columns(pl.col("played").fill_null(False), pl.lit(d).alias("game_date"))
    )


def calibrate(rows: pl.DataFrame) -> pl.DataFrame:
    return (
        rows.group_by("status")
        .agg(pl.len().alias("listed"), (~pl.col("played")).sum().alias("sat"),
             (~pl.col("played")).mean().alias("p_sits"),
             pl.col("minutes").mean().alias("min_if_played"))
        .with_columns(pl.col("status").cast(pl.Enum(STATUS_ORDER)))
        .sort("status")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season-id", default="22025")
    args = ap.parse_args()
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet").filter(
        pl.col("season_id") == args.season_id
    )
    days = pg["game_date"].unique().sort().to_list()
    out, missing = [], 0
    for i, d in enumerate(days, 1):
        rep = day_report(d)
        if rep is None or rep.is_empty():
            missing += 1
            continue
        out.append(outcomes(rep, pg.filter(pl.col("game_date") == d), d))
        if i % 25 == 0:
            print(f"{i}/{len(days)} days ({missing} without a report)", flush=True)
    rows = pl.concat(out)
    path = config.DATA_PROCESSED / f"status_outcomes_{args.season_id}.parquet"
    rows.write_parquet(path)
    print(f"wrote {path}: {rows.height:,} listings over {len(out)} days ({missing} days missing)")
    with pl.Config(tbl_hide_column_data_types=True):
        print(calibrate(rows).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
