"""Player-tracking stats per game: potential assists, passes, rebound chances, contested boards.

stats.nba.com's LeagueDashPtStats only aggregates over a date range, so we query one game day
at a time (each player plays at most once a day, so GP = 1 rows are single-game numbers).
Cached at data/raw/nba_stats/ptstats/<measure>/<YYYY-MM-DD>.json; empty responses (the current
season's tracking lags) are not cached, so they are retried next run.

    uv run python -m nba_ladders.sources.tracking --seasons 2024-25 2025-26
"""

import argparse
import json
from datetime import date
from pathlib import Path

import polars as pl
import requests

from nba_ladders import config
from nba_ladders.sources.nba_stats import RAW_ROOT, _throttle

PT_DIR = RAW_ROOT / "ptstats"
MEASURES = ("Passing", "Rebounding")
SEASON_IDS = {"2019-20": "22019", "2020-21": "22020", "2021-22": "22021", "2022-23": "22022",
              "2023-24": "22023", "2024-25": "22024", "2025-26": "22025", "2026-27": "22026"}
KEEP = {
    "Passing": ["PASSES_MADE", "AST", "FT_AST", "SECONDARY_AST", "POTENTIAL_AST", "AST_PTS_CREATED"],
    "Rebounding": ["OREB_CHANCES", "DREB_CHANCES", "REB_CHANCES", "REB_CONTEST", "REB_UNCONTEST",
                   "REB_CHANCE_DEFER"],
}


def _fetch(season: str, measure: str, d: date) -> dict:
    from nba_api.stats.endpoints import leaguedashptstats

    _throttle()
    s = d.strftime("%m/%d/%Y")
    return leaguedashptstats.LeagueDashPtStats(
        season=season, pt_measure_type=measure, player_or_team="Player",
        per_mode_simple="Totals", date_from_nullable=s, date_to_nullable=s, timeout=60,
    ).get_dict()


def day_path(measure: str, d: date, raw_dir: Path = PT_DIR) -> Path:
    return raw_dir / measure.lower() / f"{d.isoformat()}.json"


def get_day(season: str, measure: str, d: date, *, raw_dir: Path = PT_DIR, fetch=_fetch) -> dict | None:
    path = day_path(measure, d, raw_dir)
    if path.exists():
        return json.loads(path.read_text())
    payload = fetch(season, measure, d)
    if not payload["resultSets"][0]["rowSet"]:
        return None  # not processed yet; don't cache
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))
    return payload


def parse_day(payload: dict, measure: str, d: date) -> pl.DataFrame:
    rs = payload["resultSets"][0]
    df = pl.DataFrame(rs["rowSet"], schema=rs["headers"], orient="row", infer_schema_length=None)
    return (
        df.filter(pl.col("GP") == 1)
        .select(pl.col("PLAYER_ID").alias("player_id"),
                *[pl.col(c).alias(c.lower()) for c in KEEP[measure]])
        .with_columns(pl.lit(d).alias("game_date"))
    )


def build(seasons: list[str], pg: pl.DataFrame, raw_dir: Path = PT_DIR, fetch=_fetch) -> pl.DataFrame:
    """Fetch (or read cached) every game day of the seasons; one row per player-game."""
    out = []
    for season in seasons:
        days = pg.filter(pl.col("season_id") == SEASON_IDS[season])["game_date"].unique().sort()
        missing = 0
        per_measure = {m: [] for m in MEASURES}
        for i, d in enumerate(days.to_list(), 1):
            for m in MEASURES:
                try:
                    payload = get_day(season, m, d, raw_dir=raw_dir, fetch=fetch)
                except (requests.RequestException, OSError, ValueError, KeyError) as err:
                    # network hiccup or block page: skip the day, retried next run
                    print(f"  {season} {m} {d}: {err!r}", flush=True)
                    payload = None
                if payload is None:
                    missing += 1
                    continue
                per_measure[m].append(parse_day(payload, m, d))
            if i % 25 == 0:
                print(f"{season}: {i}/{len(days)} days", flush=True)
        if not per_measure["Passing"] or not per_measure["Rebounding"]:
            print(f"{season}: no tracking data yet")
            continue
        both = pl.concat(per_measure["Passing"]).join(
            pl.concat(per_measure["Rebounding"]), on=["player_id", "game_date"], how="full",
            coalesce=True,
        )
        print(f"{season}: {both.height:,} player-games ({missing} day-measures missing)")
        out.append(both)
    return pl.concat(out) if out else pl.DataFrame()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", default=["2024-25", "2025-26"])
    args = ap.parse_args()
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    trk = build(args.seasons, pg)
    path = config.DATA_PROCESSED / "tracking.parquet"
    if path.exists():  # keep seasons not re-pulled this run
        old = pl.read_parquet(path)
        trk = pl.concat([old.join(trk.select("player_id", "game_date"), on=["player_id", "game_date"],
                                  how="anti"), trk], how="diagonal_relaxed")
    trk.write_parquet(path)
    print(f"wrote {path} ({trk.height:,} rows)")


if __name__ == "__main__":
    main()
