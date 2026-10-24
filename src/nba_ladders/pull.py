"""Pull league game logs and build data/processed/player_games.parquet.

    uv run python -m nba_ladders.pull --seasons 2023-24 2024-25 2025-26 2026-27 --refresh 2026-27
"""

import argparse

import polars as pl

from nba_ladders import config
from nba_ladders.player_games import parse_league_gamelog
from nba_ladders.sources.nba_stats import get_league_gamelog


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", required=True)
    ap.add_argument("--refresh", nargs="*", default=[], help="seasons to re-fetch (in progress)")
    args = ap.parse_args()

    frames = []
    for season in args.seasons:
        payload = get_league_gamelog(season, refresh=season in args.refresh)
        df = parse_league_gamelog(payload)
        print(f"{season}: {df.height:,} player-games, {df['game_id'].n_unique():,} games")
        frames.append(df)

    out = config.DATA_PROCESSED / "player_games.parquet"
    pl.concat(frames).write_parquet(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
