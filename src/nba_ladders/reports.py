"""Write screen outputs to docs/screens/ for review.

    uv run python -m nba_ladders.reports
"""

from datetime import date

import polars as pl

from nba_ladders import config
from nba_ladders.screens import backup_bigs, parse_player_index
from nba_ladders.sources.nba_stats import get_player_index

OUT_DIR = config.ROOT / "docs" / "screens"

COLUMNS = {
    "player_name": "player",
    "team_now": "team",
    "moved": "moved",
    "position": "pos",
    "min_avg": "min",
    "min_sd": "min_sd",
    "p_min25": "p_25min",
    "pts36": "pts36",
    "reb36": "reb36",
    "pf36": "pf36",
    "n_big": "n_25min",
    "p_10_10_big": "dd_25min",
    "p_10_10_big_adj": "dd_25min_adj",
    "corr_pr": "corr_pr",
    "blocker": "blocker_last_season",
    "min_blocker_in": "min_blocker_in",
    "min_blocker_out": "min_blocker_out",
    "blocker_out_gp": "blocker_out_gp",
    "starter_now": "centre_ahead_now",
    "gp_now": "gp_now",
    "min_now": "min_now",
}


def main() -> None:
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    idx = parse_player_index(get_player_index("2026-27"))
    bb = backup_bigs(
        pg, idx, tier_season="22025", history_seasons=["22024", "22025"], current_season="22026"
    )
    out = bb.select(list(COLUMNS)).rename(COLUMNS).with_columns(pl.col(pl.Float64).round(2))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"backup_bigs_{date.today().isoformat()}.csv"
    out.write_csv(path)
    print(f"wrote {path} ({out.height} players)")


if __name__ == "__main__":
    main()
