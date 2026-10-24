"""Fair prices for a player's P+R ladder in a scenario.

    uv run python -m nba_ladders.price "Andre Drummond" --starter-out
"""

import argparse

import polars as pl

from nba_ladders import config
from nba_ladders.backtest import HISTORY_GAMES
from nba_ladders.sim import fit_player, price_table, simulate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player")
    ap.add_argument("--starter-out", action="store_true")
    ap.add_argument("--sims", type=int, default=50_000)
    args = ap.parse_args()

    bg = pl.read_parquet(config.DATA_PROCESSED / "backup_big_games.parquet")
    hist = bg.filter(pl.col("player_name") == args.player).sort("game_date").tail(HISTORY_GAMES)
    fit = fit_player(hist)
    if fit is None:
        raise SystemExit(f"not enough backup-role history for {args.player!r}")
    minutes, pts, reb = simulate(fit, args.starter_out, args.sims)
    n_out = int(hist["starter_out"].sum())
    print(f"{args.player}: {hist.height} backup-role games "
          f"({hist['game_date'].min()} to {hist['game_date'].max()}), {n_out} with starter out")
    print(f"scenario: starter {'OUT' if args.starter_out else 'plays'}; "
          f"sim minutes mean {minutes.mean():.1f}, P(25+) {(minutes >= 25).mean():.2f}")
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=20):
        print(price_table(pts, reb).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
