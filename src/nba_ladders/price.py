"""Fair prices for a player's ladder or custom combos in a scenario.

    uv run python -m nba_ladders.price "Andre Drummond" --starter-out
    uv run python -m nba_ladders.price "Yaxel Lendeborg" --combos 20p/10r/3a,20p/7r/3a,18p/10r/5a

Backup bigs use their backup-role games (with starter-out flags); anyone else uses all of
their games, with no starter-out scenario.
"""

import argparse

import polars as pl

from nba_ladders import config
from nba_ladders.backtest import HISTORY_GAMES
from nba_ladders.sim import combo_probs, fit_player, parse_combo, price_table, simulate_stats


def player_history(name: str) -> tuple[pl.DataFrame, str]:
    bg = pl.read_parquet(config.DATA_PROCESSED / "backup_big_games.parquet")
    hist = bg.filter(pl.col("player_name") == name)
    if hist.height:
        return hist.sort("game_date").tail(HISTORY_GAMES), "backup-role games"
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    hist = pg.filter(pl.col("player_name") == name)
    return hist.sort("game_date").tail(HISTORY_GAMES), "games (all roles)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player")
    ap.add_argument("--starter-out", action="store_true")
    ap.add_argument("--combos", help="comma-separated, e.g. 20p/10r/3a,18p/10r/5a")
    ap.add_argument("--sims", type=int, default=50_000)
    args = ap.parse_args()

    hist, kind = player_history(args.player)
    fit = fit_player(hist)
    if fit is None:
        raise SystemExit(f"not enough history for {args.player!r} ({hist.height} games)")
    stats = simulate_stats(fit, args.starter_out, args.sims)
    n_out = int(hist["starter_out"].sum()) if "starter_out" in hist.columns else 0
    print(f"{args.player}: {hist.height} {kind} "
          f"({hist['game_date'].min()} to {hist['game_date'].max()}), {n_out} with starter out")
    if hist.height < 20:
        print(f"WARNING: only {hist.height} games; prices rest on a tiny sample")
    m = stats["minutes"]
    print(f"scenario: starter {'OUT' if args.starter_out else 'plays'}; "
          f"sim minutes mean {m.mean():.1f}, P(25+) {(m >= 25).mean():.2f}")
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=20, fmt_str_lengths=60,
                   tbl_width_chars=160):
        if args.combos:
            combos = [parse_combo(c) for c in args.combos.split(",")]
            print(combo_probs(stats, combos).with_columns(pl.col(pl.Float64).round(3)))
        else:
            print(price_table(stats["pts"], stats["reb"]).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
