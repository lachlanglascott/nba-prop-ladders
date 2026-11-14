"""Fair prices for a player's ladder or custom combos in a scenario.

    uv run python -m nba_ladders.price "Andre Drummond" --starter-out
    uv run python -m nba_ladders.price "Yaxel Lendeborg" --combos 20p/10r/3a,20p/7r/3a,18p/10r/5a
    uv run python -m nba_ladders.price "Precious Achiuwa" --opponent NOP --as-of 2026-10-29

Backup bigs use their backup-role games (with starter-out flags); anyone else uses all of
their games, with no starter-out scenario. --as-of replays a past game using only earlier
data (luck uses the tracking snapshot taken before that game, if one exists).
"""

import argparse
from datetime import date
from functools import lru_cache

import polars as pl

from nba_ladders import config
from nba_ladders.backtest import HISTORY_GAMES
from nba_ladders.sim import combo_probs, fit_player, parse_combo, price_table, simulate_stats

MIN_BACKUP_GAMES = 20  # fewer backup-role games than this: use all his games instead


@lru_cache(maxsize=4)
def _frame(name: str) -> pl.DataFrame:
    """Processed tables, read once per process (pricing many players reuses them)."""
    return pl.read_parquet(config.DATA_PROCESSED / f"{name}.parquet")


@lru_cache(maxsize=4)
def _reb_table(as_of: date | None) -> pl.DataFrame:
    from nba_ladders.rebounding import live_multiplier_table
    from nba_ladders.scenarios import positions

    pg = _frame("player_games")
    return live_multiplier_table(pg.filter(pl.col("game_date") < as_of) if as_of else pg, positions())


def player_history(name: str, as_of: date | None = None) -> tuple[pl.DataFrame, str]:
    def before(df: pl.DataFrame) -> pl.DataFrame:
        return df.filter(pl.col("game_date") < as_of) if as_of else df

    pg = before(_frame("player_games"))
    bg = before(_frame("backup_big_games"))
    hist = bg.filter(pl.col("player_name") == name)
    latest_any = pg.filter(pl.col("player_name") == name)["game_date"].max()
    # Backup-role history only if it's his current role: its latest game must be his latest game.
    # (Al Horford's backup-role rows were all from 2021-22.)
    if hist.height >= MIN_BACKUP_GAMES and hist["game_date"].max() == latest_any:
        return hist.sort("game_date").tail(HISTORY_GAMES), "backup-role games"
    hist = pg.filter(pl.col("player_name") == name)
    return hist.sort("game_date").tail(HISTORY_GAMES), "games (all roles)"


def _today_et() -> date:
    from datetime import datetime
    from zoneinfo import ZoneInfo

    return datetime.now(ZoneInfo("America/New_York")).date()


def simulate_player(name: str, *, starter_out: bool = False, opponent: str | None = None,
                    as_of: date | None = None, luck: bool = True, n: int = 50_000,
                    verbose: bool = True, extra_minutes: float = 0.0,
                    rate_mult: dict | None = None,
                    history_adjust=None) -> tuple[dict, dict] | None:
    """Simulated stats for one player tonight (or as of a past date), with all live adjustments.

    Returns (stats, info) or None if there is too little history.
    history_adjust: callable(hist) -> hist, e.g. backing out past absences of current
    teammates (context_adjust); extra_minutes / rate_mult then add tonight's absences.
    """
    hist, kind = player_history(name, as_of)
    if history_adjust is not None and hist.height:
        hist = history_adjust(hist)
    fit = fit_player(hist)
    if fit is None:
        return None
    pid = int(hist["player_id"][0])
    info = {"player_id": pid, "games": hist.height, "kind": kind,
            "from": hist["game_date"].min(), "to": hist["game_date"].max(),
            "luck_ast": 1.0, "luck_reb": 1.0, "opp_reb": 1.0,
            "opp_pace": 1.0, "opp_pts": 1.0, "opp_ast": 1.0}
    if luck:
        from nba_ladders.luck import effective_luck, live_luck

        try:
            row = live_luck(as_of=as_of).filter(pl.col("player_id") == pid)
        except FileNotFoundError:
            row = pl.DataFrame()
            if verbose:
                print(f"no tracking snapshot for {as_of}; pricing without luck")
        if row.height:
            cur = (hist["season_id"] == hist["season_id"].max()).to_numpy()
            info["luck_ast"] = effective_luck(row["luck_ast"][0], fit.weights, cur)
            info["luck_reb"] = effective_luck(row["luck_reb"][0], fit.weights, cur)
            fit.ast_rate = fit.ast_rate * info["luck_ast"]
            fit.reb_rate = fit.reb_rate * info["luck_reb"]
    if opponent:
        from nba_ladders.rebounding import live_rebound_multiplier
        from nba_ladders.scenarios import positions

        pg_all = _frame("player_games")
        if as_of:
            pg_all = pg_all.filter(pl.col("game_date") < as_of)
        info["opp_reb"] = live_rebound_multiplier(pg_all, positions(), pid, opponent,
                                                  _reb_table(as_of))
        fit.reb_rate = fit.reb_rate * info["opp_reb"]
        # pace and defence: points and assists per minute vs this opponent
        from nba_ladders.absences import player_groups
        from nba_ladders.matchup import multipliers, player_multipliers

        grp = player_groups().filter(pl.col("player_id") == pid)["grp"]
        mm = (player_multipliers(opponent, grp[0], as_of or _today_et()) if grp.len()
              else multipliers(opponent, as_of or _today_et()))
        info.update(opp_pace=mm["pace"], opp_pts=mm["pts"], opp_ast=mm["ast"])
        fit.pts_rate = fit.pts_rate * mm["pts"]
        fit.ast_rate = fit.ast_rate * mm["ast"]
    if extra_minutes:  # e.g. minutes unlocked by tonight's absences (absences.context_effects)
        fit.role_shift += extra_minutes
    for s, m in (rate_mult or {}).items():
        setattr(fit, f"{s}_rate", getattr(fit, f"{s}_rate") * m)
    stats = simulate_stats(fit, starter_out, n)
    if verbose:
        n_out = int(hist["starter_out"].sum()) if "starter_out" in hist.columns else 0
        print(f"{name}: {hist.height} {kind} ({info['from']} to {info['to']}), {n_out} with starter out")
        if hist.height < 20:
            print(f"WARNING: only {hist.height} games; prices rest on a tiny sample")
        opp = (f", vs {opponent} rebounds x{info['opp_reb']:.3f}, pace x{info['opp_pace']:.3f}, "
               f"points x{info['opp_pts']:.3f}, assists x{info['opp_ast']:.3f}") if opponent else ""
        print(f"adjustments: luck AST x{info['luck_ast']:.3f} REB x{info['luck_reb']:.3f}{opp}")
        m = stats["minutes"]
        print(f"scenario: starter {'OUT' if starter_out else 'plays'}; "
              f"sim minutes mean {m.mean():.1f}, P(25+) {(m >= 25).mean():.2f}")
    return stats, info


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player")
    ap.add_argument("--starter-out", action="store_true")
    ap.add_argument("--combos", help="comma-separated, e.g. 20p/10r/3a,18p/10r/5a")
    ap.add_argument("--sims", type=int, default=50_000)
    ap.add_argument("--opponent", help="team abbreviation: applies the opponent rebound effect")
    ap.add_argument("--no-luck", action="store_true", help="skip the current-season luck adjustment")
    ap.add_argument("--as-of", type=date.fromisoformat, help="replay: only use data before this date")
    args = ap.parse_args()

    out = simulate_player(args.player, starter_out=args.starter_out, opponent=args.opponent,
                          as_of=args.as_of, luck=not args.no_luck, n=args.sims)
    if out is None:
        raise SystemExit(f"not enough history for {args.player!r}")
    stats, _ = out
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=20, fmt_str_lengths=60,
                   tbl_width_chars=160):
        if args.combos:
            combos = [parse_combo(c) for c in args.combos.split(",")]
            print(combo_probs(stats, combos).with_columns(pl.col(pl.Float64).round(3)))
        else:
            print(price_table(stats["pts"], stats["reb"]).with_columns(pl.col(pl.Float64).round(3)))


if __name__ == "__main__":
    main()
