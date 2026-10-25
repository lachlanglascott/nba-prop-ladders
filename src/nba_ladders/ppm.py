"""Per-minute report: a player's full-strength minutes and PPM/RPM/APM, and which current
teammates' absences unlock minutes or change his per-minute rates.

    uv run python -m nba_ladders.ppm "Tari Eason"
"""

import argparse

import polars as pl

from nba_ladders import config
from nba_ladders.absences import (
    RATES,
    context_effects,
    game_absences,
    pair_effects,
    priors,
    shrunk_effects,
)
from nba_ladders.scenarios import positions
from nba_ladders.screens import parse_player_index
from nba_ladders.sources.nba_stats import get_player_index

HISTORY_GAMES = 120


def report(name: str) -> None:
    pg = pl.read_parquet(config.DATA_PROCESSED / "player_games.parquet")
    pos = positions()
    pri = priors(pair_effects(pg, pos))
    eff = shrunk_effects(pair_effects(pg, pos), pri)
    me = pg.filter(pl.col("player_name") == name).sort("game_date").tail(HISTORY_GAMES)
    if me.is_empty():
        raise SystemExit(f"no games for {name!r}")
    pid = me["player_id"][0]
    ctx = context_effects(me, game_absences(pg), eff, pri, pos)
    h = me.join(ctx, on=["player_id", "game_id"]).with_columns(
        (pl.col("minutes") - pl.col("shift_min")).alias("base_min")
    )
    r10 = h.filter(pl.col("minutes") >= 10)
    tot_min = r10["minutes"].sum()
    print(f"{name}: last {h.height} games ({h['game_date'].min()} to {h['game_date'].max()})")
    print(f"  minutes: actual {h['minutes'].mean():.1f}, full-strength baseline "
          f"{h['base_min'].mean():.1f} (sd {h['minutes'].std():.1f}); "
          f"P(<=15) {(h['minutes'] <= 15).mean():.2f}, P(28+) {(h['minutes'] >= 28).mean():.2f}")
    print("  per minute (games 10+ min): " + ", ".join(
        f"{s.upper()[0]}PM {r10[s].sum() / tot_min:.3f} ({r10[s].sum() / tot_min * 36:.1f}/36)"
        for s in RATES))
    print(f"  games with >=4 min unlocked by absences: {(h['shift_min'] >= 4).sum()}")

    team = parse_player_index(get_player_index("2026-27")).filter(pl.col("player_id") == pid)
    team_now = team["team_now"][0] if team.height else me["team"][-1]
    roster = parse_player_index(get_player_index("2026-27")).filter(
        (pl.col("team_now") == team_now) & (pl.col("player_id") != pid)
    ).select(pl.col("player_id").alias("mate_id"), pl.col("player_name").alias("teammate"))
    mates = roster.join(eff.filter(pl.col("player_id") == pid), on="mate_id", how="left")
    with pl.Config(tbl_hide_column_data_types=True, tbl_rows=15, tbl_width_chars=140):
        print(f"\n  current {team_now} teammates — effect on {name} when they sit "
              "(shrunk; n_out = shared games they missed):")
        print(
            mates.drop_nulls("eff_min")
            .sort("eff_min", descending=True)
            .select(
                "teammate", pl.col("n_out").cast(pl.Int64), pl.col("mate_min").round(1),
                pl.col("eff_min").round(1).alias("+min"),
                *[(pl.col(f"eff_{s}") * 100).round(1).alias(f"{s.upper()[0]}PM %") for s in RATES],
            )
            .head(10)
        )
        no_hist = mates.filter(pl.col("eff_min").is_null())["teammate"].to_list()
        if no_hist:
            print(f"  no shared absences yet (league prior applies): {', '.join(no_hist[:12])}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("player")
    report(ap.parse_args().player)


if __name__ == "__main__":
    main()
